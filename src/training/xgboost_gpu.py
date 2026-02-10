"""GPU-accelerated XGBoost training for fraud detection."""

import json
from pathlib import Path
from typing import Any

import cudf
import cupy as cp
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..utils.config import TrainingConfig
from ..features.engineering import FeatureEngineer

console = Console()


class XGBoostTrainer:
    """GPU-accelerated XGBoost trainer for fraud detection.

    Uses XGBoost's native GPU support via `gpu_hist` tree method
    for fast training on NVIDIA GPUs.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.model: xgb.Booster | None = None
        self.feature_columns: list[str] = []
        self.training_history: dict[str, Any] = {}

    def prepare_data(
        self,
        df: cudf.DataFrame,
        feature_columns: list[str] | None = None,
        target_column: str = "is_fraud",
    ) -> tuple[xgb.DMatrix, xgb.DMatrix, cudf.DataFrame, cudf.DataFrame]:
        """Prepare data for XGBoost training.

        Args:
            df: Feature-engineered cuDF DataFrame.
            feature_columns: List of feature column names.
            target_column: Name of target column.

        Returns:
            Tuple of (train DMatrix, test DMatrix, train DataFrame, test DataFrame).
        """
        if feature_columns is None:
            engineer = FeatureEngineer()
            feature_columns = engineer.get_feature_columns()

        self.feature_columns = feature_columns

        # Handle categorical columns - convert to numeric codes
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=["category"]).columns.tolist()

        for col in categorical_cols:
            if col in feature_columns:
                df_processed[col] = df_processed[col].cat.codes.astype("float32")

        # Convert boolean to int
        bool_cols = df_processed.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_cols:
            if col in feature_columns:
                df_processed[col] = df_processed[col].astype("int8")

        # Filter to existing columns
        available_features = [c for c in feature_columns if c in df_processed.columns]
        console.print(f"[cyan]Using {len(available_features)} features[/cyan]")

        # Split data
        X = df_processed[available_features]
        y = df_processed[target_column]

        # Convert to numpy for sklearn split
        X_np = X.to_pandas().values
        y_np = y.to_pandas().values

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=self.config.test_size,
            stratify=y_np,
            random_state=42,
        )

        # Create DMatrix objects with GPU support
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=available_features)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=available_features)

        # Create DataFrames for later analysis
        train_df = cudf.DataFrame(X_train, columns=available_features)
        train_df[target_column] = y_train
        test_df = cudf.DataFrame(X_test, columns=available_features)
        test_df[target_column] = y_test

        console.print(f"[green]Train size: {len(X_train):,}, Test size: {len(X_test):,}[/green]")
        console.print(f"[green]Fraud rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}[/green]")

        return dtrain, dtest, train_df, test_df

    def train(
        self,
        dtrain: xgb.DMatrix,
        dtest: xgb.DMatrix,
        num_boost_round: int | None = None,
    ) -> xgb.Booster:
        """Train XGBoost model on GPU.

        Args:
            dtrain: Training DMatrix.
            dtest: Test DMatrix for evaluation.
            num_boost_round: Number of boosting rounds.

        Returns:
            Trained XGBoost Booster.
        """
        params = self.config.xgb_params.copy()

        # Remove sklearn-style params for xgb.train
        n_estimators = params.pop("n_estimators", 1000)
        num_boost_round = num_boost_round or n_estimators

        console.print("\n[bold cyan]Training XGBoost on GPU...[/bold cyan]")
        console.print(f"Parameters: {json.dumps(params, indent=2)}")

        evals = [(dtrain, "train"), (dtest, "eval")]
        evals_result: dict = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("Training in progress...", total=None)

            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                evals_result=evals_result,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=100,
            )

        self.training_history = evals_result

        # Print final metrics
        best_iteration = self.model.best_iteration
        final_train_auc = evals_result["train"]["auc"][best_iteration]
        final_eval_auc = evals_result["eval"]["auc"][best_iteration]

        console.print(f"\n[bold green]Training complete![/bold green]")
        console.print(f"Best iteration: {best_iteration}")
        console.print(f"Train AUC: {final_train_auc:.4f}")
        console.print(f"Eval AUC: {final_eval_auc:.4f}")

        return self.model

    def predict(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        """Generate predictions using the trained model.

        Args:
            dmatrix: XGBoost DMatrix with features.

        Returns:
            Array of fraud probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(dmatrix)

    def get_feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover').

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.get_score(importance_type=importance_type)

        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def save_model(self, path: Path | None = None) -> Path:
        """Save the trained model.

        Args:
            path: Output path for the model.

        Returns:
            Path where model was saved.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        path = path or self.config.model_path / "xgboost_fraud.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))
        console.print(f"[green]Model saved to {path}[/green]")

        # Save feature columns
        feature_path = path.parent / "feature_columns.json"
        with open(feature_path, "w") as f:
            json.dump(self.feature_columns, f)

        return path

    def load_model(self, path: Path) -> xgb.Booster:
        """Load a trained model.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded XGBoost Booster.
        """
        path = Path(path)
        self.model = xgb.Booster()
        self.model.load_model(str(path))

        # Load feature columns if available
        feature_path = path.parent / "feature_columns.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_columns = json.load(f)

        console.print(f"[green]Model loaded from {path}[/green]")
        return self.model


def compare_cpu_vs_gpu_training(df: cudf.DataFrame, feature_columns: list[str]) -> dict:
    """Benchmark CPU vs GPU XGBoost training.

    Returns timing comparison.
    """
    import time

    results = {}

    # Prepare data once
    trainer = XGBoostTrainer()
    dtrain, dtest, _, _ = trainer.prepare_data(df, feature_columns)

    # GPU training
    console.print("\n[bold cyan]GPU Training (gpu_hist)[/bold cyan]")
    gpu_params = {
        "tree_method": "gpu_hist",
        "device": "cuda",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
    }

    start = time.perf_counter()
    gpu_model = xgb.train(
        gpu_params, dtrain, num_boost_round=100,
        evals=[(dtest, "eval")], verbose_eval=False
    )
    gpu_time = time.perf_counter() - start
    results["gpu_seconds"] = gpu_time
    console.print(f"[green]GPU time: {gpu_time:.2f}s[/green]")

    # CPU training
    console.print("\n[bold cyan]CPU Training (hist)[/bold cyan]")
    cpu_params = {
        "tree_method": "hist",
        "device": "cpu",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
    }

    start = time.perf_counter()
    cpu_model = xgb.train(
        cpu_params, dtrain, num_boost_round=100,
        evals=[(dtest, "eval")], verbose_eval=False
    )
    cpu_time = time.perf_counter() - start
    results["cpu_seconds"] = cpu_time
    console.print(f"[green]CPU time: {cpu_time:.2f}s[/green]")

    results["speedup"] = cpu_time / gpu_time
    console.print(f"\n[bold yellow]GPU Speedup: {results['speedup']:.1f}x[/bold yellow]")

    return results
