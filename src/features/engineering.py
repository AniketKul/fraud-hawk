"""GPU-accelerated feature engineering using cuDF."""

import cudf
import cupy as cp
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..utils.config import FeatureConfig
from ..data.schema import TransactionSchema

console = Console()


class FeatureEngineer:
    """GPU-accelerated feature engineering for fraud detection.

    Features computed:
    - Rolling aggregations (transaction counts, amounts)
    - Velocity features (transactions per time window)
    - Time-based features (hour patterns, day patterns)
    - Categorical encodings (target encoding for merchants)
    - User behavior statistics
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def compute_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Compute all features for the transaction DataFrame.

        Args:
            df: Input cuDF DataFrame with raw transaction data.

        Returns:
            cuDF DataFrame with all features added.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Computing features on GPU...", total=5)

            # Sort by user and timestamp for rolling calculations
            df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            progress.advance(task)

            # Compute user-level aggregations
            df = self._compute_user_aggregations(df)
            progress.advance(task)

            # Compute velocity features
            df = self._compute_velocity_features(df)
            progress.advance(task)

            # Compute merchant risk scores
            df = self._compute_merchant_features(df)
            progress.advance(task)

            # Compute amount deviation features
            df = self._compute_amount_features(df)
            progress.advance(task)

        return df

    def _compute_user_aggregations(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Compute user-level aggregation features."""
        # User transaction history stats
        user_stats = df.groupby("user_id").agg({
            "amount": ["mean", "std", "max", "count"],
            "is_foreign": "sum",
            "distance_from_home": ["mean", "max"],
        })

        # Flatten column names
        user_stats.columns = [
            "user_avg_amount",
            "user_std_amount",
            "user_max_amount",
            "user_txn_count",
            "user_foreign_count",
            "user_avg_distance",
            "user_max_distance",
        ]
        user_stats = user_stats.reset_index()

        # Merge back to main DataFrame
        df = df.merge(user_stats, on="user_id", how="left")

        # Compute amount deviation from user's mean
        df["amount_deviation"] = (
            (df["amount"] - df["user_avg_amount"]) / (df["user_std_amount"] + 1e-6)
        ).astype("float32")

        # Is this amount unusually high for user?
        df["is_high_amount"] = (df["amount_deviation"] > 2.0).astype("int8")

        return df

    def _compute_velocity_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Compute transaction velocity features.

        Velocity = number of transactions in recent time windows.
        """
        # For each velocity window (in hours), count recent transactions
        for hours in self.config.velocity_windows:
            window_seconds = hours * 3600

            # Count transactions in window using a self-join approach
            # This is a simplified version - in production you'd use proper window functions
            feature_name = f"txn_count_{hours}h"

            # Group by user and count transactions within time buckets
            df["time_bucket"] = df["timestamp"] // window_seconds
            velocity = df.groupby(["user_id", "time_bucket"]).size().reset_index(name=feature_name)
            df = df.merge(velocity, on=["user_id", "time_bucket"], how="left")
            df = df.drop(columns=["time_bucket"])

        # High velocity flag
        df["high_velocity"] = (df["txn_count_1h"] > 5).astype("int8")

        return df

    def _compute_merchant_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Compute merchant-level features including risk scores."""
        # Merchant fraud rate (target encoding)
        merchant_stats = df.groupby("merchant_id").agg({
            "is_fraud": "mean",
            "amount": ["mean", "count"],
        })
        merchant_stats.columns = ["merchant_fraud_rate", "merchant_avg_amount", "merchant_txn_count"]
        merchant_stats = merchant_stats.reset_index()

        # Smooth merchant fraud rate with global average (to handle low-volume merchants)
        global_fraud_rate = df["is_fraud"].mean()
        smoothing_factor = 100  # Minimum transactions for full weight

        merchant_stats["merchant_risk_score"] = (
            (merchant_stats["merchant_txn_count"] * merchant_stats["merchant_fraud_rate"] +
             smoothing_factor * global_fraud_rate) /
            (merchant_stats["merchant_txn_count"] + smoothing_factor)
        ).astype("float32")

        df = df.merge(
            merchant_stats[["merchant_id", "merchant_risk_score", "merchant_avg_amount"]],
            on="merchant_id",
            how="left"
        )

        return df

    def _compute_amount_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Compute amount-based features."""
        # Log amount (useful for skewed distributions)
        df["log_amount"] = cp.log1p(df["amount"].values)

        # Amount percentile bins
        amount_quantiles = df["amount"].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_pandas()

        def assign_amount_bin(amount_series):
            bins = cp.array([-cp.inf, amount_quantiles[0.25], amount_quantiles[0.5],
                    amount_quantiles[0.75], amount_quantiles[0.9],
                    amount_quantiles[0.95], amount_quantiles[0.99], cp.inf])
            return cp.digitize(amount_series.values, bins)

        df["amount_bin"] = assign_amount_bin(df["amount"])

        # Ratio to merchant average
        df["amount_vs_merchant_avg"] = (
            df["amount"] / (df["merchant_avg_amount"] + 1e-6)
        ).astype("float32")

        return df

    def get_feature_columns(self) -> list[str]:
        """Get list of feature columns for model training."""
        base_features = [
            # Original features
            "amount",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "distance_from_home",
            "is_foreign",
            # User features
            "user_avg_amount",
            "user_std_amount",
            "user_max_amount",
            "user_txn_count",
            "user_foreign_count",
            "user_avg_distance",
            "user_max_distance",
            "amount_deviation",
            "is_high_amount",
            # Velocity features
            "txn_count_1h",
            "txn_count_6h",
            "txn_count_24h",
            "high_velocity",
            # Merchant features
            "merchant_risk_score",
            # Amount features
            "log_amount",
            "amount_bin",
            "amount_vs_merchant_avg",
        ]
        return base_features

    def save_features(self, df: cudf.DataFrame, output_path: Path | None = None) -> Path:
        """Save feature-engineered data to parquet."""
        output_path = output_path or self.config.output_path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full dataset
        df.to_parquet(output_path / "features.parquet")

        console.print(f"[green]Features saved to {output_path}[/green]")
        return output_path


def compare_pandas_vs_cudf(n_rows: int = 1_000_000) -> dict:
    """Benchmark cuDF vs Pandas for feature engineering.

    Returns timing comparison for key operations.
    """
    import pandas as pd
    import time

    from ..data.generator import FraudDataGenerator
    from ..utils.config import DataConfig

    results = {}

    # Generate data
    config = DataConfig(n_transactions=n_rows, n_users=10000, n_merchants=1000)
    generator = FraudDataGenerator(config)

    console.print(f"\n[bold]Benchmarking with {n_rows:,} rows[/bold]\n")

    # cuDF feature engineering
    console.print("[cyan]Running cuDF feature engineering...[/cyan]")
    df_gpu = generator.generate()

    start = time.perf_counter()
    engineer = FeatureEngineer()
    df_gpu_features = engineer.compute_features(df_gpu)
    cudf_time = time.perf_counter() - start
    results["cudf_seconds"] = cudf_time

    console.print(f"[green]cuDF: {cudf_time:.2f}s[/green]")

    # Convert to Pandas and run equivalent operations
    console.print("[cyan]Running Pandas feature engineering...[/cyan]")
    df_cpu = df_gpu.to_pandas()

    start = time.perf_counter()
    # Simplified Pandas operations for comparison
    df_cpu = df_cpu.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    user_stats = df_cpu.groupby("user_id")["amount"].agg(["mean", "std", "max", "count"])
    df_cpu = df_cpu.merge(user_stats, left_on="user_id", right_index=True, how="left")
    pandas_time = time.perf_counter() - start
    results["pandas_seconds"] = pandas_time

    console.print(f"[green]Pandas: {pandas_time:.2f}s[/green]")
    console.print(f"\n[bold yellow]Speedup: {pandas_time/cudf_time:.1f}x[/bold yellow]")

    results["speedup"] = pandas_time / cudf_time
    results["n_rows"] = n_rows

    return results
