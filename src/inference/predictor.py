"""GPU-accelerated fraud prediction."""

import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from rich.console import Console

from ..utils.config import InferenceConfig

console = Console()


@dataclass
class PredictionResult:
    """Result of a fraud prediction."""
    transaction_id: str | int
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


@dataclass
class BatchPredictionResult:
    """Result of batch fraud prediction."""
    predictions: list[PredictionResult]
    total_latency_ms: float
    avg_latency_ms: float
    throughput_per_sec: float


class FraudPredictor:
    """GPU-accelerated fraud prediction using XGBoost.

    Supports both single and batch predictions with automatic
    batching for optimal GPU utilization.
    """

    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self.model: xgb.Booster | None = None
        self.feature_columns: list[str] = []
        self.threshold = self.config.threshold

    def load_model(self, model_path: Path | None = None) -> None:
        """Load the trained XGBoost model.

        Args:
            model_path: Path to the saved model.
        """
        model_path = model_path or self.config.model_path
        model_path = Path(model_path)

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Load feature columns
        feature_path = model_path.parent / "feature_columns.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_columns = json.load(f)

        console.print(f"[green]Model loaded from {model_path}[/green]")

    def predict_single(
        self,
        features: dict[str, float],
        transaction_id: str | int = 0,
    ) -> PredictionResult:
        """Predict fraud probability for a single transaction.

        Args:
            features: Dictionary of feature values.
            transaction_id: Optional transaction identifier.

        Returns:
            PredictionResult with fraud probability and decision.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        start = time.perf_counter()

        # Create feature array
        feature_array = np.array([
            [features.get(col, 0.0) for col in self.feature_columns]
        ], dtype=np.float32)

        dmatrix = xgb.DMatrix(feature_array, feature_names=self.feature_columns)
        probability = float(self.model.predict(dmatrix)[0])

        latency_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            transaction_id=transaction_id,
            fraud_probability=probability,
            is_fraud=probability >= self.threshold,
            latency_ms=latency_ms,
        )

    def predict_batch(
        self,
        batch: list[dict[str, float]],
        transaction_ids: list[str | int] | None = None,
    ) -> BatchPredictionResult:
        """Predict fraud probabilities for a batch of transactions.

        Args:
            batch: List of feature dictionaries.
            transaction_ids: Optional list of transaction identifiers.

        Returns:
            BatchPredictionResult with all predictions and timing.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if transaction_ids is None:
            transaction_ids = list(range(len(batch)))

        start = time.perf_counter()

        # Create feature matrix
        feature_matrix = np.array([
            [txn.get(col, 0.0) for col in self.feature_columns]
            for txn in batch
        ], dtype=np.float32)

        dmatrix = xgb.DMatrix(feature_matrix, feature_names=self.feature_columns)
        probabilities = self.model.predict(dmatrix)

        total_latency_ms = (time.perf_counter() - start) * 1000

        # Create individual results
        predictions = [
            PredictionResult(
                transaction_id=tid,
                fraud_probability=float(prob),
                is_fraud=prob >= self.threshold,
                latency_ms=total_latency_ms / len(batch),
            )
            for tid, prob in zip(transaction_ids, probabilities)
        ]

        return BatchPredictionResult(
            predictions=predictions,
            total_latency_ms=total_latency_ms,
            avg_latency_ms=total_latency_ms / len(batch),
            throughput_per_sec=len(batch) / (total_latency_ms / 1000),
        )

    def set_threshold(self, threshold: float) -> None:
        """Update the fraud decision threshold.

        Args:
            threshold: New probability threshold (0-1).
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold


def benchmark_inference(
    predictor: FraudPredictor,
    n_samples: int = 10000,
    batch_sizes: list[int] | None = None,
) -> dict:
    """Benchmark inference performance across batch sizes.

    Args:
        predictor: Loaded FraudPredictor.
        n_samples: Total samples to process.
        batch_sizes: List of batch sizes to test.

    Returns:
        Dictionary with benchmark results.
    """
    if batch_sizes is None:
        batch_sizes = [1, 32, 128, 512, 1024, 4096]

    # Generate random test data
    np.random.seed(42)
    test_data = [
        {col: np.random.randn() for col in predictor.feature_columns}
        for _ in range(n_samples)
    ]

    results = {}

    console.print(f"\n[bold cyan]Inference Benchmark ({n_samples:,} samples)[/bold cyan]")

    for batch_size in batch_sizes:
        total_time = 0
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(0, n_samples, batch_size):
            batch = test_data[i:i + batch_size]
            result = predictor.predict_batch(batch)
            total_time += result.total_latency_ms

        throughput = n_samples / (total_time / 1000)
        avg_latency = total_time / n_samples

        results[batch_size] = {
            "throughput_per_sec": throughput,
            "avg_latency_ms": avg_latency,
            "total_time_ms": total_time,
        }

        console.print(
            f"  Batch size {batch_size:>4}: "
            f"{throughput:>10,.0f} txn/sec, "
            f"{avg_latency:>6.3f} ms/txn"
        )

    return results
