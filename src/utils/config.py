"""Configuration management for the fraud detection pipeline."""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data generation and storage."""
    n_transactions: int = 10_000_000
    n_users: int = 100_000
    n_merchants: int = 10_000
    fraud_rate: float = 0.015  # 1.5% fraud rate
    output_path: Path = field(default_factory=lambda: Path("data/transactions"))


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rolling_windows: list[int] = field(default_factory=lambda: [1, 7, 30])  # days
    velocity_windows: list[int] = field(default_factory=lambda: [1, 6, 24])  # hours
    output_path: Path = field(default_factory=lambda: Path("data/features"))


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    n_folds: int = 5
    early_stopping_rounds: int = 50

    # XGBoost GPU parameters
    xgb_params: dict = field(default_factory=lambda: {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "max_depth": 8,
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 66,  # ~1/fraud_rate for class imbalance
        "random_state": 42,
    })

    model_path: Path = field(default_factory=lambda: Path("models"))


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    batch_size: int = 1024
    model_path: Path = field(default_factory=lambda: Path("models/xgboost_fraud.json"))
    threshold: float = 0.5


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # GPU memory settings
    gpu_memory_limit: str = "60GB"  # H100 has 80GB

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        import os
        config = cls()

        if n_txn := os.getenv("N_TRANSACTIONS"):
            config.data.n_transactions = int(n_txn)
        if fraud_rate := os.getenv("FRAUD_RATE"):
            config.data.fraud_rate = float(fraud_rate)
        if gpu_mem := os.getenv("GPU_MEMORY_LIMIT"):
            config.gpu_memory_limit = gpu_mem

        return config
