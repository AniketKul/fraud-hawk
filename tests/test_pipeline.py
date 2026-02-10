"""Tests for the fraud detection pipeline."""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGeneration:
    """Test data generation module."""

    def test_generator_creates_correct_columns(self):
        """Test that generator creates all expected columns."""
        from src.data.generator import FraudDataGenerator
        from src.data.schema import TransactionSchema
        from src.utils.config import DataConfig

        config = DataConfig(n_transactions=1000, n_users=100, n_merchants=50)
        generator = FraudDataGenerator(config)
        df = generator.generate()

        # Check key columns exist
        assert "transaction_id" in df.columns
        assert "user_id" in df.columns
        assert "amount" in df.columns
        assert "is_fraud" in df.columns
        assert "timestamp" in df.columns

    def test_generator_fraud_rate(self):
        """Test that fraud rate is approximately correct."""
        from src.data.generator import FraudDataGenerator
        from src.utils.config import DataConfig

        target_rate = 0.02
        config = DataConfig(
            n_transactions=100000, n_users=1000, n_merchants=100, fraud_rate=target_rate
        )
        generator = FraudDataGenerator(config)
        df = generator.generate()

        actual_rate = df["is_fraud"].mean()
        # Allow some variance due to additional fraud factors
        assert 0.01 < actual_rate < 0.10

    def test_generator_amount_range(self):
        """Test that amounts are within expected range."""
        from src.data.generator import FraudDataGenerator
        from src.utils.config import DataConfig

        config = DataConfig(n_transactions=10000, n_users=100, n_merchants=50)
        generator = FraudDataGenerator(config)
        df = generator.generate()

        amounts = df["amount"].to_pandas()
        assert amounts.min() >= 0.01
        assert amounts.max() <= 10000


class TestFeatureEngineering:
    """Test feature engineering module."""

    def test_feature_columns_list(self):
        """Test that feature columns are defined."""
        from src.features.engineering import FeatureEngineer

        engineer = FeatureEngineer()
        columns = engineer.get_feature_columns()

        assert len(columns) > 10
        assert "amount" in columns
        assert "hour_of_day" in columns

    def test_compute_features_adds_columns(self):
        """Test that feature computation adds expected columns."""
        from src.data.generator import FraudDataGenerator
        from src.features.engineering import FeatureEngineer
        from src.utils.config import DataConfig

        config = DataConfig(n_transactions=5000, n_users=100, n_merchants=50)
        generator = FraudDataGenerator(config)
        df = generator.generate()

        engineer = FeatureEngineer()
        df_features = engineer.compute_features(df)

        # Check that new features were added
        assert "user_avg_amount" in df_features.columns
        assert "merchant_risk_score" in df_features.columns
        assert "log_amount" in df_features.columns


class TestTraining:
    """Test training module."""

    def test_trainer_prepare_data(self):
        """Test data preparation for training."""
        from src.data.generator import FraudDataGenerator
        from src.features.engineering import FeatureEngineer
        from src.training.xgboost_gpu import XGBoostTrainer
        from src.utils.config import DataConfig

        # Generate small dataset
        config = DataConfig(n_transactions=5000, n_users=100, n_merchants=50)
        generator = FraudDataGenerator(config)
        df = generator.generate()

        # Compute features
        engineer = FeatureEngineer()
        df = engineer.compute_features(df)

        # Prepare training data
        trainer = XGBoostTrainer()
        dtrain, dtest, train_df, test_df = trainer.prepare_data(df)

        assert dtrain.num_row() > 0
        assert dtest.num_row() > 0
        assert len(train_df) > len(test_df)  # 80/20 split


class TestInference:
    """Test inference module."""

    def test_prediction_result_dataclass(self):
        """Test PredictionResult dataclass."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            transaction_id=123,
            fraud_probability=0.75,
            is_fraud=True,
            latency_ms=1.5,
        )

        assert result.transaction_id == 123
        assert result.fraud_probability == 0.75
        assert result.is_fraud is True


class TestConfig:
    """Test configuration module."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.utils.config import PipelineConfig

        config = PipelineConfig()

        assert config.data.n_transactions == 10_000_000
        assert config.data.fraud_rate == 0.015
        assert config.training.test_size == 0.2

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        from src.utils.config import PipelineConfig

        monkeypatch.setenv("N_TRANSACTIONS", "5000000")
        monkeypatch.setenv("FRAUD_RATE", "0.02")

        config = PipelineConfig.from_env()

        assert config.data.n_transactions == 5000000
        assert config.data.fraud_rate == 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
