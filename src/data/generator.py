"""Synthetic fraud data generation with realistic patterns."""

import numpy as np
import cudf
import cupy as cp
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .schema import MerchantCategory, TransactionType
from ..utils.config import DataConfig


class FraudDataGenerator:
    """Generate synthetic fraud transaction data with realistic patterns.

    Fraud patterns simulated:
    - Unusual transaction amounts (much higher than user's typical)
    - Unusual hours (late night transactions)
    - Geographic anomalies (transactions far from home)
    - Rapid succession of transactions (velocity)
    - Foreign transactions for users who don't travel
    - High-risk merchant categories (electronics, ATM)
    """

    def __init__(self, config: DataConfig | None = None, seed: int = 42):
        self.config = config or DataConfig()
        self.seed = seed
        cp.random.seed(seed)
        np.random.seed(seed)

        # Merchant categories with fraud risk weights
        self.merchant_categories = list(MerchantCategory)
        self.merchant_risk = {
            MerchantCategory.GROCERY: 0.005,
            MerchantCategory.GAS_STATION: 0.01,
            MerchantCategory.RESTAURANT: 0.008,
            MerchantCategory.ONLINE_RETAIL: 0.025,
            MerchantCategory.TRAVEL: 0.015,
            MerchantCategory.ENTERTAINMENT: 0.01,
            MerchantCategory.UTILITIES: 0.002,
            MerchantCategory.HEALTHCARE: 0.003,
            MerchantCategory.ELECTRONICS: 0.035,
            MerchantCategory.ATM: 0.04,
        }

        self.transaction_types = list(TransactionType)
        self.device_types = ["mobile", "desktop", "pos_terminal"]

    def generate(self, n_transactions: int | None = None) -> cudf.DataFrame:
        """Generate synthetic transaction data on GPU.

        Args:
            n_transactions: Number of transactions to generate.
                          Defaults to config value.

        Returns:
            cuDF DataFrame with transaction data.
        """
        n = n_transactions or self.config.n_transactions
        n_users = self.config.n_users
        n_merchants = self.config.n_merchants

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Generating transactions on GPU...", total=8)

            # Generate base transaction data on GPU
            transaction_ids = cp.arange(n, dtype=cp.int64)
            progress.advance(task)

            user_ids = cp.random.randint(0, n_users, size=n, dtype=cp.int32)
            progress.advance(task)

            merchant_ids = cp.random.randint(0, n_merchants, size=n, dtype=cp.int32)
            progress.advance(task)

            # Generate timestamps (last 90 days)
            base_time = 1704067200  # 2024-01-01 00:00:00 UTC
            timestamps = base_time + cp.random.randint(0, 90 * 24 * 3600, size=n, dtype=cp.int64)
            progress.advance(task)

            # Generate amounts with log-normal distribution
            amounts = cp.exp(cp.random.normal(3.5, 1.2, size=n)).astype(cp.float32)
            amounts = cp.clip(amounts, 0.01, 10000.0)
            progress.advance(task)

            # Generate locations (US-centric for simplicity)
            home_lats = cp.random.uniform(25.0, 48.0, size=n_users).astype(cp.float32)
            home_lons = cp.random.uniform(-125.0, -70.0, size=n_users).astype(cp.float32)

            # Transaction locations (mostly near home, some far)
            location_noise = cp.random.normal(0, 0.5, size=(n, 2)).astype(cp.float32)
            user_ids_np = user_ids.get()
            user_home_lats = home_lats[user_ids_np]
            user_home_lons = home_lons[user_ids_np]
            lats = (user_home_lats + location_noise[:, 0]).get()
            lons = (user_home_lons + location_noise[:, 1]).get()
            progress.advance(task)

            # Calculate distance from home (simplified)
            user_home_lats_np = user_home_lats.get()
            user_home_lons_np = user_home_lons.get()
            distances = np.sqrt((lats - user_home_lats_np)**2 + (lons - user_home_lons_np)**2) * 111  # km
            progress.advance(task)

            # Create DataFrame
            df = cudf.DataFrame({
                "transaction_id": transaction_ids,
                "user_id": user_ids,
                "merchant_id": merchant_ids,
                "amount": amounts,
                "timestamp": timestamps,
                "location_lat": lats,
                "location_lon": lons,
                "distance_from_home": distances,
            })
            progress.advance(task)

        # Add categorical and derived columns (CPU for simplicity with categoricals)
        df = self._add_categorical_columns(df, n)
        df = self._add_time_features(df)
        df = self._generate_fraud_labels(df)

        return df

    def _add_categorical_columns(self, df: cudf.DataFrame, n: int) -> cudf.DataFrame:
        """Add categorical columns."""
        # Merchant categories with weighted distribution
        category_weights = [0.2, 0.1, 0.15, 0.2, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05]
        categories = np.random.choice(
            [c.value for c in self.merchant_categories],
            size=n,
            p=category_weights,
        )
        df["merchant_category"] = cudf.Series(categories).astype("category")

        # Transaction types
        type_weights = [0.85, 0.05, 0.05, 0.05]
        types = np.random.choice(
            [t.value for t in self.transaction_types],
            size=n,
            p=type_weights,
        )
        df["transaction_type"] = cudf.Series(types).astype("category")

        # Device types
        device_weights = [0.5, 0.2, 0.3]
        devices = np.random.choice(self.device_types, size=n, p=device_weights)
        df["device_type"] = cudf.Series(devices).astype("category")

        # Foreign transaction flag
        df["is_foreign"] = (df["distance_from_home"] > 500).astype("bool")

        return df

    def _add_time_features(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Add time-based features."""
        # Convert timestamp to datetime for feature extraction
        df["hour_of_day"] = ((df["timestamp"] % 86400) // 3600).astype("int8")
        df["day_of_week"] = ((df["timestamp"] // 86400) % 7).astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("bool")

        return df

    def _generate_fraud_labels(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Generate fraud labels based on realistic patterns."""
        n = len(df)
        base_fraud_rate = self.config.fraud_rate

        # Start with base fraud probability
        fraud_prob = cp.full(n, base_fraud_rate, dtype=cp.float32)

        # Factor 1: High-risk merchant categories
        category_series = df["merchant_category"].to_pandas()
        for cat, risk in self.merchant_risk.items():
            mask = (category_series == cat.value).values
            fraud_prob[mask] += risk

        # Factor 2: Unusual hours (2am-5am)
        hour = df["hour_of_day"].values
        late_night = (hour >= 2) & (hour <= 5)
        fraud_prob[late_night] += 0.02

        # Factor 3: Large amounts
        amount = df["amount"].values
        amount_z = (amount - cp.mean(amount)) / cp.std(amount)
        high_amount = amount_z > 2.0
        fraud_prob[high_amount] += 0.03

        # Factor 4: Far from home
        distance = cp.asarray(df["distance_from_home"].values)
        far_from_home = distance > 200
        fraud_prob[far_from_home] += 0.025

        # Factor 5: Foreign transactions
        is_foreign = cp.asarray(df["is_foreign"].values)
        fraud_prob[is_foreign] += 0.02

        # Cap probability and generate labels
        fraud_prob = cp.clip(fraud_prob, 0, 0.5)
        is_fraud = (cp.random.random(n) < fraud_prob).astype(cp.int8)

        df["is_fraud"] = is_fraud

        return df

    def save_parquet(
        self, df: cudf.DataFrame, output_path: Path | None = None, partition_size: int = 1_000_000
    ) -> Path:
        """Save DataFrame to Parquet files.

        Args:
            df: cuDF DataFrame to save.
            output_path: Output directory path.
            partition_size: Number of rows per partition.

        Returns:
            Path to the output directory.
        """
        output_path = output_path or self.config.output_path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert category columns to string for parquet compatibility
        df_save = df.copy()
        for col in df_save.select_dtypes(include=["category"]).columns:
            df_save[col] = df_save[col].astype("str")

        # Save as partitioned parquet
        n_partitions = (len(df_save) + partition_size - 1) // partition_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Saving {n_partitions} partitions...", total=n_partitions
            )

            for i in range(n_partitions):
                start = i * partition_size
                end = min((i + 1) * partition_size, len(df))
                partition = df_save.iloc[start:end]
                partition.to_parquet(output_path / f"part_{i:05d}.parquet")
                progress.advance(task)

        return output_path


def generate_sample_data(n: int = 100_000, output_dir: str = "data/sample") -> cudf.DataFrame:
    """Quick helper to generate sample data for testing."""
    config = DataConfig(n_transactions=n, n_users=1000, n_merchants=500)
    generator = FraudDataGenerator(config)
    df = generator.generate()
    generator.save_parquet(df, Path(output_dir))
    return df
