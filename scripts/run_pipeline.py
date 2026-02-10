#!/usr/bin/env python
"""Main pipeline orchestration for the fraud detection system."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PipelineConfig
from src.utils.metrics import BenchmarkResults, timed_operation

console = Console()


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """GPU-accelerated Fraud Detection Pipeline.

    Run the complete pipeline or individual stages.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = PipelineConfig.from_env()


@cli.command()
@click.option("--n-transactions", "-n", type=int, default=10_000_000, help="Number of transactions")
@click.option("--output", "-o", type=click.Path(), default="data/transactions", help="Output path")
@click.pass_context
def generate(ctx: click.Context, n_transactions: int, output: str) -> None:
    """Generate synthetic fraud transaction data."""
    from src.data.generator import FraudDataGenerator
    from src.utils.config import DataConfig

    console.print(Panel.fit(
        f"[bold cyan]Generating {n_transactions:,} transactions[/bold cyan]",
        title="Data Generation"
    ))

    config = DataConfig(n_transactions=n_transactions, output_path=Path(output))
    generator = FraudDataGenerator(config)

    results = BenchmarkResults()

    with timed_operation("Generate transactions", n_transactions, results):
        df = generator.generate()

    console.print(f"\nDataset statistics:")
    console.print(f"  Total transactions: {len(df):,}")
    console.print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    console.print(f"  Unique users: {df['user_id'].nunique():,}")
    console.print(f"  Unique merchants: {df['merchant_id'].nunique():,}")

    with timed_operation("Save to parquet", len(df), results):
        generator.save_parquet(df, Path(output))

    results.print_summary()


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), default="data/transactions")
@click.option("--output", "-o", type=click.Path(), default="data/features", help="Output path")
@click.pass_context
def features(ctx: click.Context, input_path: str, output: str) -> None:
    """Compute features from raw transaction data."""
    import cudf
    from src.features.engineering import FeatureEngineer
    from src.utils.config import FeatureConfig

    console.print(Panel.fit(
        "[bold cyan]Computing GPU-accelerated features[/bold cyan]",
        title="Feature Engineering"
    ))

    results = BenchmarkResults()

    # Load data
    with timed_operation("Load parquet data", results=results):
        df = cudf.read_parquet(input_path)

    n_rows = len(df)
    console.print(f"Loaded {n_rows:,} transactions")

    # Compute features
    config = FeatureConfig(output_path=Path(output))
    engineer = FeatureEngineer(config)

    with timed_operation("Compute features", n_rows, results):
        df = engineer.compute_features(df)

    # Save
    with timed_operation("Save features", n_rows, results):
        engineer.save_features(df, Path(output))

    results.print_summary()


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), default="data/features")
@click.option("--output", "-o", type=click.Path(), default="models", help="Model output path")
@click.option("--n-estimators", type=int, default=1000, help="Number of boosting rounds")
@click.pass_context
def train(ctx: click.Context, input_path: str, output: str, n_estimators: int) -> None:
    """Train XGBoost model on GPU."""
    import cudf
    from src.training.xgboost_gpu import XGBoostTrainer
    from src.training.evaluation import ModelEvaluator, print_feature_importance
    from src.utils.config import TrainingConfig

    console.print(Panel.fit(
        "[bold cyan]Training XGBoost on GPU[/bold cyan]",
        title="Model Training"
    ))

    results = BenchmarkResults()

    # Load features
    with timed_operation("Load feature data", results=results):
        df = cudf.read_parquet(f"{input_path}/features.parquet")

    console.print(f"Loaded {len(df):,} samples")

    # Configure training
    config = TrainingConfig(model_path=Path(output))
    config.xgb_params["n_estimators"] = n_estimators

    trainer = XGBoostTrainer(config)

    # Prepare data
    with timed_operation("Prepare training data", results=results):
        dtrain, dtest, train_df, test_df = trainer.prepare_data(df)

    # Train
    with timed_operation("Train XGBoost GPU", len(df), results):
        trainer.train(dtrain, dtest, n_estimators)

    # Evaluate
    console.print("\n[bold]Evaluating model...[/bold]")
    y_test = test_df["is_fraud"].to_pandas().values
    y_pred = trainer.predict(dtest)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred)
    evaluator.print_report(metrics)

    # Feature importance
    importance = trainer.get_feature_importance()
    print_feature_importance(importance)

    # Save
    trainer.save_model(Path(output) / "xgboost_fraud.json")
    evaluator.save_report(metrics, Path(output) / "evaluation.json")

    results.print_summary()


@cli.command()
@click.option("--n-transactions", "-n", type=int, default=1_000_000)
@click.pass_context
def benchmark(ctx: click.Context, n_transactions: int) -> None:
    """Run CPU vs GPU benchmarks."""
    console.print(Panel.fit(
        f"[bold cyan]Running benchmarks with {n_transactions:,} transactions[/bold cyan]",
        title="Performance Benchmark"
    ))

    # Import benchmark script
    from scripts.benchmark import run_full_benchmark
    run_full_benchmark(n_transactions)


@cli.command()
@click.option("--n-transactions", "-n", type=int, default=1_000_000, help="Number of transactions")
@click.pass_context
def run_all(ctx: click.Context, n_transactions: int) -> None:
    """Run the complete pipeline: generate -> features -> train."""
    console.print(Panel.fit(
        f"[bold cyan]Running complete pipeline[/bold cyan]\n"
        f"Transactions: {n_transactions:,}",
        title="Full Pipeline"
    ))

    # Generate
    ctx.invoke(generate, n_transactions=n_transactions)

    # Features
    ctx.invoke(features)

    # Train
    ctx.invoke(train)

    console.print("\n[bold green]Pipeline complete![/bold green]")


if __name__ == "__main__":
    cli()
