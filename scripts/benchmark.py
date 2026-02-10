#!/usr/bin/env python
"""Benchmark CPU vs GPU performance across the pipeline."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def benchmark_data_generation(n_rows: int) -> dict:
    """Benchmark data generation on GPU vs CPU simulation."""
    import cudf
    import cupy as cp
    import pandas as pd

    console.print("\n[bold cyan]1. Data Generation Benchmark[/bold cyan]")

    results = {}

    # GPU generation (cuDF/cupy)
    console.print("  Running GPU generation...")
    start = time.perf_counter()

    transaction_ids = cp.arange(n_rows, dtype=cp.int64)
    user_ids = cp.random.randint(0, 100000, size=n_rows, dtype=cp.int32)
    amounts = cp.exp(cp.random.normal(3.5, 1.2, size=n_rows)).astype(cp.float32)
    timestamps = cp.random.randint(0, 90 * 86400, size=n_rows, dtype=cp.int64)

    df_gpu = cudf.DataFrame({
        "transaction_id": transaction_ids,
        "user_id": user_ids,
        "amount": amounts,
        "timestamp": timestamps,
    })
    # Force computation
    _ = len(df_gpu)

    gpu_time = time.perf_counter() - start
    results["gpu_generation_sec"] = gpu_time
    console.print(f"    GPU: {gpu_time:.2f}s")

    # CPU generation (numpy/pandas)
    console.print("  Running CPU generation...")
    start = time.perf_counter()

    transaction_ids_cpu = np.arange(n_rows, dtype=np.int64)
    user_ids_cpu = np.random.randint(0, 100000, size=n_rows, dtype=np.int32)
    amounts_cpu = np.exp(np.random.normal(3.5, 1.2, size=n_rows)).astype(np.float32)
    timestamps_cpu = np.random.randint(0, 90 * 86400, size=n_rows, dtype=np.int64)

    df_cpu = pd.DataFrame({
        "transaction_id": transaction_ids_cpu,
        "user_id": user_ids_cpu,
        "amount": amounts_cpu,
        "timestamp": timestamps_cpu,
    })

    cpu_time = time.perf_counter() - start
    results["cpu_generation_sec"] = cpu_time
    results["generation_speedup"] = cpu_time / gpu_time
    console.print(f"    CPU: {cpu_time:.2f}s")
    console.print(f"    [yellow]Speedup: {results['generation_speedup']:.1f}x[/yellow]")

    return results


def benchmark_aggregations(n_rows: int) -> dict:
    """Benchmark aggregation operations."""
    import cudf
    import pandas as pd

    console.print("\n[bold cyan]2. Aggregation Benchmark[/bold cyan]")

    results = {}

    # Create test data
    n_users = min(n_rows // 10, 100000)

    # GPU
    console.print("  Preparing GPU data...")
    df_gpu = cudf.DataFrame({
        "user_id": cudf.Series(np.random.randint(0, n_users, size=n_rows, dtype=np.int32)),
        "amount": cudf.Series(np.random.exponential(100, size=n_rows).astype(np.float32)),
        "is_fraud": cudf.Series(np.random.binomial(1, 0.015, size=n_rows).astype(np.int8)),
    })

    console.print("  Running GPU aggregations...")
    start = time.perf_counter()

    agg_gpu = df_gpu.groupby("user_id").agg({
        "amount": ["mean", "std", "max", "count"],
        "is_fraud": "sum",
    })
    _ = len(agg_gpu)  # Force computation

    gpu_time = time.perf_counter() - start
    results["gpu_aggregation_sec"] = gpu_time
    console.print(f"    GPU: {gpu_time:.2f}s")

    # CPU
    console.print("  Running CPU aggregations...")
    df_cpu = df_gpu.to_pandas()

    start = time.perf_counter()

    agg_cpu = df_cpu.groupby("user_id").agg({
        "amount": ["mean", "std", "max", "count"],
        "is_fraud": "sum",
    })

    cpu_time = time.perf_counter() - start
    results["cpu_aggregation_sec"] = cpu_time
    results["aggregation_speedup"] = cpu_time / gpu_time
    console.print(f"    CPU: {cpu_time:.2f}s")
    console.print(f"    [yellow]Speedup: {results['aggregation_speedup']:.1f}x[/yellow]")

    return results


def benchmark_sorting(n_rows: int) -> dict:
    """Benchmark sorting operations."""
    import cudf
    import pandas as pd

    console.print("\n[bold cyan]3. Sorting Benchmark[/bold cyan]")

    results = {}

    # Create test data
    console.print("  Preparing data...")
    df_gpu = cudf.DataFrame({
        "user_id": cudf.Series(np.random.randint(0, 100000, size=n_rows, dtype=np.int32)),
        "timestamp": cudf.Series(np.random.randint(0, 90 * 86400, size=n_rows, dtype=np.int64)),
        "amount": cudf.Series(np.random.exponential(100, size=n_rows).astype(np.float32)),
    })

    # GPU sort
    console.print("  Running GPU sort...")
    start = time.perf_counter()
    sorted_gpu = df_gpu.sort_values(["user_id", "timestamp"])
    _ = len(sorted_gpu)
    gpu_time = time.perf_counter() - start
    results["gpu_sort_sec"] = gpu_time
    console.print(f"    GPU: {gpu_time:.2f}s")

    # CPU sort
    console.print("  Running CPU sort...")
    df_cpu = df_gpu.to_pandas()
    start = time.perf_counter()
    sorted_cpu = df_cpu.sort_values(["user_id", "timestamp"])
    cpu_time = time.perf_counter() - start
    results["cpu_sort_sec"] = cpu_time
    results["sort_speedup"] = cpu_time / gpu_time
    console.print(f"    CPU: {cpu_time:.2f}s")
    console.print(f"    [yellow]Speedup: {results['sort_speedup']:.1f}x[/yellow]")

    return results


def benchmark_xgboost_training(n_rows: int) -> dict:
    """Benchmark XGBoost GPU vs CPU training."""
    import xgboost as xgb

    console.print("\n[bold cyan]4. XGBoost Training Benchmark[/bold cyan]")

    results = {}

    # Create synthetic training data
    console.print("  Preparing training data...")
    n_features = 20
    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = np.random.binomial(1, 0.015, size=n_rows)

    dtrain = xgb.DMatrix(X, label=y)
    num_rounds = 100

    # GPU training
    console.print("  Running GPU training...")
    gpu_params = {
        "tree_method": "gpu_hist",
        "device": "cuda",
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
    }

    start = time.perf_counter()
    gpu_model = xgb.train(gpu_params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    gpu_time = time.perf_counter() - start
    results["gpu_xgb_sec"] = gpu_time
    console.print(f"    GPU: {gpu_time:.2f}s ({num_rounds} rounds)")

    # CPU training
    console.print("  Running CPU training...")
    cpu_params = {
        "tree_method": "hist",
        "device": "cpu",
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
    }

    start = time.perf_counter()
    cpu_model = xgb.train(cpu_params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    cpu_time = time.perf_counter() - start
    results["cpu_xgb_sec"] = cpu_time
    results["xgb_speedup"] = cpu_time / gpu_time
    console.print(f"    CPU: {cpu_time:.2f}s ({num_rounds} rounds)")
    console.print(f"    [yellow]Speedup: {results['xgb_speedup']:.1f}x[/yellow]")

    return results


def benchmark_inference(n_rows: int) -> dict:
    """Benchmark inference throughput."""
    import xgboost as xgb

    console.print("\n[bold cyan]5. Inference Benchmark[/bold cyan]")

    results = {}

    # Train a quick model
    n_features = 20
    X_train = np.random.randn(10000, n_features).astype(np.float32)
    y_train = np.random.binomial(1, 0.015, size=10000)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    model = xgb.train(
        {"tree_method": "gpu_hist", "device": "cuda", "objective": "binary:logistic"},
        dtrain, num_boost_round=50, verbose_eval=False
    )

    # Inference data
    X_test = np.random.randn(n_rows, n_features).astype(np.float32)

    # Batch inference
    batch_sizes = [1, 100, 1000, 10000]

    console.print("  Testing batch sizes...")
    for batch_size in batch_sizes:
        dtest = xgb.DMatrix(X_test[:batch_size])

        # Warm up
        _ = model.predict(dtest)

        # Timed run
        n_iterations = min(1000, n_rows // batch_size)
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = model.predict(dtest)
        elapsed = time.perf_counter() - start

        throughput = (batch_size * n_iterations) / elapsed
        latency_ms = (elapsed / n_iterations) * 1000

        results[f"batch_{batch_size}_throughput"] = throughput
        results[f"batch_{batch_size}_latency_ms"] = latency_ms

        console.print(f"    Batch {batch_size:>5}: {throughput:>10,.0f} txn/sec, {latency_ms:.3f}ms latency")

    return results


def run_full_benchmark(n_rows: int = 1_000_000) -> dict:
    """Run all benchmarks and print summary."""
    console.print(Panel.fit(
        f"[bold]GPU vs CPU Performance Benchmark[/bold]\n"
        f"Dataset size: {n_rows:,} rows",
        title="Benchmark Suite"
    ))

    all_results = {}

    # Run benchmarks
    all_results.update(benchmark_data_generation(n_rows))
    all_results.update(benchmark_aggregations(n_rows))
    all_results.update(benchmark_sorting(n_rows))
    all_results.update(benchmark_xgboost_training(min(n_rows, 500_000)))  # Cap XGB training size
    all_results.update(benchmark_inference(n_rows))

    # Print summary table
    console.print("\n")
    table = Table(title="Benchmark Summary")
    table.add_column("Operation", style="cyan")
    table.add_column("GPU Time", justify="right")
    table.add_column("CPU Time", justify="right")
    table.add_column("Speedup", justify="right", style="yellow")

    operations = [
        ("Data Generation", "gpu_generation_sec", "cpu_generation_sec", "generation_speedup"),
        ("Aggregations", "gpu_aggregation_sec", "cpu_aggregation_sec", "aggregation_speedup"),
        ("Sorting", "gpu_sort_sec", "cpu_sort_sec", "sort_speedup"),
        ("XGBoost Training", "gpu_xgb_sec", "cpu_xgb_sec", "xgb_speedup"),
    ]

    for name, gpu_key, cpu_key, speedup_key in operations:
        table.add_row(
            name,
            f"{all_results[gpu_key]:.2f}s",
            f"{all_results[cpu_key]:.2f}s",
            f"{all_results[speedup_key]:.1f}x",
        )

    console.print(table)

    # Inference summary
    console.print("\n[bold]Inference Throughput (GPU):[/bold]")
    for batch_size in [1, 100, 1000, 10000]:
        throughput = all_results.get(f"batch_{batch_size}_throughput", 0)
        console.print(f"  Batch {batch_size:>5}: {throughput:>10,.0f} transactions/second")

    return all_results


if __name__ == "__main__":
    import sys

    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    run_full_benchmark(n_rows)
