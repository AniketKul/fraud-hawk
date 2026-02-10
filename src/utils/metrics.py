"""Performance metrics and timing utilities."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    duration_seconds: float
    rows_processed: int = 0

    @property
    def rows_per_second(self) -> float:
        if self.duration_seconds > 0 and self.rows_processed > 0:
            return self.rows_processed / self.duration_seconds
        return 0.0


@dataclass
class BenchmarkResults:
    """Collection of benchmark results for comparison."""
    results: list[TimingResult] = field(default_factory=list)

    def add(self, result: TimingResult) -> None:
        self.results.append(result)

    def print_summary(self) -> None:
        table = Table(title="Benchmark Results")
        table.add_column("Operation", style="cyan")
        table.add_column("Duration (s)", justify="right", style="green")
        table.add_column("Rows", justify="right")
        table.add_column("Rows/sec", justify="right", style="yellow")

        for r in self.results:
            table.add_row(
                r.name,
                f"{r.duration_seconds:.2f}",
                f"{r.rows_processed:,}" if r.rows_processed else "-",
                f"{r.rows_per_second:,.0f}" if r.rows_per_second else "-",
            )

        console.print(table)


@contextmanager
def timed_operation(
    name: str, rows: int = 0, results: BenchmarkResults | None = None
) -> Generator[None, None, None]:
    """Context manager for timing operations."""
    console.print(f"[cyan]Starting:[/cyan] {name}")
    start = time.perf_counter()

    yield

    duration = time.perf_counter() - start
    result = TimingResult(name=name, duration_seconds=duration, rows_processed=rows)

    if results is not None:
        results.add(result)

    rate_str = f" ({result.rows_per_second:,.0f} rows/sec)" if rows else ""
    console.print(f"[green]Completed:[/green] {name} in {duration:.2f}s{rate_str}")
