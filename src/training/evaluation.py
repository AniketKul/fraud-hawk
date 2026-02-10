"""Model evaluation utilities for fraud detection."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    auc_roc: float
    auc_pr: float
    precision_at_threshold: dict[float, float] = field(default_factory=dict)
    recall_at_threshold: dict[float, float] = field(default_factory=dict)
    confusion_matrices: dict[float, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "precision_at_threshold": self.precision_at_threshold,
            "recall_at_threshold": self.recall_at_threshold,
        }


class ModelEvaluator:
    """Evaluate fraud detection model performance."""

    def __init__(self, thresholds: list[float] | None = None):
        """Initialize evaluator.

        Args:
            thresholds: List of probability thresholds to evaluate.
        """
        self.thresholds = thresholds or [0.3, 0.5, 0.7, 0.9]

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> EvaluationMetrics:
        """Evaluate model predictions.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            EvaluationMetrics object.
        """
        # Core metrics
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)

        metrics = EvaluationMetrics(auc_roc=auc_roc, auc_pr=auc_pr)

        # Threshold-specific metrics
        for threshold in self.thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            metrics.precision_at_threshold[threshold] = precision
            metrics.recall_at_threshold[threshold] = recall
            metrics.confusion_matrices[threshold] = {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            }

        return metrics

    def print_report(self, metrics: EvaluationMetrics) -> None:
        """Print formatted evaluation report."""
        console.print("\n[bold cyan]Model Evaluation Report[/bold cyan]")
        console.print("=" * 50)

        # Core metrics
        console.print(f"\n[bold]Overall Performance:[/bold]")
        console.print(f"  AUC-ROC: {metrics.auc_roc:.4f}")
        console.print(f"  AUC-PR:  {metrics.auc_pr:.4f}")

        # Threshold analysis
        table = Table(title="\nPerformance at Different Thresholds")
        table.add_column("Threshold", style="cyan", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("FP", justify="right")
        table.add_column("FN", justify="right")
        table.add_column("TN", justify="right")

        for threshold in self.thresholds:
            cm = metrics.confusion_matrices[threshold]
            table.add_row(
                f"{threshold:.1f}",
                f"{metrics.precision_at_threshold[threshold]:.3f}",
                f"{metrics.recall_at_threshold[threshold]:.3f}",
                f"{cm['tp']:,}",
                f"{cm['fp']:,}",
                f"{cm['fn']:,}",
                f"{cm['tn']:,}",
            )

        console.print(table)

        # Business interpretation
        console.print("\n[bold]Business Impact:[/bold]")
        best_threshold = max(
            self.thresholds,
            key=lambda t: (
                metrics.precision_at_threshold[t] * metrics.recall_at_threshold[t]
            )
        )
        cm = metrics.confusion_matrices[best_threshold]
        console.print(f"  Recommended threshold: {best_threshold}")
        console.print(f"  At this threshold:")
        console.print(f"    - Catch {cm['tp']:,} frauds (True Positives)")
        console.print(f"    - Miss {cm['fn']:,} frauds (False Negatives)")
        console.print(f"    - {cm['fp']:,} false alarms (False Positives)")

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        target_precision: float = 0.9,
    ) -> float:
        """Find threshold that achieves target precision.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            target_precision: Desired precision level.

        Returns:
            Optimal threshold.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        # Find threshold that achieves target precision
        valid_idx = np.where(precision >= target_precision)[0]
        if len(valid_idx) == 0:
            console.print(f"[yellow]Cannot achieve {target_precision:.0%} precision[/yellow]")
            return 0.5

        # Among valid thresholds, pick one with highest recall
        best_idx = valid_idx[np.argmax(recall[valid_idx])]

        # Handle edge case where best_idx equals len(thresholds)
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1

        optimal_threshold = thresholds[best_idx]
        achieved_precision = precision[best_idx]
        achieved_recall = recall[best_idx]

        console.print(f"\n[bold]Optimal Threshold Analysis[/bold]")
        console.print(f"  Target precision: {target_precision:.0%}")
        console.print(f"  Optimal threshold: {optimal_threshold:.3f}")
        console.print(f"  Achieved precision: {achieved_precision:.3f}")
        console.print(f"  Achieved recall: {achieved_recall:.3f}")

        return optimal_threshold

    def save_report(
        self,
        metrics: EvaluationMetrics,
        output_path: Path,
    ) -> None:
        """Save evaluation report to JSON.

        Args:
            metrics: Evaluation metrics.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        console.print(f"[green]Report saved to {output_path}[/green]")


def print_feature_importance(
    importance: dict[str, float],
    top_n: int = 15,
) -> None:
    """Print feature importance table."""
    table = Table(title=f"Top {top_n} Feature Importances")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right", style="green")

    total = sum(importance.values())
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for i, (feature, score) in enumerate(sorted_features[:top_n], 1):
        pct = score / total * 100 if total > 0 else 0
        table.add_row(str(i), feature, f"{score:.1f} ({pct:.1f}%)")

    console.print(table)
