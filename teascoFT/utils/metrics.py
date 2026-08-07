"""
metrics.py
==========
Evaluation utilities for the TeasCo-FT framework.

Functions
---------
- macro_f1                 : Compute Macro F1 score.
- classification_report_rich : Pretty-printed per-class metrics table.
- confidence_calibration_plot : Plot confidence vs. accuracy to assess calibration.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(
    y_true: List[str],
    y_pred: List[str],
    zero_division: float = 0.0,
) -> float:
    """
    Compute Macro-averaged F1 score.

    Each class gets equal weight regardless of support.
    This is the standard metric for imbalanced classification tasks.

    Parameters
    ----------
    y_true : list of str — ground truth labels
    y_pred : list of str — predicted labels
    zero_division : float — value for undefined F1 (default: 0.0)

    Returns
    -------
    float in [0, 1]
    """
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=zero_division)


# ─────────────────────────────────────────────────────────────────────────────
# Rich Classification Report
# ─────────────────────────────────────────────────────────────────────────────

def classification_report_rich(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
    title: str = "Classification Report",
) -> None:
    """
    Print a rich-formatted per-class precision / recall / F1 table.

    Falls back to sklearn's plain text report if rich is not installed.

    Parameters
    ----------
    y_true : list of str
    y_pred : list of str
    classes : list of str, optional — class names in display order
    title : str — table title
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f = f1.mean()
    accuracy = accuracy_score(y_true, y_pred)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title=f"[bold cyan]{title}[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Class", style="cyan", no_wrap=True)
        table.add_column("Precision", justify="right", style="green")
        table.add_column("Recall", justify="right", style="green")
        table.add_column("F1 Score", justify="right", style="yellow")
        table.add_column("Support", justify="right", style="dim")

        for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
            table.add_row(cls, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", str(int(s)))

        table.add_section()
        table.add_row(
            "[bold]Macro avg[/bold]",
            f"[bold]{macro_p:.4f}[/bold]",
            f"[bold]{macro_r:.4f}[/bold]",
            f"[bold]{macro_f:.4f}[/bold]",
            f"[bold]{sum(support)}[/bold]",
        )

        console.print()
        console.print(table)
        console.print(f"  Overall Accuracy: [bold green]{accuracy:.4f}[/bold green]\n")

    except ImportError:
        # Plain text fallback
        header = f"\n{'─' * 60}\n{title}\n{'─' * 60}"
        print(header)
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("─" * 60)
        for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
            print(f"{cls:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f} {int(s):>10}")
        print("─" * 60)
        print(f"{'Macro avg':<20} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f} {sum(support):>10}")
        print(f"\nAccuracy: {accuracy:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Plot
# ─────────────────────────────────────────────────────────────────────────────

def confidence_calibration_plot(
    confidences: np.ndarray,
    y_true: List[str],
    y_pred: List[str],
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Confidence Calibration",
) -> None:
    """
    Plot model confidence vs. actual accuracy (calibration curve).

    A well-calibrated model has confidence ≈ accuracy in each bin.
    Useful for tuning the `confidence_threshold` parameter.

    Parameters
    ----------
    confidences : np.ndarray of float, shape (n_samples,)
        Max probability output by the Base Student.
    y_true : list of str — ground truth labels
    y_pred : list of str — predicted labels
    n_bins : int — number of confidence bins (default: 10)
    output_path : str, optional — if provided, save the figure to this path
    title : str — plot title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed — skipping calibration plot.")
        return

    confidences = np.array(confidences)
    correct = np.array(y_true) == np.array(y_pred)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs, bin_confs, bin_counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(mask.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
    ax1.bar(bin_confs, bin_accs, width=0.08, alpha=0.6, color="#4A90D9", label="Model")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Confidence histogram
    ax2.hist(confidences, bins=20, color="#7B68EE", alpha=0.8, edgecolor="white")
    ax2.axvline(
        confidences.mean(), color="red", linestyle="--",
        label=f"Mean: {confidences.mean():.2f}"
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved to %s", output_path)
        print(f"📊 Calibration plot saved → {output_path}")
    else:
        plt.show()
