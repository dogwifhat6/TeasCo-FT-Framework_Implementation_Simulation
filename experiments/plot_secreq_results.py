"""
plot_secreq_results.py
======================
Reads secreq_actual_results.json (written by secreq_benchmark.py)
and generates a matplotlib bar chart comparing all models.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")      # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_path = Path(__file__).parent / "secreq_actual_results.json"
    if not results_path.exists():
        print(f"❌ {results_path} not found — run secreq_benchmark.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    scenarios      = data["scenarios"]
    lr_scores      = data["lr_scores"]
    rf_scores      = data["rf_scores"]
    teasco_scores  = data["teasco_scores"]

    x     = np.arange(len(scenarios))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_lr = ax.bar(x - width, lr_scores,     width, label="Logistic Regression",
                     color="#e74c3c", edgecolor="white", linewidth=0.5)
    bars_rf = ax.bar(x,          rf_scores,     width, label="Random Forest",
                     color="#f39c12", edgecolor="white", linewidth=0.5)
    bars_tc = ax.bar(x + width,  teasco_scores, width, label="TeasCo-FT Cascade",
                     color="#2ecc71", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Macro F1 Score", fontsize=13, fontweight="bold")
    ax.set_title(
        "SecReq Cross-Project Benchmark — ACTUAL Results\n"
        "(Binary: sec vs nonsec  |  Leave-one-project-out)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("Test: ", "Test:\n") for s in scenarios],
        fontsize=11,
    )

    # Dynamic y-axis range
    all_vals = lr_scores + rf_scores + teasco_scores
    y_min = max(0, min(all_vals) - 0.08)
    y_max = max(all_vals) + 0.08
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Label bars
    for bars in [bars_lr, bars_rf, bars_tc]:
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    # Add average annotation
    avg_lr = np.mean(lr_scores)
    avg_rf = np.mean(rf_scores)
    avg_tc = np.mean(teasco_scores)
    note = f"Average — LR: {avg_lr:.3f}  |  RF: {avg_rf:.3f}  |  TeasCo-FT: {avg_tc:.3f}"
    ax.text(0.5, 0.02, note, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()

    out_path = Path(__file__).parent / "secreq_actual_results.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    print(f"✅ Chart saved to: {out_path}")


if __name__ == "__main__":
    main()
