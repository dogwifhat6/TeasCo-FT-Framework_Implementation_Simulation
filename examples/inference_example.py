"""
inference_example.py
====================
Demonstrates fast LLM-free inference using a saved CascadePipeline.

Usage
-----
    # First train a model:
    python examples/train_example.py

    # Then run inference:
    python examples/inference_example.py

    # With a custom model path:
    python examples/inference_example.py --model models/my_classifier.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make teascoFT importable from examples/ ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT import CascadePipeline


DEMO_TEXTS = [
    "The new vaccine trials show promising results in reducing cancer mortality.",
    "NASA's Artemis mission successfully launched its lunar lander yesterday.",
    "The hockey team won the championship with a stunning overtime goal.",
    "The senator proposed new restrictions on automatic weapons.",
    "Scientists discovered a new treatment for Alzheimer's disease using CRISPR.",
    "SpaceX's Starship reached orbit for the first time in a historic test.",
    "The goalkeeper made an incredible save to keep the playoff hopes alive.",
    "The Supreme Court ruled on the Second Amendment case this morning.",
]


def main():
    parser = argparse.ArgumentParser(description="TeasCo-FT Inference Example")
    parser.add_argument(
        "--model",
        default="models/newsgroups_classifier.pkl",
        help="Path to saved pipeline .pkl file",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Model not found at: {model_path.resolve()}")
        print("   Run: python examples/train_example.py  to train first.\n")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  TeasCo-FT: Fast LLM-Free Inference")
    print("=" * 65 + "\n")

    # ── Load Pipeline ─────────────────────────────────────────────────────
    pipeline = CascadePipeline.load(model_path)

    # ── Single Prediction ─────────────────────────────────────────────────
    print("─" * 55)
    print("Single Prediction:")
    single_text = DEMO_TEXTS[0]
    pred = pipeline.predict_one(single_text)
    print(f'  Text  : "{single_text}"')
    print(f"  Label : {pred}\n")

    # ── Batch Prediction with Probabilities ───────────────────────────────
    print("─" * 55)
    print("Batch Prediction with Probabilities:\n")
    preds, probs = pipeline.predict_with_proba(DEMO_TEXTS)

    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Text (truncated)", style="dim", max_width=42)
        table.add_column("Prediction", style="bold green")
        table.add_column("Confidence", justify="right")

        for text, pred, prob in zip(DEMO_TEXTS, preds, probs):
            conf = prob.max()
            table.add_row(text[:42] + "…", pred, f"{conf:.3f}")

        console.print(table)

    except ImportError:
        for text, pred, prob in zip(DEMO_TEXTS, preds, probs):
            conf = prob.max()
            print(f"  [{conf:.2f}] {pred:30s} | {text[:50]}...")

    # ── Cascade Statistics ────────────────────────────────────────────────
    print(f"\n📊 {pipeline.cascade_stats}")

    print("\n✅ Inference complete — no LLM calls were made!\n")


if __name__ == "__main__":
    main()
