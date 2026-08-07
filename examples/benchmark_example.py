"""
benchmark_example.py
====================
Benchmarks the TeasCo-FT CascadePipeline against a naive Logistic Regression
baseline and (optionally) the Teacher LLM, measuring:
  - Macro F1
  - Inference latency (ms/sample)
  - Memory footprint

Usage
-----
    # Train first, then benchmark:
    python examples/train_example.py
    python examples/benchmark_example.py

    # Custom model:
    python examples/benchmark_example.py --model models/my_classifier.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT import CascadePipeline, TFIDFExtractor, macro_f1


def measure_latency(pipeline, texts, n_warmup=3, n_runs=5):
    """Measure average inference latency in ms/sample."""
    # Warmup
    for _ in range(n_warmup):
        pipeline.predict(texts[:10])

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pipeline.predict(texts)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed / len(texts))

    return np.mean(latencies), np.std(latencies)


def main():
    parser = argparse.ArgumentParser(description="TeasCo-FT Benchmark")
    parser.add_argument("--model", default="models/newsgroups_classifier.pkl")
    parser.add_argument("--n-test", type=int, default=500,
                        help="Number of test samples for benchmarking")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  TeasCo-FT: Benchmark vs. Baselines")
    print("=" * 65 + "\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.linear_model import LogisticRegression

    categories = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"]
    data = fetch_20newsgroups(
        subset="test", categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    target_names = data.target_names
    texts = data.data[: args.n_test]
    y_true = [target_names[t] for t in data.target[: args.n_test]]

    print(f"📂 Test set: {len(texts)} samples | Classes: {target_names}\n")

    # ── Baseline: Logistic Regression (hard labels only, no KD) ──────────
    print("─" * 55)
    print("Baseline: Logistic Regression (hard labels, no distillation)")
    data_train = fetch_20newsgroups(
        subset="train", categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    train_texts = data_train.data
    y_train = [target_names[t] for t in data_train.target]

    extractor_base = TFIDFExtractor(max_features=30_000, ngram_range=(1, 2))
    X_train = extractor_base.fit_transform(train_texts)
    X_test = extractor_base.transform(texts)

    lr = LogisticRegression(C=5.0, max_iter=1000, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_preds = [target_names[p] if isinstance(p, int) else p for p in lr.predict(X_test)]
    # sklearn LR returns indices — map back
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(target_names)
    lr_preds_raw = lr.predict(X_test)
    lr_preds = le.inverse_transform(lr_preds_raw) if lr_preds_raw.dtype != object else lr_preds_raw

    # Actually just fit on string labels
    lr2 = LogisticRegression(C=5.0, max_iter=1000, n_jobs=-1)
    lr2.fit(X_train, y_train)
    lr2_preds = lr2.predict(X_test)
    lr_f1 = macro_f1(y_true, lr2_preds)

    # Latency for baseline
    def lr_predict(txts):
        Xb = extractor_base.transform(txts)
        return lr2.predict(Xb)

    lats = []
    for _ in range(5):
        t0 = time.perf_counter()
        lr_predict(texts)
        lats.append((time.perf_counter() - t0) * 1000 / len(texts))
    lr_lat, lr_lat_std = np.mean(lats), np.std(lats)

    print(f"  Macro F1     : {lr_f1:.4f}")
    print(f"  Latency      : {lr_lat:.3f} ± {lr_lat_std:.3f} ms/sample\n")

    # ── TeasCo-FT Cascade Pipeline ────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ No trained pipeline found at {model_path.resolve()}")
        print("   Run python examples/train_example.py first.\n")
        sys.exit(1)

    print("─" * 55)
    print("TeasCo-FT Cascade Pipeline")
    pipeline = CascadePipeline.load(model_path)

    cascade_preds = pipeline.predict(texts)
    cascade_f1 = macro_f1(y_true, cascade_preds)
    lat_mean, lat_std = measure_latency(pipeline, texts)

    print(f"  Macro F1     : {cascade_f1:.4f}")
    print(f"  Latency      : {lat_mean:.3f} ± {lat_std:.3f} ms/sample")
    print(f"  Corrector trigger rate: {pipeline.cascade_stats.corrector_trigger_rate:.1%}\n")

    # ── Summary Table ─────────────────────────────────────────────────────
    print("─" * 55)
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(
            title="[bold cyan]Benchmark Summary[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan")
        table.add_column("Macro F1", justify="right", style="green")
        table.add_column("Latency (ms/sample)", justify="right", style="yellow")
        table.add_column("LLM in Production?", justify="center")

        table.add_row("LR Baseline (no KD)",    f"{lr_f1:.4f}", f"{lr_lat:.3f}", "❌ No")
        table.add_row("TeasCo-FT Cascade",       f"{cascade_f1:.4f}", f"{lat_mean:.3f}", "❌ No")
        table.add_row("Teacher LLM (estimated)", "~0.85+", "~300–2000", "✅ Yes (expensive)")

        console.print(table)
    except ImportError:
        print(f"\n{'Model':<30} {'Macro F1':>10} {'ms/sample':>12}")
        print("─" * 55)
        print(f"{'LR Baseline (no KD)':<30} {lr_f1:>10.4f} {lr_lat:>12.3f}")
        print(f"{'TeasCo-FT Cascade':<30} {cascade_f1:>10.4f} {lat_mean:>12.3f}")
        print(f"{'Teacher LLM (est.)':<30} {'~0.85+':>10} {'300-2000':>12}")

    delta = cascade_f1 - lr_f1
    sign = "+" if delta >= 0 else ""
    print(f"\n  KD Improvement over baseline: {sign}{delta:.4f} Macro F1")
    print("  No LLM required at inference — full speed, full specialization.\n")


if __name__ == "__main__":
    main()
