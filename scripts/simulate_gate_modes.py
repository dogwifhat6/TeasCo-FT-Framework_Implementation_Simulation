"""
simulate_gate_modes.py — Offline simulations of the tree adapters
=================================================================

Two controlled experiments on synthetic 385-dim cascade features, so they run
in a few minutes with no LLM, no API key and no model download:

1. **Gate modes.** How the "learn only from mistakes" rule should be ported to
   boosting: XGBoost's gradient-masking objective (with and without the
   ``gate_margin`` that prevents collapse), staged sample-weight gating with a
   strict (0.0) and a soft (0.1) floor, and the ungated control.
2. **Distillation signal.** Training on gold labels vs a noisy Teacher's argmax
   pseudo-labels vs the same Teacher's full soft distributions.

The synthetic data mimics the real feature matrix: 384 "embedding" columns with
the class signal in a low-dimensional subspace, plus a 385th Student-confidence
column that is informative about difficulty. Labels carry noise so no model can
separate them perfectly, which is what makes the gate's behaviour visible.

    python scripts/simulate_gate_modes.py              # 3 seeds
    python scripts/simulate_gate_modes.py --seeds 1    # quick look

Writes ``results/simulation_*.json`` and ``results/simulation_*.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import teasco_tree  # noqa: E402,F401  (must precede anything that imports torch)

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from teasco_tree import plots  # noqa: E402
from teasco_tree.adapters import get_adapter  # noqa: E402
from teasco_tree.pipeline import expand_soft_labels  # noqa: E402

RESULTS = ROOT / "results"

SCENARIOS = {
    "Binary, balanced": {"k": 2, "priors": [0.5, 0.5]},
    "Binary, 13% minority": {"k": 2, "priors": [0.87, 0.13]},
    "4-class, skewed": {"k": 4, "priors": [0.45, 0.25, 0.2, 0.1]},
}

GATE_CONFIGS = [
    ("XGBoost · objective gate", "xgboost", {"gate_mode": "objective"}),
    ("XGBoost · objective, margin 0", "xgboost", {"gate_mode": "objective", "gate_margin": 0.0}),
    ("XGBoost · staged, floor 0", "xgboost", {"gate_mode": "staged", "gate_floor": 0.0}),
    ("XGBoost · staged, floor 0.1", "xgboost", {"gate_mode": "staged", "gate_floor": 0.1}),
    ("XGBoost · ungated", "xgboost", {"gate_mode": "none"}),
    ("CatBoost · staged, floor 0", "catboost", {"gate_mode": "staged", "gate_floor": 0.0}),
    ("CatBoost · staged, floor 0.1", "catboost", {"gate_mode": "staged", "gate_floor": 0.1}),
    ("CatBoost · ungated", "catboost", {"gate_mode": "none"}),
    ("LightGBM · staged, floor 0", "lightgbm", {"gate_mode": "staged", "gate_floor": 0.0}),
    ("LightGBM · staged, floor 0.1", "lightgbm", {"gate_mode": "staged", "gate_floor": 0.1}),
    ("LightGBM · ungated", "lightgbm", {"gate_mode": "none"}),
]


def make_task(k, priors, n=1500, seed=0, signal_dims=8, label_noise=0.12):
    """Synthetic cascade features, gold labels, and a noisy Teacher's soft labels."""
    rng = np.random.RandomState(seed)
    y = rng.choice(k, size=n, p=priors)
    centers = rng.randn(k, signal_dims) * 1.1
    latent = centers[y] + rng.randn(n, signal_dims)

    # Spread the signal across the 384 "embedding" columns, as MiniLM would.
    mixing = rng.randn(384, signal_dims) / np.sqrt(signal_dims)
    emb = latent @ mixing.T + rng.randn(n, 384) * 0.9

    # Class posterior from the latent geometry: what an ideal model believes.
    logits = -0.5 * ((latent[:, None, :] - centers[None]) ** 2).sum(-1) + np.log(priors)
    posterior = np.exp(logits - logits.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)

    # Student confidence: informative about difficulty, but noisy.
    student_conf = np.clip(posterior.max(1) + rng.randn(n) * 0.12, 1.0 / k, 1.0)
    X = np.hstack([emb, student_conf[:, None]]).astype(np.float32)

    # Gold labels with symmetric noise.
    flip = rng.rand(n) < label_noise
    y_gold = y.copy()
    y_gold[flip] = rng.randint(0, k, flip.sum())

    # Teacher: a tempered, noisy view of the posterior — right most of the time,
    # and its confidence carries information its argmax throws away.
    t_logits = np.log(posterior + 1e-9) / 1.6 + rng.randn(n, k) * 0.9
    soft = np.exp(t_logits - t_logits.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)
    return X, y_gold, y, soft


def split(n, seed):
    idx = np.random.RandomState(seed + 99).permutation(n)
    a, b = int(0.6 * n), int(0.75 * n)
    return idx[:a], idx[a:b], idx[b:]


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def run_gate_modes(seeds, n_estimators):
    rows = []
    for scen, spec in SCENARIOS.items():
        classes = [f"c{i}" for i in range(spec["k"])]
        for seed in range(seeds):
            X, y_gold, y_clean, _ = make_task(spec["k"], spec["priors"], seed=seed)
            tr, va, te = split(len(X), seed)
            for label, name, kwargs in GATE_CONFIGS:
                adapter = get_adapter(name, classes, n_estimators=n_estimators,
                                      class_weight="balanced", random_state=seed,
                                      verbose=False, **kwargs)
                t0 = time.time()
                adapter.fit(X[tr], y_gold[tr], X[va], y_gold[va])
                val = adapter.history["val_macro_f1"]
                rows.append({
                    "scenario": scen, "seed": seed, "config": label, "adapter": name,
                    # Scored against the clean labels: what the model actually learned.
                    "test_macro_f1": macro_f1(y_clean[te], adapter.predict(X[te])),
                    "val_swing": float(np.max(np.abs(np.diff(val)))) if len(val) > 1 else 0.0,
                    "best_n_trees": int(adapter.best_n_trees),
                    "seconds": time.time() - t0,
                })
            print(f"  gate modes · {scen} · seed {seed} done")
    return rows


def run_distillation(seeds, n_estimators):
    rows = []
    for scen, spec in SCENARIOS.items():
        classes = [f"c{i}" for i in range(spec["k"])]
        for seed in range(seeds):
            X, y_gold, y_clean, soft = make_task(spec["k"], spec["priors"], seed=seed)
            tr, va, te = split(len(X), seed)
            hard = soft.argmax(1)
            teacher_f1 = macro_f1(y_clean[te], hard[te])
            for name in ("xgboost", "catboost", "lightgbm"):
                for supervision in ("ground_truth", "teacher_hard", "teacher_soft"):
                    adapter = get_adapter(name, classes, n_estimators=n_estimators,
                                          class_weight="balanced", random_state=seed,
                                          gate_floor=0.1, verbose=False)
                    if supervision == "ground_truth":
                        adapter.fit(X[tr], y_gold[tr], X[va], y_gold[va])
                    elif supervision == "teacher_hard":
                        adapter.fit(X[tr], hard[tr], X[va], hard[va])
                    else:
                        Xe, ye, we, src = expand_soft_labels(X[tr], soft[tr])
                        adapter.fit(Xe, ye, X[va], hard[va], sample_weight=we,
                                    gate_labels=hard[tr][src])
                    rows.append({
                        "scenario": scen, "seed": seed, "adapter": name,
                        "supervision": supervision, "teacher_macro_f1": teacher_f1,
                        "test_macro_f1": macro_f1(y_clean[te], adapter.predict(X[te])),
                    })
            print(f"  distillation · {scen} · seed {seed} done")
    return rows


def mean_grid(rows, row_key, row_order, col_order, value="test_macro_f1"):
    grid = np.full((len(row_order), len(col_order)), np.nan)
    for i, r in enumerate(row_order):
        for j, c in enumerate(col_order):
            vals = [x[value] for x in rows if row_key(x) == r and x["scenario"] == c]
            if vals:
                grid[i, j] = np.mean(vals)
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    scenarios = list(SCENARIOS)

    print("Simulation 1/2: gate modes")
    gate_rows = run_gate_modes(args.seeds, args.n_estimators)
    labels = [c[0] for c in GATE_CONFIGS]
    grid = mean_grid(gate_rows, lambda r: r["config"], labels, scenarios)
    swing = mean_grid(gate_rows, lambda r: r["config"], labels, scenarios, value="val_swing")
    (RESULTS / "simulation_gate_modes.json").write_text(json.dumps(
        {"seeds": args.seeds, "n_estimators": args.n_estimators, "rows": gate_rows}, indent=2))
    fig = plots.grid_heatmap(
        grid, labels, scenarios,
        title="Gate modes on synthetic cascade features",
        subtitle=f"Test macro F1 vs clean labels · mean of {args.seeds} seed(s) · "
                 f"12% label noise · class-balanced",
        vmin=float(np.nanmin(grid)) - 0.03,
    )
    fig.savefig(RESULTS / "simulation_gate_modes.png", dpi=160, bbox_inches="tight",
                facecolor=plots.SURFACE)
    fig = plots.grid_heatmap(
        swing, labels, scenarios,
        title="Training stability: largest stage-to-stage jump in validation F1",
        subtitle="Lower is steadier. A strict gate retrains on only the current mistakes "
                 "and overshoots.",
        vmin=0.0, vmax=max(0.3, float(np.nanmax(swing))), value_label="max |Δ val macro F1|",
    )
    fig.savefig(RESULTS / "simulation_gate_stability.png", dpi=160, bbox_inches="tight",
                facecolor=plots.SURFACE)

    print("Simulation 2/2: distillation signal")
    kd_rows = run_distillation(args.seeds, args.n_estimators)
    order = [(a, s) for s in plots.SUPERVISION_ORDER for a in plots.ADAPTER_ORDER]
    grid = mean_grid(kd_rows, lambda r: (r["adapter"], r["supervision"]), order, scenarios)
    teacher = [[np.mean([r["teacher_macro_f1"] for r in kd_rows if r["scenario"] == c])
                for c in scenarios]]
    (RESULTS / "simulation_distillation.json").write_text(json.dumps(
        {"seeds": args.seeds, "n_estimators": args.n_estimators, "rows": kd_rows}, indent=2))
    fig = plots.grid_heatmap(
        grid, [plots.ADAPTER_LABEL[a] for a, _ in order], scenarios,
        title="Distillation signal: gold vs Teacher hard vs Teacher soft labels",
        subtitle=f"Test macro F1 vs clean labels · mean of {args.seeds} seed(s) · "
                 f"staged gates use floor 0.1",
        reference_values=np.array(teacher), reference_labels=["Teacher's own argmax"],
        blocks=[(plots.SUPERVISION_LABEL[s], 3) for s in plots.SUPERVISION_ORDER],
        vmin=float(np.nanmin(grid)) - 0.03,
    )
    fig.savefig(RESULTS / "simulation_distillation.png", dpi=160, bbox_inches="tight",
                facecolor=plots.SURFACE)

    print("\nGate modes — mean test macro F1 (max val swing):")
    for i, label in enumerate(labels):
        cells = "  ".join(f"{grid_v:.3f} ({sw:.2f})" for grid_v, sw in zip(
            mean_grid(gate_rows, lambda r: r["config"], labels, scenarios)[i], swing[i]))
        print(f"  {label:32} {cells}")
    print("\nDistillation — mean test macro F1:")
    for (a, s), vals in zip(order, grid):
        print(f"  {a:9} {s:13} " + "  ".join(f"{v:.3f}" for v in vals))
    print("  teacher argmax          " + "  ".join(f"{v:.3f}" for v in teacher[0]))
    print(f"\nWrote results/simulation_*.json and .png")


if __name__ == "__main__":
    main()
