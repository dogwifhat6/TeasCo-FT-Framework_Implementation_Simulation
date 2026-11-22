"""
plots.py — Matplotlib figures for a results payload
===================================================

Every function takes the JSON payload from ``ExperimentResult.payload()`` (or a
saved ``results.json``) and returns a ``matplotlib.figure.Figure``, so the same
code renders in Streamlit, in the scripts, and from a saved run.

Design rules, so the next edit does not undo them:

* **Magnitude is one hue.** Macro F1 heatmaps and confusion matrices use a single
  blue ramp, light -> dark. A rainbow ramp invents boundaries the data lacks.
* **Every heatmap cell carries its number** — the grid is also the table view.
* **Heatmap scales start at the collapse floor, not zero.** A model predicting one
  class already scores the majority-class baseline, so the ramp starts just below
  the lowest baseline and the grey reference rows show that floor.
* **Adapters keep fixed colours** (XGBoost blue, CatBoost orange, LightGBM aqua),
  whatever subset is plotted: colour follows the entity, never its position.
  The three were validated as a set for colour-vision deficiency; aqua is below
  3:1 against the surface, so charts using it always carry a legend.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
REFERENCE_FILL = "#e6e5e1"

ADAPTER_ORDER = ["xgboost", "catboost", "lightgbm"]
ADAPTER_COLOR = {"xgboost": "#2a78d6", "catboost": "#eb6834", "lightgbm": "#1baf7a"}
ADAPTER_LABEL = {"xgboost": "XGBoost", "catboost": "CatBoost", "lightgbm": "LightGBM"}
SUPERVISION_ORDER = ["ground_truth", "teacher_hard", "teacher_soft"]
SUPERVISION_LABEL = {
    "ground_truth": "Ground truth",
    "teacher_hard": "Teacher pseudo-labels",
    "teacher_soft": "Teacher soft labels (KD)",
}

CMAP = LinearSegmentedColormap.from_list("teasco_blue", BLUE_RAMP)


def _style(fig, axes) -> None:
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)


def _ink_for(facecolor) -> str:
    """White or near-black text, whichever contrasts better with the cell."""
    def lum(rgb):
        ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    bg = lum(facecolor)
    contrast = lambda fg: (max(bg, fg) + 0.05) / (min(bg, fg) + 0.05)
    return "#ffffff" if contrast(1.0) >= contrast(lum((0.043, 0.043, 0.043))) else INK


def _ordered(values: Sequence[str], order: Sequence[str]) -> List[str]:
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + [v for v in present if v not in order]


def _test_sets(payload: Dict) -> List[str]:
    return list(dict.fromkeys(r["test_set"] for r in payload["results"]))


def grid_heatmap(
    values: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    subtitle: str = "",
    reference_values: Optional[np.ndarray] = None,
    reference_labels: Sequence[str] = (),
    blocks: Optional[Sequence[tuple]] = None,
    vmin: Optional[float] = None,
    vmax: float = 1.0,
    value_label: str = "Macro F1",
    fmt: str = "{:.3f}",
):
    """
    Generic annotated heatmap. ``reference_values`` rows are drawn in grey below
    the data rows; ``blocks`` is ``[(label, n_rows), ...]`` naming row groups.
    """
    values = np.asarray(values, dtype=float)
    n_rows, n_cols = values.shape
    refs = np.zeros((0, n_cols)) if reference_values is None else np.asarray(reference_values, float)
    total_rows = n_rows + len(refs)
    if vmin is None:
        finite = np.concatenate([values[np.isfinite(values)], refs[np.isfinite(refs)]])
        vmin = max(0.0, (finite.min() if finite.size else 0.0) - 0.05)
    vmax = max(vmax, vmin + 0.05)

    left_room = 2.2 + (0.35 if blocks else 0.0)
    fig, ax = plt.subplots(figsize=(min(18, 1.6 * n_cols + left_room + 1.8),
                                    0.55 * total_rows + 2.0))
    _style(fig, ax)

    def cell(col, row, value, face, ink):
        ax.add_patch(mpl.patches.Rectangle((col + 0.03, row + 0.03), 0.94, 0.94,
                                           facecolor=face, edgecolor=SURFACE, linewidth=1.5))
        text = "—" if not np.isfinite(value) else fmt.format(value)
        ax.text(col + 0.5, row + 0.5, text, ha="center", va="center", fontsize=10, color=ink)

    for r in range(n_rows):
        for c in range(n_cols):
            v = values[r, c]
            face = CMAP(np.clip((v - vmin) / (vmax - vmin), 0, 1)) if np.isfinite(v) else REFERENCE_FILL
            cell(c, r, v, face, _ink_for(face) if np.isfinite(v) else INK_3)
    for i, ref in enumerate(refs):
        for c, v in enumerate(ref):
            cell(c, n_rows + i, v, REFERENCE_FILL, INK_2)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(total_rows, 0)
    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=10, color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([r + 0.5 for r in range(total_rows)])
    ax.set_yticklabels(list(row_labels) + list(reference_labels), fontsize=10, color=INK)
    for label in ax.get_yticklabels()[n_rows:]:
        label.set_color(INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if blocks:
        start = 0
        for i, (label, size) in enumerate(blocks):
            ax.annotate(label, xy=(0, start + size / 2), xycoords="data",
                        xytext=(-118, 0), textcoords="offset points", rotation=90,
                        ha="center", va="center", fontsize=9, color=INK_2, annotation_clip=False)
            if i:
                ax.axhline(start, color=INK_2, linewidth=0.8, alpha=0.4)
            start += size
    if len(refs):
        ax.axhline(n_rows, color=INK_2, linewidth=0.8, alpha=0.4)

    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(value_label, fontsize=9, color=INK_2)
    cbar.ax.tick_params(labelsize=8, length=0, colors=INK_2)
    cbar.outline.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.995)
    if subtitle:
        fig.text(0.02, 0.955, subtitle, fontsize=9, color=INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if subtitle else 0.96))
    return fig


def _subtitle(payload: Dict) -> str:
    cfg = payload.get("config", {})
    student = cfg.get("student", {})
    teacher = cfg.get("teacher", {})
    parts = [f"Student {student.get('model', '?')}" if student.get("backend") != "mock" else "Student mock",
             f"Teacher {teacher.get('model', '?')}" if teacher.get("backend") == "api"
             else f"Teacher {teacher.get('backend', '?')}",
             f"trained on {cfg.get('train_dataset', '?')} ({payload.get('train_size', '?')} rows)"]
    return "  ·  ".join(parts)


def plot_macro_f1_heatmap(payload: Dict):
    """Adapter x supervision rows, test sets across, reference rows at the foot."""
    rows = payload["results"]
    if not rows:
        raise ValueError("No scored results to plot (test sets need labels).")
    tests = _test_sets(payload)
    lookup = {(r["adapter"], r["supervision"], r["test_set"]): r["macro_f1"] for r in rows}
    adapters = _ordered([r["adapter"] for r in rows], ADAPTER_ORDER)
    supervisions = _ordered([r["supervision"] for r in rows], SUPERVISION_ORDER)

    values, labels, blocks = [], [], []
    for sup in supervisions:
        for a in adapters:
            values.append([lookup.get((a, sup, t), np.nan) for t in tests])
            labels.append(ADAPTER_LABEL.get(a, a))
        blocks.append((SUPERVISION_LABEL.get(sup, sup), len(adapters)))

    ref = {(r["kind"], r["test_set"]): r["macro_f1"] for r in payload.get("reference", [])}
    ref_values = [[ref.get(("student_zero_shot", t), np.nan) for t in tests],
                  [ref.get(("majority_baseline", t), np.nan) for t in tests]]
    baseline_min = np.nanmin(ref_values[1]) if np.isfinite(ref_values[1]).any() else 0.0
    return grid_heatmap(
        np.array(values), labels, tests,
        title="Macro F1 by tree adapter, supervision and test set",
        subtitle=_subtitle(payload),
        reference_values=np.array(ref_values),
        reference_labels=["Student SLM zero-shot", "Majority-class baseline"],
        blocks=blocks if len(supervisions) > 1 else None,
        vmin=max(0.0, baseline_min - 0.05),
    )


def plot_macro_f1_bars(payload: Dict):
    """Small multiples, one panel per test set: supervision groups, adapter bars,
    and the majority-class baseline as a reference line."""
    rows = payload["results"]
    tests = _test_sets(payload)
    adapters = _ordered([r["adapter"] for r in rows], ADAPTER_ORDER)
    supervisions = _ordered([r["supervision"] for r in rows], SUPERVISION_ORDER)
    lookup = {(r["adapter"], r["supervision"], r["test_set"]): r["macro_f1"] for r in rows}
    ref = {(r["kind"], r["test_set"]): r["macro_f1"] for r in payload.get("reference", [])}

    n = len(tests)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.9 * nrows + 0.6),
                             squeeze=False, sharey=True)
    _style(fig, axes)
    width = 0.8 / len(adapters)
    for idx, test in enumerate(tests):
        ax = axes.ravel()[idx]
        x = np.arange(len(supervisions))
        for j, a in enumerate(adapters):
            vals = [lookup.get((a, s, test), np.nan) for s in supervisions]
            ax.bar(x + (j - (len(adapters) - 1) / 2) * width, vals, width * 0.92,
                   color=ADAPTER_COLOR.get(a, INK_3), label=ADAPTER_LABEL.get(a, a),
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        baseline = ref.get(("majority_baseline", test))
        title = test
        if baseline is not None:
            ax.axhline(baseline, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
            title = f"{test}   ·   dashed line = majority baseline {baseline:.2f}"

        ax.set_xticks(x)
        ax.set_xticklabels([SUPERVISION_LABEL.get(s, s) for s in supervisions], fontsize=9, color=INK)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        if idx % ncols == 0:
            ax.set_ylabel("Macro F1", fontsize=9, color=INK_2)
    for extra in axes.ravel()[n:]:
        extra.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(adapters), frameon=False,
               fontsize=9, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Macro F1 per test set", fontsize=13, fontweight="bold", color=INK,
                 x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_training_curves(payload: Dict):
    """Validation macro F1 as trees are added, one panel per supervision."""
    fits = payload["fit_summaries"]
    supervisions = _ordered([f["supervision"] for f in fits], SUPERVISION_ORDER)
    fig, axes = plt.subplots(1, len(supervisions), figsize=(5.2 * len(supervisions), 3.8),
                             squeeze=False, sharey=True)
    _style(fig, axes)
    for ax, sup in zip(axes.ravel(), supervisions):
        for f in sorted((f for f in fits if f["supervision"] == sup),
                        key=lambda f: ADAPTER_ORDER.index(f["adapter"])
                        if f["adapter"] in ADAPTER_ORDER else 99):
            h = f["history"]
            color = ADAPTER_COLOR.get(f["adapter"], INK_3)
            ax.plot(h["n_trees"], h["val_macro_f1"], color=color, linewidth=2,
                    solid_capstyle="round", label=f"{ADAPTER_LABEL.get(f['adapter'])} validation")
            ax.plot(h["n_trees"], h["train_macro_f1"], color=color, linewidth=1.2,
                    linestyle=(0, (3, 2)), alpha=0.8, label=f"{ADAPTER_LABEL.get(f['adapter'])} train")
            if f["best_n_trees"] in h["n_trees"]:
                i = h["n_trees"].index(f["best_n_trees"])
                ax.scatter([h["n_trees"][i]], [h["val_macro_f1"][i]], s=48, color=color,
                           edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.set_title(SUPERVISION_LABEL.get(sup, sup), fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Trees (boosting rounds)", fontsize=9, color=INK_2)
        ax.grid(color=GRID, linewidth=1)
        ax.set_ylim(0, 1.02)
    axes.ravel()[0].set_ylabel("Macro F1", fontsize=9, color=INK_2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Early stopping: macro F1 as trees are added (dot = kept size)",
                 fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def plot_confusion_matrices(payload: Dict, test_set: str):
    """Row-normalised confusion matrix for every model on one test set."""
    rows = [r for r in payload["results"] if r["test_set"] == test_set]
    if not rows:
        raise ValueError(f"No scored results for {test_set!r}.")
    classes = payload["classes"]
    adapters = _ordered([r["adapter"] for r in rows], ADAPTER_ORDER)
    supervisions = _ordered([r["supervision"] for r in rows], SUPERVISION_ORDER)
    lookup = {(r["adapter"], r["supervision"]): r for r in rows}
    k = len(classes)
    size = max(2.6, 0.55 * k + 1.2)
    fig, axes = plt.subplots(len(supervisions), len(adapters),
                             figsize=(size * len(adapters) + 0.6, size * len(supervisions) + 0.8),
                             squeeze=False)
    _style(fig, axes)
    for i, sup in enumerate(supervisions):
        for j, a in enumerate(adapters):
            ax = axes[i, j]
            r = lookup.get((a, sup))
            if r is None:
                ax.set_visible(False)
                continue
            cm = np.asarray(r["confusion"], dtype=float)
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
            if k <= 8:
                for (y, x), count in np.ndenumerate(cm):
                    face = CMAP(norm[y, x])
                    ax.text(x, y, f"{int(count)}\n{norm[y, x]:.0%}", ha="center", va="center",
                            fontsize=8, color=_ink_for(face))
            ax.set_xticks(range(k))
            ax.set_yticks(range(k))
            ax.set_xticklabels(classes, fontsize=8, rotation=45 if k > 3 else 0)
            ax.set_yticklabels(classes, fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{ADAPTER_LABEL.get(a, a)} · F1 {r['macro_f1']:.3f}", fontsize=9.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"{SUPERVISION_LABEL.get(sup, sup)}\ntrue label", fontsize=8.5, color=INK_2)
            if i == len(supervisions) - 1:
                ax.set_xlabel("predicted label", fontsize=8.5, color=INK_2)
    fig.suptitle(f"Confusion matrices — {test_set} (share of each true class)",
                 fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_all(payload: Dict, out_dir) -> Dict[str, str]:
    """Render every chart for a payload to PNGs in ``out_dir``."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    if not payload.get("results"):
        return written
    figures = {
        "macro_f1_heatmap": plot_macro_f1_heatmap(payload),
        "macro_f1_bars": plot_macro_f1_bars(payload),
        "training_curves": plot_training_curves(payload),
    }
    for test in _test_sets(payload):
        slug = "".join(ch if ch.isalnum() else "_" for ch in test).strip("_")
        figures[f"confusion_{slug}"] = plot_confusion_matrices(payload, test)
    for name, fig in figures.items():
        path = out / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        written[name] = str(path)
    return written
