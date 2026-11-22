"""
run_benchmark.py — The full cascade from the command line
=========================================================

Same pipeline as the Streamlit app, for reproducible runs: train on one dataset,
evaluate on a held-out split plus any number of other test sets, save models,
predictions, results.json and every chart under runs/.

    # PROMISE -> held-out + SecReq + NFR-SO, qwen2.5:0.5b Student, Llama 3.3 70B Teacher
    python scripts/run_benchmark.py

    # your own data
    python scripts/run_benchmark.py --train my.csv --text-col sentence --label-col category \\
        --test other.csv --test-text-col sentence --test-label-col category

    # no Ollama / API key / downloads: mock models, meaningless numbers
    python scripts/run_benchmark.py --offline

The Teacher key is read from OPENROUTER_API_KEY (or --api-key-env) in the
environment or in .env. Nothing about the key is written to disk.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import teasco_tree  # noqa: E402,F401  (must precede anything that imports torch)

from sklearn.model_selection import train_test_split  # noqa: E402

from teasco_tree import plots  # noqa: E402
from teasco_tree.adapters import available_adapters  # noqa: E402
from teasco_tree.data import (  # noqa: E402
    SAMPLE_DATASETS, clean_dataset, load_sample, read_table, to_dataset,
)
from teasco_tree.llm import TEACHER_PRESETS  # noqa: E402
from teasco_tree.pipeline import (  # noqa: E402
    SUPERVISIONS, StudentConfig, TeacherConfig, TrainConfig, new_run_dir, run_experiment,
)
from teasco_tree.system import load_env  # noqa: E402

SAMPLE_ALIASES = {"promise": "PROMISE (sec vs nonsec)", "secreq": "SecReq (CPN + GPS + ePurse)",
                  "nfr-so": "NFR-SO (400-row sample)"}


def load(spec: str, text_col: str, label_col: str):
    if spec.lower() in SAMPLE_ALIASES:
        name = SAMPLE_ALIASES[spec.lower()]
        return name.split(" (")[0], load_sample(name)
    path = Path(spec)
    return path.stem, clean_dataset(read_table(path), text_col, label_col or None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="promise", help="CSV/XLSX path, or promise|secreq|nfr-so")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label", help="'' for unlabelled data")
    p.add_argument("--classes", default=None, help="comma-separated; required if unlabelled")
    p.add_argument("--holdout", type=float, default=0.2, help="0 to disable")
    p.add_argument("--test", action="append", default=None,
                   help="repeatable; path or promise|secreq|nfr-so (default: secreq, nfr-so)")
    p.add_argument("--test-text-col", default="text")
    p.add_argument("--test-label-col", default="label")
    p.add_argument("--student", default="qwen2.5:0.5b")
    p.add_argument("--teacher-preset", default="openrouter-llama-3.3-70b", choices=list(TEACHER_PRESETS))
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--teacher-base-url", default=None)
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--adapters", default=",".join(available_adapters()))
    p.add_argument("--supervisions", default=",".join(SUPERVISIONS))
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true", help="mock Student/Teacher + hashing embedder")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_env(ROOT / ".env")

    train_name, train_df = load(args.train, args.text_col, args.label_col)
    tests = []
    if args.holdout > 0:
        stratify = train_df["label"] if "label" in train_df.columns else None
        train_df, held = train_test_split(train_df, test_size=args.holdout, stratify=stratify,
                                          random_state=args.seed)
        tests.append(to_dataset(held.reset_index(drop=True), f"{train_name} (held-out)"))
    for spec in (args.test if args.test is not None else ["secreq", "nfr-so"]):
        name, df = load(spec, args.test_text_col, args.test_label_col)
        if name != train_name:
            tests.append(to_dataset(df, name))
    train = to_dataset(train_df.reset_index(drop=True), f"{train_name} (train)")
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None

    if args.offline:
        student_cfg = StudentConfig(backend="mock", model="mock-student")
        teacher_cfg = TeacherConfig(backend="mock", model="mock-teacher", base_url=None)
        embedder = "hashing"
    else:
        preset = TEACHER_PRESETS[args.teacher_preset]
        key_env = args.api_key_env or preset["api_key_env"]
        student_cfg = StudentConfig(backend="ollama", model=args.student)
        teacher_cfg = TeacherConfig(
            backend="api", model=args.teacher_model or preset["model"],
            base_url=args.teacher_base_url or preset["base_url"],
            api_key=os.environ.get(key_env) if key_env else None,
            extra_body=preset["extra_body"], batch_size=args.batch_size,
        )
        if key_env and not teacher_cfg.api_key:
            sys.exit(f"Set {key_env} in the environment or .env for the Teacher.")
        embedder = "minilm"

    train_cfg = TrainConfig(adapters=args.adapters.split(","), supervisions=args.supervisions.split(","),
                            embedder=embedder, n_estimators=args.n_estimators, random_state=args.seed)

    last = {}

    def progress(stage, fraction, message):
        if fraction in (0.0, 1.0) or int(fraction * 10) != last.get(stage):
            last[stage] = int(fraction * 10)
            print(f"[{stage:>14}] {message}", flush=True)

    run_dir = new_run_dir(f"{train_name}_{'offline' if args.offline else args.student}")
    result = run_experiment(train, tests, classes=classes, student_cfg=student_cfg,
                            teacher_cfg=teacher_cfg, train_cfg=train_cfg,
                            progress=progress, run_dir=run_dir)
    charts = plots.save_all(result.payload(), run_dir / "plots")

    print(f"\nTraining rows: {result.train_size} · classes: {result.classes}")
    for k, v in result.diagnostics.items():
        print(f"  {k}: {v}")
    print("\nmacro F1 (majority baseline / Student zero-shot):")
    for ref in (r for r in result.reference_rows if r["kind"] == "majority_baseline"):
        student = next(r["macro_f1"] for r in result.reference_rows
                       if r["kind"] == "student_zero_shot" and r["test_set"] == ref["test_set"])
        print(f"  {ref['test_set']}  (baseline {ref['macro_f1']:.3f} · student {student:.3f})")
        for r in sorted((r for r in result.eval_rows if r["test_set"] == ref["test_set"]),
                        key=lambda r: -r["macro_f1"]):
            flag = "  COLLAPSED" if r["collapsed"] else ""
            print(f"    {r['adapter']:9} {r['supervision']:13} {r['macro_f1']:.3f}{flag}")
    for w in result.warnings:
        print(f"WARNING: {w}")
    print(f"\nTimings: " + ", ".join(f"{k} {v:.0f}s" for k, v in result.timings.items()))
    print(f"Saved run -> {run_dir.relative_to(ROOT)} ({len(charts)} charts)")


if __name__ == "__main__":
    main()
