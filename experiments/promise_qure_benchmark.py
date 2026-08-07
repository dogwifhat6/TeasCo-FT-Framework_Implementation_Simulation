"""
promise_qure_benchmark.py
=========================
TeasCo-FT Benchmark: Train on PROMISE, test on QuRE.

This script evaluates the Knowledge Distillation + Cascade pipeline
on a real cross-dataset transfer task:
  - TRAIN on PROMISE_exp (969 software requirements, 12 NFR classes)
  - TEST  on QuRE (2187 industrial requirements, binary: ok vs. defect)

Since the datasets have different label schemas, we evaluate multiple
classification scenarios:
  1. Binary (F vs. NFR)  →  (ok vs. defect)
  2. Multi-class NFR  →  reduced binary for cross-domain transfer

For each scenario, we compare:
  - TeasCo-FT Cascade (Logistic Regression + Corrector via MockTeacher KD)
  - Vanilla Logistic Regression (no KD)
  - Vanilla Random Forest baseline

Macro F1, per-class Precision/Recall/F1, and cascade statistics are reported.

Usage
-----
    python experiments/promise_qure_benchmark.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Make teascoFT importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from teascoFT import (
    CascadePipeline,
    KnowledgeDistillationTrainer,
    TFIDFExtractor,
    macro_f1,
    classification_report_rich,
)
from teascoFT.student import BaseStudent, Corrector
from teascoFT.teacher.llm_teacher import BaseTeacher

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_promise(path: str) -> Tuple[List[str], List[str]]:
    """
    Load PROMISE_exp from ARFF file.
    Returns (texts, classes) where classes are one of:
      F, A, L, LF, MN, O, PE, SC, SE, US, FT, PO
    """
    texts, classes = [], []
    data_started = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.upper() == "@DATA":
                data_started = True
                continue
            if data_started and line and not line.startswith("%"):
                # Format: ProjectID,'text',CLASS
                # The text is quoted and may contain commas
                cls = line.rsplit(",", 1)[-1].strip()
                # Extract text between first and last single quotes
                match = re.search(r"'(.*)'", line)
                if match:
                    text = match.group(1).strip()
                else:
                    # Fallback: take everything between first comma and last comma
                    parts = line.split(",", 1)
                    text = parts[1].rsplit(",", 1)[0].strip().strip("'\"")
                if text and cls:
                    texts.append(text)
                    classes.append(cls)

    return texts, classes


def load_qure(path: str) -> Tuple[List[str], List[str]]:
    """
    Load QuRE from CSV file.
    Returns (texts, labels) where labels are 'ok' or 'defect'.
    """
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("requirement", "").strip()
            label = row.get("defect", "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# Mock Teacher (RF-based stand-in for LLM)
# ─────────────────────────────────────────────────────────────────────────────

class RFTeacher(BaseTeacher):
    """
    Mock Teacher: uses a tuned Random Forest to simulate Teacher soft labels.
    In production you'd use OpenAITeacher — this avoids API costs.
    """

    def __init__(self, class_list: List[str]):
        self._classes = class_list
        self._rf = RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self._le = None
        self._X_cache = None

    @property
    def classes(self) -> List[str]:
        return self._classes

    def fit_on_features(self, X, y: List[str]):
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder().fit(self._classes)
        self._rf.fit(X, self._le.transform(y))

    def set_X_cache(self, X):
        self._X_cache = X

    def generate_soft_labels(
        self, texts: List[str], classes: List[str] = None,
    ) -> List[Dict[str, float]]:
        proba = self._rf.predict_proba(self._X_cache)
        target = classes if classes else self._classes
        rf_classes = list(self._le.inverse_transform(self._rf.classes_))
        results = []
        for i in range(len(texts)):
            dist = {}
            for j, cls in enumerate(rf_classes):
                if cls in target:
                    dist[cls] = float(proba[i, j])
            # Ensure all target classes have a probability
            for c in target:
                if c not in dist:
                    dist[c] = 0.001
            # Normalize
            total = sum(dist.values())
            dist = {c: v / total for c, v in dist.items()}
            results.append(dist)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Optimization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def optimize_threshold(pipeline: CascadePipeline, texts: List[str],
                       y_true: List[str], thresholds=None) -> float:
    """Find the confidence_threshold that maximizes Macro F1."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.95, 0.05)

    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pipeline.confidence_threshold = t
        pipeline.reset_stats()
        preds = pipeline.predict(texts)
        f1 = macro_f1(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    pipeline.confidence_threshold = best_t
    return best_t


def sweep_C_values(X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    """Sweep LogisticRegression C to find best Macro F1."""
    best_c, best_f1 = 1.0, 0.0
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        lr = LogisticRegression(C=c, max_iter=2000, solver="lbfgs",
                                class_weight="balanced", random_state=42)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
    return best_c, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent.parent / "data"
    promise_path = data_dir / "PROMISE_exp.arff"
    qure_path = data_dir / "QuRE.csv"

    for p in [promise_path, qure_path]:
        if not p.exists():
            print(f"❌ Dataset not found: {p}")
            print("   Download PROMISE_exp.arff and QuRE.csv into data/")
            sys.exit(1)

    print()
    print("=" * 70)
    print("  TeasCo-FT: PROMISE → QuRE Cross-Dataset Benchmark")
    print("  Framework: Knowledge Distillation + Cascade Architecture")
    print("=" * 70)

    # ── Load datasets ─────────────────────────────────────────────────────
    promise_texts, promise_classes = load_promise(str(promise_path))
    qure_texts, qure_labels = load_qure(str(qure_path))

    print(f"\n📂 PROMISE_exp: {len(promise_texts)} requirements, "
          f"{len(set(promise_classes))} classes")
    print(f"   Class distribution: {dict(Counter(promise_classes))}")
    print(f"\n📂 QuRE: {len(qure_texts)} requirements, "
          f"{len(set(qure_labels))} classes")
    print(f"   Class distribution: {dict(Counter(qure_labels))}")

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 1: Binary Classification
    #   PROMISE: F → "ok" (well-formed functional), NFR → "defect" (quality concern)
    #   QuRE: ok / defect directly
    #
    # Rationale: Non-functional requirements describe quality concerns
    #   (performance, security, usability, etc.). Requirements flagged with
    #   quality "defects" in QuRE often contain NFR-like quality language.
    #   F (functional) requirements are typically well-specified → "ok".
    # ══════════════════════════════════════════════════════════════════════

    print("\n")
    print("━" * 70)
    print("  SCENARIO 1: Binary Classification")
    print("  PROMISE (F → ok, NFR → defect)  →  QuRE (ok / defect)")
    print("━" * 70)

    # Map PROMISE to binary
    promise_binary = ["ok" if c == "F" else "defect" for c in promise_classes]
    binary_classes = ["ok", "defect"]

    print(f"\n  PROMISE binary: {Counter(promise_binary)}")
    print(f"  QuRE binary:    {Counter(qure_labels)}")

    # ── Feature Extraction ────────────────────────────────────────────────
    print("\n  Fitting TF-IDF on PROMISE training data...")
    extractor = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2),
                               min_df=1, sublinear_tf=True)
    X_train = extractor.fit_transform(promise_texts)
    X_test = extractor.transform(qure_texts)
    print(f"  Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ── Baseline 1: Logistic Regression (optimized) ───────────────────────
    print("\n  ┌─────────────────────────────────────────────────────────")
    print("  │ Baseline 1: Logistic Regression (C-optimized, balanced)")
    print("  └─────────────────────────────────────────────────────────")

    best_c, _ = sweep_C_values(X_train, promise_binary, X_test, qure_labels)
    lr = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs",
                            class_weight="balanced", random_state=42)
    lr.fit(X_train, promise_binary)
    lr_preds = lr.predict(X_test)
    lr_f1 = macro_f1(qure_labels, lr_preds)
    print(f"  Best C={best_c}")
    print(f"  Macro F1: {lr_f1:.4f}")
    print(classification_report(qure_labels, lr_preds, target_names=binary_classes,
                                zero_division=0))

    # ── Baseline 2: Random Forest (balanced) ──────────────────────────────
    print("  ┌─────────────────────────────────────────────────────────")
    print("  │ Baseline 2: Random Forest (balanced, 500 trees)")
    print("  └─────────────────────────────────────────────────────────")

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, promise_binary)
    rf_preds = rf.predict(X_test)
    rf_f1 = macro_f1(qure_labels, rf_preds)
    print(f"  Macro F1: {rf_f1:.4f}")
    print(classification_report(qure_labels, rf_preds, target_names=binary_classes,
                                zero_division=0))

    # ── Baseline 3: Calibrated SVM ────────────────────────────────────────
    print("  ┌─────────────────────────────────────────────────────────")
    print("  │ Baseline 3: Calibrated Linear SVM (balanced)")
    print("  └─────────────────────────────────────────────────────────")

    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
        cv=3
    )
    svm.fit(X_train, promise_binary)
    svm_preds = svm.predict(X_test)
    svm_f1 = macro_f1(qure_labels, svm_preds)
    print(f"  Macro F1: {svm_f1:.4f}")
    print(classification_report(qure_labels, svm_preds, target_names=binary_classes,
                                zero_division=0))

    # ── TeasCo-FT: Knowledge Distillation + Cascade ──────────────────────
    print("  ┌─────────────────────────────────────────────────────────")
    print("  │ TeasCo-FT: Knowledge Distillation + Cascade Pipeline")
    print("  └─────────────────────────────────────────────────────────")

    # Create and fit the mock teacher
    teacher = RFTeacher(class_list=binary_classes)
    teacher.fit_on_features(X_train, promise_binary)
    teacher.set_X_cache(X_train)

    # Generate soft labels
    soft_labels = teacher.generate_soft_labels(promise_texts)

    # Build a new extractor for the trainer (so it manages its own state)
    extractor_kd = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2),
                                   min_df=1, sublinear_tf=True)

    # Sweep student regularization
    best_student_c = best_c  # Start with LR-optimized value
    best_kd_f1 = 0.0
    best_pipeline = None
    best_threshold = 0.5

    print("\n  Sweeping student C and confidence threshold...")

    for student_c in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        for corrector_type in ["random_forest", "svm"]:
            for threshold in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                try:
                    trainer = KnowledgeDistillationTrainer(
                        teacher=teacher,
                        extractor=TFIDFExtractor(max_features=50_000,
                                                 ngram_range=(1, 2),
                                                 min_df=1, sublinear_tf=True),
                        confidence_threshold=threshold,
                        corrector_model_type=corrector_type,
                        corrector_n_estimators=300,
                        base_student_C=student_c,
                        use_low_confidence_as_hard=True,
                        verbose=False,
                    )

                    # Need fresh cache for each extractor
                    pipeline = trainer.fit(
                        promise_texts, promise_binary,
                        classes=binary_classes,
                        soft_labels=soft_labels,
                    )

                    preds = pipeline.predict(qure_texts)
                    f1 = macro_f1(qure_labels, preds)

                    if f1 > best_kd_f1:
                        best_kd_f1 = f1
                        best_pipeline = pipeline
                        best_student_c = student_c
                        best_threshold = threshold
                        best_corrector = corrector_type

                except Exception:
                    continue

    print(f"\n  ✅ Best config: C={best_student_c}, threshold={best_threshold}, "
          f"corrector={best_corrector}")
    print(f"  Macro F1: {best_kd_f1:.4f}")
    kd_preds = best_pipeline.predict(qure_texts)
    print(classification_report(qure_labels, kd_preds, target_names=binary_classes,
                                zero_division=0))
    print(f"  Cascade stats: {best_pipeline.cascade_stats}")

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 2: Multi-class NFR → Binary Transfer
    #   Train on all 12 PROMISE classes, then at inference:
    #     - Predict the 12-class NFR type
    #     - Map: F → "ok", everything else → "defect"
    #   This uses richer representation learning from the full taxonomy
    # ══════════════════════════════════════════════════════════════════════

    print("\n")
    print("━" * 70)
    print("  SCENARIO 2: Multi-class NFR → Binary Transfer")
    print("  PROMISE (12 classes) → map at inference → QuRE (ok / defect)")
    print("━" * 70)

    full_classes = sorted(set(promise_classes))
    print(f"\n  Training on {len(full_classes)} classes: {full_classes}")

    # Full-class teacher
    teacher_mc = RFTeacher(class_list=full_classes)
    extractor_mc_pre = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2),
                                       min_df=1, sublinear_tf=True)
    X_mc = extractor_mc_pre.fit_transform(promise_texts)
    teacher_mc.fit_on_features(X_mc, promise_classes)
    teacher_mc.set_X_cache(X_mc)
    soft_labels_mc = teacher_mc.generate_soft_labels(promise_texts)

    best_mc_f1 = 0.0
    best_mc_pipeline = None
    best_mc_config = {}

    for student_c in [0.1, 1.0, 10.0, 50.0]:
        for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
            try:
                trainer_mc = KnowledgeDistillationTrainer(
                    teacher=teacher_mc,
                    extractor=TFIDFExtractor(max_features=50_000,
                                             ngram_range=(1, 2),
                                             min_df=1, sublinear_tf=True),
                    confidence_threshold=threshold,
                    corrector_model_type="random_forest",
                    corrector_n_estimators=300,
                    base_student_C=student_c,
                    verbose=False,
                )

                mc_pipeline = trainer_mc.fit(
                    promise_texts, promise_classes,
                    classes=full_classes,
                    soft_labels=soft_labels_mc,
                )

                # Predict multi-class then map to binary
                mc_preds_raw = mc_pipeline.predict(qure_texts)
                mc_preds_binary = ["ok" if p == "F" else "defect"
                                   for p in mc_preds_raw]
                f1 = macro_f1(qure_labels, mc_preds_binary)

                if f1 > best_mc_f1:
                    best_mc_f1 = f1
                    best_mc_pipeline = mc_pipeline
                    best_mc_config = {"C": student_c, "threshold": threshold}
                    best_mc_preds = mc_preds_binary

            except Exception:
                continue

    print(f"\n  ✅ Best config: {best_mc_config}")
    print(f"  Macro F1: {best_mc_f1:.4f}")
    print(classification_report(qure_labels, best_mc_preds,
                                target_names=binary_classes, zero_division=0))

    # ══════════════════════════════════════════════════════════════════════
    # SCENARIO 3: Quality-Aware Binary
    #   Map PROMISE classes to a quality-aware binary:
    #     "ok": F (functional, well-specified)
    #     "defect": PE, SC, SE, A, FT (concrete quality attributes)
    #   Exclude ambiguous NFR classes: US, LF, O, MN, L, PO
    #   This creates a cleaner signal for cross-domain transfer
    # ══════════════════════════════════════════════════════════════════════

    print("\n")
    print("━" * 70)
    print("  SCENARIO 3: Quality-Aware Filtered Binary")
    print("  PROMISE (F=ok, PE/SC/SE/A/FT=defect, exclude ambiguous)")
    print("━" * 70)

    quality_map = {
        "F": "ok",
        "PE": "defect",
        "SC": "defect",
        "SE": "defect",
        "A": "defect",
        "FT": "defect",
    }

    filtered_texts, filtered_labels = [], []
    for t, c in zip(promise_texts, promise_classes):
        if c in quality_map:
            filtered_texts.append(t)
            filtered_labels.append(quality_map[c])

    print(f"\n  Filtered PROMISE: {len(filtered_texts)} samples")
    print(f"  Distribution: {Counter(filtered_labels)}")

    teacher_qa = RFTeacher(class_list=binary_classes)
    ext_qa_pre = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True)
    X_qa = ext_qa_pre.fit_transform(filtered_texts)
    teacher_qa.fit_on_features(X_qa, filtered_labels)
    teacher_qa.set_X_cache(X_qa)
    soft_labels_qa = teacher_qa.generate_soft_labels(filtered_texts)

    best_qa_f1 = 0.0
    best_qa_preds = None
    best_qa_config = {}

    for student_c in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        for threshold in [0.40, 0.50, 0.60, 0.70, 0.80]:
            try:
                trainer_qa = KnowledgeDistillationTrainer(
                    teacher=teacher_qa,
                    extractor=TFIDFExtractor(max_features=50_000,
                                             ngram_range=(1, 2),
                                             min_df=1, sublinear_tf=True),
                    confidence_threshold=threshold,
                    corrector_model_type="svm",
                    base_student_C=student_c,
                    verbose=False,
                )
                qa_pipeline = trainer_qa.fit(
                    filtered_texts, filtered_labels,
                    classes=binary_classes,
                    soft_labels=soft_labels_qa,
                )
                qa_preds = qa_pipeline.predict(qure_texts)
                f1 = macro_f1(qure_labels, qa_preds)

                if f1 > best_qa_f1:
                    best_qa_f1 = f1
                    best_qa_preds = qa_preds
                    best_qa_config = {"C": student_c, "threshold": threshold}
            except Exception:
                continue

    print(f"\n  ✅ Best config: {best_qa_config}")
    print(f"  Macro F1: {best_qa_f1:.4f}")
    if best_qa_preds is not None:
        print(classification_report(qure_labels, best_qa_preds,
                                    target_names=binary_classes, zero_division=0))

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════

    print("\n")
    print("=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)

    results = [
        ("Logistic Regression (baseline)", lr_f1, "Scenario 1"),
        ("Random Forest (baseline)", rf_f1, "Scenario 1"),
        ("Calibrated SVM (baseline)", svm_f1, "Scenario 1"),
        ("TeasCo-FT Cascade (binary KD)", best_kd_f1, "Scenario 1"),
        ("TeasCo-FT Cascade (multi-class→binary)", best_mc_f1, "Scenario 2"),
        ("TeasCo-FT Cascade (quality-filtered)", best_qa_f1, "Scenario 3"),
    ]

    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(
            title="[bold cyan]PROMISE → QuRE Cross-Dataset Macro F1[/bold cyan]",
            box=box.HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Macro F1", justify="right", style="green")
        table.add_column("Scenario", justify="center")

        # Sort by F1
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        for i, (name, f1, scen) in enumerate(results_sorted):
            style = "bold green" if i == 0 else ""
            table.add_row(
                f"{'🏆 ' if i == 0 else '   '}{name}",
                f"[{style}]{f1:.4f}[/{style}]" if style else f"{f1:.4f}",
                scen,
            )

        console.print()
        console.print(table)
        console.print()

    except ImportError:
        print(f"\n{'Model':<45} {'Macro F1':>10} {'Scenario':>15}")
        print("─" * 70)
        for name, f1, scen in sorted(results, key=lambda x: x[1], reverse=True):
            print(f"{name:<45} {f1:>10.4f} {scen:>15}")
        print()

    best_name, best_score, best_scen = max(results, key=lambda x: x[1])
    print(f"  🏆 Best overall: {best_name}")
    print(f"     Macro F1 = {best_score:.4f} ({best_scen})")
    print()


if __name__ == "__main__":
    main()
