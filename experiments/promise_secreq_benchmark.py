"""
promise_secreq_benchmark.py
===========================
TeasCo-FT Benchmark: Train on PROMISE, test on SecReq.

This script evaluates the Knowledge Distillation + Cascade pipeline
on a cross-dataset transfer task:
  - TRAIN on PROMISE_exp (969 general requirements)
    - Labels mapped to: 'sec' (if SE) and 'nonsec' (all other NFRs/F)
  - TEST  on SecReq (CPN, GPS, ePurse) 
    - 510 industry-specific security requirements

We compare:
  - TeasCo-FT Cascade (Logistic Regression + Corrector via MockTeacher KD)
  - Vanilla Logistic Regression (no KD)
  - Vanilla Random Forest baseline

Usage
-----
    python experiments/promise_secreq_benchmark.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make teascoFT importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from rich.console import Console
from rich.table import Table

from teascoFT import (
    KnowledgeDistillationTrainer,
    TFIDFExtractor,
    macro_f1,
)
from teascoFT.teacher.llm_teacher import BaseTeacher

warnings.filterwarnings("ignore")
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_promise_binary(path: str) -> Tuple[List[str], List[str]]:
    """
    Load PROMISE_exp from ARFF file and map to binary (sec vs nonsec).
    SE -> sec
    F, A, L, LF, MN, O, PE, SC, US, FT, PO -> nonsec
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
                cls_raw = line.rsplit(",", 1)[-1].strip().upper()
                match = re.search(r"'(.*)'", line)
                if match:
                    text = match.group(1).strip()
                else:
                    parts = line.split(",", 1)
                    text = parts[1].rsplit(",", 1)[0].strip().strip("'\"")
                
                if text and cls_raw:
                    texts.append(text)
                    # Binary mapping
                    if cls_raw == "SE":
                        classes.append("sec")
                    else:
                        classes.append("nonsec")
                        
    return texts, classes

def load_secreq(path: str) -> Tuple[List[str], List[str]]:
    """
    Load a SecReq CSV file. Format: <requirement text>;<label>
    Labels are: sec, nonsec.
    """
    texts, labels = [], []
    if not os.path.exists(path):
        return texts, labels
        
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip("\r")
            if not line or ";" not in line:
                continue
            text, label = line.rsplit(";", 1)
            text = text.strip().strip('"').strip("'")
            label = label.strip().lower()
            if label == "non-sec":
                label = "nonsec"
            if text and label in ("sec", "nonsec"):
                texts.append(text)
                labels.append(label)
    return texts, labels

# ─────────────────────────────────────────────────────────────────────────────
# Mock Teacher
# ─────────────────────────────────────────────────────────────────────────────

class RFTeacher(BaseTeacher):
    """Uses a Random Forest to simulate Teacher soft labels."""
    def __init__(self, class_list: List[str]):
        self._classes = class_list
        self._rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
        self._le = None
        self._fitted = False

    @property
    def classes(self) -> List[str]:
        return self._classes

    def fit_on_features(self, X, y: List[str]):
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder().fit(self._classes)
        self._rf.fit(X, self._le.transform(y))
        self._fitted = True

    def generate_soft_labels(self, texts: List[str], classes=None) -> List[Dict[str, float]]:
        raise NotImplementedError("Use generate_soft_labels_from_features")

    def generate_soft_labels_from_features(self, X) -> List[Dict[str, float]]:
        if not self._fitted:
            raise RuntimeError("Teacher not fitted")
        probas = self._rf.predict_proba(X)
        soft_labels = []
        for prob in probas:
            soft_labels.append({c: float(p) for c, p in zip(self._le.classes_, prob)})
        return soft_labels

# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold cyan]=================================================================[/]")
    console.print("[bold cyan]  TeasCo-FT: PROMISE → SecReq Cross-Dataset Benchmark[/]")
    console.print("[bold cyan]=================================================================[/]\n")

    base_dir = Path(__file__).parent.parent
    promise_path = base_dir / "data" / "PROMISE_exp.arff"
    secreq_dir = base_dir / "data" / "secreq"
    
    # 1. Load Training Data (PROMISE)
    console.print(f"[bold]📂 Loading PROMISE dataset (Training)...[/]")
    X_train, y_train = load_promise_binary(str(promise_path))
    console.print(f"   Train samples: {len(X_train)}")
    console.print(f"   Class distribution: {Counter(y_train)}\n")

    # 2. Load Testing Data (SecReq)
    console.print(f"[bold]📂 Loading SecReq datasets (Testing)...[/]")
    projects = ["CPN", "GPS", "ePurse"]
    test_datasets = {}
    
    X_test_all, y_test_all = [], []
    for proj in projects:
        path = secreq_dir / f"{proj}.csv"
        X_p, y_p = load_secreq(str(path))
        test_datasets[proj] = (X_p, y_p)
        X_test_all.extend(X_p)
        y_test_all.extend(y_p)
        console.print(f"   - {proj}: {len(X_p)} samples {Counter(y_p)}")
        
    test_datasets["Combined"] = (X_test_all, y_test_all)
    console.print(f"   [bold]Total Test samples: {len(X_test_all)}[/]\n")

    # 3. Setup Feature Extractor
    extractor = TFIDFExtractor(max_features=10000, min_df=2)
    X_train_vec = extractor.fit_transform(X_train)
    
    # 4. Teacher Simulation
    console.print("[bold yellow]🎓 Simulating Teacher LLM soft labels...[/]")
    t0 = time.time()
    teacher = RFTeacher(class_list=["sec", "nonsec"])
    teacher.fit_on_features(X_train_vec, y_train)
    soft_labels = teacher.generate_soft_labels_from_features(X_train_vec)
    console.print(f"   Done in {time.time() - t0:.2f}s\n")

    # 5. Train Models
    console.print("[bold cyan]⚙️  Training Models...[/]")
    
    # Baseline 1: Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_vec, y_train)
    
    # Baseline 2: Random Forest
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    rf.fit(X_train_vec, y_train)
    
    # TeasCo-FT
    trainer = KnowledgeDistillationTrainer(
        teacher=teacher,
        extractor=extractor,
        confidence_threshold=0.60,
        corrector_model_type="random_forest"
    )
    # We pass the precomputed soft labels to avoid the LLM API call
    pipeline = trainer.fit(X_train, y_train, soft_labels=soft_labels, classes=["sec", "nonsec"])
    
    console.print("\n[bold green]✅ Training Complete[/]\n")

    # 6. Evaluate on Test Sets
    results_table = Table(title="Cross-Dataset Macro F1 Results (Train: PROMISE)", show_header=True)
    results_table.add_column("Test Set", style="cyan", no_wrap=True)
    results_table.add_column("Logistic Reg", justify="right")
    results_table.add_column("Random Forest", justify="right")
    results_table.add_column("TeasCo-FT", justify="right", style="bold green")
    
    for proj_name, (X_t, y_t) in test_datasets.items():
        X_t_vec = extractor.transform(X_t)
        
        # Baselines
        lr_pred = lr.predict(X_t_vec)
        lr_f1 = macro_f1(y_t, lr_pred)
        
        rf_pred = rf.predict(X_t_vec)
        rf_f1 = macro_f1(y_t, rf_pred)
        
        # TeasCo-FT
        teasco_pred = pipeline.predict(X_t)
        teasco_f1 = macro_f1(y_t, teasco_pred)
        
        results_table.add_row(
            proj_name,
            f"{lr_f1:.4f}",
            f"{rf_f1:.4f}",
            f"{teasco_f1:.4f} 🏆" if teasco_f1 >= max(lr_f1, rf_f1) else f"{teasco_f1:.4f}"
        )
        
    console.print(results_table)
    console.print("\n[bold]Detailed TeasCo-FT Report on Combined SecReq dataset:[/]")
    teasco_pred_all = pipeline.predict(X_test_all)
    print(classification_report(y_test_all, teasco_pred_all, target_names=["nonsec", "sec"]))
    console.print(f"Cascade Stats: {pipeline.cascade_stats}")

if __name__ == "__main__":
    main()
