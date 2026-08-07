"""
kd_trainer.py
=============
KnowledgeDistillationTrainer — the central orchestrator that runs the
complete training pipeline from raw texts to a production-ready
CascadePipeline.

Pipeline Steps
--------------
1. Feature Extraction   : Fit the vectorizer and transform training texts.
2. Soft Label Generation: Run Teacher LLM on all training texts.
3. Base Student Training: Train Logistic Regression on Teacher soft labels (KD).
4. Hard Example Mining  : Find samples where Base Student is wrong or unsure.
5. Corrector Training   : Train Random Forest / SVM on hard examples only.
6. Pipeline Assembly    : Bundle everything into a CascadePipeline.

Usage
-----
    from teascoFT import KnowledgeDistillationTrainer, OpenAITeacher, TFIDFExtractor

    teacher   = OpenAITeacher(api_key="sk-...", model="gpt-4o-mini",
                              classes=["positive", "negative", "neutral"])
    extractor = TFIDFExtractor(max_features=30_000)

    trainer = KnowledgeDistillationTrainer(
        teacher=teacher,
        extractor=extractor,
        confidence_threshold=0.65,
        corrector_model_type="random_forest",
    )

    # Full training in one call
    pipeline = trainer.fit(texts_train, y_train)

    # Evaluate
    from teascoFT.utils import macro_f1
    preds = pipeline.predict(texts_val)
    print(f"Val Macro F1: {macro_f1(y_val, preds):.4f}")

    # Save for production
    pipeline.save("my_classifier.pkl")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score

from teascoFT.cascade.pipeline import CascadePipeline
from teascoFT.features.tfidf_extractor import TFIDFExtractor
from teascoFT.features.embedding_extractor import EmbeddingExtractor
from teascoFT.student.base_student import BaseStudent
from teascoFT.student.corrector import Corrector
from teascoFT.teacher.llm_teacher import BaseTeacher

logger = logging.getLogger(__name__)

FeatureExtractor = Union[TFIDFExtractor, EmbeddingExtractor]


# ─────────────────────────────────────────────────────────────────────────────
# Training Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingReport:
    """
    Summary of training metrics across all pipeline stages.
    Printed at the end of trainer.fit().
    """
    n_train: int = 0
    n_hard: int = 0
    hard_fraction: float = 0.0

    # Macro F1 scores measured on the training set (indicative, not held-out)
    teacher_hard_f1: float = 0.0         # Teacher hard labels vs ground truth
    base_student_f1: float = 0.0         # Base student vs ground truth
    corrector_f1_on_hard: float = 0.0    # Corrector vs ground truth on hard subset
    cascade_train_f1: float = 0.0        # Full cascade vs ground truth

    stage_durations: Dict[str, float] = field(default_factory=dict)

    def display(self) -> None:
        """Pretty-print the report using rich (falls back to plain text)."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(
                title="[bold cyan]TeasCo-FT Training Report[/bold cyan]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Stage", style="cyan", no_wrap=True)
            table.add_column("Macro F1 (train)", justify="right", style="green")
            table.add_column("Duration", justify="right", style="yellow")

            def fmt_dur(key):
                s = self.stage_durations.get(key, 0)
                return f"{s:.1f}s"

            table.add_row(
                "Teacher (hard label baseline)",
                f"{self.teacher_hard_f1:.4f}",
                fmt_dur("teacher_labeling"),
            )
            table.add_row(
                "Base Student (after KD)",
                f"{self.base_student_f1:.4f}",
                fmt_dur("base_student_training"),
            )
            table.add_row(
                f"Corrector (on {self.n_hard} hard examples)",
                f"{self.corrector_f1_on_hard:.4f}" if self.corrector_f1_on_hard else "—",
                fmt_dur("corrector_training"),
            )
            table.add_row(
                "Full Cascade",
                f"{self.cascade_train_f1:.4f}",
                "—",
            )
            console.print()
            console.print(table)
            console.print(
                f"  Hard examples: [yellow]{self.n_hard}[/yellow] / {self.n_train} "
                f"({self.hard_fraction*100:.1f}%)"
            )
            console.print()

        except ImportError:
            # Plain-text fallback
            lines = [
                "\n" + "=" * 55,
                "   TeasCo-FT Training Report",
                "=" * 55,
                f"  Teacher hard F1      : {self.teacher_hard_f1:.4f}",
                f"  Base Student F1      : {self.base_student_f1:.4f}",
                f"  Corrector F1 (hard)  : {self.corrector_f1_on_hard:.4f}",
                f"  Full Cascade F1      : {self.cascade_train_f1:.4f}",
                f"  Hard examples        : {self.n_hard}/{self.n_train} "
                f"({self.hard_fraction*100:.1f}%)",
                "=" * 55 + "\n",
            ]
            print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Distillation Trainer
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDistillationTrainer:
    """
    Orchestrates the full Knowledge Distillation + Cascade training pipeline.

    Parameters
    ----------
    teacher : BaseTeacher
        Fitted (or ready-to-use) Teacher LLM instance.
    extractor : TFIDFExtractor or EmbeddingExtractor
        Feature extractor for converting raw texts to vectors.
    confidence_threshold : float
        Confidence below which the Base Student defers to the Corrector.
        Default: 0.6.
    corrector_model_type : {"random_forest", "svm"}
        Underlying model for the Corrector. Default: "random_forest".
    corrector_C : float
        SVM regularization (only applies when model_type="svm"). Default: 1.0.
    corrector_n_estimators : int
        Number of RF trees. Default: 200.
    base_student_C : float
        Logistic Regression regularization. Default: 1.0.
    base_student_solver : str
        LR solver. Default: "lbfgs".
    use_low_confidence_as_hard : bool
        If True, samples where Base Student confidence < threshold are ALSO
        added to the hard set, even if the prediction was correct. Default: True.
    verbose : bool
        Log progress messages. Default: True.
    """

    def __init__(
        self,
        teacher: BaseTeacher,
        extractor: FeatureExtractor,
        confidence_threshold: float = 0.6,
        corrector_model_type: Literal["random_forest", "svm"] = "random_forest",
        corrector_C: float = 1.0,
        corrector_n_estimators: int = 200,
        base_student_C: float = 1.0,
        base_student_solver: str = "lbfgs",
        use_low_confidence_as_hard: bool = True,
        verbose: bool = True,
    ) -> None:
        self.teacher = teacher
        self.extractor = extractor
        self.confidence_threshold = confidence_threshold
        self.corrector_model_type = corrector_model_type
        self.corrector_C = corrector_C
        self.corrector_n_estimators = corrector_n_estimators
        self.base_student_C = base_student_C
        self.base_student_solver = base_student_solver
        self.use_low_confidence_as_hard = use_low_confidence_as_hard
        self.verbose = verbose

        # Set up logging level
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

        # State after training
        self._base_student: Optional[BaseStudent] = None
        self._corrector: Optional[Corrector] = None
        self._pipeline: Optional[CascadePipeline] = None
        self._report: Optional[TrainingReport] = None

    # ── Main Entry Point ───────────────────────────────────────────────────

    def fit(
        self,
        texts: List[str],
        y_true: List[str],
        classes: Optional[List[str]] = None,
        soft_labels: Optional[List[Dict[str, float]]] = None,
    ) -> CascadePipeline:
        """
        Run the complete KD training pipeline.

        Parameters
        ----------
        texts : list of str
            Training texts.
        y_true : list of str
            Ground truth labels for each text.
        classes : list of str, optional
            Class names. If None, inferred from unique values in y_true
            (and from teacher.classes if available).
        soft_labels : list of dict, optional
            Pre-computed Teacher soft labels. If provided, the Teacher LLM
            is NOT called, saving API costs. Useful for re-training.

        Returns
        -------
        CascadePipeline — ready for production inference.
        """
        report = TrainingReport(n_train=len(texts))
        y_true = list(y_true)

        # ── Resolve classes ───────────────────────────────────────────────
        if classes is None:
            classes = sorted(set(y_true))
            if hasattr(self.teacher, 'classes') and self.teacher.classes:
                classes = self.teacher.classes
        logger.info("Classes: %s", classes)

        # ── Step 1: Feature Extraction ────────────────────────────────────
        logger.info("─" * 50)
        logger.info("Step 1/5 — Feature Extraction")
        t0 = time.time()
        X = self.extractor.fit_transform(texts)
        report.stage_durations["feature_extraction"] = time.time() - t0
        logger.info("  Features shape: %s", X.shape)

        # ── Step 2: Teacher Soft Label Generation ─────────────────────────
        logger.info("─" * 50)
        logger.info("Step 2/5 — Teacher Soft Label Generation")
        t0 = time.time()

        if soft_labels is not None:
            logger.info("  Using pre-computed soft labels (%d provided).", len(soft_labels))
        else:
            logger.info("  Calling Teacher LLM for %d texts...", len(texts))
            soft_labels = self.teacher.generate_soft_labels(texts, classes=classes)

        report.stage_durations["teacher_labeling"] = time.time() - t0

        # Compute Teacher's own hard-label F1 as a baseline
        teacher_hard_preds = [max(sl, key=sl.get) for sl in soft_labels]
        report.teacher_hard_f1 = f1_score(
            y_true, teacher_hard_preds, average="macro", zero_division=0
        )
        logger.info("  Teacher hard-label Macro F1 vs ground truth: %.4f",
                    report.teacher_hard_f1)

        # ── Step 3: Base Student Training (Knowledge Distillation) ────────
        logger.info("─" * 50)
        logger.info("Step 3/5 — Base Student Training (Knowledge Distillation)")
        t0 = time.time()

        self._base_student = BaseStudent(
            classes=classes,
            C=self.base_student_C,
            solver=self.base_student_solver,
        )
        self._base_student.train(X, soft_labels)
        report.stage_durations["base_student_training"] = time.time() - t0

        base_preds, base_confidence = self._base_student.predict_with_confidence(X)
        report.base_student_f1 = f1_score(
            y_true, base_preds, average="macro", zero_division=0
        )
        logger.info("  Base Student Macro F1 vs ground truth: %.4f",
                    report.base_student_f1)

        # ── Step 4: Hard Example Mining ───────────────────────────────────
        logger.info("─" * 50)
        logger.info("Step 4/5 — Hard Example Mining")
        hard_mask = self._identify_hard_examples(
            base_preds, base_confidence, np.array(y_true)
        )
        hard_indices = np.where(hard_mask)[0]
        report.n_hard = len(hard_indices)
        report.hard_fraction = report.n_hard / len(texts)
        logger.info(
            "  Hard examples: %d / %d (%.1f%%)",
            report.n_hard, len(texts), report.hard_fraction * 100,
        )

        # ── Step 5: Corrector Training ────────────────────────────────────
        logger.info("─" * 50)
        logger.info("Step 5/5 — Corrector Training")
        t0 = time.time()

        self._corrector = Corrector(
            classes=classes,
            model_type=self.corrector_model_type,
            confidence_threshold=self.confidence_threshold,
            n_estimators=self.corrector_n_estimators,
            C=self.corrector_C,
        )

        if report.n_hard > 0:
            X_hard = self._index_features(X, hard_indices)
            y_hard = np.array(y_true)[hard_indices]
            self._corrector.train(X_hard, y_hard)

            if self._corrector.is_fitted:
                corrector_preds = self._corrector.predict(X_hard)
                report.corrector_f1_on_hard = f1_score(
                    y_hard, corrector_preds, average="macro", zero_division=0
                )
                logger.info(
                    "  Corrector Macro F1 on hard examples: %.4f",
                    report.corrector_f1_on_hard,
                )

        report.stage_durations["corrector_training"] = time.time() - t0

        # ── Assemble the Cascade Pipeline ─────────────────────────────────
        self._pipeline = CascadePipeline(
            extractor=self.extractor,
            base_student=self._base_student,
            corrector=self._corrector,
            confidence_threshold=self.confidence_threshold,
            classes=classes,
        )

        # Measure full cascade F1 on training data
        cascade_preds = self._pipeline.predict(texts)
        report.cascade_train_f1 = f1_score(
            y_true, cascade_preds, average="macro", zero_division=0
        )
        logger.info(
            "  Full Cascade Macro F1 vs ground truth: %.4f",
            report.cascade_train_f1,
        )

        self._report = report
        report.display()

        return self._pipeline

    # ── Convenience Methods ────────────────────────────────────────────────

    @property
    def pipeline(self) -> Optional[CascadePipeline]:
        """Return the trained CascadePipeline, or None if not yet trained."""
        return self._pipeline

    @property
    def report(self) -> Optional[TrainingReport]:
        """Return the training report from the last .fit() call."""
        return self._report

    # ── Private Helpers ────────────────────────────────────────────────────

    def _identify_hard_examples(
        self,
        predictions: np.ndarray,
        confidence: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """
        Return a boolean mask marking 'hard' examples.

        A sample is hard if:
          - The Base Student predicted incorrectly, OR
          - The Base Student confidence is below the threshold (if enabled).
        """
        wrong_mask = predictions != y_true
        if self.use_low_confidence_as_hard:
            low_conf_mask = confidence < self.confidence_threshold
            return wrong_mask | low_conf_mask
        return wrong_mask

    @staticmethod
    def _index_features(X, indices: np.ndarray):
        """Index a feature matrix (sparse or dense) by row indices."""
        from scipy.sparse import issparse
        if issparse(X):
            return X[indices]
        return X[indices]
