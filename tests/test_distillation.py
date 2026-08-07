"""
test_distillation.py
====================
Unit tests for KnowledgeDistillationTrainer (with mocked Teacher).
"""

import numpy as np
import pytest
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT.distillation import KnowledgeDistillationTrainer
from teascoFT.features import TFIDFExtractor
from teascoFT.teacher.llm_teacher import BaseTeacher
from teascoFT.cascade import CascadePipeline


CLASSES = ["tech", "sports", "politics"]
TEXTS = [
    "The new GPU has 80 billion transistors.",
    "The team won the championship yesterday.",
    "The senator proposed new tax legislation.",
    "Quantum computing breakthrough announced.",
    "Soccer player transfers for record fee.",
    "New bill passes the house committee.",
    "AI model beats human at protein folding.",
    "Marathon runner sets world record.",
    "Election results confirmed by officials.",
    "Startup raises 200 million in Series B.",
]
LABELS = [
    "tech", "sports", "politics",
    "tech", "sports", "politics",
    "tech", "sports", "politics",
    "tech",
]


class FixedTeacher(BaseTeacher):
    """Mock teacher that always returns the correct label with 80% confidence."""

    def __init__(self, class_list: List[str], y_true: List[str]):
        self._classes = class_list
        self._y_true = y_true

    @property
    def classes(self) -> List[str]:
        return self._classes

    def generate_soft_labels(
        self, texts: List[str], classes: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        target = classes or self._classes
        soft_labels = []
        for i, text in enumerate(texts):
            true_label = self._y_true[i] if i < len(self._y_true) else self._classes[0]
            sl = {c: 0.1 / (len(target) - 1) for c in target}
            sl[true_label] = 0.80
            soft_labels.append(sl)
        return soft_labels


class TestKnowledgeDistillationTrainer:

    def build_trainer(self, threshold=0.6):
        teacher = FixedTeacher(CLASSES, LABELS)
        extractor = TFIDFExtractor(max_features=100, min_df=1)
        return KnowledgeDistillationTrainer(
            teacher=teacher,
            extractor=extractor,
            confidence_threshold=threshold,
            corrector_model_type="random_forest",
            corrector_n_estimators=20,
            verbose=False,
        )

    def test_fit_returns_cascade_pipeline(self):
        trainer = self.build_trainer()
        pipeline = trainer.fit(TEXTS, LABELS, classes=CLASSES)
        assert isinstance(pipeline, CascadePipeline)

    def test_pipeline_classes_match(self):
        trainer = self.build_trainer()
        pipeline = trainer.fit(TEXTS, LABELS, classes=CLASSES)
        assert set(pipeline.classes) == set(CLASSES)

    def test_pipeline_predicts_after_fit(self):
        trainer = self.build_trainer()
        pipeline = trainer.fit(TEXTS, LABELS, classes=CLASSES)
        preds = pipeline.predict(TEXTS)
        assert len(preds) == len(TEXTS)
        assert all(p in CLASSES for p in preds)

    def test_report_is_populated(self):
        trainer = self.build_trainer()
        trainer.fit(TEXTS, LABELS, classes=CLASSES)
        report = trainer.report
        assert report is not None
        assert report.n_train == len(TEXTS)
        assert 0.0 <= report.base_student_f1 <= 1.0

    def test_precomputed_soft_labels(self):
        """Trainer should skip Teacher if soft_labels are provided."""
        teacher = FixedTeacher(CLASSES, LABELS)
        extractor = TFIDFExtractor(max_features=100, min_df=1)

        # Pre-generate soft labels
        pre_labels = teacher.generate_soft_labels(TEXTS, classes=CLASSES)

        # Use a DIFFERENT teacher (one that would fail if called)
        class FailTeacher(BaseTeacher):
            @property
            def classes(self): return CLASSES
            def generate_soft_labels(self, texts, classes=None):
                raise RuntimeError("This teacher should not have been called!")

        trainer = KnowledgeDistillationTrainer(
            teacher=FailTeacher(),
            extractor=extractor,
            confidence_threshold=0.6,
            verbose=False,
        )
        # Should not raise — uses pre_labels instead of calling teacher
        pipeline = trainer.fit(TEXTS, LABELS, classes=CLASSES, soft_labels=pre_labels)
        assert pipeline is not None

    def test_inferred_classes(self):
        """Classes should be inferred from y_true if not provided."""
        trainer = self.build_trainer()
        pipeline = trainer.fit(TEXTS, LABELS)  # No classes= argument
        assert set(pipeline.classes) == set(CLASSES)

    def test_high_threshold_more_hard_examples(self):
        """Higher threshold → more samples in hard set → more corrector training."""
        trainer_low = self.build_trainer(threshold=0.1)
        trainer_high = self.build_trainer(threshold=0.9)

        trainer_low.fit(TEXTS, LABELS, classes=CLASSES)
        trainer_high.fit(TEXTS, LABELS, classes=CLASSES)

        # Higher threshold should produce more hard examples
        assert trainer_high.report.n_hard >= trainer_low.report.n_hard

    def test_pipeline_property_after_fit(self):
        trainer = self.build_trainer()
        assert trainer.pipeline is None  # Before fit
        trainer.fit(TEXTS, LABELS, classes=CLASSES)
        assert trainer.pipeline is not None  # After fit
