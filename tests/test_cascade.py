"""
test_cascade.py
===============
Unit tests for CascadePipeline.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT.features import TFIDFExtractor
from teascoFT.student import BaseStudent, Corrector
from teascoFT.cascade import CascadePipeline


CLASSES = ["sci.med", "sci.space", "rec.sport"]
TRAIN_TEXTS = [
    "The vaccine reduced cancer cell growth significantly.",
    "NASA plans a new mission to the Moon next year.",
    "The hockey team scored three goals in overtime.",
    "Researchers discovered a new protein structure.",
    "The Mars rover found traces of ancient water.",
    "The basketball player signed a record contract.",
    "Clinical trials show promising results for the drug.",
    "SpaceX launched its newest satellite constellation.",
    "The championship match ended in a penalty shootout.",
    "A new antibiotic resistant bacteria strain emerged.",
]
TRAIN_LABELS = [
    "sci.med", "sci.space", "rec.sport",
    "sci.med", "sci.space", "rec.sport",
    "sci.med", "sci.space", "rec.sport",
    "sci.med",
]


def build_pipeline(threshold=0.5) -> CascadePipeline:
    """Build a minimal fitted CascadePipeline for testing."""
    extractor = TFIDFExtractor(max_features=200, min_df=1)
    X = extractor.fit_transform(TRAIN_TEXTS)

    # Soft labels: 70% for correct class, split rest equally
    n_classes = len(CLASSES)
    soft_labels = []
    for label in TRAIN_LABELS:
        sl = {c: 0.15 / (n_classes - 1) for c in CLASSES}
        sl[label] = 0.70
        soft_labels.append(sl)

    student = BaseStudent(classes=CLASSES, max_iter=200)
    student.train(X, soft_labels)

    corrector = Corrector(classes=CLASSES, model_type="random_forest",
                          confidence_threshold=threshold, min_hard_examples=2)
    corrector.train(X[:6], TRAIN_LABELS[:6])

    return CascadePipeline(
        extractor=extractor,
        base_student=student,
        corrector=corrector,
        confidence_threshold=threshold,
        classes=CLASSES,
    )


class TestCascadePipeline:

    def setup_method(self):
        """Build a fresh pipeline before each test."""
        self.pipeline = build_pipeline(threshold=0.5)

    def test_predict_one(self):
        pred = self.pipeline.predict_one(TRAIN_TEXTS[0])
        assert pred in CLASSES

    def test_predict_batch_length(self):
        preds = self.pipeline.predict(TRAIN_TEXTS)
        assert len(preds) == len(TRAIN_TEXTS)

    def test_predict_all_in_classes(self):
        preds = self.pipeline.predict(TRAIN_TEXTS)
        assert all(p in CLASSES for p in preds)

    def test_predict_with_proba_shapes(self):
        preds, proba = self.pipeline.predict_with_proba(TRAIN_TEXTS)
        assert len(preds) == len(TRAIN_TEXTS)
        assert proba.shape == (len(TRAIN_TEXTS), len(CLASSES))

    def test_proba_sums_to_one(self):
        _, proba = self.pipeline.predict_with_proba(TRAIN_TEXTS)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(TRAIN_TEXTS)), atol=1e-5)

    def test_cascade_stats_update(self):
        self.pipeline.reset_stats()
        self.pipeline.predict(TRAIN_TEXTS)
        stats = self.pipeline.cascade_stats
        assert stats.total_predictions == len(TRAIN_TEXTS)
        assert stats.base_student_used + stats.corrector_used == len(TRAIN_TEXTS)

    def test_reset_stats(self):
        self.pipeline.predict(TRAIN_TEXTS)
        self.pipeline.reset_stats()
        stats = self.pipeline.cascade_stats
        assert stats.total_predictions == 0

    def test_corrector_trigger_rate_in_range(self):
        self.pipeline.predict(TRAIN_TEXTS)
        rate = self.pipeline.cascade_stats.corrector_trigger_rate
        assert 0.0 <= rate <= 1.0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_pipeline.pkl"
            self.pipeline.save(path)
            assert path.exists()

            loaded = CascadePipeline.load(path)
            orig_preds = self.pipeline.predict(TRAIN_TEXTS)
            load_preds = loaded.predict(TRAIN_TEXTS)
            np.testing.assert_array_equal(orig_preds, load_preds)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            CascadePipeline.load("/nonexistent/path.pkl")

    def test_classes_property(self):
        assert self.pipeline.classes == CLASSES

    def test_evaluate_returns_dict(self):
        results = self.pipeline.evaluate(TRAIN_TEXTS, TRAIN_LABELS, verbose=False)
        assert "macro_f1" in results
        assert "weighted_f1" in results
        assert "accuracy" in results
        assert 0.0 <= results["macro_f1"] <= 1.0

    def test_high_threshold_routes_to_corrector(self):
        """With very high threshold, almost all samples go to corrector."""
        pipeline_strict = build_pipeline(threshold=0.99)
        pipeline_strict.reset_stats()
        pipeline_strict.predict(TRAIN_TEXTS)
        # Corrector should be triggered for most samples
        assert pipeline_strict.cascade_stats.corrector_used > 0
