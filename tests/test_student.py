"""
test_student.py
===============
Unit tests for BaseStudent and Corrector.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT.student import BaseStudent, Corrector


CLASSES = ["positive", "negative", "neutral"]
N_SAMPLES = 50
N_FEATURES = 100


def make_dummy_X(dense=False):
    """Create a dummy feature matrix."""
    rng = np.random.RandomState(42)
    X_dense = rng.rand(N_SAMPLES, N_FEATURES)
    if dense:
        return X_dense
    return csr_matrix(X_dense)


def make_soft_labels(classes=CLASSES, n=N_SAMPLES):
    """Generate random normalized soft labels."""
    rng = np.random.RandomState(42)
    raw = rng.dirichlet(np.ones(len(classes)), size=n)
    return [{c: float(raw[i, j]) for j, c in enumerate(classes)} for i in range(n)]


def make_hard_labels(classes=CLASSES, n=N_SAMPLES):
    """Generate random hard labels."""
    rng = np.random.RandomState(99)
    return [classes[i % len(classes)] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# BaseStudent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseStudent:

    def test_train_predict_sparse(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X(dense=False)
        soft_labels = make_soft_labels()
        student.train(X, soft_labels)
        preds = student.predict(X)
        assert len(preds) == N_SAMPLES
        assert all(p in CLASSES for p in preds)

    def test_train_predict_dense(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X(dense=True)
        soft_labels = make_soft_labels()
        student.train(X, soft_labels)
        preds = student.predict(X)
        assert len(preds) == N_SAMPLES

    def test_predict_proba_shape(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X()
        student.train(X, make_soft_labels())
        proba = student.predict_proba(X)
        assert proba.shape == (N_SAMPLES, len(CLASSES))

    def test_predict_proba_sums_to_one(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X()
        student.train(X, make_soft_labels())
        proba = student.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(N_SAMPLES), atol=1e-6)

    def test_confidence_range(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X()
        student.train(X, make_soft_labels())
        conf = student.confidence(X)
        assert conf.shape == (N_SAMPLES,)
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_predict_without_train_raises(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X()
        with pytest.raises(RuntimeError, match="not been trained"):
            student.predict(X)

    def test_mismatched_soft_labels_raises(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X()
        wrong_labels = make_soft_labels(n=10)  # Wrong count
        with pytest.raises(ValueError, match="soft_labels"):
            student.train(X, wrong_labels)

    def test_predict_with_confidence_returns_tuple(self):
        student = BaseStudent(classes=CLASSES)
        X = make_dummy_X()
        student.train(X, make_soft_labels())
        preds, conf = student.predict_with_confidence(X)
        assert len(preds) == N_SAMPLES
        assert len(conf) == N_SAMPLES

    def test_classes_property(self):
        student = BaseStudent(classes=CLASSES)
        assert student.classes == CLASSES

    def test_is_fitted_flag(self):
        student = BaseStudent(classes=CLASSES)
        assert not student.is_fitted
        X = make_dummy_X()
        student.train(X, make_soft_labels())
        assert student.is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# Corrector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrector:

    def test_random_forest_train_predict(self):
        corrector = Corrector(classes=CLASSES, model_type="random_forest")
        X = make_dummy_X(dense=True)  # RF prefers dense
        y = make_hard_labels()
        corrector.train(X, y)
        preds = corrector.predict(X)
        assert len(preds) == N_SAMPLES
        assert all(p in CLASSES for p in preds)

    def test_svm_train_predict(self):
        corrector = Corrector(classes=CLASSES, model_type="svm")
        X = make_dummy_X()
        y = make_hard_labels()
        corrector.train(X, y)
        preds = corrector.predict(X)
        assert len(preds) == N_SAMPLES

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            Corrector(classes=CLASSES, model_type="xgboost")

    def test_predict_unfitted_returns_none(self):
        corrector = Corrector(classes=CLASSES, min_hard_examples=1000)
        X = make_dummy_X(dense=True)
        y = make_hard_labels(n=5)  # Too few examples, corrector won't fit
        corrector.train(X[:5], y)
        result = corrector.predict(X[:5])
        assert result is None

    def test_predict_proba_shape(self):
        corrector = Corrector(classes=CLASSES, model_type="random_forest")
        X = make_dummy_X(dense=True)
        y = make_hard_labels()
        corrector.train(X, y)
        proba = corrector.predict_proba(X)
        assert proba.shape == (N_SAMPLES, len(CLASSES))

    def test_confidence_threshold_validation(self):
        with pytest.raises(ValueError):
            Corrector(classes=CLASSES, confidence_threshold=1.5)
        with pytest.raises(ValueError):
            Corrector(classes=CLASSES, confidence_threshold=0.0)

    def test_is_fitted_flag(self):
        corrector = Corrector(classes=CLASSES, model_type="random_forest", min_hard_examples=5)
        X = make_dummy_X(dense=True)
        y = make_hard_labels()
        assert not corrector.is_fitted
        corrector.train(X, y)
        assert corrector.is_fitted
