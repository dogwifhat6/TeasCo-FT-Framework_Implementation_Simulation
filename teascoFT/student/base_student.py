"""
base_student.py
===============
The Base Student — a Logistic Regression classifier trained via
Knowledge Distillation on the Teacher LLM's soft label probabilities.

Key Idea
--------
Instead of training on hard one-hot labels (ground truth), the student
is trained on the Teacher's *soft* probability distributions. This forces
the student to learn the inter-class relationships the LLM implicitly
encodes (e.g., "this is 80% Class A, 15% Class B, 5% Class C"), leading
to much better generalization than training on ground truth alone.

Implementation
--------------
Soft-label distillation is implemented by training a Logistic Regression
model whose loss is cross-entropy against the Teacher's probabilities.
This is done via sklearn's `warm_start` SGD training trick or by directly
minimizing KL-divergence using scipy's optimize, but the simplest and most
effective sklearn-native approach is to:
  1. Expand each sample into as many virtual rows as there are classes.
  2. Weight each virtual row by the Teacher's probability for that class.
  3. Train the LR model with `sample_weight` on this expanded dataset.

This is mathematically equivalent to minimizing the cross-entropy between
the Teacher's distribution and the Student's predicted distribution.

Usage
-----
    student = BaseStudent(classes=["pos", "neg", "neu"])
    student.train(X_features, soft_labels)

    predictions = student.predict(X_test)
    probs       = student.predict_proba(X_test)
    confidence  = student.confidence(X_test)   # max proba per sample
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Type alias for feature matrices (dense or sparse)
FeatureMatrix = Union[np.ndarray, spmatrix]


class BaseStudent:
    """
    Logistic Regression student trained on Teacher soft labels.

    Parameters
    ----------
    classes : list of str
        Ordered list of class names (must match those used by the Teacher).
    C : float
        Inverse regularization strength. Larger = less regularization.
        Default: 1.0.
    max_iter : int
        Maximum number of LR solver iterations. Default: 1000.
    solver : str
        Solver for LogisticRegression. "lbfgs" (default) handles
        multi-class well; use "saga" for very large datasets.
    class_weight : str or dict, optional
        "balanced" to auto-correct class imbalance. Default: None.
    random_state : int
        Random seed for reproducibility. Default: 42.
    """

    def __init__(
        self,
        classes: List[str],
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        class_weight: Optional[Union[str, dict]] = None,
        random_state: int = 42,
    ) -> None:
        self._classes = list(classes)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self._classes)

        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._fitted = False
        logger.info(
            "BaseStudent created | classes=%s | C=%.4f | solver=%s",
            self._classes, C, solver,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def train(
        self,
        X: FeatureMatrix,
        soft_labels: List[Dict[str, float]],
    ) -> "BaseStudent":
        """
        Train the student using Knowledge Distillation from soft labels.

        Each training example is *expanded* into N virtual samples (one per
        class), each weighted by the Teacher's probability for that class.
        This is equivalent to minimizing KL-divergence against the Teacher.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Pre-extracted feature vectors.
        soft_labels : list of dict[str, float]
            One probability distribution dict per sample.

        Returns
        -------
        self
        """
        n_samples = X.shape[0]
        n_classes = len(self._classes)

        if len(soft_labels) != n_samples:
            raise ValueError(
                f"X has {n_samples} rows but soft_labels has {len(soft_labels)} entries."
            )

        logger.info(
            "Training BaseStudent on %d samples × %d classes via soft-label KD...",
            n_samples, n_classes,
        )

        # ── Expand samples for soft-label training ─────────────────────────
        # For each sample i and class c:
        #   X_expanded[i * n_classes + c] = X[i]
        #   y_expanded[i * n_classes + c] = c
        #   weights[i * n_classes + c]    = soft_labels[i][c]

        X_expanded, y_expanded, sample_weights = self._expand_for_distillation(
            X, soft_labels
        )

        self._model.fit(X_expanded, y_expanded, sample_weight=sample_weights)
        self._fitted = True

        # Log training accuracy on hard labels derived from soft labels
        hard_y = np.array(
            [max(sl, key=sl.get) for sl in soft_labels]
        )
        train_preds = self.predict(X)
        n_correct = (train_preds == hard_y).sum()
        logger.info(
            "BaseStudent trained | train acc (vs Teacher hard) = %.4f (%d/%d)",
            n_correct / n_samples, n_correct, n_samples,
        )
        return self

    def predict(self, X: FeatureMatrix) -> np.ndarray:
        """
        Predict class labels for input features.

        Returns
        -------
        np.ndarray of str, shape (n_samples,)
        """
        self._check_fitted()
        encoded = self._model.predict(X)
        return self._label_encoder.inverse_transform(encoded)

    def predict_proba(self, X: FeatureMatrix) -> np.ndarray:
        """
        Predict class probability distributions.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Columns correspond to self.classes in order.
        """
        self._check_fitted()
        return self._model.predict_proba(X)

    def confidence(self, X: FeatureMatrix) -> np.ndarray:
        """
        Return the max probability across classes for each sample.
        Used to decide whether to invoke the Corrector.

        Returns
        -------
        np.ndarray of float, shape (n_samples,)
            Values in [0, 1]. Higher = more confident.
        """
        return self.predict_proba(X).max(axis=1)

    def predict_with_confidence(
        self, X: FeatureMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience method: returns (predictions, confidence_scores).

        Returns
        -------
        predictions : np.ndarray of str, shape (n_samples,)
        confidence  : np.ndarray of float, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        encoded_preds = proba.argmax(axis=1)
        predictions = self._label_encoder.inverse_transform(encoded_preds)
        confidence = proba.max(axis=1)
        return predictions, confidence

    # ── Private ────────────────────────────────────────────────────────────

    def _expand_for_distillation(
        self,
        X: FeatureMatrix,
        soft_labels: List[Dict[str, float]],
    ):
        """
        Expand each sample into n_classes virtual rows for soft-label training.

        Returns (X_expanded, y_expanded_encoded, sample_weights)
        """
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        total = n_samples * n_classes

        # Build expanded weight and label arrays
        y_expanded = np.empty(total, dtype=object)
        weights = np.zeros(total, dtype=np.float64)

        for i, sl in enumerate(soft_labels):
            for j, cls in enumerate(self._classes):
                idx = i * n_classes + j
                y_expanded[idx] = cls
                weights[idx] = sl.get(cls, 0.0)

        # Encode string labels to integers
        y_encoded = self._label_encoder.transform(y_expanded)

        # Expand feature matrix
        if issparse(X):
            from scipy.sparse import vstack
            X_expanded = vstack([X] * n_classes, format="csr")
            # Need to interleave, not just vstack all-of-class-0, all-of-class-1...
            # Build interleaved sparse rows
            rows = []
            for i in range(n_samples):
                for j in range(n_classes):
                    rows.append(X[i])
            X_expanded = vstack(rows, format="csr")
        else:
            # Dense: repeat each row n_classes times (interleaved)
            X_expanded = np.repeat(X, n_classes, axis=0)

        return X_expanded, y_encoded, weights

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BaseStudent has not been trained yet. Call .train(X, soft_labels) first."
            )
