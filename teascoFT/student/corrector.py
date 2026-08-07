"""
corrector.py
============
The Corrector — a secondary classifier trained exclusively on the
hard examples where the Base Student fails or is uncertain.

Role in the Cascade
-------------------
During training:
  1. Run the Base Student on the full training set.
  2. Find examples where: (a) prediction is WRONG, or (b) confidence < threshold.
  3. Train the Corrector only on these "hard examples."

During inference:
  - If Base Student confidence >= threshold  →  use Base Student prediction.
  - If Base Student confidence < threshold   →  defer to Corrector.

The Corrector uses a Random Forest by default (captures non-linear boundaries
that Logistic Regression misses), with an SVM option for high-dimensional
sparse features (e.g., TF-IDF).

Usage
-----
    corrector = Corrector(classes=["pos", "neg", "neu"], model_type="random_forest")

    # Train on hard examples only
    corrector.train(X_hard, y_hard)

    # Inference
    preds = corrector.predict(X_uncertain)
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Union

import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

FeatureMatrix = Union[np.ndarray, spmatrix]
ModelType = Literal["random_forest", "svm"]


class Corrector:
    """
    Secondary classifier trained on hard / uncertain examples.

    Parameters
    ----------
    classes : list of str
        Same class names as the BaseStudent.
    model_type : {"random_forest", "svm"}
        Underlying classifier. "random_forest" for dense embeddings;
        "svm" for high-dimensional sparse TF-IDF features. Default: "random_forest".
    confidence_threshold : float
        Samples with Base Student confidence below this value are sent to
        the Corrector during inference. Default: 0.6.
    n_estimators : int
        Number of trees (only for random_forest). Default: 200.
    max_depth : int, optional
        Max tree depth (only for random_forest). Default: None (unlimited).
    C : float
        SVM regularization (only for svm). Default: 1.0.
    class_weight : str or dict, optional
        "balanced" for imbalanced hard-example sets. Default: "balanced".
    random_state : int
        Random seed. Default: 42.
    min_hard_examples : int
        Minimum number of hard examples required to train the Corrector.
        If fewer hard examples are found, training is skipped and the
        Corrector falls back to a uniform prediction. Default: 5.
    """

    def __init__(
        self,
        classes: List[str],
        model_type: ModelType = "random_forest",
        confidence_threshold: float = 0.6,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        C: float = 1.0,
        class_weight: Optional[Union[str, dict]] = "balanced",
        random_state: int = 42,
        min_hard_examples: int = 5,
    ) -> None:
        self._classes = list(classes)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self._classes)
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.min_hard_examples = min_hard_examples
        self._fitted = False
        self._n_hard_examples = 0

        if model_type == "random_forest":
            self._model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "svm":
            # LinearSVC doesn't output probabilities natively — wrap with
            # CalibratedClassifierCV for probability support
            base_svc = LinearSVC(
                C=C,
                class_weight=class_weight,
                random_state=random_state,
                max_iter=2000,
            )
            self._model = CalibratedClassifierCV(base_svc, cv=3)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}. Use 'random_forest' or 'svm'.")

        logger.info(
            "Corrector created | model_type=%s | threshold=%.2f | classes=%s",
            model_type, confidence_threshold, self._classes,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("confidence_threshold must be in (0, 1).")
        self._threshold = value

    def train(
        self,
        X_hard: FeatureMatrix,
        y_hard: Union[np.ndarray, List[str]],
    ) -> "Corrector":
        """
        Train the Corrector on hard / uncertain examples.

        Parameters
        ----------
        X_hard : feature matrix of shape (n_hard, n_features)
            Feature vectors of the hard examples only.
        y_hard : array-like of str, shape (n_hard,)
            True labels for the hard examples.

        Returns
        -------
        self
        """
        n = X_hard.shape[0] if hasattr(X_hard, "shape") else len(X_hard)
        self._n_hard_examples = n

        if n < self.min_hard_examples:
            logger.warning(
                "Only %d hard examples found (min=%d). "
                "Corrector will not be fitted — falling back to Base Student for all inputs.",
                n, self.min_hard_examples,
            )
            return self

        logger.info("Training Corrector on %d hard examples...", n)

        y_hard_arr = np.asarray(y_hard)
        y_encoded = self._label_encoder.transform(y_hard_arr)

        # For RF, convert sparse to dense (RF doesn't accept sparse well)
        X_fit = X_hard.toarray() if issparse(X_hard) and self.model_type == "random_forest" else X_hard

        self._model.fit(X_fit, y_encoded)
        self._fitted = True

        # Log training accuracy
        preds_encoded = self._model.predict(X_fit)
        acc = (preds_encoded == y_encoded).mean()
        logger.info(
            "Corrector trained | hard_example_train_acc = %.4f | n_hard = %d",
            acc, n,
        )
        return self

    def predict(self, X: FeatureMatrix) -> np.ndarray:
        """
        Predict class labels for the given feature matrix.

        Returns
        -------
        np.ndarray of str, shape (n_samples,)
        """
        if not self._fitted:
            # Return None-array; pipeline will handle gracefully
            logger.warning("Corrector not fitted — returning base student predictions.")
            return None

        X_pred = X.toarray() if issparse(X) and self.model_type == "random_forest" else X
        encoded = self._model.predict(X_pred)
        return self._label_encoder.inverse_transform(encoded)

    def predict_proba(self, X: FeatureMatrix) -> np.ndarray:
        """
        Predict probability distributions.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        if not self._fitted:
            n = X.shape[0]
            return np.full((n, len(self._classes)), 1.0 / len(self._classes))

        X_pred = X.toarray() if issparse(X) and self.model_type == "random_forest" else X
        return self._model.predict_proba(X_pred)
