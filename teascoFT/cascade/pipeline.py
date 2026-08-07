"""
pipeline.py
===========
CascadePipeline — the production inference engine.

This is the ONLY component deployed in production. The Teacher LLM is
completely discarded after training.

Inference Flow
--------------
    Input Text
        │
        ▼
    Feature Extractor (TF-IDF / MiniLM)
        │
        ▼
    Base Student (Logistic Regression)
        │
        ├── confidence >= threshold ──► Prediction ✓
        │
        └── confidence < threshold ──► Corrector
                                           │
                                           ▼
                                       Prediction ✓

The pipeline tracks cascade statistics (how often the Corrector is triggered)
to help you monitor and tune the confidence_threshold in production.

Usage
-----
    # Typically built by KnowledgeDistillationTrainer.fit()
    pipeline = trainer.fit(texts_train, y_train)

    # Single prediction
    pred = pipeline.predict_one("The market crashed unexpectedly.")

    # Batch prediction
    preds = pipeline.predict(texts_test)

    # With probabilities
    preds, probs = pipeline.predict_with_proba(texts_test)

    # Cascade stats
    print(pipeline.cascade_stats)

    # Save / Load
    pipeline.save("classifier.pkl")
    pipeline = CascadePipeline.load("classifier.pkl")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from teascoFT.features.tfidf_extractor import TFIDFExtractor
from teascoFT.features.embedding_extractor import EmbeddingExtractor
from teascoFT.student.base_student import BaseStudent
from teascoFT.student.corrector import Corrector

logger = logging.getLogger(__name__)

FeatureExtractor = Union[TFIDFExtractor, EmbeddingExtractor]


# ─────────────────────────────────────────────────────────────────────────────
# Cascade Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CascadeStats:
    """
    Running statistics for the cascade pipeline.
    Reset with pipeline.reset_stats().
    """
    total_predictions: int = 0
    base_student_used: int = 0
    corrector_used: int = 0

    @property
    def corrector_trigger_rate(self) -> float:
        """Fraction of predictions where the Corrector was invoked."""
        if self.total_predictions == 0:
            return 0.0
        return self.corrector_used / self.total_predictions

    def __str__(self) -> str:
        return (
            f"CascadeStats("
            f"total={self.total_predictions}, "
            f"base={self.base_student_used}, "
            f"corrector={self.corrector_used}, "
            f"corrector_rate={self.corrector_trigger_rate:.2%})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cascade Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CascadePipeline:
    """
    Production inference pipeline. No LLM involved.

    Parameters
    ----------
    extractor : TFIDFExtractor or EmbeddingExtractor
        Fitted feature extractor.
    base_student : BaseStudent
        Fitted Logistic Regression student.
    corrector : Corrector
        Fitted Corrector (may be unfitted, in which case base student always used).
    confidence_threshold : float
        Base Student confidence below which the Corrector is invoked. Default: 0.6.
    classes : list of str
        Class names in canonical order.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        base_student: BaseStudent,
        corrector: Corrector,
        confidence_threshold: float = 0.6,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.extractor = extractor
        self.base_student = base_student
        self.corrector = corrector
        self.confidence_threshold = confidence_threshold
        self._classes = classes or base_student.classes
        self._stats = CascadeStats()

        logger.info(
            "CascadePipeline assembled | classes=%s | threshold=%.2f | corrector_fitted=%s",
            self._classes, confidence_threshold, corrector.is_fitted,
        )

    # ── Prediction API ─────────────────────────────────────────────────────

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def cascade_stats(self) -> CascadeStats:
        """Cumulative statistics since last reset_stats() call."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cascade statistics counters."""
        self._stats = CascadeStats()

    def predict_one(self, text: str) -> str:
        """
        Classify a single text string.

        Parameters
        ----------
        text : str

        Returns
        -------
        str — predicted class label
        """
        return self.predict([text])[0]

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Classify a batch of texts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray of str, shape (n_texts,)
        """
        preds, _ = self._cascade_predict(texts)
        return preds

    def predict_with_proba(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify texts and return class probability distributions.

        Returns
        -------
        predictions : np.ndarray of str, shape (n_texts,)
        probabilities : np.ndarray of float, shape (n_texts, n_classes)
        """
        return self._cascade_predict(texts, return_proba=True)

    def evaluate(
        self,
        texts: List[str],
        y_true: List[str],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the pipeline on labelled data and print a report.

        Parameters
        ----------
        texts : list of str
        y_true : list of str  — ground truth labels
        verbose : bool        — if True, print rich classification report

        Returns
        -------
        dict with keys: macro_f1, weighted_f1, accuracy
        """
        from sklearn.metrics import f1_score, accuracy_score

        preds = self.predict(texts)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
        accuracy = accuracy_score(y_true, preds)

        if verbose:
            try:
                from teascoFT.utils.metrics import classification_report_rich
                classification_report_rich(y_true, preds, self._classes)
            except Exception:
                from sklearn.metrics import classification_report
                print(classification_report(y_true, preds))

            print(f"\nCascade Stats: {self._stats}")

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": accuracy,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize the pipeline to disk using joblib.

        Parameters
        ----------
        path : str or Path
            File path (e.g., "my_classifier.pkl").
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "extractor": self.extractor,
            "base_student": self.base_student,
            "corrector": self.corrector,
            "confidence_threshold": self.confidence_threshold,
            "classes": self._classes,
            "_version": "0.1.0",
        }
        joblib.dump(payload, str(path))
        logger.info("Pipeline saved to %s", path)
        print(f"✅ Pipeline saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CascadePipeline":
        """
        Load a serialized pipeline from disk.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        CascadePipeline
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No pipeline found at: {path}")

        payload = joblib.load(str(path))
        logger.info("Pipeline loaded from %s", path)

        pipeline = cls(
            extractor=payload["extractor"],
            base_student=payload["base_student"],
            corrector=payload["corrector"],
            confidence_threshold=payload["confidence_threshold"],
            classes=payload["classes"],
        )
        print(f"✅ Pipeline loaded from {path}")
        return pipeline

    # ── Private ────────────────────────────────────────────────────────────

    def _cascade_predict(
        self,
        texts: List[str],
        return_proba: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Core cascade logic:
          1. Extract features.
          2. Run Base Student.
          3. For low-confidence samples, run Corrector.
          4. Merge predictions.
        """
        n = len(texts)

        # Extract features
        X = self.extractor.transform(texts)

        # Base Student predictions + confidence
        base_preds, base_conf = self.base_student.predict_with_confidence(X)
        base_proba = self.base_student.predict_proba(X) if return_proba else None

        # Identify low-confidence samples
        low_conf_mask = base_conf < self.confidence_threshold
        n_low_conf = low_conf_mask.sum()

        # Initialize output arrays from base student
        final_preds = base_preds.copy()
        final_proba = base_proba.copy() if return_proba else None

        # Update stats
        self._stats.total_predictions += n
        self._stats.base_student_used += int((~low_conf_mask).sum())
        self._stats.corrector_used += int(n_low_conf)

        # Invoke corrector on uncertain samples (if any and if fitted)
        if n_low_conf > 0 and self.corrector.is_fitted:
            X_uncertain = self._index_features(X, np.where(low_conf_mask)[0])

            corrector_preds = self.corrector.predict(X_uncertain)
            if corrector_preds is not None:
                final_preds[low_conf_mask] = corrector_preds

            if return_proba:
                corrector_proba = self.corrector.predict_proba(X_uncertain)
                final_proba[low_conf_mask] = corrector_proba

            logger.debug(
                "Corrector invoked on %d/%d samples (%.1f%%)",
                n_low_conf, n, 100 * n_low_conf / n,
            )

        return final_preds, final_proba

    @staticmethod
    def _index_features(X, indices: np.ndarray):
        from scipy.sparse import issparse
        if issparse(X):
            return X[indices]
        return X[indices]
