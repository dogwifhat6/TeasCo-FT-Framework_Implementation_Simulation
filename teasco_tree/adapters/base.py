"""
base.py — The Adapter Interface
================================

An *adapter* is the trainable head that sits on top of the frozen LLM in the
TeasCo-FT V2 cascade. It consumes the 385-dim representation
(384-dim sentence embedding + 1-dim LLM confidence scalar) and produces class
logits/probabilities.

The original V2 design hard-wired a PyTorch MLP as the adapter. This module
turns the adapter into a pluggable component so gradient-boosted trees
(XGBoost, CatBoost, LightGBM) can stand in for the MLP without any other part
of the cascade changing.

Every adapter must honour the **Gated Update Rule** from the V2 spec:

    Learn only from mistakes — samples the adapter already classifies
    correctly contribute no update signal.

How that rule is realised differs by adapter family (backprop gating for the
MLP, gradient/hessian masking or sample-weight gating for the tree ensembles),
which is exactly why the interface is defined in terms of *fit/predict* rather
than in terms of losses and gradients.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Abstract trainable head for the TeasCo-FT V2 cascade.

    Parameters
    ----------
    classes : list of str
        Ordered class names. Index in this list is the integer class id.
    random_state : int
        Seed for reproducibility. Default: 42.
    verbose : bool
        Log per-stage/per-epoch progress. Default: True.

    Notes
    -----
    Subclasses implement :meth:`fit` and :meth:`predict_proba`. Everything
    else (argmax prediction, macro-F1 evaluation) is shared.
    """

    #: Short identifier used in reports and the adapter registry.
    name: str = "base"

    #: Whether this adapter is gradient-based (MLP) or an ensemble of trees.
    family: str = "unknown"

    def __init__(
        self,
        classes: List[str],
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.random_state = random_state
        self.verbose = verbose

        self.history: Dict[str, List[float]] = {}
        self.best_val_f1: float = 0.0
        self.best_iteration: int = 0
        self._fitted = False

    # ── Contract ───────────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train the adapter under the gated update rule.

        Parameters
        ----------
        X : np.ndarray, shape (N, 385)
            Training features.
        y : np.ndarray, shape (N,)
            Integer class targets.
        X_val, y_val : np.ndarray, optional
            Validation set used for early stopping on macro F1.

        Returns
        -------
        dict
            Training summary with at least ``best_val_f1``, ``best_iteration``,
            ``elapsed_seconds`` and ``history``.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (N, num_classes)."""

    # ── Shared behaviour ───────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted integer class ids, shape (N,)."""
        return self.predict_proba(X).argmax(axis=1)

    def predict_labels(self, X: np.ndarray) -> List[str]:
        """Return predicted class *names*."""
        return [self.classes[i] for i in self.predict(X)]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate on a labelled set.

        Returns
        -------
        dict with ``macro_f1``, ``weighted_f1`` and ``accuracy``.
        """
        preds = self.predict(X)
        return {
            "macro_f1": float(
                f1_score(y, preds, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y, preds, average="weighted", zero_division=0)
            ),
            "accuracy": float((preds == y).mean()),
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call .fit(X, y) first."
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"classes={self.classes}, fitted={self._fitted})"
        )
