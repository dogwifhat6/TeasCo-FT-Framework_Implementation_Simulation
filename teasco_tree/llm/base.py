"""The frozen Student interface consumed by :class:`FeatureBuilder`."""

from __future__ import annotations

import abc
from typing import List, Tuple

import numpy as np


class BaseLLM(abc.ABC):
    """A frozen model that returns ``(labels, confidences)`` for a list of texts.

    Only the confidence reaches the adapter — it becomes column 384 of the
    385-dim feature matrix — so an implementation is judged on whether that
    number *varies* with difficulty, not on whether its label is right.
    """

    @abc.abstractmethod
    def predict(self, texts: List[str], progress_callback=None) -> Tuple[List[str], np.ndarray]:
        """Labels and float32 confidences, one per text."""

    def verify(self, text: str, prediction: str) -> str:  # pragma: no cover - optional
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Identity string; part of the feature-cache key."""
