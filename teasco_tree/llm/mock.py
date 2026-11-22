"""
Offline stand-ins for the Student and Teacher, so the pipeline, the UI and the
tests run with no Ollama server and no API key. Both are deterministic and read
a keyword signal from the text, so a demo run produces non-trivial results
rather than noise.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from teasco_tree.llm.base import BaseLLM

#: Words that pull a text toward the cue class (see ``_cue_class_index``).
_DEFAULT_CUES = (
    "secur", "encrypt", "password", "auth", "access", "privacy", "attack",
    "unauthori", "login", "permission", "protect", "confidential", "audit",
)


def _cue_class_index(classes: Sequence[str], cue_class: Optional[str]) -> int:
    """Which class the keyword cues vote for: explicit, else a security-sounding
    class name (``sec`` but not ``nonsec``), else the first class."""
    if cue_class is not None:
        return list(classes).index(cue_class)
    for i, c in enumerate(classes):
        low = c.lower()
        if "sec" in low and not low.startswith("non"):
            return i
    return 0


def _stable_unit(text: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from the text."""
    digest = hashlib.sha256(f"{salt}\x00{text}".encode("utf-8", "replace")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _cue_score(text: str, cues: Sequence[str]) -> float:
    lowered = text.lower()
    return float(sum(cue in lowered for cue in cues))


class MockStudentLLM(BaseLLM):
    """Keyword-driven Student with a confidence that varies with the evidence."""

    def __init__(self, classes: Sequence[str], cues: Sequence[str] = _DEFAULT_CUES,
                 noise: float = 0.15, cue_class: Optional[str] = None):
        self.classes = list(classes)
        self._cue = _cue_class_index(self.classes, cue_class)
        self.cues = tuple(cues)
        self.noise = noise
        self.failures = 0

    @property
    def model_name(self) -> str:
        return "mock-student"

    def predict(self, texts: List[str], progress_callback=None) -> Tuple[List[str], np.ndarray]:
        labels, confs = [], []
        k = len(self.classes)
        for i, text in enumerate(texts):
            score = _cue_score(text, self.cues)
            jitter = (_stable_unit(text, "student") - 0.5) * 2 * self.noise
            if score > 0:
                label = self.classes[self._cue]
                conf = min(0.99, 0.6 + 0.12 * score + jitter)
            else:
                label = self.classes[(self._cue + 1) % k]
                conf = min(0.99, 0.55 + abs(jitter))
            labels.append(label)
            confs.append(max(1.0 / k, conf))
            if progress_callback is not None:
                progress_callback(i + 1, len(texts))
        return labels, np.asarray(confs, dtype=np.float32)


class MockTeacher:
    """Soft-label Teacher with the same interface as :class:`OpenAITeacher`."""

    def __init__(self, classes: Sequence[str], cues: Sequence[str] = _DEFAULT_CUES,
                 fail_indices: Optional[Sequence[int]] = None,
                 cue_class: Optional[str] = None):
        self._classes = list(classes)
        self._cue = _cue_class_index(self._classes, cue_class)
        self.cues = tuple(cues)
        self.model = "mock-teacher"
        self._fail = set(fail_indices or [])
        self.failed_indices: List[int] = []
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "rate_limit_hits": 0}

    @property
    def classes(self) -> List[str]:
        return self._classes

    def generate_soft_labels(self, texts: List[str], classes=None,
                             progress_callback=None) -> List[Dict[str, float]]:
        k = len(self._classes)
        out, self.failed_indices = [], []
        for i, text in enumerate(texts):
            if i in self._fail:
                self.failed_indices.append(i)
                out.append({c: 1.0 / k for c in self._classes})
                continue
            score = _cue_score(text, self.cues)
            first = 0.15 + 0.3 * min(score, 2.5) + 0.1 * _stable_unit(text, "teacher")
            first = min(0.97, first)
            rest = (1.0 - first) / (k - 1)
            out.append({c: (first if j == self._cue else rest) for j, c in enumerate(self._classes)})
            if progress_callback is not None:
                progress_callback(i + 1, len(texts))
        self.usage["calls"] = len(texts)
        return out

    def filter_failed(self, *sequences):
        drop = set(self.failed_indices)
        return tuple([x for i, x in enumerate(seq) if i not in drop] for seq in sequences)
