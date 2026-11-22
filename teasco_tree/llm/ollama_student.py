"""
ollama_llm.py
=============
Frozen local Student SLM served by `Ollama <https://ollama.com>`_.

Implements :class:`teasco_tree.llm.base.BaseLLM`, so it is a drop-in
replacement for ``MockLLM`` / ``FrozenHuggingFaceLLM``: ``FeatureBuilder`` and
``AdapterCascadeTrainer`` consume it unchanged. It supplies the
``(prediction, confidence)`` pair whose confidence scalar becomes the 385th
feature column.

Confidence comes from the first generated token's logprob distribution,
renormalised over the class names (``scoring="logprobs"``, the default). That
matters most for the small models this wrapper is aimed at: asked to *state* a
confidence, a 0.5B model says 0.9 or 1.0 to everything and the 385th feature
column goes constant. Pass ``scoring="json"`` for the older self-reported
behaviour.

Every call is cached to disk under a hash of ``(model, scoring, text)``. That cache is
deliberately independent of ``FeatureBuilder``'s own ``.npz`` cache, which is
keyed on the *whole* text list: a run interrupted partway through a long
sequential pass resumes from the per-text cache instead of starting over.

Usage
-----
    from teasco_tree.llm import OllamaLLM

    llm = OllamaLLM(classes=["sec", "nonsec"], model_name="llama3.2:latest")
    predictions, confidences = llm.predict(["The system shall encrypt ..."])
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from teasco_tree.llm.base import BaseLLM

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
#: Option letters used by the forced-choice logprob prompt.
_LETTERS = [chr(ord("A") + i) for i in range(26)]
DEFAULT_MODEL = "llama3.2:latest"


class OllamaLLM(BaseLLM):
    """
    Frozen local Student SLM served by Ollama.

    Parameters
    ----------
    classes : list of str
        Ordered class names. ``classes[0]`` doubles as the fallback label when
        a call cannot be parsed.
    model_name : str
        Ollama model tag, e.g. ``"llama3.2:latest"``. This string is part of
        ``FeatureBuilder``'s cache key, so changing it invalidates cached
        feature matrices.
    host : str
        Ollama base URL.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Attempts per text before falling back to ``(classes[0], 0.5)``.
    cache_path : str or Path, optional
        JSON file for the per-text response cache. ``None`` disables caching.
    flush_every : int
        Write the cache to disk after this many new results.
    scoring : {"logprobs", "json"}
        How the confidence scalar is obtained. ``"logprobs"`` (default) reads
        the first generated token's own top-k distribution and renormalises it
        over ``classes`` — continuous, calibrated, and one generated token per
        text. ``"json"`` asks the model to state a confidence number in a
        schema-constrained reply; small models answer ~1.0 to everything, so
        the resulting feature column is near-constant. ``"logprobs"``
        auto-falls back to ``"json"`` if the server returns none (Ollama
        < 0.12).
    top_logprobs : int
        How many candidate tokens to request per step in ``"logprobs"`` mode.
    debias : bool
        In ``"logprobs"`` mode, ask twice with the options in both orders and
        average, cancelling the position bias small models show. Costs a
        second call per text and is worth it: without it ``qwen2.5:0.5b``
        answered 45/50 one way and 44/50 the other way purely on option order.
    class_descriptions : dict, optional
        Human-readable text for each class in the forced-choice prompt, e.g.
        ``{"sec": "security-related", "nonsec": "not security-related"}``.
        Classes not listed are shown under their own name.
    """

    def __init__(
        self,
        classes: Sequence[str],
        model_name: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_path: Optional[str] = None,
        flush_every: int = 25,
        scoring: str = "logprobs",
        top_logprobs: int = 20,
        debias: bool = True,
        class_descriptions: Optional[dict] = None,
    ) -> None:
        try:
            import requests  # noqa: F401
        except ImportError as e:  # pragma: no cover - environment guard
            raise ImportError(
                "requests is required for OllamaLLM. Install it with: "
                "pip install requests"
            ) from e

        self.classes = list(classes)
        self._model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.flush_every = flush_every
        self._fallback = self.classes[0]

        if scoring not in ("logprobs", "json"):
            raise ValueError(
                f"scoring must be 'logprobs' or 'json', got {scoring!r}"
            )
        self.scoring = scoring
        self.top_logprobs = top_logprobs
        self.debias = debias
        self.class_descriptions = dict(class_descriptions or {})
        if len(self.classes) > len(_LETTERS):
            raise ValueError(
                f"logprob scoring offers one letter per class and there are "
                f"only {len(_LETTERS)}; got {len(self.classes)} classes."
            )
        # Flipped to True the first time a server reply carries no logprobs,
        # so the fallback to self-reported JSON confidence happens once and
        # every later call takes the cheap path straight away.
        self._logprobs_unsupported = False

        # Ollama enforces this schema server-side, so the reply is already
        # constrained to a valid label rather than free-form prose.
        self._schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": self.classes},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
        }

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
                logger.info(
                    "Loaded %d cached Ollama responses from %s",
                    len(self._cache), self.cache_path.name,
                )
            except json.JSONDecodeError:
                logger.warning(
                    "Ollama cache at %s is corrupt; starting empty.", self.cache_path
                )
        self._dirty = 0
        self.failures = 0

    @property
    def model_name(self) -> str:
        # The suffix is part of FeatureBuilder's cache key (it hashes
        # ``llm.model_name``): the two scoring modes produce different
        # confidence columns, so they must not share a cached matrix.
        if self.scoring == "logprobs" and not self._logprobs_unsupported:
            suffix = "#logprobs-debiased" if self.debias else "#logprobs"
            return f"{self._model_name}{suffix}"
        return self._model_name

    def is_available(self) -> bool:
        """True if the Ollama server is up and serving this model."""
        import requests

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = {m["name"] for m in resp.json().get("models", [])}
        except Exception:
            return False
        return self._model_name in tags

    # ── Cache ──────────────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.model_name}\x00{text}".encode("utf-8", errors="replace")
        ).hexdigest()

    def _save_cache(self) -> None:
        if self.cache_path and self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))
            self._dirty = 0

    # ── Inference ──────────────────────────────────────────────────────────

    def _classify_one(self, text: str) -> Tuple[str, float]:
        """Classify one text, dispatching on the configured scoring mode."""
        if self.scoring == "logprobs" and not self._logprobs_unsupported:
            result = self._classify_one_logprobs(text)
            if result is not None:
                return result
        return self._classify_one_json(text)

    # -- logprob scoring ---------------------------------------------------

    def _option_block(self, order: List[str]) -> str:
        return "  ".join(
            f"{_LETTERS[i]}) {self.class_descriptions.get(c, c)}"
            for i, c in enumerate(order)
        )

    def _letter_distribution(
        self, text: str, order: List[str]
    ) -> Optional[dict]:
        """
        One forced-choice call: probability of each class, or None on failure.

        The classes are presented as lettered options and only the first token
        is generated, so the answer is read straight out of that token's top-k
        distribution. Letters are used rather than the class names themselves
        because the names are not tokenised symmetrically -- ``"sec"`` is one
        token while ``"nonsec"`` starts with ``"non"``, and the model pays for
        that asymmetry before it has considered the requirement at all.
        """
        import requests

        system = (
            "You classify software requirements. Answer with a single letter "
            "only, no punctuation and no explanation."
        )
        user = (
            f'Requirement: "{text[:1000]}"\n\n'
            f"Which option applies?\n{self._option_block(order)}\n\nAnswer:"
        )
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # One token is all that gets scored, which also makes this the
            # cheapest possible call - about 3x faster per call than
            # generating the JSON object the other mode asks for.
            "options": {"temperature": 0, "num_predict": 1},
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "stream": False,
        }

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                steps = resp.json().get("logprobs") or []
                if not steps or not steps[0].get("top_logprobs"):
                    self._logprobs_unsupported = True
                    logger.warning(
                        "Ollama at %s returned no logprobs for %s (needs "
                        "Ollama >= 0.12); falling back to self-reported JSON "
                        "confidence for the rest of this run.",
                        self.host, self._model_name,
                    )
                    return None

                valid = _LETTERS[:len(order)]
                mass = {c: 0.0 for c in order}
                for entry in steps[0]["top_logprobs"]:
                    token = str(entry.get("token", "")).strip().upper()
                    if token in valid:
                        mass[order[valid.index(token)]] += math.exp(
                            float(entry["logprob"])
                        )
                total = sum(mass.values())
                if total <= 0.0:
                    # A real answer, just not one of the offered letters.
                    return None
                return {c: m / total for c, m in mass.items()}
            except Exception as exc:
                if attempt == self.max_retries:
                    logger.warning(
                        "Ollama logprob call failed after %d attempts "
                        "(len=%d): %s", attempt, len(text), exc,
                    )
                    return None
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    def _classify_one_logprobs(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Score the label from the first generated token's own distribution.

        Two things make this the default over the self-reported JSON number,
        both measured on 50 balanced PROMISE requirements with
        ``qwen2.5:0.5b``:

        * **The confidence is real.** Asked to *state* a confidence, a 0.5B
          model answers 0.9 or 1.0 to everything -- 2 distinct values across
          50 texts, so the 385th feature column is a constant and hands the
          adapter nothing. Read off the token distribution it is continuous
          (50 distinct values, sd 0.11) and correlates with being right.
        * **The label stops collapsing.** Single-order forced choice inherits
          the position bias small models have (``qwen2.5:0.5b`` answered 45/50
          one way, then 44/50 the other way when the options were swapped).
          Averaging the two orderings cancels it: 27/23, macro F1 0.80 against
          0.33 for the JSON path -- and better than ``llama3.2:3b``, at ~9x
          the speed.

        Returns ``None`` when no call yielded a usable distribution, leaving
        the caller to fall back to the JSON path.
        """
        orders = [list(self.classes)]
        if self.debias and len(self.classes) > 1:
            orders.append(list(reversed(self.classes)))

        totals = {c: 0.0 for c in self.classes}
        n_ok = 0
        for order in orders:
            probs = self._letter_distribution(text, order)
            if probs is None:
                if self._logprobs_unsupported:
                    return None
                continue
            for c, v in probs.items():
                totals[c] += v
            n_ok += 1

        if n_ok == 0:
            return None
        label = max(totals, key=totals.get)
        return label, float(totals[label] / n_ok)

    # -- JSON scoring ------------------------------------------------------

    def _classify_one_json(self, text: str) -> Tuple[str, float]:
        import requests

        system = (
            f"You are a requirements-engineering classifier. Classify the "
            f"requirement as one of: {', '.join(self.classes)}. "
            f"Respond with strict JSON only."
        )
        user = (
            f'Requirement: "{text[:1000]}"\n'
            f'Respond as JSON: {{"label": one of {self.classes}, '
            f'"confidence": number between 0 and 1}}'
        )
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": self._schema,
            "options": {"temperature": 0},
            "stream": False,
        }

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                parsed = json.loads(resp.json()["message"]["content"])
                label = str(parsed["label"]).strip().lower()
                if label not in self.classes:
                    label = self._fallback
                confidence = min(max(float(parsed["confidence"]), 0.0), 1.0)
                return label, confidence
            except Exception as exc:
                if attempt == self.max_retries:
                    logger.warning(
                        "Ollama failed after %d attempts (len=%d): %s",
                        attempt, len(text), exc,
                    )
                    self.failures += 1
                    return self._fallback, 0.5
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    def predict(
        self, texts: List[str], progress_callback=None
    ) -> Tuple[List[str], np.ndarray]:
        """Classify each text, returning labels and a confidence array.

        ``progress_callback(done, total)`` is called after every text, so a UI
        can show progress for what is the longest step of a run.
        """
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(
                texts, desc=f"Student SLM ({self._model_name})", unit="req"
            )
        except ImportError:
            iterator = texts

        predictions: List[str] = []
        confidences: List[float] = []
        for text in iterator:
            key = self._cache_key(text)
            if key in self._cache:
                label, confidence = self._cache[key]
            else:
                label, confidence = self._classify_one(text)
                self._cache[key] = [label, confidence]
                self._dirty += 1
                if self._dirty >= self.flush_every:
                    self._save_cache()
            predictions.append(label)
            confidences.append(confidence)
            if progress_callback is not None:
                progress_callback(len(predictions), len(texts))

        self._save_cache()
        return predictions, np.asarray(confidences, dtype=np.float32)

    def verify(self, text: str, prediction: str) -> str:
        """Teacher-verifier interface: does a fresh call agree with *prediction*?"""
        label, _ = self._classify_one(text)
        return "Agree" if label == prediction else "Disagree"
