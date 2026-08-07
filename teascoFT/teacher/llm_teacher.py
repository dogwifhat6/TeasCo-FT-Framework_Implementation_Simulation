"""
llm_teacher.py
==============
Teacher LLM wrappers that generate *soft labels* (probability distributions
over classes) for a text classification task.

The OpenAITeacher works with any OpenAI-compatible endpoint:
  - OpenAI   (api_key="sk-...", base_url=None)
  - Ollama   (api_key="ollama", base_url="http://localhost:11434/v1")
  - Azure    (use AzureOpenAI client variant)
  - Groq / Together / any proxy

Usage
-----
    teacher = OpenAITeacher(
        api_key="sk-...",
        model="gpt-4o-mini",
        classes=["positive", "negative", "neutral"],
    )
    soft_labels = teacher.generate_soft_labels(
        texts=["Great product!", "Terrible experience."],
    )
    # → [{"positive": 0.91, "negative": 0.05, "neutral": 0.04},
    #    {"positive": 0.02, "negative": 0.94, "neutral": 0.04}]
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseTeacher(ABC):
    """Abstract interface every Teacher must implement."""

    @abstractmethod
    def generate_soft_labels(
        self,
        texts: List[str],
        classes: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Return a probability distribution over *classes* for each input text.

        Parameters
        ----------
        texts : list of str
            Raw input texts to classify.
        classes : list of str, optional
            Target class names. If None, uses the classes set at init.

        Returns
        -------
        list of dict
            One dict per input text, mapping class name → probability.
            Probabilities sum to 1.0 for each text.
        """
        ...

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """Return the list of class names this teacher knows about."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-Compatible Teacher
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a precise text classification assistant.
Your task is to classify the given text into one of these categories: {classes}.

You MUST respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.
The JSON must have exactly these keys: {classes_json}
The values must be floating-point probabilities that sum to exactly 1.0.

Example response for classes ["positive", "negative", "neutral"]:
{{"positive": 0.85, "negative": 0.10, "neutral": 0.05}}
"""

_USER_PROMPT_TEMPLATE = """\
Classify the following text:

\"\"\"{text}\"\"\"

Respond with ONLY the JSON probability distribution.
"""


class OpenAITeacher(BaseTeacher):
    """
    Teacher that uses any OpenAI-compatible LLM to generate soft labels.

    Parameters
    ----------
    api_key : str
        Your API key.
    model : str
        Model name, e.g. "gpt-4o-mini", "llama3", "claude-3-haiku".
    classes : list of str
        The classification target labels.
    base_url : str, optional
        Override the API endpoint (for Ollama, Together, Groq, etc.).
        Defaults to the OpenAI endpoint.
    batch_size : int
        Number of texts to process per batch (each text is one API call).
        Default: 10.
    max_retries : int
        Number of retry attempts on API errors. Default: 3.
    retry_delay : float
        Base delay (seconds) between retries (doubles each attempt). Default: 1.0.
    temperature : float
        LLM temperature. Use 0.0 for deterministic outputs. Default: 0.0.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        classes: List[str],
        base_url: Optional[str] = None,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAITeacher. "
                "Install it with: pip install openai"
            ) from e

        self._classes = list(classes)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

        client_kwargs: Dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

        # Pre-build the system prompt (doesn't change between calls)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            classes=", ".join(f'"{c}"' for c in self._classes),
            classes_json=json.dumps(self._classes),
        )
        logger.info(
            "OpenAITeacher initialized | model=%s | classes=%s",
            self.model, self._classes,
        )

    @property
    def classes(self) -> List[str]:
        return self._classes

    def generate_soft_labels(
        self,
        texts: List[str],
        classes: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Generate soft label distributions for a list of texts.

        Each text is sent to the LLM with a structured JSON-output prompt.
        Results are parsed and validated; malformed outputs are retried.

        Parameters
        ----------
        texts : list of str
        classes : list of str, optional — overrides instance classes for this call.

        Returns
        -------
        list of dict[str, float]  — one per input text, probabilities sum to 1.
        """
        target_classes = classes if classes is not None else self._classes

        results: List[Dict[str, float]] = []
        total = len(texts)

        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Teacher labeling", unit="text", total=total)
        except ImportError:
            iterator = texts

        for text in iterator:
            soft_label = self._call_with_retry(text, target_classes)
            results.append(soft_label)

        logger.info("Generated %d soft labels.", len(results))
        return results

    # ── Private helpers ────────────────────────────────────────────────────

    def _call_with_retry(
        self, text: str, classes: List[str]
    ) -> Dict[str, float]:
        """Call the LLM with exponential back-off on failure."""
        delay = self.retry_delay
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._single_call(text, classes)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Attempt %d/%d: parse error — %s", attempt, self.max_retries, exc
                )
                last_exc = exc
            except Exception as exc:  # network / API error
                logger.warning(
                    "Attempt %d/%d: API error — %s", attempt, self.max_retries, exc
                )
                last_exc = exc

            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2  # exponential back-off

        # Fallback: uniform distribution
        logger.error(
            "All %d attempts failed for text (len=%d). "
            "Using uniform distribution. Last error: %s",
            self.max_retries, len(text), last_exc,
        )
        uniform = 1.0 / len(classes)
        return {c: uniform for c in classes}

    def _single_call(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Make one API call and parse the JSON soft-label response."""
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            classes=", ".join(f'"{c}"' for c in classes),
            classes_json=json.dumps(classes),
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(text=text[:4000])  # truncate

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed: Dict = json.loads(raw)

        # Validate: all classes present
        missing = [c for c in classes if c not in parsed]
        if missing:
            raise KeyError(f"Missing classes in LLM response: {missing}")

        # Normalize to ensure probabilities sum to 1.0
        dist = {c: float(parsed[c]) for c in classes}
        total = sum(dist.values())
        if total <= 0:
            raise ValueError("Probabilities sum to zero.")
        return {c: v / total for c, v in dist.items()}
