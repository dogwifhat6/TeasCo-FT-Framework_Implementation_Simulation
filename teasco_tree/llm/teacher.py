"""
teacher.py — Offline Teacher LLM (soft-label generator)
=======================================================

Ported from ``teascoFT/teacher/llm_teacher.py`` in the original TeasCo-FT repo,
which is where the throttle, retry budgets and failure accounting below were
built and measured. Changes for this project:

* ``TEACHER_PRESETS`` targets a **70B-parameter** Teacher (Llama 3.3 70B
  Instruct) — hosted on OpenRouter by default, or local Ollama for machines
  with the ~48 GB of memory a 4-bit 70B model needs.
* ``extra_body`` is forwarded to the provider (OpenRouter uses it for provider
  routing, e.g. ``{"provider": {"require_parameters": True}}`` so a request is
  never routed to a host that silently ignores JSON mode).
* ``class_descriptions`` explains terse label codes ("PE", "SE") to the model.
* ``generate_soft_labels(progress_callback=...)`` for the Streamlit UI.

Works with any OpenAI-compatible endpoint (OpenRouter, Together, Groq, Ollama's
``/v1``, vLLM, ...).

Failures are *not* silently swallowed. A text whose call never succeeds is
recorded in :attr:`OpenAITeacher.failed_indices`, and if more than
``max_failure_rate`` of the run fails the teacher raises
:class:`TeacherFailureError` rather than handing back a training set full of
uniform distributions: a uniform distribution is a *tie*, argmax breaks ties
toward the first class, and a mostly-failed pass therefore distils into a
constant-label model.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


#: 70B-parameter Teacher configurations. ``api_key_env`` names the environment
#: variable holding the key (``None`` for local servers).
TEACHER_PRESETS: Dict[str, Dict[str, object]] = {
    "openrouter-llama-3.3-70b": {
        "label": "Llama 3.3 70B Instruct — OpenRouter (hosted)",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "extra_body": {"provider": {"require_parameters": True}},
        "min_memory_gb": 0,
        "note": "Non-reasoning model, so no hidden chain-of-thought tokens. "
                "~$0.10/M prompt + $0.32/M completion tokens: a ~800-text "
                "training set costs about one US cent.",
    },
    "ollama-llama3.3-70b": {
        "label": "Llama 3.3 70B — local Ollama (needs ~48 GB RAM)",
        "model": "llama3.3:70b",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "extra_body": None,
        "min_memory_gb": 48,
        "note": "`ollama pull llama3.3:70b` (43 GB, Q4_K_M). Will not load on "
                "machines with less unified memory than the weights.",
    },
    "custom": {
        "label": "Custom OpenAI-compatible endpoint",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "extra_body": None,
        "min_memory_gb": 0,
        "note": "Any provider serving a 70B model behind /v1/chat/completions.",
    },
}


class TeacherFailureError(RuntimeError):
    """
    Raised when too large a share of a labeling run failed.

    Distillation on a mostly-failed teacher pass is worse than not distilling at
    all: the fallback uniform distribution is a *tie*, and ``max(d, key=d.get)``
    breaks ties toward whichever class happens to be listed first. A run that
    quietly fails 96% of its calls therefore produces a training set that is 96%
    a single constant label, and every student trained on it collapses to
    predicting that one class — which looks like a modelling problem, not an
    API problem.
    """


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
        progress_callback=None,
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
# Client-side rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class _RateLimiter:
    """
    Sliding-window throttle over the trailing 60 seconds.

    Tracks both tokens and requests because providers cap both, and the binding
    constraint is usually tokens: Groq's free tier allows 1000 requests/min but
    only 8000 tokens/min for ``openai/gpt-oss-20b``, so a naive loop exhausts
    the token budget in its first ten calls and then 429s for the rest of the
    minute.

    Limits left as ``None`` are not enforced; :meth:`configure` fills them in
    from the provider's response headers once the first call comes back.
    """

    def __init__(
        self,
        tokens_per_minute: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        safety_factor: float = 0.9,
    ) -> None:
        self.safety_factor = safety_factor
        self.tokens_per_minute = self._scale(tokens_per_minute)
        self.requests_per_minute = self._scale(requests_per_minute)
        # (timestamp, tokens) — tokens are an estimate until record() lands.
        self._window: Deque[List[float]] = deque()
        self._blocked_until = 0.0

    def _scale(self, limit: Optional[int]) -> Optional[float]:
        return None if limit is None else max(1.0, limit * self.safety_factor)

    def configure(
        self,
        tokens_per_minute: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
    ) -> None:
        """Adopt provider-advertised limits for any budget not already set."""
        if self.tokens_per_minute is None and tokens_per_minute:
            self.tokens_per_minute = self._scale(tokens_per_minute)
            logger.info(
                "Rate limiter: adopting provider TPM=%s (throttling at %.0f)",
                tokens_per_minute, self.tokens_per_minute,
            )
        if self.requests_per_minute is None and requests_per_minute:
            self.requests_per_minute = self._scale(requests_per_minute)
            logger.info(
                "Rate limiter: adopting provider RPM=%s (throttling at %.0f)",
                requests_per_minute, self.requests_per_minute,
            )

    def _prune(self, now: float) -> None:
        while self._window and now - self._window[0][0] >= 60.0:
            self._window.popleft()

    def acquire(self, estimated_tokens: float) -> List[float]:
        """
        Block until *estimated_tokens* fit in the current window, then reserve
        them. Returns a handle to pass to :meth:`record`.
        """
        while True:
            now = time.monotonic()
            if now < self._blocked_until:
                time.sleep(self._blocked_until - now)
                continue

            self._prune(now)
            wait = 0.0
            oldest = self._window[0][0] if self._window else now

            if self.tokens_per_minute is not None and self._window:
                used = sum(entry[1] for entry in self._window)
                if used + estimated_tokens > self.tokens_per_minute:
                    wait = max(wait, 60.0 - (now - oldest))
            if self.requests_per_minute is not None and self._window:
                if len(self._window) + 1 > self.requests_per_minute:
                    wait = max(wait, 60.0 - (now - oldest))

            if wait <= 0.0:
                handle = [now, float(estimated_tokens)]
                self._window.append(handle)
                return handle

            logger.debug("Rate limiter: sleeping %.1fs to stay under quota", wait)
            time.sleep(min(wait, 60.0) + 0.05)

    @staticmethod
    def record(handle: List[float], actual_tokens: Optional[int]) -> None:
        """Replace a reservation's estimate with the real usage."""
        if actual_tokens is not None:
            handle[1] = float(actual_tokens)

    def penalize(self, seconds: float) -> None:
        """Hard-stop all traffic for *seconds* (used on a 429)."""
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)


class DailyQuotaExhausted(RuntimeError):
    """
    Raised when the provider reports a *daily* budget as spent.

    A per-minute 429 is a "come back in a moment" and is worth retrying. A
    per-day one is not: the window refills over hours, so retrying grinds
    through the whole retry budget, spends what little quota trickles back,
    and still ends with a half-labeled training set. Failing immediately with
    the provider's own numbers is both faster and more useful.
    """


#: Groq spells the daily bucket "tokens per day (TPD)"; OpenAI and Azure use
#: "per day" / "requests per day" in the same position.
_DAILY_LIMIT_RE = re.compile(r"per\s+day|\bTPD\b|\bRPD\b", re.IGNORECASE)


def _is_daily_limit(exc: Exception) -> bool:
    """True if a 429 names a per-day budget rather than a per-minute one."""
    return bool(_DAILY_LIMIT_RE.search(str(exc)))


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    Extract the provider's requested cool-off from a rate-limit error.

    Checks the ``retry-after`` header first, then falls back to parsing the
    duration out of the message body (Groq words it as "Please try again in
    7.038s" and does not always set the header).
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        for name in ("retry-after", "x-ratelimit-reset-tokens",
                     "x-ratelimit-reset-requests"):
            raw = headers.get(name)
            if not raw:
                continue
            parsed = _parse_duration(str(raw))
            if parsed is not None:
                return parsed

    match = re.search(r"try again in ([\d.]+)\s*([a-z]*)", str(exc), re.IGNORECASE)
    if match:
        return _parse_duration(match.group(1) + match.group(2))
    return None


def _parse_duration(raw: str) -> Optional[float]:
    """Parse ``"7.038s"`` / ``"1m30s"`` / ``"12"`` (bare seconds) into seconds."""
    raw = raw.strip()
    match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)?s?", raw)
    if not match or not any(match.groups()):
        return None
    minutes = float(match.group(1) or 0.0)
    seconds = float(match.group(2) or 0.0)
    return minutes * 60.0 + seconds


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return status == 429 or type(exc).__name__ == "RateLimitError"


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
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

_BATCH_SYSTEM_PROMPT_TEMPLATE = """\
You are a precise text classification assistant.
You will be given a numbered list of texts. Classify EVERY one of them into one
of these categories: {classes}.

Judge each numbered text independently — do not let one text influence another.

You MUST respond with ONLY a valid JSON object — no explanation, no markdown, no extra text:
{{"results": [{{"i": <index>, {example_keys}}}, ...]}}

Rules:
- Emit exactly one entry per input index, using the same indices you were given.
- Each entry's probabilities must cover all of {classes_json} and sum to 1.0.

Example for classes ["positive", "negative"] and 2 inputs:
{{"results": [{{"i": 0, "positive": 0.85, "negative": 0.15}}, {{"i": 1, "positive": 0.03, "negative": 0.97}}]}}
"""

_BATCH_USER_PROMPT_TEMPLATE = """\
Classify each of the following {n} texts:

{items}

Respond with ONLY the JSON object containing all {n} results.
"""


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-Compatible Teacher
# ─────────────────────────────────────────────────────────────────────────────

class OpenAITeacher(BaseTeacher):
    """
    Teacher that uses any OpenAI-compatible LLM to generate soft labels.

    Parameters
    ----------
    api_key : str
        Your API key.
    model : str
        Model name, e.g. "gpt-4o-mini", "openai/gpt-oss-20b", "llama3".
        See ``TEACHER_PRESETS`` for the 70B configurations.
    classes : list of str
        The classification target labels.
    base_url : str, optional
        Override the API endpoint (for Ollama, Together, Groq, etc.).
        Defaults to the OpenAI endpoint.
    batch_size : int
        Number of texts sent **per API call**. Values > 1 ask the model for a
        JSON array of per-index distributions, which amortizes the system
        prompt across the batch and cuts token cost per text roughly 2-3x.
        Set to 1 for the one-call-per-text behaviour. Default: 25.
    max_retries : int
        Retry attempts for non-rate-limit errors (parse failures, transient
        network errors). Default: 3.
    retry_delay : float
        Base delay (seconds) between retries (doubles each attempt). Default: 1.0.
    max_rate_limit_retries : int
        Retry attempts reserved for HTTP 429s, which are waited out using the
        provider's own ``retry-after`` hint rather than counted against
        ``max_retries``. A quota refills on a fixed schedule, so a 429 is a
        "come back later", not a failure — burning three 1-second retries on it
        (the pre-fix behaviour) turns a solvable wait into a lost label.
        Default: 8.
    max_rate_limit_wait : float
        Upper bound, in seconds, on how long a single ``retry-after`` is
        honoured. Providers sometimes ask for far more than they need — Groq
        has answered a first 25-text batch with ``retry-after`` ~974s and then
        served an equivalent request normally a minute later, which stalls an
        otherwise healthy run for the better part of an hour. Retrying too
        early merely earns another 429 and another capped wait, so the cost of
        capping is bounded while the cost of not capping is not.
        Default: 120.
    temperature : float
        LLM temperature. Use 0.0 for deterministic outputs. Default: 0.0.
    max_tokens : int, optional
        Max completion tokens per call. ``None`` (default) sizes the budget from
        the batch. Reasoning models spend part of this budget on hidden
        chain-of-thought before emitting JSON, so a too-tight cap shows up as
        parse failures.
    reasoning_effort : str, optional
        Forwarded as-is to providers that support it (e.g. Groq's
        ``"low"``/``"medium"``/``"high"``) to cap reasoning-token spend.
        Ignored by providers/models that don't recognize it. Default: None.
    tokens_per_minute, requests_per_minute : int, optional
        Client-side throttle budgets. ``None`` (default) auto-discovers them
        from the provider's ``x-ratelimit-limit-*`` headers.
    extra_body : dict, optional
        Provider-specific request fields, merged into every call.
    class_descriptions : dict, optional
        ``{class: meaning}`` appended to the system prompt, for label sets whose
        names alone do not say what they mean.
    max_failure_rate : float
        Fraction of texts allowed to fall back to a uniform distribution before
        :meth:`generate_soft_labels` raises :class:`TeacherFailureError`.
        Default: 0.05.

    Attributes
    ----------
    failed_indices : list of int
        Indices (into the most recent ``texts`` argument) that fell back to a
        uniform distribution. Drop these rows before training rather than
        distilling on them — see :meth:`filter_failed`.
    usage : dict
        Token/call accounting for the most recent run.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        classes: List[str],
        base_url: Optional[str] = None,
        batch_size: int = 25,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        tokens_per_minute: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        max_rate_limit_retries: int = 8,
        max_rate_limit_wait: float = 120.0,
        max_failure_rate: float = 0.05,
        progress_desc: str = "Teacher labeling",
        extra_body: Optional[Dict] = None,
        class_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAITeacher. "
                "Install it with: pip install openai"
            ) from e

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._classes = list(classes)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_rate_limit_retries = max_rate_limit_retries
        self.max_rate_limit_wait = max_rate_limit_wait
        self.max_failure_rate = max_failure_rate
        # The same transport backs the V2 API Student, where a progress bar
        # reading "Teacher labeling" would be actively misleading.
        self.progress_desc = progress_desc
        self.extra_body = dict(extra_body) if extra_body else None
        self.class_descriptions = dict(class_descriptions or {})

        self.failed_indices: List[int] = []
        self.usage: Dict[str, int] = {"calls": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "total_tokens": 0,
                                      "rate_limit_hits": 0}

        self._limiter = _RateLimiter(tokens_per_minute, requests_per_minute)
        self._limits_discovered = tokens_per_minute is not None
        # Adaptive per-text completion cost, refined from observed usage.
        self._tokens_per_item = 80.0

        client_kwargs: Dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

        logger.info(
            "OpenAITeacher initialized | model=%s | classes=%s | batch_size=%d",
            self.model, self._classes, self.batch_size,
        )

    @property
    def classes(self) -> List[str]:
        return self._classes

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_soft_labels(
        self,
        texts: List[str],
        classes: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Generate soft label distributions for a list of texts.

        Texts are sent in groups of ``batch_size`` with a structured JSON-output
        prompt, paced to stay inside the provider's rate limit. Entries the
        model omits or mangles are re-tried individually before falling back to
        a uniform distribution.

        Returns
        -------
        list of dict[str, float]  — one per input text, probabilities sum to 1.

        Raises
        ------
        TeacherFailureError
            If more than ``max_failure_rate`` of the texts failed. Inspect
            :attr:`failed_indices` and lower ``batch_size`` / switch models /
            wait for quota to reset.
        """
        target_classes = classes if classes is not None else self._classes
        texts = list(texts)
        total = len(texts)

        self.failed_indices = []
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "rate_limit_hits": 0}
        if total == 0:
            return []

        results: List[Optional[Dict[str, float]]] = [None] * total
        chunks = [
            list(range(start, min(start + self.batch_size, total)))
            for start in range(0, total, self.batch_size)
        ]

        try:
            from tqdm import tqdm
            progress = tqdm(total=total, desc=self.progress_desc, unit="text")
        except ImportError:
            progress = None

        started = time.monotonic()
        try:
            for chunk in chunks:
                if len(chunk) == 1:
                    idx = chunk[0]
                    results[idx] = self._label_one(texts[idx], target_classes, idx)
                else:
                    self._label_chunk(texts, chunk, target_classes, results)
                if progress is not None:
                    progress.update(len(chunk))
                    progress.set_postfix(
                        failed=len(self.failed_indices),
                        tok_per_text=f"{self.usage['total_tokens'] / max(1, sum(1 for r in results if r is not None)):.0f}",
                    )
                if progress_callback is not None:
                    progress_callback(chunk[-1] + 1, total)
                # Once the absolute allowance is spent the run cannot recover —
                # later successes do not un-fail earlier texts — so stop here
                # rather than grinding through the remaining chunks.
                self._check_failure_budget(total)
        finally:
            if progress is not None:
                progress.close()

        elapsed = time.monotonic() - started
        final: List[Dict[str, float]] = [
            r if r is not None else self._uniform(target_classes) for r in results
        ]

        failure_rate = len(self.failed_indices) / total
        logger.info(
            "Generated %d soft labels in %.1fs | %d API calls | %d tokens "
            "(%.1f/text) | %d rate-limit waits | %d failed (%.1f%%)",
            total, elapsed, self.usage["calls"], self.usage["total_tokens"],
            self.usage["total_tokens"] / total, self.usage["rate_limit_hits"],
            len(self.failed_indices), 100 * failure_rate,
        )

        self._check_failure_budget(total)
        if self.failed_indices:
            logger.warning(
                "%d text(s) fell back to a uniform distribution: %s. "
                "Drop them with filter_failed() rather than distilling on them.",
                len(self.failed_indices), self.failed_indices[:20],
            )
        return final

    def _check_failure_budget(self, total: int) -> None:
        """Raise once more than ``max_failure_rate`` of *total* has failed."""
        n_failed = len(self.failed_indices)
        if n_failed <= self.max_failure_rate * total:
            return
        raise TeacherFailureError(
            f"{n_failed}/{total} texts ({100 * n_failed / total:.1f}%) could not be "
            f"labeled by {self.model} and fell back to a uniform distribution, "
            f"exceeding max_failure_rate={self.max_failure_rate:.0%}. Uniform soft "
            f"labels are a tie that argmax resolves to a single constant class, so "
            f"training on them collapses the student. Hit the rate limit "
            f"{self.usage['rate_limit_hits']} times — try a smaller model, a larger "
            f"batch_size, or an explicit tokens_per_minute budget. "
            f"Failed indices: {self.failed_indices[:20]}"
            + ("..." if n_failed > 20 else "")
        )

    def filter_failed(self, *sequences: Sequence) -> Tuple[List, ...]:
        """
        Drop the rows recorded in :attr:`failed_indices` from parallel sequences.

        A uniform soft label carries no signal but is not neutral — argmax turns
        it into a confident vote for whichever class sorts first — so failed
        rows should leave the training set entirely::

            soft = teacher.generate_soft_labels(texts)
            texts, labels, soft = teacher.filter_failed(texts, labels, soft)
        """
        drop = set(self.failed_indices)
        if not drop:
            return tuple(list(s) for s in sequences)
        return tuple(
            [item for i, item in enumerate(seq) if i not in drop] for seq in sequences
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _uniform(classes: List[str]) -> Dict[str, float]:
        return {c: 1.0 / len(classes) for c in classes}

    def _normalize(self, raw: Dict, classes: List[str]) -> Dict[str, float]:
        """Validate that every class is present and rescale to sum to 1."""
        missing = [c for c in classes if c not in raw]
        if missing:
            raise KeyError(f"Missing classes in LLM response: {missing}")
        dist = {c: float(raw[c]) for c in classes}
        total = sum(dist.values())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("Probabilities sum to zero or are non-finite.")
        return {c: v / total for c, v in dist.items()}

    def _label_one(
        self, text: str, classes: List[str], index: int
    ) -> Dict[str, float]:
        """Label a single text, recording a failure if every attempt fails."""
        try:
            return self._call_with_retry(text, classes)
        except DailyQuotaExhausted:
            # Not this text's failure, and the next 700 would fail identically.
            raise
        except Exception as exc:
            logger.error(
                "All attempts failed for text index %d (len=%d): %s",
                index, len(text), exc,
            )
            self.failed_indices.append(index)
            return self._uniform(classes)

    def _label_chunk(
        self,
        texts: List[str],
        chunk: List[int],
        classes: List[str],
        results: List[Optional[Dict[str, float]]],
    ) -> None:
        """
        Label one batch, then repair whatever the batch call left behind.

        A batch response can come back short (the model skipped an index, or the
        completion was truncated mid-array). Rather than discard the whole
        batch, keep the entries that parsed and re-issue only the stragglers as
        single-text calls.
        """
        try:
            parsed = self._call_batch_with_retry(
                [texts[i] for i in chunk], classes
            )
        except DailyQuotaExhausted:
            # Per-text retries would each hit the same wall, turning one clear
            # error into `len(chunk)` uniform distributions.
            raise
        except Exception as exc:
            logger.warning(
                "Batch of %d failed (%s); falling back to per-text calls.",
                len(chunk), exc,
            )
            parsed = {}

        for position, index in enumerate(chunk):
            dist = parsed.get(position)
            results[index] = (
                dist if dist is not None
                else self._label_one(texts[index], classes, index)
            )

    def _call_with_retry(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Call the LLM for one text, with back-off. Raises if all attempts fail."""
        return self._execute_with_retry(
            lambda: self._single_call(text, classes),
            estimated_tokens=self._estimate_tokens([text]),
            label=f"single(len={len(text)})",
        )

    def _call_batch_with_retry(
        self, batch: List[str], classes: List[str]
    ) -> Dict[int, Dict[str, float]]:
        """Call the LLM for a batch. Raises if all attempts fail."""
        return self._execute_with_retry(
            lambda: self._batch_call(batch, classes),
            estimated_tokens=self._estimate_tokens(batch),
            label=f"batch(n={len(batch)})",
        )

    def _execute_with_retry(self, operation, estimated_tokens: float, label: str):
        """
        Run *operation* under the throttle, retrying on failure.

        Rate-limit errors get their own, larger retry budget and sleep for the
        duration the provider asks for; everything else uses ``max_retries``
        with exponential back-off. Keeping the two budgets separate is the whole
        point: a 429 means "your quota refills in N seconds", which three
        one-second retries cannot outlast, so mixing them lets a temporary quota
        stall consume the retries meant for genuine errors.
        """
        delay = self.retry_delay
        attempts = 0
        rate_limit_attempts = 0

        while True:
            handle = self._limiter.acquire(estimated_tokens)
            before = self.usage["total_tokens"]
            try:
                result = operation()
            except Exception as exc:
                # A 429 consumed no tokens, but holding the reservation is the
                # conservative choice while the quota is already under strain.
                self._limiter.record(handle, int(estimated_tokens))

                if _is_rate_limit(exc):
                    if _is_daily_limit(exc):
                        raise DailyQuotaExhausted(
                            f"{self.model}: the provider's daily quota is "
                            f"spent, so retrying cannot help — it would only "
                            f"grind through {self.max_rate_limit_retries} "
                            f"waits and still return a half-labeled set. "
                            f"Re-run once the window resets; the soft-label "
                            f"cache means completed texts are not re-paid "
                            f"for. Provider said: {exc}"
                        ) from exc
                    self.usage["rate_limit_hits"] += 1
                    rate_limit_attempts += 1
                    if rate_limit_attempts > self.max_rate_limit_retries:
                        raise
                    requested = _retry_after_seconds(exc)
                    if requested is None:
                        cool_off = min(60.0, 2.0 ** rate_limit_attempts)
                    else:
                        # Cap what the provider asks for. Groq has answered a
                        # first batch with retry-after ~974s and then served an
                        # equivalent request fine a minute later; sleeping the
                        # full value stalls an otherwise healthy run for the
                        # better part of an hour. Capped, a too-early retry just
                        # 429s again and costs another cap's worth of wait, so
                        # the downside is bounded and the upside is not.
                        cool_off = min(requested, self.max_rate_limit_wait)
                    if requested is not None and cool_off < requested:
                        logger.warning(
                            "Rate limited on %s (%d/%d) — provider asked for "
                            "%.0fs, waiting %.0fs (max_rate_limit_wait) and "
                            "retrying",
                            label, rate_limit_attempts,
                            self.max_rate_limit_retries, requested, cool_off,
                        )
                    else:
                        logger.warning(
                            "Rate limited on %s (%d/%d) — waiting %.1fs",
                            label, rate_limit_attempts,
                            self.max_rate_limit_retries, cool_off,
                        )
                    self._limiter.penalize(cool_off + 0.25)
                    continue

                attempts += 1
                if attempts >= self.max_retries:
                    raise
                logger.warning(
                    "Attempt %d/%d on %s failed — %s",
                    attempts, self.max_retries, label, exc,
                )
                time.sleep(delay)
                delay *= 2  # exponential back-off
                continue
            else:
                # Replace the pre-flight guess with what the call actually cost,
                # so the window reflects real spend rather than our estimate.
                self._limiter.record(handle, self.usage["total_tokens"] - before)
                return result

    def _estimate_tokens(self, batch: List[str]) -> float:
        """
        Rough pre-flight token cost, used to size the throttle reservation.

        Deliberately approximate — :meth:`_record_usage` overwrites the estimate
        with the provider's real number as soon as the call returns, so a wrong
        guess costs at most one mis-paced request.
        """
        prompt_chars = sum(len(t[:4000]) for t in batch) + 600
        return prompt_chars / 4.0 + len(batch) * self._tokens_per_item

    def _record_usage(self, response, n_items: int) -> None:
        """Fold a response's real token usage into accounting and the throttle."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += usage.prompt_tokens or 0
        self.usage["completion_tokens"] += usage.completion_tokens or 0
        self.usage["total_tokens"] += usage.total_tokens or 0
        if n_items > 0 and usage.completion_tokens:
            observed = usage.completion_tokens / n_items
            # EWMA so a single odd batch doesn't swing the pacing.
            self._tokens_per_item = 0.7 * self._tokens_per_item + 0.3 * observed

    def _discover_limits(self, headers) -> None:
        """Adopt the provider's advertised quota from response headers, once."""
        if self._limits_discovered or not headers:
            return
        self._limits_discovered = True

        def as_int(name):
            raw = headers.get(name)
            try:
                return int(float(raw)) if raw else None
            except (TypeError, ValueError):
                return None

        self._limiter.configure(
            tokens_per_minute=as_int("x-ratelimit-limit-tokens"),
            requests_per_minute=as_int("x-ratelimit-limit-requests"),
        )

    def _extra_body(self) -> Optional[Dict]:
        body = dict(self.extra_body or {})
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body or None

    def _descriptions_block(self, classes: List[str]) -> str:
        described = [c for c in classes if self.class_descriptions.get(c, c) != c]
        if not described:
            return ""
        lines = "\n".join(f'- "{c}": {self.class_descriptions[c]}' for c in described)
        return f"\nWhat each category means:\n{lines}\n"

    def _create(self, messages: List[Dict], max_tokens: int, n_items: int):
        """Issue one chat completion, recording usage and rate-limit headers."""
        raw = self._client.chat.completions.with_raw_response.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=self._extra_body(),
        )
        self._discover_limits(raw.headers)
        response = raw.parse()
        self._record_usage(response, n_items)
        return response

    def _single_call(self, text: str, classes: List[str]) -> Dict[str, float]:
        """Make one API call for one text and parse the JSON soft-label response."""
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            classes=", ".join(f'"{c}"' for c in classes),
            classes_json=json.dumps(classes),
        ) + self._descriptions_block(classes)
        user_prompt = _USER_PROMPT_TEMPLATE.format(text=text[:4000])  # truncate

        response = self._create(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens or 512,
            n_items=1,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty completion (max_tokens likely too low).")
        return self._normalize(json.loads(content.strip()), classes)

    def _batch_call(
        self, batch: List[str], classes: List[str]
    ) -> Dict[int, Dict[str, float]]:
        """
        Make one API call for *batch* texts.

        Returns a mapping of ``position within the batch -> distribution``. The
        mapping may be incomplete: entries the model skipped or returned
        unparseably are simply absent, and the caller re-issues those
        individually.
        """
        system_prompt = _BATCH_SYSTEM_PROMPT_TEMPLATE.format(
            classes=", ".join(f'"{c}"' for c in classes),
            classes_json=json.dumps(classes),
            example_keys=", ".join(f'"{c}": <prob>' for c in classes),
        ) + self._descriptions_block(classes)
        items = "\n".join(
            f"{i}. {text[:4000]}" for i, text in enumerate(batch)
        )
        user_prompt = _BATCH_USER_PROMPT_TEMPLATE.format(n=len(batch), items=items)

        budget = self.max_tokens or (
            300 + len(batch) * (40 + 12 * len(classes))
        )
        response = self._create(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            max_tokens=budget,
            n_items=len(batch),
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty completion (max_tokens likely too low).")
        payload = json.loads(content.strip())
        entries = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError(f"Expected a 'results' list, got {type(entries).__name__}")

        parsed: Dict[int, Dict[str, float]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                position = int(entry["i"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= position < len(batch) or position in parsed:
                continue
            try:
                parsed[position] = self._normalize(entry, classes)
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Dropping malformed batch entry %d: %s", position, exc)

        if not parsed:
            raise ValueError("Batch response contained no usable entries.")
        return parsed
