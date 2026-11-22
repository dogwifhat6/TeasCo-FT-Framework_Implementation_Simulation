"""
Tests for teasco_tree.llm.teacher.

These focus on the failure mode that silently wrecked a real distillation run:
the teacher hit its provider's token-per-minute quota, every retry was consumed
inside three seconds, 96% of texts fell back to a uniform distribution, and
``max(d, key=d.get)`` turned each of those ties into a confident vote for the
first class — producing a training set that was 97% one constant label. The
tests pin the throttling, the 429 handling, and the loud failure that now
replaces the silent collapse.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from teasco_tree.llm.teacher import (  # noqa: E402
    OpenAITeacher,
    TeacherFailureError,
    _is_daily_limit,
    _is_rate_limit,
    _parse_duration,
    _RateLimiter,
    _retry_after_seconds,
    DailyQuotaExhausted,
)

CLASSES = ["sec", "nonsec"]


@pytest.fixture
def fake_clock(monkeypatch):
    """
    Swap the module's ``time`` for a virtual clock that *advances* on sleep.

    A stub that merely records sleeps is not enough: ``_RateLimiter.acquire``
    re-checks the wall clock after each sleep, so with frozen time it spins
    until the real 60-second window expires. Advancing virtual time makes the
    throttle resolve instantly and keeps its timing assertions exact.
    """
    import time as real_time

    class Clock:
        def __init__(self):
            self.now = real_time.monotonic()
            self.slept = []

        def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds

        def monotonic(self):
            return self.now

    clock = Clock()
    monkeypatch.setattr("teasco_tree.llm.teacher.time", clock)
    return clock


@pytest.fixture(autouse=True)
def _no_real_sleeping(fake_clock):
    """Back-off delays are asserted on, never actually waited out."""
    return fake_clock


class FakeRateLimitError(Exception):
    """Stands in for openai.RateLimitError (matched by name and status)."""

    status_code = 429

    def __init__(self, message="Rate limit reached. Please try again in 4.5s",
                 headers=None):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=429, headers=headers or {})


class FakeCompletions:
    """Minimal stand-in for ``client.chat.completions``."""

    def __init__(self, responder, headers=None):
        self._responder = responder
        self._headers = headers or {}
        self.calls = []

    @property
    def with_raw_response(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self._responder(len(self.calls), kwargs)
        if isinstance(payload, Exception):
            raise payload
        content, usage = payload
        parsed = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=usage[0], completion_tokens=usage[1],
                total_tokens=usage[0] + usage[1],
            ),
        )
        return SimpleNamespace(headers=self._headers, parse=lambda: parsed)


def build_teacher(responder, headers=None, **kwargs):
    """An OpenAITeacher whose HTTP client is replaced by a scripted fake."""
    kwargs.setdefault("tokens_per_minute", 10 ** 9)  # effectively unthrottled
    teacher = OpenAITeacher(
        api_key="test-key", model="fake-model", classes=CLASSES, **kwargs
    )
    fake = FakeCompletions(responder, headers)
    teacher._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return teacher, fake


def batch_json(indices, sec_prob=0.9):
    entries = ", ".join(
        f'{{"i": {i}, "sec": {sec_prob}, "nonsec": {round(1 - sec_prob, 4)}}}'
        for i in indices
    )
    return '{"results": [' + entries + ']}'


# ─────────────────────────────────────────────────────────────────────────────
# Batching
# ─────────────────────────────────────────────────────────────────────────────

class TestBatching:
    def test_batches_many_texts_into_one_call(self):
        """25 texts must cost one API call, not 25 — this is the quota lever."""
        teacher, fake = build_teacher(
            lambda n, kw: (batch_json(range(25)), (700, 600)), batch_size=25
        )
        labels = teacher.generate_soft_labels([f"requirement {i}" for i in range(25)])

        assert len(labels) == 25
        assert len(fake.calls) == 1
        assert teacher.failed_indices == []

    def test_batch_results_map_back_to_input_order(self):
        """A shuffled 'results' array must still land on the right texts."""
        def responder(n, kw):
            body = ('{"results": ['
                    '{"i": 2, "sec": 0.9, "nonsec": 0.1}, '
                    '{"i": 0, "sec": 0.1, "nonsec": 0.9}, '
                    '{"i": 1, "sec": 0.5, "nonsec": 0.5}]}')
            return body, (100, 80)

        teacher, _ = build_teacher(responder, batch_size=3)
        labels = teacher.generate_soft_labels(["a", "b", "c"])

        assert [round(d["sec"], 2) for d in labels] == [0.1, 0.5, 0.9]

    def test_probabilities_are_renormalized(self):
        teacher, _ = build_teacher(
            lambda n, kw: ('{"results": [{"i": 0, "sec": 3.0, "nonsec": 1.0}, '
                           '{"i": 1, "sec": 1.0, "nonsec": 1.0}]}', (50, 20)),
            batch_size=2,
        )
        labels = teacher.generate_soft_labels(["a", "b"])
        assert labels[0]["sec"] == pytest.approx(0.75)
        assert labels[1]["sec"] == pytest.approx(0.5)
        assert all(sum(d.values()) == pytest.approx(1.0) for d in labels)

    def test_batch_size_one_uses_the_single_text_prompt(self):
        teacher, fake = build_teacher(
            lambda n, kw: ('{"sec": 0.8, "nonsec": 0.2}', (60, 20)), batch_size=1
        )
        labels = teacher.generate_soft_labels(["a", "b"])

        assert len(fake.calls) == 2
        assert all(round(d["sec"], 2) == 0.8 for d in labels)

    def test_missing_batch_entries_are_retried_individually(self):
        """
        A truncated batch must not poison the whole group: the entries that did
        parse are kept, and only the stragglers are re-issued one at a time.
        """
        def responder(n, kw):
            if n == 1:
                return batch_json([0, 1], sec_prob=0.8), (200, 100)  # index 2 missing
            return '{"sec": 0.3, "nonsec": 0.7}', (60, 20)

        teacher, fake = build_teacher(responder, batch_size=3)
        labels = teacher.generate_soft_labels(["a", "b", "c"])

        assert len(fake.calls) == 2               # one batch + one repair call
        assert round(labels[0]["sec"], 2) == 0.8
        assert round(labels[2]["sec"], 2) == 0.3  # repaired, not uniform
        assert teacher.failed_indices == []

    def test_malformed_entries_are_dropped_not_fatal(self):
        def responder(n, kw):
            if n == 1:
                return ('{"results": [{"i": 0, "sec": 0.9, "nonsec": 0.1}, '
                        '{"i": 1, "sec": "banana"}]}'), (200, 100)
            return '{"sec": 0.4, "nonsec": 0.6}', (60, 20)

        teacher, _ = build_teacher(responder, batch_size=2)
        labels = teacher.generate_soft_labels(["a", "b"])
        assert round(labels[1]["sec"], 2) == 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimitHandling:
    def test_429_is_waited_out_not_burned_through(self):
        """
        The original bug: a 429 consumed one of only three retries with a 1s
        sleep, so a quota that refills in seconds still exhausted the budget.
        429s now draw on their own, larger allowance.
        """
        def responder(n, kw):
            if n <= 5:
                return FakeRateLimitError()
            return batch_json([0, 1]), (200, 100)

        teacher, fake = build_teacher(
            responder, batch_size=2, max_retries=3, max_rate_limit_retries=8,
        )
        teacher._limiter.penalize = lambda seconds: None  # don't actually sleep

        labels = teacher.generate_soft_labels(["a", "b"])

        assert len(fake.calls) == 6
        assert teacher.usage["rate_limit_hits"] == 5
        assert teacher.failed_indices == []
        assert all(d["sec"] > d["nonsec"] for d in labels)

    def test_rate_limit_budget_is_finite(self):
        teacher, _ = build_teacher(
            lambda n, kw: FakeRateLimitError(), batch_size=1,
            max_rate_limit_retries=2, max_failure_rate=1.0,
        )
        teacher._limiter.penalize = lambda seconds: None

        labels = teacher.generate_soft_labels(["a"])

        assert teacher.failed_indices == [0]
        assert labels[0] == {"sec": 0.5, "nonsec": 0.5}

    def test_retry_after_header_is_honored(self):
        err = FakeRateLimitError("rate limited", headers={"retry-after": "12"})
        assert _retry_after_seconds(err) == 12.0

    def test_retry_after_falls_back_to_the_message_body(self):
        """Groq states the cool-off in prose and omits the header."""
        err = FakeRateLimitError("Rate limit reached... Please try again in 7.038s")
        assert _retry_after_seconds(err) == pytest.approx(7.038)

    @pytest.mark.parametrize("raw, expected", [
        ("12", 12.0), ("7.038s", 7.038), ("1m30s", 90.0), ("2m", 120.0),
        ("", None), ("Wed, 21 Oct 2015 07:28:00 GMT", None),
    ])
    def test_parse_duration(self, raw, expected):
        assert _parse_duration(raw) == expected

    def test_is_rate_limit_detects_429(self):
        assert _is_rate_limit(FakeRateLimitError())
        assert not _is_rate_limit(ValueError("nope"))

    def test_limits_are_discovered_from_response_headers(self):
        teacher, _ = build_teacher(
            lambda n, kw: ('{"sec": 0.6, "nonsec": 0.4}', (100, 50)),
            headers={"x-ratelimit-limit-tokens": "8000",
                     "x-ratelimit-limit-requests": "1000"},
            batch_size=1, tokens_per_minute=None,
        )
        teacher.generate_soft_labels(["a"])

        # 0.9 safety factor keeps a margin under the advertised ceiling.
        assert teacher._limiter.tokens_per_minute == pytest.approx(7200.0)
        assert teacher._limiter.requests_per_minute == pytest.approx(900.0)


class TestRateLimiter:
    def test_window_blocks_once_the_token_budget_is_spent(self, fake_clock):
        limiter = _RateLimiter(tokens_per_minute=1000, safety_factor=1.0)
        limiter.acquire(600)   # fits
        assert fake_clock.slept == []
        limiter.acquire(600)   # 1200 > 1000 -> must wait out the window
        assert fake_clock.slept and sum(fake_clock.slept) >= 60.0

    def test_recording_actual_usage_frees_over_reserved_budget(self, fake_clock):
        limiter = _RateLimiter(tokens_per_minute=1000, safety_factor=1.0)
        handle = limiter.acquire(900)
        limiter.record(handle, 100)   # the call was far cheaper than estimated
        limiter.acquire(800)          # 100 + 800 <= 1000, so no wait
        assert fake_clock.slept == []

    def test_unset_limits_are_not_enforced(self, fake_clock):
        limiter = _RateLimiter()
        for _ in range(50):
            limiter.acquire(10 ** 6)
        assert fake_clock.slept == []


# ─────────────────────────────────────────────────────────────────────────────
# Failure visibility — the actual regression
# ─────────────────────────────────────────────────────────────────────────────

class TestFailuresAreNotSilent:
    def test_mass_failure_raises_instead_of_returning_uniform(self):
        """
        The exact shape of the original incident: nearly every call fails, so
        nearly every soft label is a uniform tie. Returning that quietly hands
        back a constant-label training set, so it must raise instead.
        """
        teacher, fake = build_teacher(
            lambda n, kw: FakeRateLimitError(), batch_size=1,
            max_rate_limit_retries=0, max_failure_rate=0.05,
        )
        teacher._limiter.penalize = lambda seconds: None

        with pytest.raises(TeacherFailureError) as excinfo:
            teacher.generate_soft_labels([f"text {i}" for i in range(20)])

        message = str(excinfo.value)
        assert "uniform" in message
        assert "/20" in message
        # Aborts as soon as the budget is blown (5% of 20 allows one failure),
        # rather than burning the API on the remaining 18 doomed texts.
        assert len(fake.calls) < 20

    def test_failure_budget_is_absolute_not_a_running_average(self):
        """
        Later successes must not rescue a run that already blew its budget:
        a failed text stays failed, so the check is against the total, not
        against the texts processed so far.
        """
        def responder(n, kw):
            return ValueError("bad") if n <= 2 else ('{"sec": 0.9, "nonsec": 0.1}', (60, 20))

        teacher, _ = build_teacher(
            responder, batch_size=1, max_retries=1, max_failure_rate=0.10,
        )
        with pytest.raises(TeacherFailureError):
            teacher.generate_soft_labels([f"t{i}" for i in range(10)])

    def test_failure_budget_allows_the_exact_allowance(self):
        """One failure in ten is exactly 10% — at the limit, not over it."""
        def responder(n, kw):
            return ValueError("bad") if n == 1 else ('{"sec": 0.9, "nonsec": 0.1}', (60, 20))

        teacher, _ = build_teacher(
            responder, batch_size=1, max_retries=1, max_failure_rate=0.10,
        )
        labels = teacher.generate_soft_labels([f"t{i}" for i in range(10)])
        assert teacher.failed_indices == [0]
        assert len(labels) == 10

    def test_a_few_failures_are_tolerated_but_recorded(self):
        def responder(n, kw):
            if n == 3:
                return ValueError("truncated JSON")
            return '{"sec": 0.7, "nonsec": 0.3}', (60, 20)

        teacher, _ = build_teacher(
            responder, batch_size=1, max_retries=1, max_failure_rate=0.2,
        )
        labels = teacher.generate_soft_labels([f"t{i}" for i in range(10)])

        assert teacher.failed_indices == [2]
        assert labels[2] == {"sec": 0.5, "nonsec": 0.5}
        assert round(labels[0]["sec"], 2) == 0.7

    def test_filter_failed_drops_the_bad_rows_from_parallel_sequences(self):
        def responder(n, kw):
            if n == 2:
                return ValueError("bad")
            return '{"sec": 0.7, "nonsec": 0.3}', (60, 20)

        teacher, _ = build_teacher(
            responder, batch_size=1, max_retries=1, max_failure_rate=0.5,
        )
        texts = ["a", "b", "c", "d"]
        gold = ["sec", "nonsec", "sec", "nonsec"]
        soft = teacher.generate_soft_labels(texts)

        kept_texts, kept_gold, kept_soft = teacher.filter_failed(texts, gold, soft)

        assert kept_texts == ["a", "c", "d"]
        assert kept_gold == ["sec", "sec", "nonsec"]
        assert len(kept_soft) == 3
        assert all(d != {"sec": 0.5, "nonsec": 0.5} for d in kept_soft)

    def test_uniform_fallback_argmax_collapses_to_one_class(self):
        """
        Documents *why* the fallback is dangerous rather than merely useless:
        a tie is not neutral once argmax gets hold of it.
        """
        uniform = {"sec": 0.5, "nonsec": 0.5}
        assert max(uniform, key=uniform.get) == "sec"  # always the first key

    def test_usage_accounting_is_reported(self):
        teacher, _ = build_teacher(
            lambda n, kw: (batch_json(range(5)), (400, 300)), batch_size=5
        )
        teacher.generate_soft_labels([f"t{i}" for i in range(5)])

        assert teacher.usage["calls"] == 1
        assert teacher.usage["total_tokens"] == 700
        assert teacher.usage["rate_limit_hits"] == 0

    def test_state_resets_between_runs(self):
        calls = {"n": 0}

        def responder(n, kw):
            calls["n"] += 1
            return (ValueError("bad") if calls["n"] == 1
                    else ('{"sec": 0.7, "nonsec": 0.3}', (60, 20)))

        teacher, _ = build_teacher(
            responder, batch_size=1, max_retries=1, max_failure_rate=1.0,
        )
        teacher.generate_soft_labels(["a"])
        assert teacher.failed_indices == [0]

        teacher.generate_soft_labels(["b"])
        assert teacher.failed_indices == []

    def test_empty_input_is_a_no_op(self):
        teacher, fake = build_teacher(lambda n, kw: (batch_json([0]), (10, 10)))
        assert teacher.generate_soft_labels([]) == []
        assert fake.calls == []

    def test_invalid_batch_size_rejected(self):
        with pytest.raises(ValueError, match="batch_size"):
            OpenAITeacher(api_key="k", model="m", classes=CLASSES, batch_size=0)


class TestRateLimitWaitCap:
    """
    A provider's ``retry-after`` is a hint, not a contract.

    Groq has answered a first 25-text batch with ``retry-after`` ~974s and then
    served an equivalent request normally a minute later. Honouring that value
    literally parks a healthy run for the better part of an hour — which is how
    three separate full-dataset runs stalled at 0/775 and 75/775 without ever
    raising anything.
    """

    @staticmethod
    def _run(retry_after, n_429s=1, **kwargs):
        """Label one text through ``n_429s`` rate limits; return the waits taken."""
        def responder(n, kw):
            if n <= n_429s:
                return FakeRateLimitError(
                    headers={"retry-after": str(retry_after)}
                )
            return batch_json([0, 1]), (200, 100)

        teacher, fake = build_teacher(responder, batch_size=2, **kwargs)
        waits = []
        # Record the cool-off the retry loop decided on, without sleeping it.
        teacher._limiter.penalize = lambda seconds: waits.append(seconds)

        soft = teacher.generate_soft_labels(["a", "b"])
        return teacher, fake, waits, soft

    def test_an_oversized_retry_after_is_capped(self):
        teacher, fake, waits, _ = self._run(974.0, max_rate_limit_wait=120.0)

        assert len(fake.calls) == 2                 # it retried rather than gave up
        assert waits and max(waits) <= 121.0        # never committed to the 974s
        assert teacher.failed_indices == []

    def test_a_short_retry_after_is_honored_as_given(self):
        """The cap must not shorten a wait the provider genuinely needs."""
        _, _, waits, _ = self._run(21.0, max_rate_limit_wait=120.0)

        assert any(21.0 <= w <= 22.0 for w in waits)

    def test_the_run_completes_through_repeated_oversized_waits(self):
        teacher, _, waits, soft = self._run(
            974.0, n_429s=3, max_rate_limit_wait=120.0,
        )

        assert teacher.failed_indices == []
        assert teacher.usage["rate_limit_hits"] == 3
        assert sum(waits) < 4 * 121.0               # bounded: 3 caps, not 3 x 974s
        assert max(soft[0], key=soft[0].get) in CLASSES

    def test_no_retry_after_still_backs_off_exponentially(self):
        """With no hint to cap, the fallback schedule must be untouched."""
        def responder(n, kw):
            if n <= 3:
                return FakeRateLimitError(message="slow down")
            return batch_json([0, 1]), (200, 100)

        teacher, _ = build_teacher(responder, batch_size=2)
        waits = []
        teacher._limiter.penalize = lambda seconds: waits.append(seconds)

        teacher.generate_soft_labels(["a", "b"])

        assert [round(w) for w in waits] == [2, 4, 8]


class TestDailyQuota:
    """
    A per-day 429 is categorically different from a per-minute one.

    Observed on Groq: ``on tokens per day (TPD): Limit 200000, Used 199395 ...
    try again in 17m51s``. Treated as an ordinary rate limit it burns all 8
    retries over ~16 minutes, spends the trickle of quota that comes back, and
    still returns a half-labeled training set — which then distils into a
    partly-uniform, partly-real supervision signal.
    """

    GROQ_TPD = (
        "Rate limit reached for model `openai/gpt-oss-20b` in organization "
        "`org_x` service tier `on_demand` on tokens per day (TPD): "
        "Limit 200000, Used 199395, Requested 3086. "
        "Please try again in 17m51.79s"
    )
    GROQ_TPM = (
        "Rate limit reached for model `openai/gpt-oss-20b` on tokens per "
        "minute (TPM): Limit 8000, Used 7900, Requested 300. "
        "Please try again in 4.5s"
    )

    def test_daily_and_minute_limits_are_told_apart(self):
        assert _is_daily_limit(FakeRateLimitError(self.GROQ_TPD))
        assert not _is_daily_limit(FakeRateLimitError(self.GROQ_TPM))

    def test_a_daily_limit_raises_instead_of_retrying(self):
        def responder(n, kw):
            return FakeRateLimitError(self.GROQ_TPD)

        teacher, fake = build_teacher(
            responder, batch_size=2, max_rate_limit_retries=8,
        )

        with pytest.raises(DailyQuotaExhausted, match="daily quota"):
            teacher.generate_soft_labels(["a", "b"])

        assert len(fake.calls) == 1          # not 9 — it did not grind

    def test_a_minute_limit_is_still_retried(self):
        """The fail-fast path must not swallow ordinary, recoverable 429s."""
        def responder(n, kw):
            if n <= 2:
                return FakeRateLimitError(self.GROQ_TPM)
            return batch_json([0, 1]), (200, 100)

        teacher, fake = build_teacher(responder, batch_size=2)
        teacher._limiter.penalize = lambda seconds: None

        soft = teacher.generate_soft_labels(["a", "b"])

        assert len(fake.calls) == 3
        assert teacher.failed_indices == []
        assert len(soft) == 2
