"""
test_ollama_llm.py — Tests for the local Ollama Student SLM wrapper
===================================================================

The wrapper's job is to produce the confidence scalar that becomes the 385th
feature column. These tests pin the two things that determine whether that
column carries any information at all when the frozen base is a very small
model:

1. Confidence is read from the first token's logprob distribution, not from a
   number the model states about itself — a 0.5B model states ~1.0 for
   everything, which flattens the column to a constant.
2. The forced choice is asked in both option orders and averaged, because
   small models answer largely from option position. Measured on 50 balanced
   PROMISE requirements, ``qwen2.5:0.5b`` answered 45/50 one way and 44/50 the
   other way when only the ordering changed.

No Ollama server is needed: every test drives a fake HTTP transport.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from teasco_tree.llm.ollama_student import OllamaLLM

CLASSES = ["sec", "nonsec"]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _logprob_reply(letter_probs):
    """A /api/chat reply whose first token carries ``letter_probs``."""
    top = [
        {"token": letter, "logprob": math.log(p)}
        for letter, p in letter_probs.items()
    ]
    top.sort(key=lambda e: -e["logprob"])
    return {
        "message": {"role": "assistant", "content": top[0]["token"]},
        "logprobs": [{"token": top[0]["token"], "top_logprobs": top}],
    }


def _install(monkeypatch, handler):
    """Route ``requests.post`` to ``handler(payload) -> reply dict``."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return _FakeResponse(handler(json))

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def _user_message(payload):
    return payload["messages"][-1]["content"]


class TestLogprobScoring:
    def test_confidence_comes_from_the_token_distribution(self, monkeypatch):
        # A) is the first-listed class in whichever order was sent.
        _install(monkeypatch, lambda p: _logprob_reply({"A": 0.7, "B": 0.3}))

        llm = OllamaLLM(CLASSES, cache_path=None, debias=False)
        preds, confs = llm.predict(["some requirement"])

        assert preds == ["sec"]
        assert confs[0].item() == pytest.approx(0.7, abs=1e-5)

    def test_probabilities_are_renormalised_over_the_classes(self, monkeypatch):
        # Most of the mass sits on tokens that are not offered letters; what
        # survives must still be a distribution over the classes.
        _install(
            monkeypatch,
            lambda p: _logprob_reply({"A": 0.1, "B": 0.3, "The": 0.6}),
        )

        llm = OllamaLLM(CLASSES, cache_path=None, debias=False)
        preds, confs = llm.predict(["some requirement"])

        assert preds == ["nonsec"]
        assert confs[0].item() == pytest.approx(0.75, abs=1e-5)

    def test_confidence_varies_across_texts(self, monkeypatch):
        """The failure this mode exists to prevent: a constant 385th column."""
        splits = iter([(0.95, 0.05), (0.55, 0.45), (0.72, 0.28)])
        _install(
            monkeypatch,
            lambda p: _logprob_reply(dict(zip("AB", next(splits)))),
        )

        llm = OllamaLLM(CLASSES, cache_path=None, debias=False)
        _, confs = llm.predict(["a", "b", "c"])

        assert len(set(round(c, 4) for c in confs.tolist())) == 3


class TestOrderDebiasing:
    def test_both_orderings_are_asked(self, monkeypatch):
        calls = _install(
            monkeypatch, lambda p: _logprob_reply({"A": 0.6, "B": 0.4})
        )

        llm = OllamaLLM(
            CLASSES,
            cache_path=None,
            class_descriptions={"sec": "security-related",
                                "nonsec": "not security-related"},
        )
        llm.predict(["some requirement"])

        assert len(calls) == 2
        first, second = (_user_message(c) for c in calls)
        assert first.index("A) security-related") < first.index("B) not security")
        assert second.index("A) not security") < second.index("B) security-related")

    def test_position_bias_is_cancelled(self, monkeypatch):
        """A model that always answers 'A' must come out undecided, not certain."""
        _install(monkeypatch, lambda p: _logprob_reply({"A": 0.9, "B": 0.1}))

        llm = OllamaLLM(CLASSES, cache_path=None)
        _, confs = llm.predict(["some requirement"])

        assert confs[0].item() == pytest.approx(0.5, abs=1e-5)

    def test_single_order_would_not_cancel_it(self, monkeypatch):
        """The same fake model, ungated by debiasing, reads as 90% confident."""
        _install(monkeypatch, lambda p: _logprob_reply({"A": 0.9, "B": 0.1}))

        llm = OllamaLLM(CLASSES, cache_path=None, debias=False)
        preds, confs = llm.predict(["some requirement"])

        assert preds == ["sec"]
        assert confs[0].item() == pytest.approx(0.9, abs=1e-5)


class TestJsonFallback:
    def test_falls_back_when_the_server_returns_no_logprobs(self, monkeypatch):
        """Ollama < 0.12 ignores the flag; degrade instead of failing."""
        def handler(payload):
            if payload.get("logprobs"):
                return {"message": {"content": "A"}}     # no "logprobs" key
            return {"message": {"content": '{"label": "nonsec", '
                                           '"confidence": 0.8}'}}

        _install(monkeypatch, handler)

        llm = OllamaLLM(CLASSES, cache_path=None)
        preds, confs = llm.predict(["some requirement"])

        assert preds == ["nonsec"]
        assert confs[0].item() == pytest.approx(0.8, abs=1e-5)
        assert llm._logprobs_unsupported is True

    def test_fallback_is_decided_once(self, monkeypatch):
        """After the first miss, later texts must not re-probe for logprobs."""
        def handler(payload):
            if payload.get("logprobs"):
                return {"message": {"content": "A"}}
            return {"message": {"content": '{"label": "sec", '
                                           '"confidence": 0.6}'}}

        calls = _install(monkeypatch, handler)

        llm = OllamaLLM(CLASSES, cache_path=None)
        llm.predict(["a", "b", "c"])

        assert sum(1 for c in calls if c.get("logprobs")) == 1

    def test_falls_back_when_no_class_letter_is_offered(self, monkeypatch):
        def handler(payload):
            if payload.get("logprobs"):
                return _logprob_reply({"The": 0.6, "It": 0.4})
            return {"message": {"content": '{"label": "nonsec", '
                                           '"confidence": 0.55}'}}

        _install(monkeypatch, handler)

        llm = OllamaLLM(CLASSES, cache_path=None)
        preds, confs = llm.predict(["some requirement"])

        assert preds == ["nonsec"]
        assert confs[0].item() == pytest.approx(0.55, abs=1e-5)
        # Not a server capability problem — later texts still try logprobs.
        assert llm._logprobs_unsupported is False


class TestCacheIdentity:
    def test_model_name_carries_the_scoring_mode(self):
        """FeatureBuilder hashes model_name; the modes must not share a cache."""
        ids = {
            OllamaLLM(CLASSES, cache_path=None).model_name,
            OllamaLLM(CLASSES, cache_path=None, debias=False).model_name,
            OllamaLLM(CLASSES, cache_path=None, scoring="json").model_name,
        }
        assert len(ids) == 3

    def test_per_text_cache_keys_differ_across_modes(self):
        a = OllamaLLM(CLASSES, cache_path=None)
        b = OllamaLLM(CLASSES, cache_path=None, scoring="json")
        assert a._cache_key("same text") != b._cache_key("same text")

    def test_cached_results_are_reused(self, monkeypatch, tmp_path):
        calls = _install(
            monkeypatch, lambda p: _logprob_reply({"A": 0.8, "B": 0.2})
        )

        cache = tmp_path / "ollama_cache.json"
        first = OllamaLLM(CLASSES, cache_path=cache, debias=False)
        first.predict(["repeated requirement"])
        assert len(calls) == 1

        second = OllamaLLM(CLASSES, cache_path=cache, debias=False)
        preds, confs = second.predict(["repeated requirement"])
        assert len(calls) == 1                      # served from disk
        assert preds == ["sec"]
        assert confs[0].item() == pytest.approx(0.8, abs=1e-5)


class TestValidation:
    def test_rejects_an_unknown_scoring_mode(self):
        with pytest.raises(ValueError, match="scoring"):
            OllamaLLM(CLASSES, scoring="vibes")

    def test_rejects_more_classes_than_letters(self):
        with pytest.raises(ValueError, match="letter"):
            OllamaLLM([f"c{i}" for i in range(27)])
