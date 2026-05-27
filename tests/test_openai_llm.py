"""
OpenAILLM adapter tests with mocked openai SDK (no API calls).

We monkey-patch openai.AsyncOpenAI so the harness's request shape is verified
without hitting the network. The adapter uses `with_raw_response.create()` so
it can read response headers (for gateway cost detection), so we stub that
specific path.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("openai")
from harness.llm.openai import (  # noqa: E402
    OpenAILLM,
    _prepend_system,
    _read_gateway_cost,
)

# ── Fake openai SDK surface ──────────────────────────────────────────────────


class _FakeUsage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


def _fake_response(content: str, *, model="gpt-test", p=10, c=20):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = _FakeUsage(p, c)
    resp.model = model
    return resp


def _fake_chunk(text: str | None = None, *, usage: _FakeUsage | None = None, model="gpt-test"):
    chunk = MagicMock()
    if text is None:
        chunk.choices = []
    else:
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
    chunk.usage = usage
    chunk.model = model
    return chunk


class _FakeRawResponse:
    """Mimics openai's APIResponse[T] returned by `with_raw_response.create()`."""

    def __init__(self, *, body, headers: dict[str, str] | None = None):
        self._body = body
        self.headers = headers or {}

    def parse(self):
        return self._body


class _FakeRawCreate:
    """Stand-in for client.chat.completions.with_raw_response."""

    def __init__(self):
        self.calls: list[dict] = []
        self.next_body = None
        self.next_headers: dict[str, str] = {}
        self.next_stream_chunks: list = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _FakeRawResponse(body=self._make_stream(), headers=self.next_headers)
        return _FakeRawResponse(body=self.next_body, headers=self.next_headers)

    async def _make_stream(self):
        for chunk in self.next_stream_chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self):
        self.with_raw_response = _FakeRawCreate()


def _build(monkeypatch, **kwargs):
    """Build an OpenAILLM whose client.chat.completions.with_raw_response is faked."""
    completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = types.SimpleNamespace(completions=completions)

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeClient)
    return OpenAILLM(model="gpt-test", **kwargs), completions.with_raw_response


# ── Pure helpers ─────────────────────────────────────────────────────────────


def test_prepend_system_noop_when_none():
    assert _prepend_system(None, [{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"},
    ]


def test_prepend_system_injects_system_message():
    out = _prepend_system("you are X", [{"role": "user", "content": "y"}])
    assert out == [
        {"role": "system", "content": "you are X"},
        {"role": "user", "content": "y"},
    ]


def test_read_gateway_cost_litellm_header():
    assert _read_gateway_cost({"x-litellm-response-cost": "0.0123"}) == pytest.approx(0.0123)


def test_read_gateway_cost_helicone_header():
    assert _read_gateway_cost({"x-helicone-cost-usd": "0.005"}) == pytest.approx(0.005)


def test_read_gateway_cost_missing_or_bad():
    assert _read_gateway_cost({}) is None
    assert _read_gateway_cost({"x-cost-usd": "not-a-number"}) is None
    assert _read_gateway_cost(None) is None


# ── complete() ───────────────────────────────────────────────────────────────


async def test_complete_returns_text_and_usage(monkeypatch):
    llm, raw = _build(monkeypatch)
    raw.next_body = _fake_response('{"action":"finish","answer":"ok"}', p=12, c=34)

    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])

    assert out["text"] == '{"action":"finish","answer":"ok"}'
    assert out["usage"]["tokens_in"] == 12
    assert out["usage"]["tokens_out"] == 34
    assert out["usage"]["total_tokens"] == 46
    assert out["usage"]["model"] == "gpt-test"
    # No cost_fn or gateway header → no cost reported.
    assert "cost_usd" not in out["usage"]
    # last_usage is mirrored on the adapter for streaming callers.
    assert llm.last_usage["tokens_in"] == 12


async def test_complete_prepends_system(monkeypatch):
    llm, raw = _build(monkeypatch)
    raw.next_body = _fake_response("ok")
    await llm.complete(
        system="be helpful",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert raw.calls[0]["messages"][0] == {"role": "system", "content": "be helpful"}


async def test_complete_forwards_response_format(monkeypatch):
    llm, raw = _build(monkeypatch)
    raw.next_body = _fake_response("{}")
    await llm.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert raw.calls[0]["response_format"] == {"type": "json_object"}


# ── cost: gateway header takes precedence ────────────────────────────────────


async def test_gateway_header_drives_cost(monkeypatch):
    llm, raw = _build(monkeypatch, cost_fn=lambda u: 999.0)  # cost_fn should be ignored
    raw.next_body = _fake_response("ok", p=100, c=50)
    raw.next_headers = {"x-litellm-response-cost": "0.0042"}

    out = await llm.complete(system=None, messages=[])

    assert out["usage"]["cost_usd"] == pytest.approx(0.0042)


async def test_cost_fn_used_when_no_gateway_header(monkeypatch):
    def my_cost(usage):
        return usage["tokens_in"] * 1e-4 + usage["tokens_out"] * 2e-4

    llm, raw = _build(monkeypatch, cost_fn=my_cost)
    raw.next_body = _fake_response("ok", p=100, c=50)

    out = await llm.complete(system=None, messages=[])

    expected = 100 * 1e-4 + 50 * 2e-4
    assert out["usage"]["cost_usd"] == pytest.approx(expected)


async def test_cost_fn_exception_swallowed(monkeypatch):
    def broken(usage):
        raise RuntimeError("pricing service offline")

    llm, raw = _build(monkeypatch, cost_fn=broken)
    raw.next_body = _fake_response("ok", p=10, c=5)

    out = await llm.complete(system=None, messages=[])
    assert "cost_usd" not in out["usage"]  # we still return usage, just no cost
    assert out["text"] == "ok"


# ── BudgetGuard integration ─────────────────────────────────────────────────


async def test_set_budget_routes_cost_to_guard(monkeypatch):
    from harness.runtime import BudgetGuard, GuardrailConfig

    llm, raw = _build(monkeypatch, cost_fn=lambda u: 0.0125)
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0))
    llm.set_budget(guard)

    raw.next_body = _fake_response("ok")
    await llm.complete(system=None, messages=[])
    assert guard.cost == pytest.approx(0.0125)


async def test_no_budget_means_no_crash(monkeypatch):
    llm, raw = _build(monkeypatch, cost_fn=lambda u: 0.99)
    raw.next_body = _fake_response("ok")
    out = await llm.complete(system=None, messages=[])
    assert out["usage"]["cost_usd"] == pytest.approx(0.99)


# ── stream_complete ─────────────────────────────────────────────────────────


async def test_stream_complete_yields_tokens(monkeypatch):
    llm, raw = _build(monkeypatch)
    raw.next_stream_chunks = [_fake_chunk("he"), _fake_chunk("llo")]
    tokens = [t async for t in llm.stream_complete(system=None, messages=[])]
    assert tokens == ["he", "llo"]
    # Without a final usage chunk, last_usage is left untouched.
    assert llm.last_usage is None


async def test_stream_complete_captures_final_usage(monkeypatch):
    llm, raw = _build(monkeypatch, cost_fn=lambda u: 0.007)
    # Final chunk has empty choices and a usage block — that's what OpenAI sends
    # when stream_options.include_usage=True.
    raw.next_stream_chunks = [
        _fake_chunk("he"),
        _fake_chunk("llo"),
        _fake_chunk(usage=_FakeUsage(7, 3)),
    ]
    tokens = [t async for t in llm.stream_complete(system=None, messages=[])]
    assert tokens == ["he", "llo"]
    assert llm.last_usage["tokens_in"] == 7
    assert llm.last_usage["tokens_out"] == 3
    assert llm.last_usage["cost_usd"] == pytest.approx(0.007)


async def test_stream_complete_sets_include_usage(monkeypatch):
    llm, raw = _build(monkeypatch)
    raw.next_stream_chunks = [_fake_chunk(usage=_FakeUsage(1, 1))]
    [t async for t in llm.stream_complete(system=None, messages=[])]
    assert raw.calls[0]["stream"] is True
    assert raw.calls[0]["stream_options"] == {"include_usage": True}
