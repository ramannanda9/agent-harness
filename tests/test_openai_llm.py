"""
OpenAILLM adapter tests with mocked openai SDK (no API calls).

We monkey-patch openai.AsyncOpenAI so the harness's request shape is verified
without hitting the network.
"""
from __future__ import annotations

import types
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("openai")
from harness.llm.openai import OpenAILLM, _prepend_system  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────


class _FakeChatCompletions:
    """Stand-in for client.chat.completions."""

    def __init__(self, *, response_content: str, stream_chunks: list[str]):
        self.calls: list[dict] = []
        self._response_content = response_content
        self._stream_chunks = stream_chunks

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._make_stream()
        msg = MagicMock()
        msg.content = self._response_content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    async def _make_stream(self) -> AsyncGenerator:
        for tok in self._stream_chunks:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = tok
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk


def _build(monkeypatch, *, content="ok", chunks=None) -> tuple[OpenAILLM, _FakeChatCompletions]:
    chunks = chunks or []
    fake = _FakeChatCompletions(response_content=content, stream_chunks=chunks)

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.chat = types.SimpleNamespace(completions=fake)

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeClient)
    return OpenAILLM(model="gpt-test"), fake


# ── _prepend_system ───────────────────────────────────────────────────────────


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


# ── complete ──────────────────────────────────────────────────────────────────


async def test_complete_returns_text_dict(monkeypatch):
    llm, fake = _build(monkeypatch, content='{"action":"finish","answer":"ok"}')
    out = await llm.complete(system=None, messages=[{"role": "user", "content": "hi"}])
    assert out == {"text": '{"action":"finish","answer":"ok"}'}
    assert fake.calls[0]["model"] == "gpt-test"
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


async def test_complete_prepends_system(monkeypatch):
    llm, fake = _build(monkeypatch, content="ok")
    await llm.complete(
        system="be helpful", messages=[{"role": "user", "content": "hi"}],
    )
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": "be helpful"}


async def test_complete_forwards_response_format(monkeypatch):
    llm, fake = _build(monkeypatch, content="{}")
    await llm.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


# ── stream_complete ──────────────────────────────────────────────────────────


async def test_stream_complete_yields_tokens(monkeypatch):
    llm, _fake = _build(monkeypatch, chunks=["he", "llo"])
    tokens = [t async for t in llm.stream_complete(system=None, messages=[])]
    assert tokens == ["he", "llo"]
