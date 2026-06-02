"""Tests for ``RoutingLLM`` — bring-your-own-selector LLM dispatcher.

The shipped surface is intentionally small: a selector callable, a routes
dict, and a default. No baked-in selectors that encourage misrouting.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from harness.llm.routing import RoutingLLM


class _StubLLM:
    def __init__(self, *, text: str = "", usage: dict | None = None) -> None:
        self._text = text
        self.last_usage = usage
        self.calls = 0
        self.stream_calls = 0
        self.budget: Any = None

    def set_budget(self, guard: Any) -> None:
        self.budget = guard

    async def complete(self, system, messages, **kwargs) -> dict:
        self.calls += 1
        return {"text": self._text, "usage": self.last_usage or {}}

    async def stream_complete(self, system, messages, **_kwargs) -> AsyncGenerator[str, None]:
        self.stream_calls += 1
        if self._text:
            yield self._text


class _CompleteOnlyLLM:
    """Adapter without stream_complete — verifies RoutingLLM's fallback path."""

    def __init__(self, *, text: str = "") -> None:
        self._text = text
        self.last_usage: dict | None = None
        self.calls = 0

    async def complete(self, system, messages, **kwargs) -> dict:
        self.calls += 1
        return {"text": self._text, "usage": {}}


# ── Constructor validation ───────────────────────────────────────────────────


def test_constructor_rejects_empty_routes():
    with pytest.raises(ValueError):
        RoutingLLM({}, selector=lambda s, m: "x", default_route="x")


def test_constructor_rejects_default_not_in_routes():
    with pytest.raises(ValueError, match="default_route"):
        RoutingLLM({"a": _StubLLM()}, selector=lambda s, m: "a", default_route="missing")


# ── complete() ───────────────────────────────────────────────────────────────


async def test_complete_dispatches_to_selected_route():
    cheap = _StubLLM(text="from cheap", usage={"tokens_in": 1})
    pricey = _StubLLM(text="from pricey", usage={"tokens_in": 99})
    llm = RoutingLLM(
        {"cheap": cheap, "pricey": pricey},
        selector=lambda s, m: "cheap",
        default_route="pricey",
    )

    result = await llm.complete(None, [])
    assert result["text"] == "from cheap"
    assert llm.last_route == "cheap"
    assert llm.last_usage == {"tokens_in": 1}
    assert pricey.calls == 0


async def test_complete_falls_back_to_default_when_selector_returns_unknown_key():
    a = _StubLLM(text="a")
    default = _StubLLM(text="default")
    llm = RoutingLLM(
        {"a": a, "default": default},
        selector=lambda s, m: "nonexistent",
        default_route="default",
    )

    result = await llm.complete(None, [])
    assert result["text"] == "default"
    assert llm.last_route == "default"


async def test_complete_falls_back_to_default_when_selector_raises():
    a = _StubLLM(text="a")
    default = _StubLLM(text="default")

    def explodes(_system, _messages):
        raise RuntimeError("boom")

    llm = RoutingLLM(
        {"a": a, "default": default},
        selector=explodes,
        default_route="default",
    )

    result = await llm.complete(None, [])
    assert result["text"] == "default"


async def test_set_budget_forwards_to_every_route():
    a = _StubLLM()
    b = _StubLLM()
    llm = RoutingLLM({"a": a, "b": b}, selector=lambda s, m: "a", default_route="a")
    guard = object()
    llm.set_budget(guard)
    assert a.budget is guard
    assert b.budget is guard


async def test_selector_receives_system_and_messages():
    """Custom selectors get the full call context — verify the wiring."""
    seen: list[tuple[str | None, list[dict]]] = []

    def remember(system, messages):
        seen.append((system, messages))
        return "x"

    a = _StubLLM(text="x")
    llm = RoutingLLM({"x": a}, selector=remember, default_route="x")
    msgs = [{"role": "user", "content": "hello"}]
    await llm.complete("a system", msgs)

    assert seen == [("a system", msgs)]


# ── stream_complete() ────────────────────────────────────────────────────────


async def test_stream_dispatches_to_selected_route():
    cheap = _StubLLM(text="hello cheap")
    pricey = _StubLLM(text="hello pricey")
    llm = RoutingLLM(
        {"cheap": cheap, "pricey": pricey},
        selector=lambda s, m: "cheap",
        default_route="pricey",
    )

    chunks = [c async for c in llm.stream_complete(None, [])]
    assert chunks == ["hello cheap"]
    assert llm.last_route == "cheap"
    assert pricey.stream_calls == 0


async def test_stream_falls_back_to_complete_for_route_without_streaming():
    no_stream = _CompleteOnlyLLM(text="non-streaming result")
    llm = RoutingLLM({"x": no_stream}, selector=lambda s, m: "x", default_route="x")

    chunks = [c async for c in llm.stream_complete(None, [])]
    assert chunks == ["non-streaming result"]
    assert no_stream.calls == 1
