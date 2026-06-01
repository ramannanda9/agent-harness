"""Tests for ``RoutingLLM`` — dispatch each call to a different adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from harness.llm.routing import RoutingLLM, by_role, by_system_keyword, by_token_count


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

    async def stream_complete(self, system, messages) -> AsyncGenerator[str, None]:
        self.stream_calls += 1
        if self._text:
            yield self._text


class _CompleteOnlyLLM:
    """Adapter without stream_complete — to test fallback in RoutingLLM."""

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


async def test_complete_strips_call_role_kwarg_before_forwarding():
    """call_role is consumed by the selector — don't pass it through to the
    SDK or it'll reject the unknown kwarg."""

    class _StrictLLM(_StubLLM):
        async def complete(self, system, messages, **kwargs):
            assert "call_role" not in kwargs, "call_role must not be forwarded"
            return await super().complete(system, messages, **kwargs)

    strict = _StrictLLM(text="ok")
    llm = RoutingLLM({"x": strict}, selector=lambda s, m: "x", default_route="x")
    result = await llm.complete(None, [], call_role="planner")
    assert result["text"] == "ok"


async def test_set_budget_forwards_to_every_route():
    a = _StubLLM()
    b = _StubLLM()
    llm = RoutingLLM({"a": a, "b": b}, selector=lambda s, m: "a", default_route="a")
    guard = object()
    llm.set_budget(guard)
    assert a.budget is guard
    assert b.budget is guard


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


# ── by_system_keyword ────────────────────────────────────────────────────────


def test_by_system_keyword_matches_first_keyword():
    sel = by_system_keyword({"planner": "cheap", "agent": "pricey"}, default="default")
    assert sel("You are a planner that...", []) == "cheap"
    assert sel("You are an agent that...", []) == "pricey"
    assert sel("You are something else", []) == "default"
    assert sel(None, []) == "default"


def test_by_system_keyword_is_case_insensitive():
    sel = by_system_keyword({"Planner": "cheap"}, default="default")
    assert sel("YOU ARE A PLANNER", []) == "cheap"


# ── by_token_count ───────────────────────────────────────────────────────────


def test_by_token_count_routes_small_to_cheap_large_to_pricey():
    sel = by_token_count(threshold=100, small="cheap", large="pricey")
    assert sel("short", [{"content": "hi"}]) == "cheap"
    assert sel("x" * 1000, [{"content": "y" * 1000}]) == "pricey"


def test_by_token_count_handles_multimodal_content_blocks():
    """messages may carry a list of content blocks (e.g. vision) — token
    estimate must walk into them."""
    sel = by_token_count(threshold=10, small="cheap", large="pricey")
    msgs = [{"content": [{"type": "text", "text": "x" * 200}]}]
    assert sel(None, msgs) == "pricey"


def test_by_token_count_uses_custom_counter_when_supplied():
    sel = by_token_count(
        threshold=10,
        small="cheap",
        large="pricey",
        counter=lambda s: 100,  # always says 100 tokens
    )
    assert sel("a", []) == "pricey"


# ── by_role ──────────────────────────────────────────────────────────────────


def test_by_role_routes_from_call_role_metadata_on_message():
    sel = by_role({"planner": "cheap", "agent": "pricey"}, default="default")
    msgs = [{"role": "user", "content": "x", "call_role": "planner"}]
    assert sel(None, msgs) == "cheap"


def test_by_role_falls_back_to_default_without_metadata():
    sel = by_role({"planner": "cheap"}, default="default")
    msgs = [{"role": "user", "content": "x"}]
    assert sel(None, msgs) == "default"
