"""Tests for ``FallbackLLM`` — try each adapter in order on transient failures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from harness.llm.fallback import FallbackLLM, _default_is_transient


class _StubLLM:
    """Configurable LLM stub.

    Set ``exc`` to raise on ``complete`` / ``stream_complete``; set ``text`` to
    return successfully. ``stream_chunks`` controls what stream_complete emits.
    Records the budget guard injected by ``set_budget``.
    """

    def __init__(
        self,
        *,
        text: str = "",
        exc: BaseException | None = None,
        usage: dict | None = None,
        stream_chunks: list[str] | None = None,
        stream_exc: BaseException | None = None,
        stream_exc_after: int = 0,
    ) -> None:
        self._text = text
        self._exc = exc
        if stream_chunks is not None:
            self._stream_chunks = stream_chunks
        else:
            self._stream_chunks = [text] if text else []
        self._stream_exc = stream_exc
        self._stream_exc_after = stream_exc_after
        self.last_usage = usage
        self.calls = 0
        self.stream_calls = 0
        self.budget: Any = None

    def set_budget(self, guard: Any) -> None:
        self.budget = guard

    async def complete(self, system, messages, **kwargs) -> dict:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return {"text": self._text, "usage": self.last_usage or {}}

    async def stream_complete(self, system, messages) -> AsyncGenerator[str, None]:
        self.stream_calls += 1
        for i, chunk in enumerate(self._stream_chunks):
            if self._stream_exc is not None and i == self._stream_exc_after:
                raise self._stream_exc
            yield chunk


class _RateLimitError(Exception):
    pass


class _PermissionError(Exception):
    pass


# ── Transient classifier ─────────────────────────────────────────────────────


def test_default_is_transient_on_status_code():
    e = Exception()
    e.status_code = 429
    assert _default_is_transient(e) is True


def test_default_is_transient_on_5xx_status_code():
    for code in (500, 502, 503, 504):
        e = Exception()
        e.status_code = code
        assert _default_is_transient(e) is True


def test_default_is_transient_on_class_name_suffix():
    assert _default_is_transient(_RateLimitError()) is True


def test_default_is_transient_on_timeout_error():
    assert _default_is_transient(asyncio.TimeoutError()) is True


def test_default_is_transient_on_connection_error():
    assert _default_is_transient(ConnectionError("reset")) is True


def test_default_is_transient_false_for_permanent_error():
    e = _PermissionError("denied")
    assert _default_is_transient(e) is False


def test_default_is_transient_false_for_4xx_other_than_408_425_429():
    for code in (400, 401, 403, 404, 422):
        e = Exception()
        e.status_code = code
        assert _default_is_transient(e) is False, f"status {code} should not be transient"


# ── complete() ───────────────────────────────────────────────────────────────


async def test_complete_uses_first_llm_on_success():
    a = _StubLLM(text="from a", usage={"tokens_in": 1})
    b = _StubLLM(text="from b")
    llm = FallbackLLM([a, b])

    result = await llm.complete(None, [])
    assert result["text"] == "from a"
    assert llm.last_route == 0
    assert llm.last_usage == {"tokens_in": 1}
    assert b.calls == 0


async def test_complete_falls_through_on_transient_error():
    a = _StubLLM(exc=_RateLimitError())
    b = _StubLLM(text="from b", usage={"tokens_in": 2})
    llm = FallbackLLM([a, b])

    result = await llm.complete(None, [])
    assert result["text"] == "from b"
    assert llm.last_route == 1
    assert llm.last_usage == {"tokens_in": 2}


async def test_complete_re_raises_permanent_error_from_first():
    """A non-transient error from the primary must not fall through —
    auth failures, schema errors, etc. should surface immediately."""
    a = _StubLLM(exc=_PermissionError("403"))
    b = _StubLLM(text="from b")
    llm = FallbackLLM([a, b])

    with pytest.raises(_PermissionError):
        await llm.complete(None, [])
    assert b.calls == 0


async def test_complete_re_raises_when_all_adapters_fail():
    a = _StubLLM(exc=_RateLimitError("a"))
    b = _StubLLM(exc=_RateLimitError("b"))
    llm = FallbackLLM([a, b])

    with pytest.raises(_RateLimitError, match="b"):
        await llm.complete(None, [])


async def test_set_budget_forwards_to_all_inner_llms():
    a = _StubLLM(text="x")
    b = _StubLLM(text="y")
    llm = FallbackLLM([a, b])
    guard = object()
    llm.set_budget(guard)
    assert a.budget is guard
    assert b.budget is guard


def test_constructor_rejects_empty_list():
    with pytest.raises(ValueError):
        FallbackLLM([])


async def test_custom_transient_classifier_overrides_default():
    a = _StubLLM(exc=_PermissionError("403"))
    b = _StubLLM(text="from b")
    # Treat _PermissionError as transient via the custom classifier.
    llm = FallbackLLM(
        [a, b],
        transient_errors=lambda e: isinstance(e, _PermissionError),
    )
    result = await llm.complete(None, [])
    assert result["text"] == "from b"


# ── stream_complete() ────────────────────────────────────────────────────────


async def test_stream_falls_through_on_pre_first_token_failure():
    """Adapter that raises before yielding any chunk should fall through."""
    a = _StubLLM(
        stream_chunks=["never-yielded"],
        stream_exc=_RateLimitError(),
        stream_exc_after=0,
    )
    b = _StubLLM(stream_chunks=["hel", "lo"])
    llm = FallbackLLM([a, b])

    chunks = [c async for c in llm.stream_complete(None, [])]
    assert chunks == ["hel", "lo"]
    assert llm.last_route == 1


async def test_stream_does_not_retry_after_partial_output():
    """Once a chunk has been yielded, mid-stream failure must propagate —
    switching adapters would corrupt the response."""
    # Two chunks so the for loop reaches i=1 and raises; the second chunk is
    # never yielded because the exception fires first.
    a = _StubLLM(
        stream_chunks=["partial", "never-yielded"],
        stream_exc=_RateLimitError(),
        stream_exc_after=1,
    )
    b = _StubLLM(stream_chunks=["fallback"])
    llm = FallbackLLM([a, b])

    chunks = []
    with pytest.raises(_RateLimitError):
        async for c in llm.stream_complete(None, []):
            chunks.append(c)
    assert chunks == ["partial"]
    # Fallback was never invoked because the failure was mid-stream.
    assert b.stream_calls == 0
