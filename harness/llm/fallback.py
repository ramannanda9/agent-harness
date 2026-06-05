"""``FallbackLLM`` — try multiple LLM clients in order on transient failures.

Wraps any number of LLM adapters that share the standard harness contract
(``complete``, optionally ``stream_complete``, ``set_budget``, ``last_usage``).
On a transient error (rate limit, timeout, 5xx) the next adapter in the list
is tried. The first non-transient error — or exhausting the list — re-raises.

Example::

    from harness.llm.openai import OpenAILLM
    from harness.llm.anthropic import AnthropicLLM
    from harness.llm.fallback import FallbackLLM

    primary = AnthropicLLM(model="claude-sonnet-4-6")
    backup = OpenAILLM(model="gpt-4o-mini")
    llm = FallbackLLM([primary, backup])

    runtime = AgentRuntime(..., llm=llm)

Set ``transient_errors`` to a callable that returns True when the exception
should trigger the next fallback. The default heuristic catches rate-limit
and 5xx-class errors from the OpenAI and Anthropic SDKs and any
``asyncio.TimeoutError`` / ``ConnectionError`` / ``OSError`` raised by the
transport.

``last_route`` exposes the index of the adapter that actually answered the
most recent call, so callers can see which one was hit::

    await llm.complete(system, messages)
    print(llm.last_route)   # 0 if primary worked, 1 if backup did, ...
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

logger = logging.getLogger(__name__)


def _default_is_transient(exc: BaseException) -> bool:
    """Best-effort classifier for retryable upstream errors.

    Detects without importing the SDKs (so the fallback adapter has no
    optional-dep coupling):
      - ``status_code`` attr in {408, 425, 429, 500, 502, 503, 504}
      - class name suffixed with ``RateLimitError`` / ``ServiceUnavailableError``
        / ``APITimeoutError`` / ``InternalServerError`` / ``OverloadedError``
      - ``asyncio.TimeoutError``, ``ConnectionError``, ``OSError``
    """
    if isinstance(exc, asyncio.TimeoutError | ConnectionError | OSError):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in {408, 425, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__
    transient_suffixes = (
        "RateLimitError",
        "ServiceUnavailableError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
        "TimeoutError",
    )
    return any(name.endswith(s) for s in transient_suffixes)


class FallbackLLM:
    def __init__(
        self,
        llms: list[Any],
        *,
        transient_errors: Callable[[BaseException], bool] | None = None,
    ) -> None:
        if not llms:
            raise ValueError("FallbackLLM requires at least one inner LLM")
        self._llms = list(llms)
        self._is_transient = transient_errors or _default_is_transient
        self.last_route: int = -1
        self.last_usage: dict | None = None

    def set_budget(self, guard: Any) -> None:
        """Forward the budget guard to every inner LLM."""
        for llm in self._llms:
            if hasattr(llm, "set_budget"):
                llm.set_budget(guard)

    # ── Non-streaming ────────────────────────────────────────────────────────

    async def complete(
        self,
        system: str | None,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict:
        last_exc: BaseException | None = None
        for i, llm in enumerate(self._llms):
            try:
                result = await llm.complete(system, messages, **kwargs)
            except BaseException as exc:
                if i == len(self._llms) - 1 or not self._is_transient(exc):
                    raise
                logger.warning(
                    "FallbackLLM: adapter %d (%s) raised transient %s — trying next",
                    i,
                    type(llm).__name__,
                    type(exc).__name__,
                )
                last_exc = exc
                continue
            self.last_route = i
            self.last_usage = getattr(llm, "last_usage", None)
            return result
        # Unreachable in practice — the loop always returns or re-raises.
        assert last_exc is not None
        raise last_exc

    # ── Streaming ────────────────────────────────────────────────────────────

    async def stream_complete(
        self,
        system: str | None,
        messages: list[dict],
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream from the first adapter that doesn't fail before yielding.

        We can only retry until the first token has been emitted — once the
        caller has seen partial output, a switch mid-stream would corrupt the
        response. The transient check therefore runs against errors raised
        before the generator yields anything.
        """
        last_exc: BaseException | None = None
        for i, llm in enumerate(self._llms):
            if not hasattr(llm, "stream_complete"):
                continue
            try:
                gen = llm.stream_complete(system, messages, source=source, **kwargs)
                first = await _peek_first(gen)
            except BaseException as exc:
                if i == len(self._llms) - 1 or not self._is_transient(exc):
                    raise
                logger.warning(
                    "FallbackLLM(stream): adapter %d (%s) raised transient %s "
                    "before first token — trying next",
                    i,
                    type(llm).__name__,
                    type(exc).__name__,
                )
                last_exc = exc
                continue
            self.last_route = i
            if first is not None:
                yield first
            async for chunk in gen:
                yield chunk
            self.last_usage = getattr(llm, "last_usage", None)
            return
        assert last_exc is not None
        raise last_exc


async def _peek_first(gen: AsyncGenerator[str, None]) -> str | None:
    """Pull the first item from an async generator, or None if exhausted."""
    try:
        return await gen.__anext__()
    except StopAsyncIteration:
        return None
