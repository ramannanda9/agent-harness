"""Multi-producer event fan-in.

Used by ``BaseAgent`` when the LLM emits parallel ``actions: [...]`` whose
tools stream events (today: ``SubAgentTool``). Each driver pushes events
into a shared ``asyncio.Queue``; the consumer drains in arrival order so
the parent agent's stream stays a single sequence even when several
sub-agents are working in parallel.

The orchestrator has its own near-identical fan-in in
``_run_batch`` — but it's coupled to task / DAG / TaskResult logic.
Rather than refactor both call sites at once, this module hosts the
clean shape; the orchestrator can adopt it later without changing
``_run_batch``'s public behaviour.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

T = TypeVar("T")

# Returned from drivers to signal "I'm done, nothing more to yield."
# Identity-comparable so consumers can distinguish from legitimate yields.
_DRIVER_DONE = object()


async def fan_in(
    drivers: list[Callable[[], AsyncIterator[T]]],
) -> AsyncIterator[tuple[int, T]]:
    """Drain N async iterators into a single stream, preserving arrival order.

    Each driver is a zero-arg factory that returns an async iterator. The
    factory shape (vs. passing iterators directly) lets the caller capture
    closure state per-driver cleanly; same pattern as the orchestrator's
    ``drive(task)`` closures.

    Yields ``(driver_index, item)`` so the consumer can distinguish which
    sub-agent emitted each event without inspecting payload.
    """
    if not drivers:
        return

    bus: asyncio.Queue = asyncio.Queue()

    async def _drive(idx: int, factory: Callable[[], AsyncIterator[T]]) -> None:
        try:
            async for item in factory():
                await bus.put((idx, item))
        except Exception as exc:  # noqa: BLE001 — caller decides how to surface
            await bus.put((idx, _DriverError(exc)))
        finally:
            await bus.put((idx, _DRIVER_DONE))

    tasks = [asyncio.create_task(_drive(i, f)) for i, f in enumerate(drivers)]
    finished = 0
    try:
        while finished < len(drivers):
            idx, payload = await bus.get()
            if payload is _DRIVER_DONE:
                finished += 1
                continue
            if isinstance(payload, _DriverError):
                # Re-raise on the consumer side so caller can decide whether to
                # crash the parent run or treat as a sub-agent failure.
                raise payload.exc
            yield idx, payload
    finally:
        # Cancel any drivers still running if the consumer exits early
        # (e.g. caller broke out of the loop). Without this, sub-agent tasks
        # would dangle until the event loop exits.
        for t in tasks:
            if not t.done():
                t.cancel()
        # Drain cancellations so we don't leave warnings on the event loop.
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


class _DriverError:
    """Wraps a driver exception so it survives the bus queue verbatim."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


async def gather_with_events(
    drivers: list[Callable[[], AsyncIterator[Any]]],
    *,
    is_result: Callable[[Any], bool],
) -> AsyncIterator[Any]:
    """Convenience layer on top of ``fan_in`` for the
    "streaming tools that also produce a final result" pattern.

    Each driver yields a mix of events and exactly one terminal "result"
    object. ``is_result(item)`` distinguishes the two. The consumer sees
    events bubble up in arrival order, then receives ``("results", [r_0,
    r_1, …])`` as the final yield once every driver has finished.

    Designed for ``BaseAgent``'s parallel-action handler where each action
    is a streaming tool: the parent stream interleaves sub-agent events
    while running, then gets one observation list when the batch is done.
    """
    results: dict[int, Any] = {}
    async for idx, item in fan_in(drivers):
        if is_result(item):
            results[idx] = item
        else:
            yield item
    yield ("results", [results.get(i) for i in range(len(drivers))])


async def collect_with_events(
    driver: Callable[[], AsyncIterator[Any]],
    *,
    is_result: Callable[[Any], bool],
) -> AsyncIterator[Any]:
    """Single-driver variant: pass through events; the terminal "result"
    item is yielded last as ``("result", obj)``.

    Used by ``BaseAgent``'s sequential single-action handler when the tool
    streams. Avoids the fan-in machinery for the no-parallelism case.
    """
    result = None
    async for item in driver():
        if is_result(item):
            result = item
        else:
            yield item
    yield ("result", result)
