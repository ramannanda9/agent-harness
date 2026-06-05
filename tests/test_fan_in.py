"""``harness.streaming.fan_in`` — multi-producer event fan-in.

Validates the fan-in invariants the SubAgentTool path depends on:
  - All items from all drivers reach the consumer
  - Per-driver order is preserved (no reordering within a single driver)
  - Driver exceptions surface to the consumer, not silently swallowed
  - Consumer cancellation cleans up still-running drivers
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from harness.streaming import fan_in


async def _driver(items: list, *, delay: float = 0.0) -> AsyncIterator:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


@pytest.mark.asyncio
async def test_fan_in_drains_all_items():
    collected = []
    async for idx, item in fan_in(
        [
            lambda: _driver(["a1", "a2"]),
            lambda: _driver(["b1"]),
        ]
    ):
        collected.append((idx, item))
    assert sorted(collected) == [(0, "a1"), (0, "a2"), (1, "b1")]


@pytest.mark.asyncio
async def test_fan_in_preserves_per_driver_order():
    """Within one driver, items must arrive in the order produced."""
    seen: dict[int, list] = {0: [], 1: []}
    async for idx, item in fan_in(
        [
            lambda: _driver(["a", "b", "c"], delay=0.01),
            lambda: _driver(["x", "y"], delay=0.005),
        ]
    ):
        seen[idx].append(item)
    assert seen[0] == ["a", "b", "c"]
    assert seen[1] == ["x", "y"]


@pytest.mark.asyncio
async def test_fan_in_empty_input_returns_immediately():
    items = [(i, v) async for i, v in fan_in([])]
    assert items == []


@pytest.mark.asyncio
async def test_fan_in_surfaces_driver_exception():
    async def _bomb() -> AsyncIterator:
        yield "first"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in fan_in([lambda: _bomb()]):
            pass


@pytest.mark.asyncio
async def test_fan_in_cancels_drivers_on_consumer_break():
    """If the consumer exits the loop early, still-running drivers must be
    cancelled — otherwise they'd dangle in the event loop."""
    cancelled = False

    async def _slow() -> AsyncIterator:
        nonlocal cancelled
        try:
            yield "first"
            await asyncio.sleep(1.0)  # consumer breaks before this resolves
            yield "second"
        except asyncio.CancelledError:
            cancelled = True
            raise

    async for _idx, item in fan_in([lambda: _slow()]):
        if item == "first":
            break

    # Give the cancellation a tick to propagate.
    await asyncio.sleep(0.05)
    assert cancelled, "driver should have been cancelled when consumer exited early"
