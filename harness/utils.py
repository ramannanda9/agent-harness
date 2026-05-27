"""Shared utilities for harness, orchestrator, and memory packages."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncGenerator, AsyncIterable, Coroutine
from typing import Any

_log = logging.getLogger(__name__)


def fire(coro: Coroutine) -> asyncio.Task:
    """Schedule coro as a background task; log but don't propagate exceptions.

    Use for best-effort writes (memory, metrics) that must not block the
    critical path. The returned Task can be ignored — errors are logged.
    """
    task = asyncio.create_task(coro)

    def _on_done(t: asyncio.Task) -> None:
        if not t.cancelled() and (exc := t.exception()):
            _log.warning("Background task failed: %s", exc)

    task.add_done_callback(_on_done)
    return task


async def stream_tokens_inline(
    events: AsyncIterable[Any],
    *,
    prefix: str = "[stream]    ",
    show_agent_id: bool = False,
    out: Any = None,
) -> AsyncGenerator[Any, None]:
    """Wrap a BusEvent stream, consuming TOKEN events to stdout inline.

    Each `EventType.TOKEN` event is printed live (no newline between
    deltas) under a single `prefix` line, so the LLM response
    materialises in place. The very next non-TOKEN event closes the
    line with a newline and is yielded through to the caller unchanged.

    Callers iterate the returned stream as usual and never need a
    `TOKEN` branch of their own; the streaming concern is fully hidden.

    Args:
        events:        Source async iterable of `BusEvent`s
                       (e.g. `runtime.dispatch_stream(goal)`).
        prefix:        Header shown once at the start of each streamed
                       reply, before the first delta.
        show_agent_id: If True, append `<agent_id>: ` after the prefix
                       — useful in orchestrated multi-agent demos where
                       knowing which agent is talking matters.
        out:           Stream to write to. Defaults to `sys.stdout`.

    Yields:
        Every non-TOKEN event from the source, in order.
    """
    from harness.events import EventType  # local import: avoid cycle

    stream = out or sys.stdout
    active = False
    async for event in events:
        if event.type == EventType.TOKEN:
            if not active:
                if show_agent_id and event.agent_id:
                    stream.write(f"{prefix}{event.agent_id}: ")
                else:
                    stream.write(prefix)
                active = True
            stream.write(event.token)
            stream.flush()
            continue
        if active:
            stream.write("\n")
            stream.flush()
            active = False
        yield event
    if active:
        stream.write("\n")
        stream.flush()


def parse_llm_json(response: Any) -> dict:
    """Unwrap an LLM adapter response into a plain dict.

    LLM adapters may return:
      - a dict with a "text" key containing a JSON string
      - a raw JSON string
      - a dict already (json_object mode with some adapters)

    Raises json.JSONDecodeError if the content is not valid JSON.
    """
    if isinstance(response, dict) and "text" in response:
        return json.loads(response["text"])
    if isinstance(response, str):
        return json.loads(response)
    if isinstance(response, dict):
        return response
    return {}
