"""JSONL trace recorder + replayer for ``BusEvent`` streams.

Two halves:

  - ``record_trace(stream, path)`` — async iterator that copies every
    ``BusEvent`` through unchanged while also writing one JSON object per
    line to ``path``. Drop-in around any harness streaming call.

  - ``replay(path)`` — async iterator that yields ``BusEvent`` objects
    reconstructed from the JSONL file, preserving inter-event timing by
    default (``realtime=True``) so timeline displays match the original
    run. Set ``realtime=False`` to drain as fast as possible.

The on-disk format is one JSON object per line, fields::

    {"type": "thought", "agent_id": "planner", "payload": {...},
     "token": "", "error": "", "timestamp": 1748823051.42}

Forward-compatible: unknown keys are ignored on replay, ``EventType`` values
fall back to their raw string when not in the current enum.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from harness.events import BusEvent, EventType

logger = logging.getLogger(__name__)


async def record_trace(
    stream: AsyncIterator[BusEvent],
    path: str | Path,
    *,
    append: bool = False,
) -> AsyncIterator[BusEvent]:
    """Wrap a BusEvent stream and persist each event to ``path`` as JSONL.

    Yields every event through unchanged so the caller's own loop sees the
    same stream. Writes are flushed per-event so partial traces survive
    crashes — replay against an interrupted run shows everything up to the
    failure point.

    Args:
        stream: Any async iterator of BusEvents (``runtime.dispatch_stream``,
                ``runtime.run_stream``, …).
        path: Destination file. Parent directories are created.
        append: When True, append to an existing file instead of truncating.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as fh:
        async for event in stream:
            try:
                fh.write(_event_to_json(event) + "\n")
                fh.flush()
            except (TypeError, ValueError) as e:
                # Don't crash the live stream because tracing failed —
                # the agent's run is more important than the trace file.
                logger.warning("trace write failed for %s: %s", event.type, e)
            yield event


async def replay(
    path: str | Path,
    *,
    realtime: bool = True,
    speed: float = 1.0,
) -> AsyncIterator[BusEvent]:
    """Yield ``BusEvent`` objects reconstructed from a JSONL trace file.

    Args:
        path: Path to a ``.jsonl`` file produced by ``record_trace``.
        realtime: When True (default), sleep between events to match the
                  recorded timing — useful for UI timelines that want to
                  see the run \"as it happened\". When False, drain
                  immediately.
        speed: Multiplier when ``realtime=True``. ``2.0`` plays back at
               double speed; ``0.5`` at half.
    """
    src = Path(path).expanduser()
    last_ts: float | None = None
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed trace line: %s", e)
                continue
            event = _event_from_json(obj)
            if event is None:
                continue
            if realtime and last_ts is not None and speed > 0:
                delay = max(0.0, (event.timestamp - last_ts) / speed)
                if delay > 0:
                    await asyncio.sleep(delay)
            last_ts = event.timestamp
            yield event


# ── Serialisation ────────────────────────────────────────────────────────────


def _event_to_json(event: BusEvent) -> str:
    """Render a BusEvent as a one-line JSON string."""
    return json.dumps(
        {
            "type": event.type.value if isinstance(event.type, EventType) else str(event.type),
            "agent_id": event.agent_id,
            "payload": _safe_payload(event.payload),
            "token": event.token,
            "error": event.error,
            "timestamp": event.timestamp,
        },
        ensure_ascii=False,
        default=_json_fallback,
    )


def _event_from_json(obj: dict[str, Any]) -> BusEvent | None:
    """Reconstruct a BusEvent from a parsed JSON object."""
    raw_type = obj.get("type")
    if not isinstance(raw_type, str):
        return None
    try:
        event_type = EventType(raw_type)
    except ValueError:
        # Forward-compat: unknown event types pass through as strings so
        # consumers can still inspect them, even if EventType.<X> is missing.
        event_type = raw_type  # type: ignore[assignment]
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    return BusEvent(
        type=event_type,
        agent_id=str(obj.get("agent_id") or ""),
        payload=payload,
        token=str(obj.get("token") or ""),
        error=str(obj.get("error") or ""),
        timestamp=float(obj.get("timestamp") or 0.0),
    )


def _safe_payload(payload: Any) -> Any:
    """Best-effort coerce a payload to JSON-safe primitives."""
    if isinstance(payload, dict):
        return {str(k): _safe_payload(v) for k, v in payload.items()}
    if isinstance(payload, list | tuple):
        return [_safe_payload(v) for v in payload]
    if isinstance(payload, str | int | float | bool) or payload is None:
        return payload
    return repr(payload)


def _json_fallback(obj: Any) -> Any:
    """Last-resort encoder for arbitrary objects in payloads."""
    if hasattr(obj, "model_dump"):  # pydantic
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)
