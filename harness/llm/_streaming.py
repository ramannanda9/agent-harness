"""Shared SSE helpers for streaming-capable LLM adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any


async def aiter_sse_events(response: Any) -> AsyncGenerator[tuple[str, str], None]:
    """Yield (event_type, data) pairs from an SSE response.

    Parses the standard `event:` / `data:` line format. Blank lines
    terminate events. The default event type for unlabelled events is
    `"message"`. Trailing buffered data (no terminating blank line) is
    flushed when the stream ends.
    """
    current_event = "message"
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield current_event, "\n".join(data_lines)
                current_event = "message"
                data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        yield current_event, "\n".join(data_lines)


async def read_error_body(response: Any) -> bytes:
    """Drain the body of an error response, returning at most 4 KiB."""
    out: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        if total >= 4096:
            break
        out.append(chunk)
        total += len(chunk)
    return b"".join(out)[:4096]


def format_streaming_error(status_code: int, body: bytes, *, provider: str) -> str:
    """Build a user-facing error message from an error response body.

    Truncates aggressively because error bodies sometimes echo request
    payloads — we don't want bearer tokens or full prompts in tracebacks.
    """
    text = body.decode(errors="replace").strip()
    if not text:
        return f"{provider} backend returned HTTP {status_code}"
    return f"{provider} backend returned {status_code}: {text[:500]}"
