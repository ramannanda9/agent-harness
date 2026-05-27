"""Tests for harness/utils.py helpers."""

from __future__ import annotations

import io

import pytest

from harness.events import BusEvent, EventType
from harness.utils import stream_tokens_inline


async def _emit(events: list[BusEvent]):
    for ev in events:
        yield ev


@pytest.mark.asyncio
async def test_stream_tokens_inline_consumes_tokens_and_yields_others():
    out = io.StringIO()
    events = [
        BusEvent(type=EventType.DISPATCH, payload={"complexity": "simple"}),
        BusEvent(type=EventType.TOKEN, token="he"),
        BusEvent(type=EventType.TOKEN, token="llo"),
        BusEvent(type=EventType.THOUGHT, payload={"thought": "done"}),
    ]
    yielded: list[BusEvent] = []
    async for ev in stream_tokens_inline(_emit(events), out=out):
        yielded.append(ev)

    # Two non-TOKEN events come through; TOKEN events are consumed.
    assert [ev.type for ev in yielded] == [EventType.DISPATCH, EventType.THOUGHT]
    # The streamed deltas land on stdout under the prefix on one line.
    assert out.getvalue() == "[stream]    hello\n"


@pytest.mark.asyncio
async def test_stream_tokens_inline_no_tokens_is_passthrough():
    out = io.StringIO()
    events = [
        BusEvent(type=EventType.DISPATCH, payload={}),
        BusEvent(type=EventType.TASK_DONE, payload={"answer": "ok"}),
    ]
    yielded = [ev async for ev in stream_tokens_inline(_emit(events), out=out)]
    assert [ev.type for ev in yielded] == [EventType.DISPATCH, EventType.TASK_DONE]
    assert out.getvalue() == ""


@pytest.mark.asyncio
async def test_stream_tokens_inline_flushes_trailing_stream_with_newline():
    """If the source ends mid-stream (no following non-TOKEN event), wrapper
    still terminates the stream line so stdout isn't left dangling."""
    out = io.StringIO()
    events = [
        BusEvent(type=EventType.TOKEN, token="he"),
        BusEvent(type=EventType.TOKEN, token="llo"),
    ]
    yielded = [ev async for ev in stream_tokens_inline(_emit(events), out=out)]
    assert yielded == []  # both tokens consumed
    assert out.getvalue() == "[stream]    hello\n"


@pytest.mark.asyncio
async def test_stream_tokens_inline_multiple_token_runs_get_separate_prefixes():
    """A non-TOKEN event between two runs closes the line; next TOKEN starts
    a new [stream] line with its own prefix."""
    out = io.StringIO()
    events = [
        BusEvent(type=EventType.TOKEN, token="abc"),
        BusEvent(type=EventType.OBSERVATION, payload={"observation": "ok"}),
        BusEvent(type=EventType.TOKEN, token="xyz"),
    ]
    yielded = [ev async for ev in stream_tokens_inline(_emit(events), out=out)]
    assert [ev.type for ev in yielded] == [EventType.OBSERVATION]
    assert out.getvalue() == "[stream]    abc\n[stream]    xyz\n"


@pytest.mark.asyncio
async def test_stream_tokens_inline_show_agent_id():
    """show_agent_id=True annotates the [stream] line with the agent_id."""
    out = io.StringIO()
    events = [
        BusEvent(type=EventType.TOKEN, agent_id="researcher", token="hi "),
        BusEvent(type=EventType.TOKEN, agent_id="researcher", token="there"),
    ]
    yielded = [ev async for ev in stream_tokens_inline(_emit(events), show_agent_id=True, out=out)]
    assert yielded == []
    assert out.getvalue() == "[stream]    researcher: hi there\n"


@pytest.mark.asyncio
async def test_stream_tokens_inline_custom_prefix():
    out = io.StringIO()
    events = [BusEvent(type=EventType.TOKEN, token="hello")]
    [_ async for _ in stream_tokens_inline(_emit(events), prefix=">>> ", out=out)]
    assert out.getvalue() == ">>> hello\n"
