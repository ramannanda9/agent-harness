"""Tests for the JSONL trace recorder + replay + viewer.

The recorder/replay pair has to round-trip BusEvents losslessly, survive
crashes mid-stream, and degrade gracefully on garbage input. The viewer
test just confirms the HTTP server boots and serves the expected routes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import pytest

from harness.events import BusEvent, EventType
from harness.trace import record_trace, replay


async def _events() -> AsyncIterator[BusEvent]:
    """A short, varied event stream covering every interesting payload shape."""
    now = time.time()
    yield BusEvent(
        type=EventType.DISPATCH,
        payload={"complexity": "simple", "path": "routed"},
        timestamp=now,
    )
    yield BusEvent(
        type=EventType.ROUTE,
        agent_id="planner",
        payload={"agent_id": "planner", "rationale": "only one registered"},
        timestamp=now + 0.01,
    )
    yield BusEvent(
        type=EventType.THOUGHT,
        agent_id="planner",
        payload={"thought": "I should fetch the docs"},
        timestamp=now + 0.02,
    )
    yield BusEvent(type=EventType.TOKEN, agent_id="planner", token="hel", timestamp=now + 0.03)
    yield BusEvent(
        type=EventType.ACTION,
        agent_id="planner",
        payload={"tool": "fetch", "args": {"url": "https://example.com"}},
        timestamp=now + 0.04,
    )
    yield BusEvent(
        type=EventType.OBSERVATION,
        agent_id="planner",
        payload={"observation": "ok"},
        timestamp=now + 0.05,
    )
    yield BusEvent(type=EventType.ERROR, error="boom", timestamp=now + 0.06)
    yield BusEvent(
        type=EventType.DONE,
        payload={"answer": "42"},
        timestamp=now + 0.07,
    )


# ── Recorder ─────────────────────────────────────────────────────────────────


async def test_record_writes_jsonl_and_passes_events_through(tmp_path):
    out = tmp_path / "trace.jsonl"
    seen = []
    async for event in record_trace(_events(), out):
        seen.append(event.type)

    assert seen == [
        EventType.DISPATCH,
        EventType.ROUTE,
        EventType.THOUGHT,
        EventType.TOKEN,
        EventType.ACTION,
        EventType.OBSERVATION,
        EventType.ERROR,
        EventType.DONE,
    ]
    lines = out.read_text().splitlines()
    assert len(lines) == 8
    first = json.loads(lines[0])
    assert first["type"] == "dispatch"
    assert first["payload"] == {"complexity": "simple", "path": "routed"}
    err = json.loads(lines[-2])
    assert err["error"] == "boom"


async def test_record_flushes_per_event_so_partial_traces_survive(tmp_path):
    """If the consumer crashes mid-stream, the trace file still has everything
    written up to the failure."""
    out = tmp_path / "trace.jsonl"

    async def crashing_consumer():
        count = 0
        async for _event in record_trace(_events(), out):
            count += 1
            if count == 3:
                raise RuntimeError("consumer died")
        return count

    with pytest.raises(RuntimeError, match="consumer died"):
        await crashing_consumer()

    lines = out.read_text().splitlines()
    # First three events should be on disk despite the crash.
    assert len(lines) == 3
    assert json.loads(lines[2])["type"] == "thought"


async def test_record_append_mode(tmp_path):
    """append=True keeps existing trace content and adds new events after it."""
    out = tmp_path / "trace.jsonl"
    out.write_text('{"type": "dispatch", "payload": {}, "timestamp": 1.0}\n')

    async def one_event():
        yield BusEvent(type=EventType.DONE, payload={"x": 1}, timestamp=2.0)

    async for _ in record_trace(one_event(), out, append=True):
        pass

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["type"] == "done"


async def test_record_does_not_crash_on_unserialisable_payload(tmp_path, caplog):
    """Payloads that contain objects without a JSON encoder fall back to repr().
    The stream itself must keep flowing — tracing is best-effort."""
    out = tmp_path / "trace.jsonl"

    class Weird:
        def __repr__(self) -> str:
            return "<Weird>"

    async def stream():
        yield BusEvent(type=EventType.THOUGHT, payload={"obj": Weird()}, timestamp=1.0)

    received = []
    async for ev in record_trace(stream(), out):
        received.append(ev)

    assert len(received) == 1
    lines = out.read_text().splitlines()
    assert "<Weird>" in lines[0]


# ── Replay ───────────────────────────────────────────────────────────────────


async def test_replay_round_trips_events(tmp_path):
    out = tmp_path / "trace.jsonl"
    async for _ in record_trace(_events(), out):
        pass

    replayed = [ev async for ev in replay(out, realtime=False)]
    assert [e.type for e in replayed] == [
        EventType.DISPATCH,
        EventType.ROUTE,
        EventType.THOUGHT,
        EventType.TOKEN,
        EventType.ACTION,
        EventType.OBSERVATION,
        EventType.ERROR,
        EventType.DONE,
    ]
    # Agent IDs, payloads, tokens, errors all preserved.
    assert replayed[1].agent_id == "planner"
    assert replayed[3].token == "hel"
    assert replayed[4].payload == {"tool": "fetch", "args": {"url": "https://example.com"}}
    assert replayed[6].error == "boom"


async def test_replay_skips_malformed_lines(tmp_path):
    out = tmp_path / "trace.jsonl"
    out.write_text(
        "\n"
        '{"type": "dispatch", "payload": {}, "timestamp": 1.0}\n'
        "this is not json\n"
        '{"type": "done", "payload": {"x": 1}, "timestamp": 2.0}\n'
    )

    events = [ev async for ev in replay(out, realtime=False)]
    assert [e.type for e in events] == [EventType.DISPATCH, EventType.DONE]


async def test_replay_preserves_unknown_event_types(tmp_path):
    """Forward-compat: a trace from a newer harness with new EventType values
    should still load — the type passes through as a raw string."""
    out = tmp_path / "trace.jsonl"
    out.write_text('{"type": "future_event", "payload": {"k": 1}, "timestamp": 1.0}\n')

    events = [ev async for ev in replay(out, realtime=False)]
    assert len(events) == 1
    assert events[0].type == "future_event"
    assert events[0].payload == {"k": 1}


async def test_replay_realtime_honours_inter_event_delays(tmp_path):
    """With realtime=True, the iterator should sleep so the visible cadence
    matches the recorded timestamps. Use speed=10 to keep the test fast."""
    out = tmp_path / "trace.jsonl"
    out.write_text(
        '{"type": "dispatch", "payload": {}, "timestamp": 100.0}\n'
        '{"type": "done", "payload": {}, "timestamp": 100.3}\n'
    )

    start = asyncio.get_running_loop().time()
    events = [ev async for ev in replay(out, realtime=True, speed=10.0)]
    elapsed = asyncio.get_running_loop().time() - start

    assert len(events) == 2
    # 0.3s recorded gap @ 10x speed = ~0.03s; allow generous slack for CI.
    assert 0.02 <= elapsed <= 0.5


# ── Viewer ───────────────────────────────────────────────────────────────────


def test_viewer_serves_html_and_trace(tmp_path):
    import urllib.request

    from harness.trace_viewer import serve

    out = tmp_path / "trace.jsonl"
    out.write_text('{"type": "dispatch", "payload": {"x": 1}, "timestamp": 1.0}\n')

    server = serve(out, port=0, open_browser=False, block=False)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            html = resp.read().decode("utf-8")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/trace.jsonl") as resp:
            jsonl = resp.read().decode("utf-8")
    finally:
        server.shutdown()

    assert "trace viewer" in html.lower()
    assert "dispatch" in jsonl
