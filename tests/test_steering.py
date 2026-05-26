"""Async agent steering — core API, file watcher, stdin router.

Three groups:
  - Core: BaseAgent.steer() + drain at step boundary
  - FileSteer: file-watcher shim
  - StdinRouter / StdinSteer: terminal shim with HITL coordination

Tests use a stub agent for the shim groups so they exercise just the
shim behavior; the core group uses a real BaseAgent with ScriptedLLM.
"""

from __future__ import annotations

import asyncio

import pytest

from agents.base import AgentConfig
from harness.events import EventType
from harness.steering import (
    FileSteer,
    StdinRouter,
    StdinSteer,
    get_active_router,
)

# ── Test helpers ─────────────────────────────────────────────────────────────


class _StubAgent:
    """Minimal stand-in for BaseAgent — records steer() calls."""

    def __init__(self, agent_id: str = "a"):
        self.config = type("C", (), {"agent_id": agent_id})()
        self.steered: list[str] = []

    def steer(self, text: str) -> None:
        self.steered.append(text)


def _make_react_routes(*scripted_responses: dict) -> dict:
    """Build ScriptedLLM routes that emit a deterministic sequence of actions.

    The first len(scripted_responses)-1 calls return tool actions; the last
    returns a finish. Routed via the agent's system prompt prefix.
    """
    state = {"i": 0}

    def handler(system, messages, kwargs):
        i = state["i"]
        state["i"] = min(i + 1, len(scripted_responses) - 1)
        return scripted_responses[i]

    return {"react": handler}  # matches any system prompt (broad fallback)


# ── Core: BaseAgent.steer() + drain ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_drained_at_step_boundary(agent_factory, llm):
    """Guidance queued before run lands as a WM user message on step 0."""
    llm.routes = {
        "react": lambda *_: {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }
    }
    config = AgentConfig(agent_id="a", role="r", system_prompt="react", allowed_tools=[])
    agent = agent_factory(config)

    agent.steer("focus on X")
    await agent.run("task")

    msgs = agent._working_memory.get_messages()
    guidance = [
        m
        for m in msgs
        if isinstance(m["content"], str) and m["content"].startswith("Human guidance:")
    ]
    assert len(guidance) == 1
    assert guidance[0]["content"] == "Human guidance: focus on X"
    assert guidance[0]["role"] == "user"


@pytest.mark.asyncio
async def test_steer_fifo_order(agent_factory, llm):
    """Multiple queued items land in FIFO order in a single drain."""
    llm.routes = {
        "react": lambda *_: {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }
    }
    agent = agent_factory(
        AgentConfig(agent_id="a", role="r", system_prompt="react", allowed_tools=[])
    )
    agent.steer("first")
    agent.steer("second")
    agent.steer("third")
    await agent.run("task")

    msgs = agent._working_memory.get_messages()
    guidance_texts = [
        m["content"]
        for m in msgs
        if isinstance(m["content"], str) and m["content"].startswith("Human guidance:")
    ]
    assert guidance_texts == [
        "Human guidance: first",
        "Human guidance: second",
        "Human guidance: third",
    ]


@pytest.mark.asyncio
async def test_steer_emits_human_guidance_event(agent_factory, llm):
    """A HUMAN_GUIDANCE event fires with step + text for each drained item."""
    llm.routes = {
        "react": lambda *_: {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }
    }
    agent = agent_factory(
        AgentConfig(agent_id="a", role="r", system_prompt="react", allowed_tools=[])
    )
    agent.steer("pivot now")

    events: list = []
    async for ev in agent.run_stream("task"):
        events.append(ev)

    guidance_events = [e for e in events if e.type == EventType.HUMAN_GUIDANCE]
    assert len(guidance_events) == 1
    assert guidance_events[0].payload == {"step": 0, "text": "pivot now"}
    assert guidance_events[0].agent_id == "a"


@pytest.mark.asyncio
async def test_steer_empty_text_ignored(agent_factory, llm):
    """Empty / whitespace-only text never reaches WorkingMemory."""
    llm.routes = {
        "react": lambda *_: {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }
    }
    agent = agent_factory(
        AgentConfig(agent_id="a", role="r", system_prompt="react", allowed_tools=[])
    )
    agent.steer("")
    agent.steer("   ")
    agent.steer("\n\t")
    await agent.run("task")

    msgs = agent._working_memory.get_messages()
    guidance = [
        m for m in msgs if isinstance(m["content"], str) and "Human guidance:" in m["content"]
    ]
    assert guidance == []


@pytest.mark.asyncio
async def test_steer_drained_between_steps(agent_factory, llm):
    """Items queued during step N appear in step N+1's drain, in order."""
    from tests.conftest import EchoTool

    call_count = {"n": 0}
    queued = {"done": False}

    def handler(system, messages, kwargs):
        call_count["n"] += 1
        # Queue steering AFTER the first think but BEFORE the next iteration.
        if call_count["n"] == 1 and not queued["done"]:
            queued["done"] = True
            agent.steer("between steps")
        if call_count["n"] >= 2:
            return {
                "thought": "done",
                "action": "finish",
                "answer": "ok",
                "confidence": 1.0,
            }
        return {
            "thought": "use echo",
            "action": "echo",
            "args": {"message": "hi"},
        }

    llm.routes = {"react": handler}
    agent = agent_factory(
        AgentConfig(agent_id="a", role="r", system_prompt="react", allowed_tools=["echo"]),
        tools={"echo": EchoTool()},
    )

    events: list = []
    async for ev in agent.run_stream("task"):
        events.append(ev)

    # The steering call was made during step 0; it must drain at the top of step 1.
    guidance_events = [e for e in events if e.type == EventType.HUMAN_GUIDANCE]
    assert len(guidance_events) == 1
    assert guidance_events[0].payload["step"] == 1
    assert guidance_events[0].payload["text"] == "between steps"


# ── FileSteer ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_steer_picks_up_new_lines(tmp_path):
    path = tmp_path / "steer.txt"
    path.write_text("")  # exists, empty
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        path.write_text("first\nsecond\n")
        await asyncio.sleep(0.1)
        with open(path, "a") as f:
            f.write("third\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_file_steer_starts_at_eof_for_existing_file(tmp_path):
    """Pre-existing content is not replayed; only post-enter appends are picked up."""
    path = tmp_path / "steer.txt"
    path.write_text("stale1\nstale2\n")
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        await asyncio.sleep(0.05)  # would catch any premature replay
        with open(path, "a") as f:
            f.write("fresh\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["fresh"]


@pytest.mark.asyncio
async def test_file_steer_handles_missing_file(tmp_path):
    """Missing file is a no-op until it appears."""
    path = tmp_path / "not-yet.txt"
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        await asyncio.sleep(0.05)  # file still missing
        path.write_text("hi\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["hi"]


@pytest.mark.asyncio
async def test_file_steer_truncation_resets_offset(tmp_path):
    """If the file shrinks (truncate/recreate), the watcher reads from the start."""
    path = tmp_path / "steer.txt"
    path.write_text("orig1\norig2\n")
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        await asyncio.sleep(0.05)
        # Truncate and write fresh content
        path.write_text("brand new\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["brand new"]


@pytest.mark.asyncio
async def test_file_steer_stops_cleanly(tmp_path):
    """Lines written after context exit are not delivered."""
    path = tmp_path / "steer.txt"
    path.write_text("")
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        with open(path, "a") as f:
            f.write("inside\n")
        await asyncio.sleep(0.05)

    # Wait some time after exit; any further writes must NOT be picked up.
    with open(path, "a") as f:
        f.write("after exit\n")
    await asyncio.sleep(0.1)

    assert agent.steered == ["inside"]


def test_file_steer_default_path_uses_run_id_and_agent_id():
    agent = _StubAgent(agent_id="researcher")
    fs = FileSteer(agent, run_id="run123")
    assert fs.path == "/tmp/ah-run123-researcher.steer"


def test_file_steer_requires_path_or_run_id():
    agent = _StubAgent()
    with pytest.raises(ValueError):
        FileSteer(agent)


# ── StdinRouter / StdinSteer ──────────────────────────────────────────────────


class _FakeStdin:
    """Async readline that pops from a list. Returns None when exhausted."""

    def __init__(self, lines: list[str]):
        self._lines = list(lines)
        self._exhausted_event = asyncio.Event()

    async def readline(self) -> str | None:
        if not self._lines:
            # Block on an event so the router doesn't busy-loop after exhaustion.
            # Tests cancel the router before this matters.
            await self._exhausted_event.wait()
            return None
        return self._lines.pop(0)


@pytest.mark.asyncio
async def test_router_routes_to_steer_by_default():
    fake = _FakeStdin(["hello\n"])
    received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.set_steer_callback(received.append)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    assert received == ["hello"]


@pytest.mark.asyncio
async def test_router_routes_to_hitl_when_claimed():
    fake = _FakeStdin(["yes\n"])
    received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.set_steer_callback(received.append)
    await router.start()
    fut = router.claim_next_line()
    line = await asyncio.wait_for(fut, timeout=0.5)
    await router.stop()
    assert line == "yes"
    assert received == []  # HITL claimed, steer callback not invoked


@pytest.mark.asyncio
async def test_router_skips_empty_lines():
    fake = _FakeStdin(["\n", "   \n", "real\n"])
    received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.set_steer_callback(received.append)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    assert received == ["real"]


@pytest.mark.asyncio
async def test_stdin_single_agent_no_prefix_needed():
    a = _StubAgent("a")
    fake = _FakeStdin(["just do it\n"])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        async with StdinSteer(a, router=router):
            await asyncio.sleep(0.05)
    finally:
        await router.stop()
    assert a.steered == ["just do it"]


@pytest.mark.asyncio
async def test_stdin_multi_agent_prefix_routes():
    a = _StubAgent("a")
    b = _StubAgent("b")
    fake = _FakeStdin(["a: do A\n", "b: do B\n"])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        async with StdinSteer([a, b], router=router):
            await asyncio.sleep(0.05)
    finally:
        await router.stop()
    assert a.steered == ["do A"]
    assert b.steered == ["do B"]


@pytest.mark.asyncio
async def test_stdin_multi_agent_broadcast():
    a = _StubAgent("a")
    b = _StubAgent("b")
    fake = _FakeStdin(["*: stop now\n"])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        async with StdinSteer([a, b], router=router):
            await asyncio.sleep(0.05)
    finally:
        await router.stop()
    assert a.steered == ["stop now"]
    assert b.steered == ["stop now"]


@pytest.mark.asyncio
async def test_stdin_multi_agent_unprefixed_is_ignored(capsys):
    a = _StubAgent("a")
    b = _StubAgent("b")
    fake = _FakeStdin(["no prefix here\n"])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        async with StdinSteer([a, b], router=router):
            await asyncio.sleep(0.05)
    finally:
        await router.stop()
    assert a.steered == []
    assert b.steered == []
    err = capsys.readouterr().err
    assert "no agent prefix" in err


@pytest.mark.asyncio
async def test_stdin_multi_agent_unknown_prefix_is_ignored(capsys):
    a = _StubAgent("a")
    b = _StubAgent("b")
    fake = _FakeStdin(["zzz: nope\n"])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        async with StdinSteer([a, b], router=router):
            await asyncio.sleep(0.05)
    finally:
        await router.stop()
    assert a.steered == []
    assert b.steered == []
    err = capsys.readouterr().err
    assert "unknown agent 'zzz'" in err


@pytest.mark.asyncio
async def test_stdin_steer_registers_as_active_router():
    """get_active_router returns the router while inside the context."""
    a = _StubAgent("a")
    fake = _FakeStdin([])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        assert get_active_router() is None
        async with StdinSteer(a, router=router):
            assert get_active_router() is router
        assert get_active_router() is None
    finally:
        await router.stop()


def test_stdin_steer_rejects_empty_agent_list():
    with pytest.raises(ValueError):
        StdinSteer([])
