"""Async agent steering — core API, file watcher, stdin router, factory.

Four groups:
  - Core BaseAgent.steer() + drain at step boundary
  - FileSteer file-watcher shim
  - StdinRouter pub/sub + HITL coordination
  - StdinSteer convenience wrapper + steering_source_factory integration
"""

from __future__ import annotations

import asyncio

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from harness.steering import (
    FileSteer,
    StdinAgentSource,
    StdinRouter,
    StdinSteer,
    file_steering_factory,
    get_active_router,
    stdin_steering_factory,
)
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── Test helpers ─────────────────────────────────────────────────────────────


class _StubAgent:
    """Minimal stand-in for BaseAgent — records steer() calls."""

    def __init__(self, agent_id: str = "a"):
        self.config = type("C", (), {"agent_id": agent_id})()
        self.steered: list[str] = []

    def steer(self, text: str) -> None:
        self.steered.append(text)


class _FakeStdin:
    """Async readline that pops from a list. Blocks on an event when exhausted."""

    def __init__(self, lines: list[str]):
        self._lines = list(lines)
        self._exhausted = asyncio.Event()

    async def readline(self) -> str | None:
        if not self._lines:
            await self._exhausted.wait()
            return None
        return self._lines.pop(0)


# ── Core: BaseAgent.steer() + drain ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_drained_at_step_boundary(agent_factory, llm):
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
    texts = [
        m["content"]
        for m in msgs
        if isinstance(m["content"], str) and m["content"].startswith("Human guidance:")
    ]
    assert texts == [
        "Human guidance: first",
        "Human guidance: second",
        "Human guidance: third",
    ]


@pytest.mark.asyncio
async def test_steer_emits_human_guidance_event(agent_factory, llm):
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
    assert [
        m for m in msgs if isinstance(m["content"], str) and "Human guidance:" in m["content"]
    ] == []


@pytest.mark.asyncio
async def test_steer_drained_between_steps(agent_factory, llm):
    from tests.conftest import EchoTool

    call_count = {"n": 0}

    def handler(system, messages, kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
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

    guidance_events = [e for e in events if e.type == EventType.HUMAN_GUIDANCE]
    assert len(guidance_events) == 1
    assert guidance_events[0].payload["step"] == 1
    assert guidance_events[0].payload["text"] == "between steps"


# ── FileSteer ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_steer_picks_up_new_lines(tmp_path):
    path = tmp_path / "steer.txt"
    path.write_text("")
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
    path = tmp_path / "steer.txt"
    path.write_text("stale1\nstale2\n")
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        await asyncio.sleep(0.05)
        with open(path, "a") as f:
            f.write("fresh\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["fresh"]


@pytest.mark.asyncio
async def test_file_steer_handles_missing_file(tmp_path):
    path = tmp_path / "not-yet.txt"
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        await asyncio.sleep(0.05)
        path.write_text("hi\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["hi"]


@pytest.mark.asyncio
async def test_file_steer_truncation_resets_offset(tmp_path):
    path = tmp_path / "steer.txt"
    path.write_text("orig1\norig2\n")
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        await asyncio.sleep(0.05)
        path.write_text("brand new\n")
        await asyncio.sleep(0.1)

    assert agent.steered == ["brand new"]


@pytest.mark.asyncio
async def test_file_steer_stops_cleanly(tmp_path):
    path = tmp_path / "steer.txt"
    path.write_text("")
    agent = _StubAgent()

    async with FileSteer(agent, str(path), interval=0.02):
        with open(path, "a") as f:
            f.write("inside\n")
        await asyncio.sleep(0.05)

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


# ── StdinRouter pub/sub ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_routes_to_catchall_subscriber():
    fake = _FakeStdin(["plain line\n"])
    received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.subscribe(None, received.append)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    assert received == ["plain line"]


@pytest.mark.asyncio
async def test_router_routes_by_prefix():
    fake = _FakeStdin(["a: hello a\n", "b: hello b\n"])
    a_received: list[str] = []
    b_received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.subscribe("a", a_received.append)
    router.subscribe("b", b_received.append)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    assert a_received == ["hello a"]
    assert b_received == ["hello b"]


@pytest.mark.asyncio
async def test_router_broadcast_with_star():
    fake = _FakeStdin(["*: stop everyone\n"])
    a_received: list[str] = []
    b_received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.subscribe("a", a_received.append)
    router.subscribe("b", b_received.append)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    assert a_received == ["stop everyone"]
    assert b_received == ["stop everyone"]


@pytest.mark.asyncio
async def test_router_unknown_prefix_warns(capsys):
    fake = _FakeStdin(["zzz: nope\n"])
    router = StdinRouter(readline=fake.readline)
    router.subscribe("a", lambda _t: None)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    err = capsys.readouterr().err
    assert "no subscriber for prefix 'zzz'" in err


@pytest.mark.asyncio
async def test_router_unprefixed_with_no_catchall_warns(capsys):
    fake = _FakeStdin(["no prefix here\n"])
    router = StdinRouter(readline=fake.readline)
    router.subscribe("a", lambda _t: None)
    await router.start()
    await asyncio.sleep(0.05)
    await router.stop()
    err = capsys.readouterr().err
    assert "no catch-all subscriber" in err


@pytest.mark.asyncio
async def test_router_unsubscribe_removes_subscription():
    """Unsubscribing while the router is running stops further deliveries."""

    class _GatedStdin:
        """Returns lines one at a time, gated by per-line release events."""

        def __init__(self, lines: list[str]):
            self._lines = list(lines)
            self.gates = [asyncio.Event() for _ in lines]  # keyed by original index
            self._next = 0
            self._exhausted = asyncio.Event()

        def release(self, i: int) -> None:
            self.gates[i].set()

        async def readline(self) -> str | None:
            if self._next >= len(self._lines):
                await self._exhausted.wait()
                return None
            i = self._next
            self._next += 1
            await self.gates[i].wait()
            return self._lines[i]

    fake = _GatedStdin(["a: first\n", "a: second\n"])
    received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    sid = router.subscribe("a", received.append)
    await router.start()

    fake.release(0)
    await asyncio.sleep(0.05)
    assert received == ["first"]

    router.unsubscribe(sid)
    fake.release(1)
    await asyncio.sleep(0.05)

    await router.stop()
    assert received == ["first"]


@pytest.mark.asyncio
async def test_router_routes_to_hitl_when_claimed():
    fake = _FakeStdin(["yes\n"])
    received: list[str] = []
    router = StdinRouter(readline=fake.readline)
    router.subscribe(None, received.append)
    await router.start()
    fut = router.claim_next_line()
    line = await asyncio.wait_for(fut, timeout=0.5)
    await router.stop()
    assert line == "yes"
    assert received == []  # HITL took it, subscriber didn't fire


def test_router_rejects_star_as_subscription_prefix():
    router = StdinRouter()
    with pytest.raises(ValueError):
        router.subscribe("*", lambda _t: None)


# ── StdinSteer convenience wrapper ────────────────────────────────────────────


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
async def test_stdin_single_agent_prefix_also_works():
    """Single-agent also accepts the explicit `agent_id:` form."""
    a = _StubAgent("a")
    fake = _FakeStdin(["a: explicit\n"])
    router = StdinRouter(readline=fake.readline)
    await router.start()
    try:
        async with StdinSteer(a, router=router):
            await asyncio.sleep(0.05)
    finally:
        await router.stop()
    assert a.steered == ["explicit"]


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
async def test_stdin_steer_registers_as_active_router():
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


# ── steering_source_factory integration with BaseAgent ────────────────────────


@pytest.mark.asyncio
async def test_factory_invoked_during_run_stream(agent_factory, llm, tmp_path):
    """A factory passed to BaseAgent is invoked once per run."""
    llm.routes = {
        "react": lambda *_: {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }
    }

    calls: list[BaseAgent] = []

    class _ProbeSource:
        def __init__(self, agent: BaseAgent):
            calls.append(agent)
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, *exc):
            self.exited = True

    sources: list[_ProbeSource] = []

    def factory(agent: BaseAgent) -> _ProbeSource:
        s = _ProbeSource(agent)
        sources.append(s)
        return s

    config = AgentConfig(agent_id="a", role="r", system_prompt="react", allowed_tools=[])

    # Build manually so we can pass the factory (agent_factory fixture doesn't).
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    agent = BaseAgent(
        config=config,
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=1.0, max_wall_time_seconds=30)),
        llm=llm,
        steering_source_factory=factory,
    )
    await agent.run("task")

    assert calls == [agent]
    assert sources[0].entered is True
    assert sources[0].exited is True


@pytest.mark.asyncio
async def test_file_steering_factory_drives_agent_via_file(llm, tmp_path):
    """A FileSteer source produced by file_steering_factory steers the agent mid-run."""
    from tests.conftest import SlowTool

    def handler(system, messages, kwargs):
        saw = any(
            isinstance(m.get("content"), str) and "Human guidance" in m["content"] for m in messages
        )
        if saw:
            return {"thought": "got steered", "action": "finish", "answer": "ok", "confidence": 1.0}
        return {"thought": "loop", "action": "slow", "args": {"label": "x"}}

    llm.routes = {"react": handler}

    config = AgentConfig(
        agent_id="counter",
        role="r",
        system_prompt="react",
        allowed_tools=["slow"],
        max_steps=50,  # loop is paced by SlowTool; want headroom for steer to land
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    template = str(tmp_path / "ah-{run_id}-{agent_id}.steer")
    factory = file_steering_factory(template, interval=0.02)
    agent = BaseAgent(
        config=config,
        tools={"slow": SlowTool(delay=0.05)},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=1.0, max_wall_time_seconds=30)),
        llm=llm,
        steering_source_factory=factory,
    )

    run_id = "test-run"
    expected_path = template.format(run_id=run_id, agent_id="counter")
    open(expected_path, "w").close()  # pre-exist so __aenter__ records offset=0

    async def steer_after_delay():
        await asyncio.sleep(0.2)  # let agent take a couple of slow-tool steps
        with open(expected_path, "a") as f:
            f.write("stop and finish\n")

    async def runner():
        return await agent.run("loop things", run_id=run_id)

    _, result = await asyncio.gather(steer_after_delay(), runner())
    assert result["answer"] == "ok", f"agent did not pivot — answer={result.get('answer')!r}"
    msgs = agent._working_memory.get_messages()
    assert any(isinstance(m["content"], str) and "stop and finish" in m["content"] for m in msgs)


@pytest.mark.asyncio
async def test_stdin_steering_factory_subscribes_each_agent():
    """stdin_steering_factory produces a source that subscribes to the agent's prefix."""
    a = _StubAgent("a")
    router = StdinRouter()
    factory = stdin_steering_factory(router)

    source = factory(a)
    assert isinstance(source, StdinAgentSource)
    async with source:
        # Verify it actually subscribed.
        assert "a" in router.active_prefixes()
    assert "a" not in router.active_prefixes()
