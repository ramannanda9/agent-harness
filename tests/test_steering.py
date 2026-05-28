"""Async agent steering — core API, file watcher, stdin router, factory.

Four groups:
  - Core BaseAgent.steer() + drain at step boundary
  - FileSteer file-watcher shim
  - StdinRouter pub/sub + HITL coordination
  - StdinSteer convenience wrapper + steering_source_factory integration
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.runtime import (
    AgentRegistry,
    AgentRuntime,
    BudgetGuard,
    GuardrailConfig,
    ToolRegistry,
    Tracer,
)
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


@contextlib.contextmanager
def _piped_router():
    """Yield (router, pipe_input). Caller writes via pipe_input.send_text().

    Submit sequences: "\\r" (Enter). To insert a newline mid-input:
    "\\n" (Ctrl+J binding) or "\\x1b\\r" (Esc-Enter). patch_stdout is
    disabled so pytest's stdout capture still works.
    """
    with create_pipe_input() as pipe_in:
        router = StdinRouter(
            input_=pipe_in,
            output=DummyOutput(),
            patch_stdout_=False,
        )
        yield router, pipe_in


async def _drain(timeout: float = 0.1) -> None:
    """Let the router process pending input. Small await for the read loop."""
    await asyncio.sleep(timeout)


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


# ── StdinRouter pub/sub (prompt_toolkit-backed) ──────────────────────────────


@pytest.mark.asyncio
async def test_router_routes_to_catchall_subscriber():
    received: list[str] = []
    with _piped_router() as (router, pipe_in):
        router.subscribe(None, received.append)
        await router.start()
        pipe_in.send_text("plain line\r")
        await _drain()
        await router.stop()
    assert received == ["plain line"]


@pytest.mark.asyncio
async def test_router_default_patch_stdout_context_starts():
    """Default patch_stdout path uses a sync context manager but still runs in async loop."""
    received: list[str] = []
    with create_pipe_input() as pipe_in:
        router = StdinRouter(input_=pipe_in, output=DummyOutput())
        router.subscribe(None, received.append)
        await router.start()
        pipe_in.send_text("plain line\r")
        await _drain()
        await router.stop()
    assert received == ["plain line"]


@pytest.mark.asyncio
async def test_router_routes_by_prefix():
    a_received: list[str] = []
    b_received: list[str] = []
    with _piped_router() as (router, pipe_in):
        router.subscribe("a", a_received.append)
        router.subscribe("b", b_received.append)
        await router.start()
        pipe_in.send_text("a: hello a\r")
        await _drain()
        pipe_in.send_text("b: hello b\r")
        await _drain()
        await router.stop()
    assert a_received == ["hello a"]
    assert b_received == ["hello b"]


@pytest.mark.asyncio
async def test_router_broadcast_with_star():
    a_received: list[str] = []
    b_received: list[str] = []
    with _piped_router() as (router, pipe_in):
        router.subscribe("a", a_received.append)
        router.subscribe("b", b_received.append)
        await router.start()
        pipe_in.send_text("*: stop everyone\r")
        await _drain()
        await router.stop()
    assert a_received == ["stop everyone"]
    assert b_received == ["stop everyone"]


@pytest.mark.asyncio
async def test_router_multiline_input_via_ctrl_j():
    """Ctrl+J (\\n) inserts a newline; Enter (\\r) submits the whole block."""
    received: list[str] = []
    with _piped_router() as (router, pipe_in):
        router.subscribe("a", received.append)
        await router.start()
        # Type "a: line one" + Ctrl+J + "line two" + Enter
        pipe_in.send_text("a: line one\nline two\r")
        await _drain()
        await router.stop()
    assert received == ["line one\nline two"]


@pytest.mark.asyncio
async def test_router_unknown_prefix_warns(capsys):
    with _piped_router() as (router, pipe_in):
        router.subscribe("a", lambda _t: None)
        await router.start()
        pipe_in.send_text("zzz: nope\r")
        await _drain()
        await router.stop()
    err = capsys.readouterr().err
    assert "no subscriber for prefix 'zzz'" in err


@pytest.mark.asyncio
async def test_router_unprefixed_with_no_catchall_warns(capsys):
    with _piped_router() as (router, pipe_in):
        router.subscribe("a", lambda _t: None)
        await router.start()
        pipe_in.send_text("no prefix here\r")
        await _drain()
        await router.stop()
    err = capsys.readouterr().err
    assert "no catch-all subscriber" in err


@pytest.mark.asyncio
async def test_router_unsubscribe_removes_subscription():
    """After unsubscribe, further input doesn't reach the old callback."""
    received: list[str] = []
    with _piped_router() as (router, pipe_in):
        sid = router.subscribe("a", received.append)
        await router.start()
        pipe_in.send_text("a: first\r")
        await _drain()
        router.unsubscribe(sid)
        pipe_in.send_text("a: second\r")
        await _drain()
        await router.stop()
    assert received == ["first"]


@pytest.mark.asyncio
async def test_router_claim_next_line_resolves_with_typed_answer():
    """HITL claims pre-empt steering; the typed answer resolves HITL's Future."""
    received: list[str] = []
    with _piped_router() as (router, pipe_in):
        router.subscribe(None, received.append)
        await router.start()
        # Give the steering prompt a moment to start, then claim for HITL.
        await asyncio.sleep(0.02)
        fut = router.claim_next_line(prompt="approve? ")
        # User types "y" + Enter; HITL gets it, not the catch-all subscriber.
        pipe_in.send_text("y\r")
        line = await asyncio.wait_for(fut, timeout=1.0)
        await router.stop()
    assert line == "y"
    assert received == []


def test_router_rejects_star_as_subscription_prefix():
    router = StdinRouter(patch_stdout_=False)
    with pytest.raises(ValueError):
        router.subscribe("*", lambda _t: None)


# ── StdinSteer convenience wrapper ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stdin_single_agent_no_prefix_needed():
    a = _StubAgent("a")
    with _piped_router() as (router, pipe_in):
        await router.start()
        try:
            async with StdinSteer(a, router=router):
                pipe_in.send_text("just do it\r")
                await _drain()
        finally:
            await router.stop()
    assert a.steered == ["just do it"]


@pytest.mark.asyncio
async def test_stdin_single_agent_prefix_also_works():
    a = _StubAgent("a")
    with _piped_router() as (router, pipe_in):
        await router.start()
        try:
            async with StdinSteer(a, router=router):
                pipe_in.send_text("a: explicit\r")
                await _drain()
        finally:
            await router.stop()
    assert a.steered == ["explicit"]


@pytest.mark.asyncio
async def test_stdin_multi_agent_prefix_routes():
    a = _StubAgent("a")
    b = _StubAgent("b")
    with _piped_router() as (router, pipe_in):
        await router.start()
        try:
            async with StdinSteer([a, b], router=router):
                pipe_in.send_text("a: do A\r")
                await _drain()
                pipe_in.send_text("b: do B\r")
                await _drain()
        finally:
            await router.stop()
    assert a.steered == ["do A"]
    assert b.steered == ["do B"]


@pytest.mark.asyncio
async def test_stdin_multi_agent_broadcast():
    a = _StubAgent("a")
    b = _StubAgent("b")
    with _piped_router() as (router, pipe_in):
        await router.start()
        try:
            async with StdinSteer([a, b], router=router):
                pipe_in.send_text("*: stop now\r")
                await _drain()
        finally:
            await router.stop()
    assert a.steered == ["stop now"]
    assert b.steered == ["stop now"]


@pytest.mark.asyncio
async def test_stdin_steer_registers_as_active_router():
    a = _StubAgent("a")
    with _piped_router() as (router, _pipe_in):
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


# ── AgentRuntime.run_agent_stream steering lifecycle ─────────────────────────


@pytest.mark.asyncio
async def test_run_agent_stream_starts_steering_lifecycle(llm, memory):
    """run_agent_stream must enter the steering lifecycle so the factory's
    __aenter__/__aexit__ are called — regression test for the single-agent
    steering bug where the lifecycle wrapper was missing."""
    llm.routes = {
        "react": lambda *_: {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }
    }

    entered = False
    exited = False

    class _LifecycleFactory:
        """Looks like a stdin_steering_factory — has both lifecycle and per-agent call."""

        async def __aenter__(self):
            nonlocal entered
            entered = True
            return self

        async def __aexit__(self, *exc):
            nonlocal exited
            exited = True

        def __call__(self, agent: BaseAgent):
            import contextlib

            @contextlib.asynccontextmanager
            async def _noop():
                yield

            return _noop()

    config = AgentConfig(agent_id="solo", role="r", system_prompt="react", allowed_tools=[])
    agent_reg = AgentRegistry()
    agent_reg.register(config)

    runtime = AgentRuntime(
        agent_registry=agent_reg,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        steering_source_factory=_LifecycleFactory(),
    )

    events = [ev async for ev in runtime.run_agent_stream("solo", "test task")]
    assert any(ev.type == EventType.TASK_DONE for ev in events)
    assert entered, "steering lifecycle __aenter__ was never called"
    assert exited, "steering lifecycle __aexit__ was never called"
