"""
Tests for the unified streaming/blocking execution model.

Verifies that BaseAgent.run_stream() and Orchestrator.run_stream() yield the
expected BusEvent sequence, and that the blocking run() drains to the same
result the stream's DONE event carries.
"""
from __future__ import annotations

from agents.base import AgentConfig
from harness.events import BusEvent, EventType
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tests.conftest import EchoTool, ScriptedLLM

# ── BaseAgent.run_stream event sequence ───────────────────────────────────────


async def test_agent_run_stream_finish_yields_task_done(agent_factory):
    """Finish on first step → just one TASK_DONE event (no THOUGHT/ACTION pairs)."""
    cfg = AgentConfig(
        agent_id="a", role="r", system_prompt="finish.",
        allowed_tools=[], working_memory_max_tokens=2000,
    )
    agent = agent_factory(cfg)
    events = [e async for e in agent.run_stream("hi")]

    types = [e.type for e in events]
    # The default ScriptedLLM finishes immediately; we expect THOUGHT then TASK_DONE.
    assert EventType.THOUGHT in types
    assert types[-1] == EventType.TASK_DONE
    assert events[-1].payload["answer"].startswith("done:")


async def test_agent_run_stream_tool_call_yields_action_and_observation(
    agent_factory, llm: ScriptedLLM,
):
    """A tool-using step should yield THOUGHT → ACTION → OBSERVATION → ... → TASK_DONE."""
    step = {"n": 0}

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {"thought": "use tool", "action": "echo", "args": {"message": "hi"}}
        return {"action": "finish", "answer": "done", "confidence": 0.9, "thought": ""}

    llm.routes = {"react": react}
    cfg = AgentConfig(
        agent_id="a", role="r", system_prompt="ReAct format.",
        allowed_tools=["echo"],
    )
    agent = agent_factory(cfg, tools={"echo": EchoTool()})

    events = [e async for e in agent.run_stream("do it")]
    types = [e.type for e in events]
    assert types.count(EventType.THOUGHT) == 2
    assert types.count(EventType.ACTION) == 1
    assert types.count(EventType.OBSERVATION) == 1
    assert types[-1] == EventType.TASK_DONE

    # ACTION event carries the tool name + args
    action_evt = next(e for e in events if e.type == EventType.ACTION)
    assert action_evt.payload["tool"] == "echo"
    assert action_evt.payload["args"] == {"message": "hi"}


async def test_agent_run_is_drain_of_run_stream(agent_factory):
    """run() and the TASK_DONE payload from run_stream() must agree."""
    cfg = AgentConfig(
        agent_id="a", role="r", system_prompt="finish.", allowed_tools=[],
    )
    agent = agent_factory(cfg)

    stream_result: dict = {}
    async for event in agent.run_stream("hi"):
        if event.type == EventType.TASK_DONE:
            stream_result = event.payload

    blocking_result = await agent.run("hi")

    assert stream_result["answer"] == blocking_result["answer"]
    assert stream_result["confidence"] == blocking_result["confidence"]


# ── Token streaming when the LLM supports it ────────────────────────────────


async def test_agent_forwards_token_events_when_llm_streams(agent_factory):
    """If the LLM exposes stream_complete, BaseAgent yields TOKEN events."""

    class StreamingLLM:
        async def complete(self, system, messages, **kwargs):
            return {"action": "finish", "answer": "x", "confidence": 1.0}

        async def stream_complete(self, system, messages):
            for tok in ['{"action":"', "finish", '","answer":"', "ok", '","confidence":1.0}']:
                yield tok

    cfg = AgentConfig(
        agent_id="a", role="r", system_prompt="ReAct.", allowed_tools=[],
    )
    agent = agent_factory(cfg)
    agent._llm = StreamingLLM()

    events = [e async for e in agent.run_stream("hi")]
    tokens = [e.token for e in events if e.type == EventType.TOKEN]
    assert tokens == ['{"action":"', "finish", '","answer":"', "ok", '","confidence":1.0}']
    assert events[-1].type == EventType.TASK_DONE
    assert events[-1].payload["answer"] == "ok"


# ── Orchestrator.run_stream event sequence ──────────────────────────────────


def _orchestrator_routes():
    """Routes for a 2-task DAG that successfully completes."""

    def planner(system, messages, kwargs):
        return {
            "tasks": [
                {"id": "t1", "agent_id": "analyst", "instruction": "do x",
                 "depends_on": [], "on_failure": "skip"},
                {"id": "t2", "agent_id": "reporter", "instruction": "do y",
                 "depends_on": ["t1"], "on_failure": "skip"},
            ],
            "rationale": "two tasks",
        }

    def synth(system, messages, kwargs):
        return {"answer": "all good", "confidence": 0.9, "conflicts": [], "unknowns": []}

    def extract(system, messages, kwargs):
        return {
            "semantic_facts": {}, "episodic_summary": "ok",
            "metadata": {}, "ttl_seconds": None,
        }

    return {
        "decomposes goals": planner,
        "synthesis agent": synth,
        "memory extraction": extract,
    }


def _build_runtime(llm):
    tools = ToolRegistry().register(EchoTool())
    agents = (
        AgentRegistry()
        .register(AgentConfig(
            agent_id="analyst", role="r", system_prompt="ReAct.",
            allowed_tools=["echo"], max_steps=2,
        ))
        .register(AgentConfig(
            agent_id="reporter", role="r", system_prompt="ReAct.",
            allowed_tools=["echo"], max_steps=2,
        ))
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    return AgentRuntime(
        agent_registry=agents, tool_registry=tools, memory=memory, llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=5.0, max_wall_time_seconds=30,
            max_replan_count=1, confidence_threshold=0.5,
        ),
    )


async def test_runtime_run_stream_yields_full_lifecycle():
    llm = ScriptedLLM(routes=_orchestrator_routes())
    runtime = _build_runtime(llm)

    events: list[BusEvent] = [e async for e in runtime.run_stream("the goal")]
    types = [e.type for e in events]

    # Lifecycle: PLAN → (per-task agent events) → TASK_DONE (x2) → SYNTHESIS → DONE
    assert types[0] == EventType.PLAN
    assert types.count(EventType.TASK_DONE) == 2
    assert EventType.SYNTHESIS in types
    assert types[-1] == EventType.DONE

    # PLAN payload contains the parsed plan dict
    plan_evt = events[0]
    assert "tasks" in plan_evt.payload["plan"]
    assert len(plan_evt.payload["plan"]["tasks"]) == 2

    # DONE payload matches what runtime.run() would return
    done = events[-1].payload
    assert done["answer"] == "all good"
    assert done["confidence"] == 0.9


async def test_runtime_run_is_drain_of_run_stream():
    """runtime.run() must agree with the DONE event from runtime.run_stream()."""
    routes = _orchestrator_routes()

    # Two independent runtimes (memory state isolation)
    stream_result: dict = {}
    llm1 = ScriptedLLM(routes=routes)
    runtime1 = _build_runtime(llm1)
    async for event in runtime1.run_stream("goal"):
        if event.type == EventType.DONE:
            stream_result = event.payload

    llm2 = ScriptedLLM(routes=routes)
    runtime2 = _build_runtime(llm2)
    blocking_result = await runtime2.run("goal")

    assert stream_result["answer"] == blocking_result["answer"]
    assert stream_result["confidence"] == blocking_result["confidence"]
    # blocking adds trace+budget
    assert "trace" in blocking_result
    assert "trace" not in stream_result
