"""Orchestrator + AgentRuntime end-to-end smoke tests."""
from __future__ import annotations

from agents.base import AgentConfig
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from orchestrator.planner import (
    EvalConfig,
    OnFailure,
    Plan,
    TaskResult,
    _parse_plan,
    should_replan,
)
from tests.conftest import EchoTool, ScriptedLLM

# ── Pure helpers ──────────────────────────────────────────────────────────────


def test_should_replan_low_confidence():
    r = TaskResult(
        task_id="t", agent_id="a", answer="x", confidence=0.3, steps=1, success=True
    )
    assert should_replan(r, EvalConfig(confidence_threshold=0.6))


def test_should_replan_failed():
    r = TaskResult(
        task_id="t", agent_id="a", answer="", confidence=1.0, steps=0, success=False
    )
    assert should_replan(r, EvalConfig(confidence_threshold=0.6))


def test_should_not_replan_when_confident_and_successful():
    r = TaskResult(
        task_id="t", agent_id="a", answer="ok", confidence=0.9, steps=1, success=True
    )
    assert not should_replan(r, EvalConfig(confidence_threshold=0.6))


def test_parse_plan_from_dict():
    plan = _parse_plan(
        {
            "tasks": [
                {"id": "t1", "agent_id": "a", "instruction": "do x", "depends_on": []},
                {
                    "id": "t2",
                    "agent_id": "b",
                    "instruction": "do y",
                    "depends_on": ["t1"],
                    "on_failure": "skip",
                },
            ],
            "rationale": "test",
        }
    )
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 2
    assert plan.tasks[1].depends_on == ["t1"]
    assert plan.tasks[1].on_failure == OnFailure.SKIP


def test_parse_plan_from_json_string():
    raw = '{"tasks": [{"id": "t1", "agent_id": "a", "instruction": "x"}], "rationale": ""}'
    plan = _parse_plan(raw)
    assert plan.tasks[0].id == "t1"


# ── End-to-end via AgentRuntime ───────────────────────────────────────────────


def _build_runtime(llm: ScriptedLLM) -> tuple[AgentRuntime, MemoryManager]:
    tools = ToolRegistry().register(EchoTool())
    agents = (
        AgentRegistry()
        .register(
            AgentConfig(
                agent_id="analyst",
                role="analyses",
                system_prompt="You analyse. ReAct format.",
                allowed_tools=["echo"],
                max_steps=3,
            )
        )
        .register(
            AgentConfig(
                agent_id="reporter",
                role="reports",
                system_prompt="You report. ReAct format.",
                allowed_tools=["echo"],
                max_steps=3,
            )
        )
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    runtime = AgentRuntime(
        agent_registry=agents,
        tool_registry=tools,
        memory=memory,
        llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=5.0,
            max_wall_time_seconds=30,
            max_replan_count=1,
            confidence_threshold=0.5,
        ),
    )
    return runtime, memory


def _routes_for_runtime() -> dict:
    """Routes that produce a 2-task DAG, finish on each agent, simple synthesis."""

    def planner(system, messages, kwargs):
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return {
            "tasks": [
                {
                    "id": "t1",
                    "agent_id": "analyst",
                    "instruction": f"analyse: {last_user[:40]}",
                    "depends_on": [],
                    "on_failure": "skip",
                },
                {
                    "id": "t2",
                    "agent_id": "reporter",
                    "instruction": "report findings",
                    "depends_on": ["t1"],
                    "on_failure": "skip",
                },
            ],
            "rationale": "analyse then report",
        }

    def synth(system, messages, kwargs):
        return {
            "answer": "all good",
            "confidence": 0.9,
            "conflicts": [],
            "unknowns": [],
        }

    def extract(system, messages, kwargs):
        return {
            "semantic_facts": {"run:last_goal": "test"},
            "episodic_summary": "completed test run",
            "metadata": {},
            "ttl_seconds": None,
        }

    return {
        # PLAN_SYSTEM contains "decomposes goals into tasks"
        "decomposes goals": planner,
        # SYNTHESIZE_SYSTEM contains "synthesis agent"
        "synthesis agent": synth,
        # MemoryManager extraction prompt routed via system="memory extraction agent"
        "memory extraction": extract,
    }


async def test_runtime_runs_two_task_dag_to_completion():
    llm = ScriptedLLM(routes=_routes_for_runtime())
    runtime, memory = _build_runtime(llm)

    result = await runtime.run("test goal")

    assert result["answer"] == "all good"
    assert result["confidence"] == 0.9
    assert result["replan_count"] == 0
    # both tasks should have run
    task_ids = {tr["task_id"] for tr in result["task_results"]}
    assert task_ids == {"t1", "t2"}
    # episodic memory should have the run-end summary
    assert memory._episodic.count() == 1


# ── Router tests ──────────────────────────────────────────────────────────────


async def test_route_single_agent_skips_llm():
    """With one agent registered, route() returns it without calling the LLM."""
    llm = ScriptedLLM()
    tools = ToolRegistry().register(EchoTool())
    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="analyst", role="analyses",
            system_prompt="You analyse.", allowed_tools=["echo"],
        )
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    runtime = AgentRuntime(agent_registry=agents, tool_registry=tools, memory=memory, llm=llm)

    agent_id, rationale = await runtime.route("investigate something")

    assert agent_id == "analyst"
    assert rationale == "only one agent registered"
    assert llm.calls == []   # no LLM call made


async def test_route_multi_agent_calls_llm():
    """With multiple agents, route() makes one LLM call and returns the chosen id."""

    def router(system, messages, kwargs):
        return {"agent_id": "reporter", "rationale": "goal needs reporting"}

    llm = ScriptedLLM(routes={"routing agent": router})
    runtime, _ = _build_runtime(llm)

    agent_id, rationale = await runtime.route("summarise findings")

    assert agent_id == "reporter"
    assert rationale == "goal needs reporting"
    router_calls = [c for c in llm.calls if "routing agent" in (c["system"] or "").lower()]
    assert len(router_calls) == 1


async def test_route_falls_back_on_unknown_agent_id():
    """If router returns an unregistered agent_id, fall back to first registered."""

    def router(system, messages, kwargs):
        return {"agent_id": "ghost", "rationale": "should not exist"}

    llm = ScriptedLLM(routes={"routing agent": router})
    runtime, _ = _build_runtime(llm)

    agent_id, _ = await runtime.route("anything")
    assert agent_id in runtime._agent_registry.all_ids()


async def test_run_routed_returns_task_done_payload():
    """run_routed ends at TASK_DONE — no synthesis, no DONE event."""
    llm = ScriptedLLM(routes=_routes_for_runtime())
    runtime, _ = _build_runtime(llm)

    result = await runtime.run_routed("analyse something")

    assert "answer" in result
    assert "confidence" in result
    # routed path produces no orchestrator DONE payload fields
    assert "replan_count" not in result
    assert "task_results" not in result


async def test_run_routed_stream_emits_route_event():
    """run_routed_stream yields a ROUTE event before agent events."""
    from harness.events import EventType

    llm = ScriptedLLM(routes=_routes_for_runtime())
    runtime, _ = _build_runtime(llm)

    events = []
    async for event in runtime.run_routed_stream("analyse something"):
        events.append(event)

    types = [e.type for e in events]
    assert types[0] == EventType.ROUTE
    assert EventType.TASK_DONE in types
    # no PLAN or SYNTHESIS — those belong to the orchestrated path
    assert EventType.PLAN not in types
    assert EventType.SYNTHESIS not in types


async def test_runtime_handles_unknown_agent_in_plan():
    """Planner emits a task for an unregistered agent — should not crash, task fails."""
    routes = _routes_for_runtime()

    def planner(system, messages, kwargs):
        return {
            "tasks": [
                {
                    "id": "t1",
                    "agent_id": "ghost",
                    "instruction": "x",
                    "depends_on": [],
                    "on_failure": "skip",
                }
            ],
            "rationale": "",
        }

    routes["decomposes goals"] = planner
    llm = ScriptedLLM(routes=routes)
    runtime, _ = _build_runtime(llm)

    result = await runtime.run("ghost goal")

    assert result["task_results"][0]["success"] is False
    # synthesis still runs
    assert "answer" in result
