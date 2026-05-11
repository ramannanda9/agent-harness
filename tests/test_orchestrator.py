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
