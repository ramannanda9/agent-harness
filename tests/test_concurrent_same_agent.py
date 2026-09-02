"""Two tasks on one agent, in one batch, must not corrupt each other.

``BaseAgent`` keeps a run's working memory, task and checkpoint key as
instance attributes, and ``run_stream`` assigns them on entry. The
orchestrator held one instance per ``agent_id`` and drove every ready task
concurrently, so a plan putting two independent tasks on the same agent had
the second run overwrite the first mid-flight: both continued in the *same*
working memory and both returned the second task's answer — reported as
success, with nothing surfacing the loss.

Nothing serialised this. The batch's ``asyncio.Queue`` is a fan-in channel for
events, not a lock, and the drivers are separate ``asyncio.Task``s; the only
lock on the path guards stdout during an approval prompt.
"""

from __future__ import annotations

import asyncio

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from orchestrator.planner import Orchestrator, Plan, Task, validate_plan
from tests.conftest import ScriptedLLM


class SlowTool:
    """Awaits, so the two drivers genuinely interleave rather than each
    running to completion before the other starts."""

    name = "slow"

    async def execute(self, **_kwargs) -> dict:
        await asyncio.sleep(0.02)
        return {"ok": True}


def _guard() -> BudgetGuard:
    return BudgetGuard(GuardrailConfig(max_total_cost_usd=99.0, max_wall_time_seconds=60))


@pytest.fixture
def seen() -> list[str]:
    return []


@pytest.fixture
def llm(seen: list[str]) -> ScriptedLLM:
    """Reports which task's instruction the agent currently believes it has."""

    def react(system, messages, kwargs):
        user = [m for m in messages if m["role"] == "user"]
        current = user[0]["content"] if user else "?"
        seen.append(current[:6])
        if len(seen) <= 2:
            return {"thought": "work", "action": "slow", "args": {}}
        return {
            "thought": "done",
            "action": "finish",
            "answer": f"answer-for-{current[:6]}",
            "confidence": 1.0,
        }

    return ScriptedLLM(routes={"react": react})


@pytest.fixture
def orchestration(llm: ScriptedLLM):
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    registered = BaseAgent(
        config=AgentConfig(
            agent_id="worker",
            role="r",
            system_prompt="ReAct.",
            allowed_tools=["slow"],
            max_steps=6,
        ),
        tools={"slow": SlowTool()},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
    )
    orchestrator = Orchestrator(
        agents={"worker": registered},
        memory=memory,
        llm=llm,
        tracer=Tracer(),
        guard=_guard(),
        run_id="dup-run",
    )
    plan = Plan(
        tasks=[
            Task(id="tA", agent_id="worker", instruction="ALPHA-do-the-first-thing"),
            Task(id="tB", agent_id="worker", instruction="BETA--do-the-second-thing"),
        ],
        rationale="two independent tasks, one agent",
    )
    return orchestrator, registered, plan


async def _run(orchestrator, plan) -> dict:
    results: dict = {}
    async for _ in orchestrator._run_batch("goal", plan.tasks, results, {}):
        pass
    return results


def test_a_plan_may_put_two_tasks_on_one_agent(orchestration):
    """This is a legitimate plan shape — and one the planner produces — so the
    fix has to make it work, not reject it."""
    _orchestrator, _registered, plan = orchestration

    validate_plan(plan, {"worker"})  # must not raise


async def test_concurrent_tasks_on_one_agent_keep_their_own_answers(orchestration):
    orchestrator, _registered, plan = orchestration

    results = await _run(orchestrator, plan)

    assert results["tA"].answer == "answer-for-ALPHA-"[: len(results["tA"].answer)]
    assert results["tA"].answer != results["tB"].answer, (
        "both tasks returned the same answer — the runs shared working memory"
    )
    assert results["tA"].success and results["tB"].success


async def test_each_task_keeps_its_own_instruction_throughout(orchestration, seen):
    """The clearest signature of the corruption: after the first await, the
    agent was thinking about the other task's instruction."""
    orchestrator, _registered, plan = orchestration

    await _run(orchestrator, plan)

    assert sorted(seen) == ["ALPHA-", "ALPHA-", "BETA--", "BETA--"], (
        f"each task should think twice about its own instruction, saw {seen}"
    )


async def test_the_registered_agent_is_left_untouched(orchestration):
    """Tasks drive copies; the instance in the registry is a template and
    should not end a batch holding one task's leftovers."""
    orchestrator, registered, plan = orchestration

    await _run(orchestrator, plan)

    assert registered._task == ""
    assert registered._working_memory is None
    assert registered._ckp_scope is None


# ── The same hazard one level down ───────────────────────────────────────────


def test_cloning_an_agent_also_clones_tools_that_hold_run_state(llm):
    """A delegating tool carries per-invocation state and owns a nested agent,
    so sharing one between two concurrent parents repeats the bug a level
    deeper. Stateless tools stay shared — most are, and some deliberately hold
    a connection."""
    from tools.builtin.subagent import SubAgentTool

    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    inner = BaseAgent(
        config=AgentConfig(agent_id="inner", role="r", system_prompt="p", allowed_tools=[]),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
    )
    shared_stateless = SlowTool()
    outer = BaseAgent(
        config=AgentConfig(
            agent_id="outer", role="r", system_prompt="p", allowed_tools=["slow", "delegate_inner"]
        ),
        tools={"slow": shared_stateless, "delegate_inner": SubAgentTool(inner)},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
    )

    a = outer.clone_for_run(ckp_scope="t1")
    b = outer.clone_for_run(ckp_scope="t2")

    assert a._tools["delegate_inner"] is not b._tools["delegate_inner"]
    assert a._tools["delegate_inner"]._agent is not b._tools["delegate_inner"]._agent
    assert a._tools["slow"] is shared_stateless, "stateless tools stay shared"
    assert b._tools["slow"] is shared_stateless


def test_a_clone_shares_the_budget_guard(llm):
    """One budget covers the whole run — cloning must not hand each task its
    own allowance."""
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    agent = BaseAgent(
        config=AgentConfig(agent_id="w", role="r", system_prompt="p", allowed_tools=[]),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
    )

    clone = agent.clone_for_run(ckp_scope="t1")

    assert clone._guard is agent._guard
    assert clone._llm is agent._llm
    assert clone._memory is agent._memory
    assert clone.config is agent.config


def test_a_clone_gets_its_own_steering_queue(llm):
    """Steering is bound per run: run_stream calls the source factory with the
    instance being driven, so a copy must not share the template's queue."""
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    agent = BaseAgent(
        config=AgentConfig(agent_id="w", role="r", system_prompt="p", allowed_tools=[]),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
    )

    clone = agent.clone_for_run()
    clone.steer("only for the clone")

    assert clone._steering.qsize() == 1
    assert agent._steering.qsize() == 0
