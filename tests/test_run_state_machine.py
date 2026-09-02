"""Tests for the explicit ReAct state machine and what it makes resumable.

The loop used to keep its position in the Python stack, so a crash could only
be recovered at one granularity: a single pending tool call, replayed by a
second implementation of the loop. These cover the cases that granularity
could not express — a batch interrupted partway, a budget that must survive a
resume, and a run that failed for a reason resume can actually act on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.hitl import ApprovalResponse
from harness.runstate import ActionStatus, ErrorKind, Phase, RunState, load_state
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tests.conftest import ScriptedLLM


class RecordingTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    async def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return f"{self.name} ran"


class Store:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.writes: list[str] = []

    async def write(self, key: str, data: dict) -> None:
        self.data[key] = data
        self.writes.append(key)

    async def read(self, key: str) -> dict | None:
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def memory(llm: ScriptedLLM) -> MemoryManager:
    return MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )


def _agent(llm, memory, tools, store=None, *, hitl=(), max_steps=6, guard=None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="sm",
            role="r",
            system_prompt="ReAct.",
            allowed_tools=list(tools),
            hitl_tools=list(hitl),
            max_steps=max_steps,
        ),
        tools=tools,
        memory=memory,
        tracer=Tracer(),
        guard=guard
        or BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60)),
        llm=llm,
        checkpoint_store=store,
    )


def _batch_then_finish(tools: list[str]):
    """An LLM that asks for one parallel batch, then finishes."""
    seen = {"n": 0}

    def react(system, messages, kwargs):
        seen["n"] += 1
        if seen["n"] == 1:
            return {
                "thought": "fan out",
                "actions": [{"tool": t, "args": {"i": i}} for i, t in enumerate(tools)],
            }
        return {"thought": "done", "action": "finish", "answer": "batch ok", "confidence": 1.0}

    return {"react": react}


# ── A batch interrupted partway through approval ─────────────────────────────


async def test_crash_mid_batch_keeps_earlier_approvals_and_reprompts_only_the_rest(
    llm, memory, store
):
    """The failure this replaces: the old checkpoint held a single 'pending'
    tool, overwritten by each gate in turn. A crash partway through a batch
    resumed by replaying only the last one gated and then advancing the step,
    silently dropping every other action the model had asked for."""
    tools = {n: RecordingTool(n) for n in ("one", "two", "three")}
    llm.routes = _batch_then_finish(list(tools))
    agent = _agent(llm, memory, tools, store, hitl=list(tools))

    prompts: list[str] = []

    async def approve_then_die(request, guard):
        prompts.append(request.tool)
        if len(prompts) == 2:
            raise RuntimeError("process died at the prompt")
        return ApprovalResponse(approval_id=request.approval_id, approved=True)

    with patch("harness.hitl.request_approval", approve_then_die):
        async for _ in agent.run_stream(task="t", run_id="r1"):
            pass

    assert prompts == ["one", "two"]
    state = load_state(store.data["r1:sm"])
    assert [a.status for a in state.actions] == [
        ActionStatus.APPROVED,  # decided before the crash — must not be re-asked
        ActionStatus.PENDING,  # died at this prompt
        ActionStatus.PENDING,  # never reached
    ]
    assert [t.calls for t in tools.values()] == [[], [], []], "nothing ran yet"

    # Resume: only the undecided actions are re-prompted, and the whole batch runs.
    resumed_prompts: list[str] = []

    async def approve_all(request, guard):
        resumed_prompts.append(request.tool)
        return ApprovalResponse(approval_id=request.approval_id, approved=True)

    fresh_tools = {n: RecordingTool(n) for n in tools}
    resumed = _agent(llm, memory, fresh_tools, store, hitl=list(tools))
    with patch("harness.hitl.request_approval", approve_all):
        async for _ in resumed._resume_stream(state):
            pass

    assert resumed_prompts == ["two", "three"], "an already-approved action was re-asked"
    assert all(t.calls for t in fresh_tools.values()), "the whole batch should have run"


async def test_rejected_action_in_a_batch_is_reported_back_to_the_model(llm, memory, store):
    """The parallel path used to drop rejected calls from the batch silently
    while still recording the full response as the assistant message — telling
    the model it made three calls and showing it two results."""
    tools = {n: RecordingTool(n) for n in ("keep", "refuse")}
    llm.routes = _batch_then_finish(list(tools))
    agent = _agent(llm, memory, tools, store, hitl=["refuse"])

    async def reject_refuse(request, guard):
        return ApprovalResponse(approval_id=request.approval_id, approved=request.tool != "refuse")

    observations: list[str] = []
    with patch("harness.hitl.request_approval", reject_refuse):
        async for event in agent.run_stream(task="t", run_id="r1"):
            if event.type == EventType.OBSERVATION:
                observations.append(f"{event.payload['tool']}:{event.payload['observation']}")

    assert tools["refuse"].calls == []
    assert any(o.startswith("refuse:") and "rejected by human" in o for o in observations)
    assert any(o.startswith("keep:") for o in observations)


# ── Budget across a resume ───────────────────────────────────────────────────


async def test_spending_carries_across_a_resume(llm, memory, store):
    tools = {"one": RecordingTool("one")}
    llm.routes = {"react": lambda *a: {"thought": "x", "action": "one", "args": {}}}
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60))
    guard.add_cost(4.0)
    agent = _agent(llm, memory, tools, store, max_steps=1, guard=guard)

    async for _ in agent.run_stream(task="t", run_id="r1"):
        pass

    state = load_state(store.data["r1:sm"]) if "r1:sm" in store.data else None
    if state is None:  # max_steps=1 with no gated tool writes no checkpoint
        state = RunState(
            run_id="r1",
            agent_id="sm",
            task="t",
            budget=guard.snapshot(),
            memory=agent._working_memory.to_dict(),
        )

    resumed = _agent(llm, memory, tools, store, max_steps=1)
    async for _ in resumed._resume_stream(state):
        pass

    assert resumed._guard.cost >= 4.0, "a resumed run must not start the budget over"


async def test_a_budget_failure_does_not_restore_the_budget_that_tripped(llm, memory, store):
    """Otherwise 'resume with a larger budget' is an infinite loop: the guard
    is restored, re-raises at the same point, and nothing advances."""
    tools = {"one": RecordingTool("one")}
    llm.routes = {"react": lambda *a: {"thought": "x", "action": "one", "args": {}}}
    state = RunState(
        run_id="r1",
        agent_id="sm",
        task="t",
        phase=Phase.FAILED,
        error="Cost budget exceeded",
        error_kind=ErrorKind.BUDGET,
        budget={
            "cost_usd": 99.0,
            "elapsed_seconds": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "breakdown": {},
        },
    )

    resumed = _agent(llm, memory, tools, store)
    async for _ in resumed._resume_stream(state):
        pass

    assert resumed._guard.cost == 0.0, "the tripped budget must not be restored"


# ── Failure kinds a resume can act on ────────────────────────────────────────


async def test_a_max_steps_failure_resumes_by_thinking_again(llm, memory, store):
    """FAILED is not a black hole. max_steps is the most common resume there
    is, and it means 'try again with more room', not 'this run is over'."""
    llm.routes = {
        "react": lambda *a: {
            "thought": "done",
            "action": "finish",
            "answer": "second chance",
            "confidence": 1.0,
        }
    }
    state = RunState(
        run_id="r1",
        agent_id="sm",
        task="t",
        step=3,
        phase=Phase.FAILED,
        error="Max steps (3) reached",
        error_kind=ErrorKind.MAX_STEPS,
    )

    resumed = _agent(llm, memory, {}, store, max_steps=10)
    result: dict = {}
    async for event in resumed._resume_stream(state):
        if event.type == EventType.TASK_DONE:
            result = event.payload

    assert result["answer"] == "second chance"


async def test_max_steps_records_why_the_run_failed(llm, memory, store):
    llm.routes = {"react": lambda *a: {"thought": "loop", "action": "one", "args": {}}}
    agent = _agent(llm, memory, {"one": RecordingTool("one")}, store, max_steps=2)

    state = RunState(run_id="r1", agent_id="sm", task="t")
    agent._ckp_id = "r1:sm"
    agent._task = "t"
    from memory.working import WorkingMemory

    agent._working_memory = WorkingMemory(llm=llm, max_tokens=8000)
    await agent._working_memory.append("system", "ReAct.", pinned=True)

    async for _ in agent._drive(state):
        pass

    assert state.phase is Phase.FAILED
    assert state.error_kind is ErrorKind.MAX_STEPS


# ── The resumed stream announces itself ──────────────────────────────────────


async def test_a_resumed_run_opens_with_a_resumed_event(llm, memory, store):
    llm.routes = {
        "react": lambda *a: {"thought": "d", "action": "finish", "answer": "ok", "confidence": 1.0}
    }
    state = RunState(run_id="r1", agent_id="sm", task="t", step=4)

    agent = _agent(llm, memory, {}, store)
    events = [e async for e in agent._resume_stream(state)]

    assert events[0].type == EventType.RESUMED
    assert events[0].payload["step"] == 4
    assert events[0].payload["phase"] == Phase.THINK.value


# ── Delegation depth ─────────────────────────────────────────────────────────


async def test_a_refused_delegation_is_never_put_to_a_human(llm, memory, store):
    """The depth check used to run inside the gate, after the prompt. Asking
    someone to approve a call that cannot run wastes their attention."""
    from tools.builtin.subagent import SubAgentTool

    class FakeSub(SubAgentTool):
        def __init__(self) -> None:
            self.name = "delegate"
            self._agent = _agent(llm, memory, {})
            self._invoking_agent_id = ""

        async def execute_stream(self, **kwargs):
            yield "should not run"

    llm.routes = {"react": lambda *a: {"thought": "x", "action": "delegate", "args": {}}}
    agent = _agent(llm, memory, {"delegate": FakeSub()}, store, hitl=["delegate"], max_steps=1)
    agent._subagent_depth = 99
    agent.config.max_subagent_depth = 1
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=True))

    observations: list[str] = []
    with patch("harness.hitl.request_approval", approval):
        async for event in agent.run_stream(task="t", run_id="r1"):
            if event.type == EventType.OBSERVATION:
                observations.append(str(event.payload["observation"]))

    assert approval.await_count == 0, "a refused delegation must not prompt"
    assert any("Refused to delegate" in o for o in observations)
