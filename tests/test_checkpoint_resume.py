"""
Tests for per-agent checkpoint namespacing and crash-resume behavior.

Agent resume covers:
  - _ckp_id is f"{run_id}:{agent_id}" (not bare run_id)
  - Two orchestrated agents sharing a run_id write to distinct checkpoint keys
  - Step checkpoints are written at the configured interval
  - Checkpoint is deleted on clean exit
  - Agent resumes from the saved step with WorkingMemory intact
  - Pending HITL approval is re-prompted on resume, then tool executes
  - AgentRuntime.resume_agent reads via ckp_id and passes outer run_id correctly

Orchestrator resume covers:
  - Orchestrator checkpoint written after each batch (bare run_id key)
  - Orchestrator checkpoint deleted on clean DONE
  - resume_orchestration skips completed tasks and runs only pending ones
  - resume_orchestration re-runs tasks not yet started
  - runtime.resume() auto-detects checkpoint type (plan vs agent_id field)
  - _resume_key on agents points to outer run_id in orchestrated context
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.hitl import ApprovalResponse
from harness.runtime import (
    AgentRegistry,
    AgentRuntime,
    BudgetGuard,
    GuardrailConfig,
    ToolRegistry,
    Tracer,
)
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from memory.working import WorkingMemory
from tests.conftest import EchoTool, ScriptedLLM

# ── Fixtures ──────────────────────────────────────────────────────────────────


class InMemoryCheckpointStore:
    """Simple in-memory checkpoint store that records every call for assertions."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.writes: list[str] = []
        self.deletes: list[str] = []

    async def write(self, key: str, data: dict) -> None:
        self.data[key] = data
        self.writes.append(key)

    async def read(self, key: str) -> dict | None:
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        self.deletes.append(key)
        self.data.pop(key, None)


@pytest.fixture
def ckp_store() -> InMemoryCheckpointStore:
    return InMemoryCheckpointStore()


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


def _make_agent(
    config: AgentConfig,
    llm: ScriptedLLM,
    memory: MemoryManager,
    checkpoint_store: InMemoryCheckpointStore | None = None,
    tools: dict | None = None,
) -> BaseAgent:
    return BaseAgent(
        config=config,
        tools=tools or {},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60)),
        llm=llm,
        checkpoint_store=checkpoint_store,
    )


async def _minimal_wm(llm: ScriptedLLM) -> WorkingMemory:
    """A WorkingMemory with a system + user message — suitable for resume tests."""
    wm = WorkingMemory(llm=llm, max_tokens=8000)
    await wm.append("system", "You are a ReAct agent.", pinned=True)
    await wm.append("user", "the task")
    return wm


# ── Checkpoint key namespacing ─────────────────────────────────────────────────


async def test_ckp_id_set_after_run(llm: ScriptedLLM, memory: MemoryManager):
    """After run_stream, _ckp_id is f'{run_id}:{agent_id}'."""
    config = AgentConfig(
        agent_id="researcher",
        role="r",
        system_prompt="finish.",
        allowed_tools=[],
    )
    agent = _make_agent(config, llm, memory)

    run_id = "run-abc123"
    await agent.run("task", run_id=run_id)

    assert agent._ckp_id == f"{run_id}:researcher"


async def test_checkpoint_store_never_receives_bare_run_id(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """The store key is always namespaced — never the bare run_id."""
    config = AgentConfig(
        agent_id="writer",
        role="w",
        system_prompt="finish.",
        allowed_tools=[],
        checkpoint_every=1,
    )
    agent = _make_agent(config, llm, memory, ckp_store)

    run_id = "bare-run"
    await agent.run("task", run_id=run_id)

    assert run_id not in ckp_store.writes
    assert f"{run_id}:writer" in ckp_store.writes


async def test_two_agents_same_run_id_write_distinct_keys(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """Orchestrated agents sharing a run_id must not overwrite each other's checkpoints."""

    def _cfg(agent_id: str) -> AgentConfig:
        return AgentConfig(
            agent_id=agent_id,
            role="r",
            system_prompt="finish.",
            allowed_tools=[],
            checkpoint_every=1,
        )

    agent_a = _make_agent(_cfg("agent-a"), llm, memory, ckp_store)
    agent_b = _make_agent(_cfg("agent-b"), llm, memory, ckp_store)

    run_id = "shared-run"
    await asyncio.gather(
        agent_a.run("task a", run_id=run_id),
        agent_b.run("task b", run_id=run_id),
    )

    assert f"{run_id}:agent-a" in ckp_store.writes
    assert f"{run_id}:agent-b" in ckp_store.writes
    assert run_id not in ckp_store.writes


# ── Step checkpointing ─────────────────────────────────────────────────────────


async def test_step_checkpoint_written_at_configured_interval(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """With checkpoint_every=2, writes happen at steps 0, 2, 4 (3 total)."""
    call_count = {"n": 0}

    def react(system, messages, kwargs):
        call_count["n"] += 1
        if call_count["n"] < 5:
            return {"thought": "keep going", "action": "echo", "args": {"message": "x"}}
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id="stepper",
        role="r",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        max_steps=10,
        checkpoint_every=2,
    )
    agent = _make_agent(config, llm, memory, ckp_store, tools={"echo": EchoTool()})
    await agent.run("step task")

    ckp_id = agent._ckp_id
    # Steps 0, 2, 4 → 3 writes
    assert ckp_store.writes.count(ckp_id) == 3
    # Step fields in written data should be 0, 2, 4
    # (data is overwritten each time; check the delete happened on clean exit)
    assert ckp_id in ckp_store.deletes


async def test_checkpoint_deleted_on_clean_exit(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """Checkpoint file is removed after the agent finishes successfully."""
    config = AgentConfig(
        agent_id="finisher",
        role="r",
        system_prompt="finish.",
        allowed_tools=[],
        checkpoint_every=1,
    )
    agent = _make_agent(config, llm, memory, ckp_store)
    await agent.run("task")

    ckp_id = agent._ckp_id
    assert ckp_id in ckp_store.deletes
    assert ckp_id not in ckp_store.data


# ── Agent resume ───────────────────────────────────────────────────────────────


async def test_resume_continues_from_saved_step(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """
    _resume_stream starts the ReAct loop at start_step, not 0.
    The pre-loaded WorkingMemory is visible to the LLM on the first call.
    """
    run_id = "resume-run"
    agent_id = "researcher"
    ckp_id = f"{run_id}:{agent_id}"

    wm = await _minimal_wm(llm)
    # Simulate two steps already completed in WM
    await wm.append("assistant", '{"thought":"step0","action":"echo","args":{"message":"x"}}')
    await wm.append("user", "Observation: step0 done")
    await wm.append("assistant", '{"thought":"step1","action":"echo","args":{"message":"y"}}')
    await wm.append("user", "Observation: step1 done")

    ckp_store.data[ckp_id] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "task": "the task",
        "step": 2,
        "memory": wm.to_dict(),
    }

    call_count = {"n": 0}
    seen_messages: list[list] = []

    def react(system, messages, kwargs):
        call_count["n"] += 1
        seen_messages.append(messages[:])
        return {"thought": "done", "action": "finish", "answer": "resumed ok", "confidence": 0.9}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id=agent_id,
        role="r",
        system_prompt="ReAct.",
        allowed_tools=[],
        max_steps=10,
    )
    agent = _make_agent(config, llm, memory, ckp_store)
    agent._working_memory = WorkingMemory.from_dict(wm.to_dict(), llm=llm)
    agent._task = "the task"

    result: dict = {}
    async for event in agent._resume_stream(run_id=run_id, start_step=2):
        if event.type == EventType.TASK_DONE:
            result = event.payload

    assert result["answer"] == "resumed ok"
    # Only one LLM call — we resumed at step 2, not step 0
    assert call_count["n"] == 1
    # _ckp_id set correctly in _resume_stream
    assert agent._ckp_id == ckp_id
    # The LLM received the pre-loaded WM messages (step history is visible)
    first_call_messages = seen_messages[0]
    combined = " ".join(str(m.get("content", "")) for m in first_call_messages)
    assert "step0 done" in combined
    assert "step1 done" in combined


async def test_resume_with_pending_hitl_replays_and_executes_tool(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """
    When a checkpoint has a 'pending' HITL approval, _resume_stream re-prompts
    the human and then executes the tool before continuing the loop.
    """
    run_id = "hitl-run"
    agent_id = "hitl-agent"
    ckp_id = f"{run_id}:{agent_id}"

    wm = await _minimal_wm(llm)
    pending = {
        "approval_id": "appr-abc",
        "tool": "echo",
        "args": {"message": "pending call"},
        "step": 1,
        "llm_response": {
            "thought": "call echo",
            "action": "echo",
            "args": {"message": "pending call"},
        },
    }
    ckp_store.data[ckp_id] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "task": "hitl task",
        "step": 1,
        "memory": wm.to_dict(),
        "pending": pending,
    }

    def react(system, messages, kwargs):
        return {"thought": "done", "action": "finish", "answer": "hitl ok", "confidence": 0.9}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id=agent_id,
        role="r",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        hitl_tools=["echo"],
        max_steps=5,
    )
    agent = _make_agent(config, llm, memory, ckp_store, tools={"echo": EchoTool()})
    agent._working_memory = WorkingMemory.from_dict(wm.to_dict(), llm=llm)
    agent._task = "hitl task"

    approval = ApprovalResponse(approval_id="appr-abc", approved=True)

    observation_events: list = []
    result: dict = {}

    with patch("harness.hitl.request_approval", AsyncMock(return_value=approval)):
        async for event in agent._resume_stream(
            run_id=run_id,
            start_step=2,
            pending=pending,
        ):
            if event.type == EventType.OBSERVATION:
                observation_events.append(event)
            elif event.type == EventType.TASK_DONE:
                result = event.payload

    assert result["answer"] == "hitl ok"
    # The pending echo tool should have produced an observation
    assert len(observation_events) >= 1
    assert observation_events[0].payload["tool"] == "echo"
    assert agent._ckp_id == ckp_id


async def test_resume_with_rejected_hitl_skips_tool(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """When a pending HITL approval is rejected, the tool is skipped and loop continues."""
    run_id = "reject-run"
    agent_id = "reject-agent"
    ckp_id = f"{run_id}:{agent_id}"

    wm = await _minimal_wm(llm)
    pending = {
        "approval_id": "appr-rej",
        "tool": "echo",
        "args": {"message": "should not run"},
        "step": 0,
        "llm_response": {
            "thought": "call echo",
            "action": "echo",
            "args": {"message": "should not run"},
        },
    }
    ckp_store.data[ckp_id] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "task": "reject task",
        "step": 0,
        "memory": wm.to_dict(),
        "pending": pending,
    }

    echo = EchoTool()
    execute_calls: list = []
    original_execute = echo.execute

    async def tracked_execute(**kwargs):
        execute_calls.append(kwargs)
        return await original_execute(**kwargs)

    echo.execute = tracked_execute

    def react(system, messages, kwargs):
        return {"thought": "done", "action": "finish", "answer": "rejected ok", "confidence": 0.9}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id=agent_id,
        role="r",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        hitl_tools=["echo"],
        max_steps=5,
    )
    agent = _make_agent(config, llm, memory, ckp_store, tools={"echo": echo})
    agent._working_memory = WorkingMemory.from_dict(wm.to_dict(), llm=llm)
    agent._task = "reject task"

    rejection = ApprovalResponse(approval_id="appr-rej", approved=False)

    result: dict = {}
    with patch("harness.hitl.request_approval", AsyncMock(return_value=rejection)):
        async for event in agent._resume_stream(run_id=run_id, start_step=1, pending=pending):
            if event.type == EventType.TASK_DONE:
                result = event.payload

    assert result["answer"] == "rejected ok"
    assert execute_calls == []  # tool was never executed


# ── AgentRuntime.resume_agent ─────────────────────────────────────────────────


async def test_runtime_resume_agent_reads_ckp_id_and_returns_result(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """
    resume_agent(ckp_id) reads the checkpoint, extracts the outer run_id,
    reconstructs the agent, and returns the TASK_DONE payload.
    """
    run_id = "rt-outer-run"
    agent_id = "rt-agent"
    ckp_id = f"{run_id}:{agent_id}"

    wm = await _minimal_wm(llm)
    ckp_store.data[ckp_id] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "task": "the task",
        "step": 0,
        "memory": wm.to_dict(),
    }

    def react(system, messages, kwargs):
        return {
            "thought": "done",
            "action": "finish",
            "answer": "runtime resume ok",
            "confidence": 0.85,
        }

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id=agent_id,
        role="r",
        system_prompt="ReAct.",
        allowed_tools=[],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(config)

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        checkpoint_store=ckp_store,
    )

    result = await runtime.resume_agent(ckp_id)

    assert result["answer"] == "runtime resume ok"
    assert result["confidence"] == 0.85


async def test_runtime_resume_agent_raises_on_missing_checkpoint(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """resume_agent raises KeyError when the ckp_id is not in the store."""
    config = AgentConfig(
        agent_id="ghost",
        role="r",
        system_prompt="ReAct.",
        allowed_tools=[],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(config)

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        checkpoint_store=ckp_store,
    )

    with pytest.raises(KeyError, match="ghost"):
        await runtime.resume_agent("nonexistent-run:ghost")


async def test_runtime_resume_agent_recomputes_ckp_id_correctly(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """
    _resume_stream inside resume_agent must recompute _ckp_id as
    f"{outer_run_id}:{agent_id}" so subsequent checkpoint ops use the same key.
    """
    run_id = "outer-run"
    agent_id = "agent-x"
    ckp_id = f"{run_id}:{agent_id}"

    wm = await _minimal_wm(llm)
    ckp_store.data[ckp_id] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "task": "task",
        "step": 0,
        "memory": wm.to_dict(),
    }

    def react(system, messages, kwargs):
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id=agent_id,
        role="r",
        system_prompt="ReAct.",
        allowed_tools=[],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(config)

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        checkpoint_store=ckp_store,
    )

    await runtime.resume_agent(ckp_id)

    # The agent cleaned up using the correct namespaced key, not the outer run_id
    assert ckp_id in ckp_store.deletes
    assert run_id not in ckp_store.deletes


# ── Orchestrator checkpoint writing ───────────────────────────────────────────


def _make_runtime(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
    agent_ids: list[str] | None = None,
) -> tuple[AgentRuntime, MemoryManager]:
    """Build a two-agent runtime wired to an in-memory checkpoint store."""
    from harness.runtime import GuardrailConfig

    ids = agent_ids or ["analyst", "reporter"]
    agent_registry = AgentRegistry()
    for aid in ids:
        agent_registry.register(
            AgentConfig(
                agent_id=aid,
                role=f"{aid} role",
                system_prompt=f"You are {aid}. ReAct format.",
                allowed_tools=[],
                max_steps=3,
            )
        )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    runtime = AgentRuntime(
        agent_registry=agent_registry,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=5.0,
            max_wall_time_seconds=30,
        ),
        checkpoint_store=ckp_store,
    )
    return runtime, memory


def _orch_routes(agent_ids: list[str] | None = None) -> dict:
    """LLM routes for a simple linear plan: agent_ids[0] → agent_ids[1]."""
    ids = agent_ids or ["analyst", "reporter"]

    def planner(system, messages, kwargs):
        return {
            "tasks": [
                {
                    "id": "t1",
                    "agent_id": ids[0],
                    "instruction": "step 1",
                    "depends_on": [],
                    "on_failure": "skip",
                },
                {
                    "id": "t2",
                    "agent_id": ids[1],
                    "instruction": "step 2",
                    "depends_on": ["t1"],
                    "on_failure": "skip",
                },
            ],
            "rationale": "linear plan",
        }

    def synth(system, messages, kwargs):
        return {"answer": "synthesised", "confidence": 0.9, "conflicts": [], "unknowns": []}

    def extract(system, messages, kwargs):
        return {"semantic_facts": {}, "episodic_summary": "ok", "metadata": {}, "ttl_seconds": None}

    return {
        "decomposes goals": planner,
        "synthesis agent": synth,
        "memory extraction": extract,
    }


async def test_orchestrator_writes_checkpoint_after_each_batch(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """After each batch completes, the orchestrator checkpoint is written to the store."""
    llm.routes = _orch_routes()
    runtime, _ = _make_runtime(llm, ckp_store)

    result = await runtime.run("test goal")

    assert result["answer"] == "synthesised"
    # Orchestrator writes checkpoint at: initial (empty) + after t1 batch + after t2 batch
    # All writes use the bare run_id (no agent suffix)
    orch_writes = [k for k in ckp_store.writes if ":" not in k]
    assert len(orch_writes) >= 2  # at least initial + after first completed batch


async def test_orchestrator_checkpoint_has_plan_field(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """The orchestrator checkpoint schema includes 'plan', 'completed', 'goal', 'replan_count'."""
    llm.routes = _orch_routes()
    runtime, _ = _make_runtime(llm, ckp_store)

    # Capture the checkpoint after t1 completes by checking what was written
    written_payloads: list[dict] = []
    original_write = ckp_store.write

    async def capturing_write(key: str, data: dict) -> None:
        await original_write(key, data)
        if "plan" in data:
            written_payloads.append(data)

    ckp_store.write = capturing_write

    await runtime.run("test goal")

    assert written_payloads, "no orchestrator checkpoint was written"
    ckp = written_payloads[0]
    assert "plan" in ckp
    assert "completed" in ckp
    assert "goal" in ckp
    assert "replan_count" in ckp
    assert "run_id" in ckp


async def test_orchestrator_checkpoint_deleted_on_done(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """Orchestrator checkpoint is removed after successful DONE."""
    llm.routes = _orch_routes()
    runtime, _ = _make_runtime(llm, ckp_store)

    await runtime.run("test goal")

    orch_writes = [k for k in ckp_store.writes if ":" not in k]
    assert orch_writes, "no orchestrator checkpoint was written"
    run_id = orch_writes[0]
    # Deleted on clean exit
    assert run_id in ckp_store.deletes
    assert run_id not in ckp_store.data


# ── Orchestrator resume ────────────────────────────────────────────────────────


async def test_resume_orchestration_skips_completed_tasks(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """
    resume_orchestration injects completed results and only runs pending tasks.
    t1 is pre-completed; only t2 should run.
    """
    from orchestrator.planner import TaskResult, _task_result_to_dict

    run_id = "orch-resume-run"
    t1_result = TaskResult(
        task_id="t1",
        agent_id="analyst",
        answer="t1 already done",
        confidence=0.95,
        steps=2,
        success=True,
    )

    plan_dict = {
        "rationale": "linear plan",
        "tasks": [
            {
                "id": "t1",
                "agent_id": "analyst",
                "instruction": "step 1",
                "depends_on": [],
                "on_failure": "skip",
            },
            {
                "id": "t2",
                "agent_id": "reporter",
                "instruction": "step 2",
                "depends_on": ["t1"],
                "on_failure": "skip",
            },
        ],
    }
    ckp_store.data[run_id] = {
        "run_id": run_id,
        "goal": "test goal",
        "plan": plan_dict,
        "completed": {"t1": _task_result_to_dict(t1_result)},
        "replan_count": 0,
    }

    agents_run: list[str] = []

    def react(system, messages, kwargs):
        # Detect which agent is calling by checking its system prompt
        routing = system or next((m["content"] for m in messages if m["role"] == "system"), "")
        if "analyst" in routing.lower():
            agents_run.append("analyst")
        elif "reporter" in routing.lower():
            agents_run.append("reporter")
        return {"thought": "done", "action": "finish", "answer": "t2 done", "confidence": 0.9}

    def synth(system, messages, kwargs):
        return {"answer": "resumed synthesis", "confidence": 0.9, "conflicts": [], "unknowns": []}

    def extract(system, messages, kwargs):
        return {"semantic_facts": {}, "episodic_summary": "ok", "metadata": {}, "ttl_seconds": None}

    llm.routes = {
        "decomposes goals": lambda *a: plan_dict,
        "synthesis agent": synth,
        "memory extraction": extract,
        "analyst": react,
        "reporter": react,
    }

    runtime, _ = _make_runtime(llm, ckp_store)
    result = await runtime.resume_orchestration(run_id)

    assert result["answer"] == "resumed synthesis"
    # analyst (t1) was already completed — should NOT have run again
    assert "analyst" not in agents_run
    # reporter (t2) was pending — must have run
    assert "reporter" in agents_run


async def test_resume_orchestration_reruns_incomplete_tasks(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """When no tasks are completed yet, resume_orchestration runs all of them."""

    run_id = "orch-fresh-resume"
    plan_dict = {
        "rationale": "both tasks fresh",
        "tasks": [
            {
                "id": "t1",
                "agent_id": "analyst",
                "instruction": "step 1",
                "depends_on": [],
                "on_failure": "skip",
            },
            {
                "id": "t2",
                "agent_id": "reporter",
                "instruction": "step 2",
                "depends_on": ["t1"],
                "on_failure": "skip",
            },
        ],
    }
    ckp_store.data[run_id] = {
        "run_id": run_id,
        "goal": "fresh goal",
        "plan": plan_dict,
        "completed": {},  # nothing done yet
        "replan_count": 0,
    }

    agents_run: list[str] = []

    def react(system, messages, kwargs):
        routing = system or next((m["content"] for m in messages if m["role"] == "system"), "")
        if "analyst" in routing.lower():
            agents_run.append("analyst")
            return {
                "thought": "done",
                "action": "finish",
                "answer": "analyst done",
                "confidence": 0.9,
            }
        agents_run.append("reporter")
        return {"thought": "done", "action": "finish", "answer": "reporter done", "confidence": 0.9}

    llm.routes = {
        "decomposes goals": lambda *a: plan_dict,
        "synthesis agent": lambda *a: {
            "answer": "all done",
            "confidence": 0.9,
            "conflicts": [],
            "unknowns": [],
        },
        "memory extraction": lambda *a: {
            "semantic_facts": {},
            "episodic_summary": "ok",
            "metadata": {},
            "ttl_seconds": None,
        },
        "analyst": react,
        "reporter": react,
    }

    runtime, _ = _make_runtime(llm, ckp_store)
    result = await runtime.resume_orchestration(run_id)

    assert result["answer"] == "all done"
    assert "analyst" in agents_run
    assert "reporter" in agents_run


async def test_resume_unified_detects_orchestrator_checkpoint(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """runtime.resume(run_id) routes to resume_orchestration when checkpoint has 'plan'."""
    from orchestrator.planner import TaskResult, _task_result_to_dict

    run_id = "unified-orch"
    t1_result = TaskResult(
        task_id="t1",
        agent_id="analyst",
        answer="done",
        confidence=0.9,
        steps=1,
        success=True,
    )
    plan_dict = {
        "rationale": "one task done one pending",
        "tasks": [
            {
                "id": "t1",
                "agent_id": "analyst",
                "instruction": "s1",
                "depends_on": [],
                "on_failure": "skip",
            },
            {
                "id": "t2",
                "agent_id": "reporter",
                "instruction": "s2",
                "depends_on": ["t1"],
                "on_failure": "skip",
            },
        ],
    }
    ckp_store.data[run_id] = {
        "run_id": run_id,
        "goal": "unified goal",
        "plan": plan_dict,
        "completed": {"t1": _task_result_to_dict(t1_result)},
        "replan_count": 0,
    }

    llm.routes = {
        "decomposes goals": lambda *a: plan_dict,
        "synthesis agent": lambda *a: {
            "answer": "unified ok",
            "confidence": 0.9,
            "conflicts": [],
            "unknowns": [],
        },
        "memory extraction": lambda *a: {
            "semantic_facts": {},
            "episodic_summary": "ok",
            "metadata": {},
            "ttl_seconds": None,
        },
        "reporter": lambda *a: {
            "thought": "done",
            "action": "finish",
            "answer": "t2 ok",
            "confidence": 0.9,
        },
    }

    runtime, _ = _make_runtime(llm, ckp_store)
    result = await runtime.resume(run_id)

    assert result["answer"] == "unified ok"


async def test_resume_unified_detects_agent_checkpoint(
    llm: ScriptedLLM,
    memory: MemoryManager,
    ckp_store: InMemoryCheckpointStore,
):
    """runtime.resume(ckp_id) routes to resume_agent when checkpoint has 'agent_id'."""
    run_id = "unified-agent-run"
    agent_id = "analyst"
    ckp_id = f"{run_id}:{agent_id}"

    wm = await _minimal_wm(llm)
    ckp_store.data[ckp_id] = {
        "run_id": run_id,
        "agent_id": agent_id,
        "task": "agent task",
        "step": 0,
        "memory": wm.to_dict(),
    }

    llm.routes = {
        "react": lambda *a: {
            "thought": "done",
            "action": "finish",
            "answer": "agent ok",
            "confidence": 0.9,
        },
    }

    config = AgentConfig(
        agent_id=agent_id,
        role="r",
        system_prompt="ReAct.",
        allowed_tools=[],
    )
    agent_registry = AgentRegistry()
    agent_registry.register(config)

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        checkpoint_store=ckp_store,
    )

    result = await runtime.resume(ckp_id)
    assert result["answer"] == "agent ok"


async def test_agent_resume_key_is_outer_run_id_in_orchestrated_context(
    llm: ScriptedLLM,
    ckp_store: InMemoryCheckpointStore,
):
    """
    When the orchestrator sets agent._resume_key, the agent uses it for the
    HITL banner — so the human sees --resume <outer_run_id>, not --resume <ckp_id>.
    """
    from agents.base import BaseAgent
    from orchestrator.planner import EvalConfig, Orchestrator

    run_id = "outer-orch-run"
    plan_dict = {
        "rationale": "one task",
        "tasks": [
            {
                "id": "t1",
                "agent_id": "analyst",
                "instruction": "go",
                "depends_on": [],
                "on_failure": "skip",
            },
        ],
    }

    captured_resume_keys: list[str] = []

    config = AgentConfig(
        agent_id="analyst",
        role="r",
        system_prompt="ReAct.",
        allowed_tools=[],
        max_steps=3,
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    from harness.runtime import BudgetGuard, GuardrailConfig, Tracer

    tracer = Tracer()
    guard = BudgetGuard(GuardrailConfig())

    agent = BaseAgent(
        config=config,
        tools={},
        memory=memory,
        tracer=tracer,
        guard=guard,
        llm=llm,
        checkpoint_store=ckp_store,
    )

    original_run_stream = agent.run_stream

    async def capturing_run_stream(task, run_id=None):
        async for event in original_run_stream(task, run_id=run_id):
            captured_resume_keys.append(agent._resume_key)
            yield event

    agent.run_stream = capturing_run_stream

    llm.routes = {
        "decomposes goals": lambda *a: plan_dict,
        "synthesis agent": lambda *a: {
            "answer": "ok",
            "confidence": 0.9,
            "conflicts": [],
            "unknowns": [],
        },
        "memory extraction": lambda *a: {
            "semantic_facts": {},
            "episodic_summary": "ok",
            "metadata": {},
            "ttl_seconds": None,
        },
        "react": lambda *a: {
            "thought": "done",
            "action": "finish",
            "answer": "agent ok",
            "confidence": 0.9,
        },
    }

    orchestrator = Orchestrator(
        agents={"analyst": agent},
        memory=memory,
        tracer=tracer,
        guard=guard,
        llm=llm,
        eval_config=EvalConfig(),
        checkpoint_store=ckp_store,
        run_id=run_id,
    )

    async for _ in orchestrator.run_stream("goal"):
        pass

    # Every captured _resume_key should be the outer run_id, not the ckp_id
    assert all(k == run_id for k in captured_resume_keys), (
        f"expected all resume keys == {run_id!r}, got {captured_resume_keys}"
    )
