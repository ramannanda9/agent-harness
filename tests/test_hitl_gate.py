"""Tests for when the human-in-the-loop approval gate actually runs.

The gate used to be conditional on checkpoint durability: with no checkpoint
store configured it returned early and the tool ran with no prompt at all.
That predicate was also inverted with respect to its intent — it honoured the
gate only for agents that had explicitly opted *out* of resumable checkpoints.

Gating and durability are separate concerns. A tool the operator listed in
``hitl_tools`` is gated; whether the pause survives a crash is a property of
having somewhere to write.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.hitl import ApprovalResponse
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tests.conftest import ScriptedLLM


class RecordingTool:
    name = "echo"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"echo": kwargs.get("message", "")}


class InMemoryCheckpointStore:
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
def llm() -> ScriptedLLM:
    """Calls the gated tool once, then finishes."""
    calls = {"n": 0}

    def react(system, messages, kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "thought": "use the tool",
                "action": "echo",
                "args": {"message": "hello"},
            }
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 0.9}

    return ScriptedLLM(routes={"react": react})


@pytest.fixture
def memory(llm: ScriptedLLM) -> MemoryManager:
    return MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )


def _agent(llm, memory, tool, checkpoint_store=None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="gated",
            role="r",
            system_prompt="ReAct.",
            allowed_tools=["echo"],
            hitl_tools=["echo"],
            max_steps=4,
        ),
        tools={"echo": tool},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60)),
        llm=llm,
        checkpoint_store=checkpoint_store,
    )


async def _drain(agent, run_id="run-1"):
    events = []
    async for event in agent.run_stream(task="do the thing", run_id=run_id):
        events.append(event)
    return events


# ── The gate must not depend on durability ───────────────────────────────────


async def test_gated_tool_prompts_even_without_a_checkpoint_store(llm, memory):
    tool = RecordingTool()
    agent = _agent(llm, memory, tool, checkpoint_store=None)
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=True))

    with patch("harness.hitl.request_approval", approval):
        await _drain(agent)

    assert approval.await_count == 1, "a tool listed in hitl_tools must be gated"
    assert tool.calls == [{"message": "hello"}]


async def test_rejection_without_a_checkpoint_store_still_blocks_the_tool(llm, memory):
    """The failure this guards against: the tool running unsupervised because
    no store happened to be configured."""
    tool = RecordingTool()
    agent = _agent(llm, memory, tool, checkpoint_store=None)
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=False))

    with patch("harness.hitl.request_approval", approval):
        events = await _drain(agent)

    assert approval.await_count == 1
    assert tool.calls == [], "a rejected tool must not execute"
    observations = [e for e in events if e.type == EventType.OBSERVATION]
    assert "rejected by human" in observations[0].payload["observation"].lower()


async def test_gated_tool_still_prompts_when_resume_is_disabled(llm, memory):
    """PersistentAgent turns opt out of crash-resume but keep the gate."""
    tool = RecordingTool()
    agent = _agent(llm, memory, tool, checkpoint_store=InMemoryCheckpointStore())
    agent._checkpoint_resume_enabled = False
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=True))

    with patch("harness.hitl.request_approval", approval):
        await _drain(agent)

    assert approval.await_count == 1


async def test_ungated_tool_is_never_prompted(llm, memory):
    tool = RecordingTool()
    agent = _agent(llm, memory, tool, checkpoint_store=None)
    agent.config.hitl_tools = []
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=True))

    with patch("harness.hitl.request_approval", approval):
        await _drain(agent)

    assert approval.await_count == 0
    assert tool.calls == [{"message": "hello"}]


# ── Durability stays opt-out ─────────────────────────────────────────────────


async def test_disabled_resume_writes_no_checkpoint_at_any_point(llm, memory):
    """``_commit_checkpoint`` used to ignore the opt-out, so a gated tool in a
    PersistentAgent turn wrote a checkpoint that only stayed invisible because
    the clean-exit path deleted it. A crash mid-turn left an orphan."""
    store = InMemoryCheckpointStore()
    agent = _agent(llm, memory, RecordingTool(), checkpoint_store=store)
    agent._checkpoint_resume_enabled = False
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=True))

    with patch("harness.hitl.request_approval", approval):
        await _drain(agent)

    assert store.writes == [], "no write should happen while resume is disabled"


async def test_enabled_resume_does_write_a_checkpoint_for_a_gated_tool(llm, memory):
    """The complement of the test above — the opt-out is opt-out, not off."""
    store = InMemoryCheckpointStore()
    agent = _agent(llm, memory, RecordingTool(), checkpoint_store=store)
    approval = AsyncMock(return_value=ApprovalResponse(approval_id="a", approved=True))

    with patch("harness.hitl.request_approval", approval):
        await _drain(agent)

    assert "run-1:gated" in store.writes
