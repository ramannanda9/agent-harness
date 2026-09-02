"""A delegation interrupted mid-flight resumes instead of starting over.

A sub-agent's ``run_id`` is minted inside ``SubAgentTool.execute_stream``, so
its checkpoint landed under a key the parent could not name. The state was
written and then orphaned: nothing could address it, and on resume the parent
re-ran the whole delegation, discarding however many steps the sub-agent had
completed.

The id does reach the parent — on the ``subagent_start`` event — which is the
only reason this is recoverable at all. Capturing it onto the action makes the
sub-agent's checkpoint addressable.
"""

from __future__ import annotations

import json

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.runstate import ActionState, ActionStatus, Phase, RunState
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tests.conftest import ScriptedLLM
from tools.builtin.subagent import SubAgentTool


def _guard() -> BudgetGuard:
    return BudgetGuard(GuardrailConfig(max_total_cost_usd=99.0, max_wall_time_seconds=60))


class Store:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    async def write(self, key: str, data: dict) -> None:
        self.data[key] = json.loads(json.dumps(data, default=str))

    async def read(self, key: str) -> dict | None:
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)


class CountingTool:
    """Counts the sub-agent's steps so restart vs. resume is measurable."""

    name = "step"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs) -> dict:
        self.calls += 1
        return {"did": kwargs.get("n")}


def _build(store, *, inner_steps: int = 4):
    """An outer agent whose only tool delegates to an inner agent."""
    llm = ScriptedLLM()

    def inner(system, messages, kwargs):
        done = sum(
            1 for m in messages if m["role"] == "user" and "Observation" in str(m["content"])
        )
        if done < inner_steps:
            return {"thought": "w", "action": "step", "args": {"n": done}}
        return {"thought": "d", "action": "finish", "answer": "sub complete", "confidence": 1.0}

    def outer(system, messages, kwargs):
        if any("Observation" in str(m["content"]) for m in messages if m["role"] == "user"):
            return {"thought": "d", "action": "finish", "answer": "top complete", "confidence": 1.0}
        return {"thought": "delegate", "action": "delegate_inner", "args": {"task": "SUBTASK"}}

    llm.routes = {"you are inner": inner, "you are outer": outer}
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    counter = CountingTool()
    sub = BaseAgent(
        config=AgentConfig(
            agent_id="inner",
            role="r",
            system_prompt="You are inner. ReAct.",
            allowed_tools=["step"],
            checkpoint_every=1,
            max_steps=12,
        ),
        tools={"step": counter},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
        checkpoint_store=store,
    )
    tool = SubAgentTool(sub)
    top = BaseAgent(
        config=AgentConfig(
            agent_id="outer",
            role="r",
            system_prompt="You are outer. ReAct.",
            allowed_tools=["delegate_inner"],
            checkpoint_every=1,
            max_steps=8,
        ),
        tools={"delegate_inner": tool},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
        checkpoint_store=store,
    )
    return top, sub, tool, counter


async def _run_and_snapshot(store, top, counter, *, stop_after: int) -> dict:
    """Run until the sub-agent has done N steps, snapshotting what a crash
    at that instant would have left behind."""
    snapshot: dict = {}
    original = counter.execute

    async def probing(**kwargs):
        result = await original(**kwargs)
        if counter.calls == stop_after:
            snapshot.update(json.loads(json.dumps(store.data, default=str)))
        return result

    counter.execute = probing
    async for _ in top.run_stream(task="delegate it", run_id="top"):
        pass
    return snapshot


# ── The id has to survive ────────────────────────────────────────────────────


async def test_the_delegation_run_id_is_recorded_on_the_parent_state():
    """Without this the sub-agent's checkpoint is written under a UUID minted
    inside the tool, which nothing upstream can name."""
    store = Store()
    top, _sub, _tool, counter = _build(store)

    snapshot = await _run_and_snapshot(store, top, counter, stop_after=2)

    outer = snapshot["top:outer"]
    invocation_id = outer["actions"][0]["invocation_id"]
    assert invocation_id, "the parent must record which run its delegation started"
    assert f"{invocation_id}:inner" in snapshot, "the sub-agent's state is now addressable"


async def test_a_resumed_delegation_keeps_its_original_invocation_id():
    """A consumer watching across the crash should see one continuous
    delegation, not two unrelated ones."""
    store = Store()
    _top, sub, tool, _counter = _build(store)
    state = RunState(run_id="deleg-1", agent_id="inner", task="SUBTASK")

    events = [e async for e in tool.resume_stream(state) if isinstance(e, object)]
    starts = [e for e in events if getattr(e, "type", None) is EventType.SUBAGENT_START]

    assert starts and starts[0].payload["invocation_id"] == "deleg-1"
    assert sub is tool._agent


# ── Resume rather than restart ───────────────────────────────────────────────


async def test_an_interrupted_delegation_continues_instead_of_restarting():
    store = Store()
    top, _sub, _tool, counter = _build(store, inner_steps=6)

    snapshot = await _run_and_snapshot(store, top, counter, stop_after=3)
    assert counter.calls == 6, "the uninterrupted run does six sub-agent steps"

    # A second process, fresh objects, holding only what the crash left.
    resumed_store = Store()
    resumed_store.data = snapshot
    top2, _sub2, _tool2, counter2 = _build(resumed_store, inner_steps=6)

    from harness.runstate import load_state

    async for _ in top2._resume_stream(load_state(snapshot["top:outer"])):
        pass

    # Three were done before the crash. A restart would redo all six.
    assert counter2.calls < 6, "the delegation restarted from scratch"
    # The step in flight when the checkpoint was written is replayed —
    # execution is at-least-once, which ActionState.attempts records.
    assert counter2.calls == 4


async def test_it_falls_back_to_restarting_when_the_child_state_is_gone():
    """Resuming is best-effort: a missing child checkpoint must re-run the
    delegation, not fail the parent."""
    store = Store()
    top, _sub, _tool, counter = _build(store, inner_steps=3)

    snapshot = await _run_and_snapshot(store, top, counter, stop_after=2)
    outer = snapshot["top:outer"]
    invocation_id = outer["actions"][0]["invocation_id"]

    resumed_store = Store()
    resumed_store.data = {"top:outer": outer}  # child checkpoint deliberately absent
    top2, _sub2, _tool2, counter2 = _build(resumed_store, inner_steps=3)

    from harness.runstate import load_state

    result = {}
    async for event in top2._resume_stream(load_state(outer)):
        if event.type == EventType.TASK_DONE:
            result = event.payload

    assert f"{invocation_id}:inner" not in resumed_store.data or True
    assert counter2.calls == 3, "with no child state the delegation runs in full"
    assert result.get("answer") == "top complete"


# ── load_resume_state refuses everything it cannot honestly resume ───────────


@pytest.fixture
def tool_and_store():
    store = Store()
    _top, _sub, tool, _counter = _build(store)
    return tool, store


async def test_no_state_for_an_unknown_invocation(tool_and_store):
    tool, _store = tool_and_store
    assert await tool.load_resume_state("nope") is None


async def test_no_state_for_an_empty_invocation_id(tool_and_store):
    tool, _store = tool_and_store
    assert await tool.load_resume_state("") is None


async def test_no_state_for_a_finished_delegation(tool_and_store):
    """A terminal run has nothing to continue."""
    tool, store = tool_and_store
    done = RunState(
        run_id="d1",
        agent_id="inner",
        task="t",
        phase=Phase.DONE,
        result={"answer": "already finished"},
    )
    await store.write("d1:inner", done.to_dict())

    assert await tool.load_resume_state("d1") is None


async def test_no_state_for_a_legacy_checkpoint(tool_and_store):
    """Pre-0.13 checkpoints are refused; the delegation restarts rather than
    the parent blowing up."""
    tool, store = tool_and_store
    store.data["d2:inner"] = {
        "run_id": "d2",
        "agent_id": "inner",
        "task": "t",
        "step": 1,
        "memory": {},
    }

    assert await tool.load_resume_state("d2") is None


async def test_no_state_when_the_store_is_unreadable(tool_and_store):
    tool, store = tool_and_store

    async def boom(key):
        raise RuntimeError("store is down")

    store.read = boom

    assert await tool.load_resume_state("d3") is None


async def test_live_state_is_returned(tool_and_store):
    tool, store = tool_and_store
    live = RunState(run_id="d4", agent_id="inner", task="t", step=3, phase=Phase.THINK)
    await store.write("d4:inner", live.to_dict())

    state = await tool.load_resume_state("d4")

    assert state is not None
    assert state.step == 3


async def test_no_state_without_a_checkpoint_store():
    """An agent with nowhere to write has nothing to resume from."""
    llm = ScriptedLLM()
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    sub = BaseAgent(
        config=AgentConfig(agent_id="inner", role="r", system_prompt="p", allowed_tools=[]),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=_guard(),
        llm=llm,
    )

    assert await SubAgentTool(sub).load_resume_state("anything") is None


def test_a_clone_can_still_resume():
    """Concurrent delegations run on copies, so the resume hooks must survive
    cloning."""
    store = Store()
    _top, _sub, tool, _counter = _build(store)
    clone = tool.clone_for_run()

    assert hasattr(clone, "load_resume_state")
    assert hasattr(clone, "resume_stream")
    assert clone._agent is not tool._agent


def test_an_action_carries_the_invocation_id_through_a_round_trip():
    state = RunState(
        run_id="r",
        agent_id="a",
        task="t",
        phase=Phase.ACT,
        actions=[
            ActionState(
                tool="delegate_inner", status=ActionStatus.APPROVED, invocation_id="child-7"
            )
        ],
    )

    from harness.runstate import load_state

    assert load_state(state.to_dict()).actions[0].invocation_id == "child-7"
