"""``SubAgentTool`` — sub-agent delegation as a parent-callable tool.

Covers:
  - Single delegation: parent emits ``delegate_X(task=...)``; sub's events
    bubble through parent's stream; sub's final answer becomes parent's
    observation.
  - Parallel delegation: parent emits ``actions: [delegate_A, delegate_B]``;
    both sub-agent streams interleave via fan-in; both observations land.
  - Recursion guard: a sub-agent attempting to delegate beyond
    ``max_subagent_depth`` is refused with an observation, not a hang.
  - Event tagging: bubbled events carry ``parent_agent_id``.
  - Sub-agent failure surfaces as an observation, not a crash.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import BusEvent, EventType
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool

# ── Test scaffolding ─────────────────────────────────────────────────────────


class _CannedLLM:
    """LLM stub that emits a pre-scripted sequence of responses, one per call."""

    def __init__(self, responses: list[dict]) -> None:
        import json

        self._responses = [json.dumps(r) for r in responses]
        self._i = 0
        self.last_usage: dict | None = None

    async def complete(self, system, messages, **kwargs) -> dict:
        if self._i >= len(self._responses):
            # Default finish — keeps tests from hanging on max_steps.
            import json

            return {
                "text": json.dumps({"action": "finish", "answer": "default", "confidence": 1.0}),
                "usage": {},
            }
        text = self._responses[self._i]
        self._i += 1
        return {"text": text, "usage": {}}


def _build_agent(
    *,
    agent_id: str,
    llm: Any,
    tools: dict[str, Any] | None = None,
    max_steps: int = 5,
    max_subagent_depth: int = 3,
) -> BaseAgent:
    semantic = InMemorySemanticStore()
    episodic = InMemoryEpisodicStore()
    memory = MemoryManager(
        semantic_store=semantic,
        episodic_store=episodic,
        llm=llm,
        reconcile_on_write=False,  # tests don't care about reconcile here
    )
    config = AgentConfig(
        agent_id=agent_id,
        role=f"{agent_id} role",
        system_prompt=f"you are {agent_id}",
        allowed_tools=list((tools or {}).keys()),
        max_steps=max_steps,
        max_subagent_depth=max_subagent_depth,
    )
    return BaseAgent(
        config=config,
        tools=tools or {},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60)),
        llm=llm,
    )


# ── Single delegation ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_delegation_returns_subagent_answer_as_observation():
    """Parent's `delegate_research(task=...)` runs the researcher; the
    researcher's final answer becomes the parent's observation, and the
    parent then finishes with its own answer."""
    researcher_llm = _CannedLLM(
        [
            {"thought": "search now", "action": "finish", "answer": "found X", "confidence": 0.9},
        ]
    )
    researcher = _build_agent(agent_id="researcher", llm=researcher_llm)
    tool = SubAgentTool(researcher)

    parent_llm = _CannedLLM(
        [
            {"thought": "need data", "action": "delegate_researcher", "args": {"task": "find X"}},
            {
                "thought": "report",
                "action": "finish",
                "answer": "parent says: found X",
                "confidence": 0.95,
            },
        ]
    )
    parent = _build_agent(agent_id="coordinator", llm=parent_llm, tools={tool.name: tool})

    events = [e async for e in parent.run_stream("delegate and report")]
    finish = next((e for e in events if e.type == EventType.TASK_DONE), None)
    assert finish is not None
    assert "found X" in finish.payload["answer"]
    # Parent should have seen an OBSERVATION whose content reflects the
    # researcher's final answer.
    parent_obs = [
        e for e in events if e.type == EventType.OBSERVATION and e.agent_id == "coordinator"
    ]
    assert parent_obs
    assert "found X" in str(parent_obs[0].payload["observation"])


@pytest.mark.asyncio
async def test_subagent_events_bubble_with_parent_agent_id_tag():
    """Sub-agent THOUGHT / ACTION / OBSERVATION events appear in the parent
    stream, tagged with ``parent_agent_id`` so renderers can distinguish."""
    researcher_llm = _CannedLLM(
        [{"thought": "delivering", "action": "finish", "answer": "data", "confidence": 0.8}]
    )
    researcher = _build_agent(agent_id="researcher", llm=researcher_llm)
    tool = SubAgentTool(researcher)

    parent_llm = _CannedLLM(
        [
            {"thought": "go", "action": "delegate_researcher", "args": {"task": "find"}},
            {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0},
        ]
    )
    parent = _build_agent(agent_id="coord", llm=parent_llm, tools={tool.name: tool})

    bubbled = [
        e
        async for e in parent.run_stream("go")
        if e.parent_agent_id  # only sub-agent-tagged events
    ]
    assert bubbled, "expected at least one bubbled sub-agent event"
    # parent_agent_id must be the INVOKING parent's id ("coord"), NOT the
    # sub-agent's own id — otherwise renderers can't group / indent by
    # actual parent, and a sub-agent's events look orphaned.
    assert all(e.parent_agent_id == "coord" for e in bubbled), (
        f"expected parent_agent_id='coord' on all bubbled events; got "
        f"{set(e.parent_agent_id for e in bubbled)!r}"
    )
    # THOUGHT from the researcher should be among them — and agent_id
    # should still be the sub's own id (researcher), separate from parent.
    assert any(
        e.type == EventType.THOUGHT and e.agent_id == "researcher" and e.parent_agent_id == "coord"
        for e in bubbled
    )


# ── Parallel delegation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_delegation_collects_both_observations():
    """Two delegations in one ``actions: [...]`` complete in parallel; both
    final answers land in the OBSERVATION the parent records."""
    researcher_llm = _CannedLLM(
        [{"thought": "go", "action": "finish", "answer": "RES-A", "confidence": 0.9}]
    )
    analyst_llm = _CannedLLM(
        [{"thought": "go", "action": "finish", "answer": "ANL-B", "confidence": 0.9}]
    )
    researcher = _build_agent(agent_id="researcher", llm=researcher_llm)
    analyst = _build_agent(agent_id="analyst", llm=analyst_llm)
    delegate_research = SubAgentTool(researcher, name="delegate_research")
    delegate_analyse = SubAgentTool(analyst, name="delegate_analyse")

    parent_llm = _CannedLLM(
        [
            {
                "thought": "parallel",
                "actions": [
                    {"tool": "delegate_research", "args": {"task": "find"}},
                    {"tool": "delegate_analyse", "args": {"task": "look"}},
                ],
            },
            {
                "thought": "done",
                "action": "finish",
                "answer": "combined",
                "confidence": 1.0,
            },
        ]
    )
    parent = _build_agent(
        agent_id="coord",
        llm=parent_llm,
        tools={
            delegate_research.name: delegate_research,
            delegate_analyse.name: delegate_analyse,
        },
    )

    events = [e async for e in parent.run_stream("go")]
    parent_obs = [e for e in events if e.type == EventType.OBSERVATION and e.agent_id == "coord"]
    assert parent_obs
    combined = " ".join(str(e.payload.get("observation", "")) for e in parent_obs)
    assert "RES-A" in combined
    assert "ANL-B" in combined


# ── Recursion guard ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recursion_guard_refuses_delegation_beyond_max_depth():
    """A sub-agent whose own tools include another SubAgentTool gets
    refused at the depth limit. The refusal lands as an observation so the
    caller can react — it doesn't hang the framework."""
    # Build a 3-level chain limited to depth 1: top → sub1 (depth 1) →
    # would-be sub2 (depth 2). max_subagent_depth=1 on the top means sub1's
    # attempt to delegate further is refused.
    sub2_llm = _CannedLLM(
        [{"thought": "fine", "action": "finish", "answer": "L2", "confidence": 1.0}]
    )
    sub2 = _build_agent(agent_id="sub2", llm=sub2_llm)
    sub2_tool = SubAgentTool(sub2, name="delegate_sub2")

    sub1_llm = _CannedLLM(
        [
            {
                "thought": "delegate further",
                "action": "delegate_sub2",
                "args": {"task": "go deeper"},
            },
            {
                "thought": "use the refusal",
                "action": "finish",
                "answer": "sub1 done",
                "confidence": 0.8,
            },
        ]
    )
    sub1 = _build_agent(
        agent_id="sub1",
        llm=sub1_llm,
        tools={sub2_tool.name: sub2_tool},
        max_subagent_depth=1,  # sub1 itself enforces; sub2 hop would be depth 2
    )
    sub1_tool = SubAgentTool(sub1, name="delegate_sub1")

    top_llm = _CannedLLM(
        [
            {"thought": "start", "action": "delegate_sub1", "args": {"task": "go"}},
            {"thought": "done", "action": "finish", "answer": "top", "confidence": 1.0},
        ]
    )
    top = _build_agent(
        agent_id="top",
        llm=top_llm,
        tools={sub1_tool.name: sub1_tool},
        max_subagent_depth=1,
    )

    events = [e async for e in top.run_stream("delegate chain")]
    # The sub1 observation about sub2 should mention the refusal — surfacing
    # depth-exceeded as a string the LLM can read.
    sub1_obs_events = [
        e for e in events if e.type == EventType.OBSERVATION and e.agent_id == "sub1"
    ]
    assert sub1_obs_events, "sub1 should have produced an observation about the refusal"
    refusal_text = " ".join(str(e.payload.get("observation", "")) for e in sub1_obs_events)
    assert "max sub-agent depth" in refusal_text.lower() or "refused" in refusal_text.lower()


# ── Failure handling ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subagent_failure_surfaces_as_observation():
    """If the sub-agent's run stream ends without TASK_DONE (e.g. budget
    exceeded, max_steps), the SubAgentTool surfaces a structured failure
    observation rather than hanging or raising."""

    # Sub-agent loops forever on its tool — exhausts max_steps without finishing.
    class _LoopTool:
        name = "loop"

        async def execute(self, **_kwargs: Any) -> str:
            return "did nothing useful"

    sub_llm = _CannedLLM(
        [
            {"thought": "loop", "action": "loop", "args": {}},
            {"thought": "loop", "action": "loop", "args": {}},
            {"thought": "loop", "action": "loop", "args": {}},
        ]
    )
    sub = _build_agent(
        agent_id="sub",
        llm=sub_llm,
        tools={"loop": _LoopTool()},
        max_steps=2,  # forced to exhaust without finishing
    )
    sub_tool = SubAgentTool(sub, name="delegate_sub")

    parent_llm = _CannedLLM(
        [
            {"thought": "go", "action": "delegate_sub", "args": {"task": "try"}},
            {
                "thought": "saw failure",
                "action": "finish",
                "answer": "handled",
                "confidence": 0.7,
            },
        ]
    )
    parent = _build_agent(agent_id="parent", llm=parent_llm, tools={sub_tool.name: sub_tool})

    events = [e async for e in parent.run_stream("go")]
    parent_obs = [e for e in events if e.type == EventType.OBSERVATION and e.agent_id == "parent"]
    assert parent_obs
    obs_text = str(parent_obs[0].payload["observation"]).lower()
    assert "success" in obs_text or "error" in obs_text or "stream ended" in obs_text


# ── Argument validation ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subagent_tool_rejects_missing_task_arg():
    """If the parent's LLM hallucinates a call with no task, the tool
    returns a structured error rather than running an empty sub-agent."""
    sub_llm = _CannedLLM([])  # never reached
    sub = _build_agent(agent_id="sub", llm=sub_llm)
    tool = SubAgentTool(sub, name="delegate_sub")

    collected: list[Any] = []
    async for item in tool.execute_stream():  # no task=
        collected.append(item)
    assert len(collected) == 1
    assert collected[0]["success"] is False
    assert "missing required arg" in collected[0]["error"]


@pytest.mark.asyncio
async def test_parallel_mixed_streaming_and_plain_tools_runs():
    """A parallel batch mixing a streaming sub-agent tool and a plain
    awaitable tool must run both branches to completion.

    Pins the fix for ``asyncio.create_task(asyncio.gather(...))`` — in
    modern Python ``gather`` returns a Future, not a coroutine, and
    ``create_task`` rejects it with TypeError. The bug was invisible to
    pure-streaming batches; this test exercises the mixed path
    specifically."""

    class _PlainEcho:
        name = "plain_echo"

        async def execute(self, msg: str = "") -> dict:
            return {"echoed": msg}

    sub_llm = _CannedLLM(
        [{"thought": "go", "action": "finish", "answer": "from-sub", "confidence": 0.9}]
    )
    sub = _build_agent(agent_id="sub", llm=sub_llm)
    delegate = SubAgentTool(sub, name="delegate_sub")

    parent_llm = _CannedLLM(
        [
            {
                "thought": "parallel mix",
                "actions": [
                    {"tool": "delegate_sub", "args": {"task": "work"}},
                    {"tool": "plain_echo", "args": {"msg": "hello"}},
                ],
            },
            {
                "thought": "done",
                "action": "finish",
                "answer": "combined",
                "confidence": 1.0,
            },
        ]
    )
    parent = _build_agent(
        agent_id="coord",
        llm=parent_llm,
        tools={delegate.name: delegate, "plain_echo": _PlainEcho()},
    )

    events = [e async for e in parent.run_stream("mixed")]
    parent_obs = [e for e in events if e.type == EventType.OBSERVATION and e.agent_id == "coord"]
    # Both observations must land in the parent's stream.
    assert parent_obs
    combined = " ".join(str(e.payload.get("observation", "")) for e in parent_obs)
    assert "from-sub" in combined
    assert "hello" in combined


@pytest.mark.asyncio
async def test_parent_guard_is_shared_with_subagent_on_delegate():
    """The sub-agent's ``_guard`` should be reassigned to the parent's
    guard when delegation starts — so ``check()`` enforces the run-level
    cap and the bubbled TASK_DONE budget snapshot reflects real usage,
    not the sub-agent's stale construction-time guard."""
    sub_llm = _CannedLLM(
        [{"thought": "go", "action": "finish", "answer": "done", "confidence": 1.0}]
    )
    sub = _build_agent(agent_id="sub", llm=sub_llm)
    sub_local_guard = sub._guard  # captured before delegation
    tool = SubAgentTool(sub, name="delegate_sub")

    parent_llm = _CannedLLM(
        [
            {"thought": "go", "action": "delegate_sub", "args": {"task": "do thing"}},
            {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0},
        ]
    )
    parent = _build_agent(agent_id="parent", llm=parent_llm, tools={tool.name: tool})

    async for _ in parent.run_stream("go"):
        pass

    # After the run, the sub-agent's _guard should now be the parent's
    # guard, not its original local one.
    assert sub._guard is parent._guard
    assert sub._guard is not sub_local_guard


# ── Lifecycle events ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subagent_start_and_done_events_emitted():
    """SUBAGENT_START fires before any sub-agent THOUGHT; SUBAGENT_DONE fires
    after the sub-agent's TASK_DONE and before the terminal result dict."""
    sub_llm = _CannedLLM(
        [{"thought": "go", "action": "finish", "answer": "result", "confidence": 0.9}]
    )
    sub = _build_agent(agent_id="worker", llm=sub_llm)
    tool = SubAgentTool(sub, name="delegate_worker")
    tool._invoking_agent_id = "coord"

    items: list = []
    async for item in tool.execute_stream(task="do work"):
        items.append(item)

    bus_events = [i for i in items if isinstance(i, BusEvent)]
    result_dicts = [i for i in items if not isinstance(i, BusEvent)]

    start_events = [e for e in bus_events if e.type == EventType.SUBAGENT_START]
    done_events = [e for e in bus_events if e.type == EventType.SUBAGENT_DONE]
    thought_events = [e for e in bus_events if e.type == EventType.THOUGHT]

    assert len(start_events) == 1, "exactly one SUBAGENT_START per delegation"
    assert len(done_events) == 1, "exactly one SUBAGENT_DONE per delegation"

    start_idx = bus_events.index(start_events[0])
    done_idx = bus_events.index(done_events[0])
    assert start_idx == 0, "SUBAGENT_START must be the first bus event"
    if thought_events:
        thought_idx = bus_events.index(thought_events[0])
        assert start_idx < thought_idx, "SUBAGENT_START before first THOUGHT"
    assert done_idx > start_idx, "SUBAGENT_DONE after SUBAGENT_START"
    assert result_dicts, "terminal result dict still yielded"

    # SUBAGENT_DONE carries success metadata.
    d = done_events[0]
    assert d.agent_id == "worker"
    assert d.parent_agent_id == "coord"
    assert d.payload["success"] is True
    assert d.payload["steps"] > 0
    assert d.payload["confidence"] == pytest.approx(0.9)
    assert "result" in d.payload["answer"]
    assert d.payload["invocation_id"] == start_events[0].payload["invocation_id"]


@pytest.mark.asyncio
async def test_subagent_done_reports_failure_when_no_task_done():
    """SUBAGENT_DONE with success=False when the sub-agent exhausts max_steps."""

    class _LoopTool:
        name = "loop"

        async def execute(self, **_: object) -> str:
            return "nothing"

    sub_llm = _CannedLLM([{"thought": "loop", "action": "loop", "args": {}}] * 3)
    sub = _build_agent(agent_id="sub", llm=sub_llm, tools={"loop": _LoopTool()}, max_steps=2)
    tool = SubAgentTool(sub, name="delegate_sub")
    tool._invoking_agent_id = "parent"

    bus_events: list[BusEvent] = []
    async for item in tool.execute_stream(task="fail"):
        if isinstance(item, BusEvent):
            bus_events.append(item)

    done_events = [e for e in bus_events if e.type == EventType.SUBAGENT_DONE]
    assert len(done_events) == 1
    assert done_events[0].payload["success"] is False
    assert done_events[0].payload["error"]


@pytest.mark.asyncio
async def test_subagent_lifecycle_events_carry_parent_agent_id():
    """SUBAGENT_START and SUBAGENT_DONE both carry the invoking parent's id."""
    sub_llm = _CannedLLM(
        [{"thought": "ok", "action": "finish", "answer": "done", "confidence": 1.0}]
    )
    sub = _build_agent(agent_id="sub", llm=sub_llm)
    tool = SubAgentTool(sub, name="delegate_sub")

    parent_llm = _CannedLLM(
        [
            {"thought": "go", "action": "delegate_sub", "args": {"task": "work"}},
            {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0},
        ]
    )
    parent = _build_agent(agent_id="parent", llm=parent_llm, tools={tool.name: tool})

    events = [e async for e in parent.run_stream("go")]
    starts = [e for e in events if e.type == EventType.SUBAGENT_START]
    dones = [e for e in events if e.type == EventType.SUBAGENT_DONE]

    assert starts and dones
    assert all(e.parent_agent_id == "parent" for e in starts)
    assert all(e.parent_agent_id == "parent" for e in dones)
    assert all(e.agent_id == "sub" for e in starts + dones)
    assert starts[0].payload["invocation_id"] == dones[0].payload["invocation_id"]


@pytest.mark.asyncio
async def test_subagent_bubbled_events_carry_invocation_id():
    sub_llm = _CannedLLM(
        [{"thought": "ok", "action": "finish", "answer": "done", "confidence": 1.0}]
    )
    sub = _build_agent(agent_id="sub", llm=sub_llm)
    tool = SubAgentTool(sub, name="delegate_sub")
    tool._invoking_agent_id = "parent"

    events: list[BusEvent] = []
    async for item in tool.execute_stream(task="work"):
        if isinstance(item, BusEvent):
            events.append(item)

    starts = [e for e in events if e.type == EventType.SUBAGENT_START]
    assert starts
    invocation_id = starts[0].payload["invocation_id"]
    assert invocation_id
    assert all(e.payload.get("invocation_id") == invocation_id for e in events)


@pytest.mark.asyncio
async def test_subagent_tool_uses_custom_task_arg_name():
    sub_llm = _CannedLLM([{"thought": "go", "action": "finish", "answer": "ok", "confidence": 1.0}])
    sub = _build_agent(agent_id="sub", llm=sub_llm)
    tool = SubAgentTool(sub, name="delegate_sub", task_arg="instruction")

    final = None
    async for item in tool.execute_stream(instruction="do thing"):
        if not isinstance(item, BusEvent):
            final = item
    assert final is not None
    assert final["success"] is True
    assert final["answer"] == "ok"
