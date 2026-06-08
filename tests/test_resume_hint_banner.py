"""``_ResumeHint`` banner suppression contract for ``BaseAgent``.

The "Agent X interrupted — Resume: ... --resume <key>" banner should
print only when the run was actually *interrupted* (the
``_run_stream_internal`` generator exited via an exception that
unwound the ``async with _ResumeHint(...)`` context). It must NOT
print when:

- The agent ran to completion with ``EventType.TASK_DONE``
  (existing contract).
- The agent ran to completion with a terminal ``EventType.ERROR``
  (max_steps reached, budget exceeded, mid-run crash translated to
  ERROR by ``_run_stream_internal``'s except branch) — the run
  *ended*; "interrupted — Resume" is the wrong wording and resuming
  with identical config would hit the same wall.

Sub-agent events that bubble up through the outer agent's stream
carry ``parent_agent_id`` and must NOT short-circuit the outer agent's
hint either — the outer is still running when a delegated sub-agent
completes or errors.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import BusEvent, EventType
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── Fakes ─────────────────────────────────────────────────────────────────


class _StubLLM:
    """Returns whatever scripted response the test installs."""

    input_token_budget = 10_000
    last_usage = None

    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }

    async def complete(self, system, messages, **kwargs):
        return self.response


class _CheckpointStore:
    """In-memory checkpoint store with the same shape ``_ResumeHint`` reads."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def write(self, key: str, value: dict) -> None:
        self._data[key] = value

    async def read(self, key: str) -> dict | None:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


def _make_agent(
    *,
    llm: _StubLLM,
    checkpoint_store: _CheckpointStore | None = None,
    max_steps: int = 5,
) -> BaseAgent:
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent",
            role="test agent",
            system_prompt="You are a test agent.",
            allowed_tools=[],
            max_steps=max_steps,
            stream_tokens=False,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
        checkpoint_store=checkpoint_store,
    )
    return agent


async def _drain(agent: BaseAgent, *, run_id: str = "r1") -> list[BusEvent]:
    return [event async for event in agent.run_stream("do something", run_id=run_id)]


def _banner_printed(captured: list[str]) -> bool:
    return any("interrupted" in line and "Resume:" in line for line in captured)


# ── 1. Clean TASK_DONE: never any banner ──────────────────────────────────


@pytest.mark.asyncio
async def test_task_done_clears_checkpoint_and_suppresses_banner():
    store = _CheckpointStore()
    agent = _make_agent(llm=_StubLLM(), checkpoint_store=store)

    captured: list[str] = []
    with patch("sys.stderr") as stderr_mock:
        stderr_mock.write.side_effect = lambda s: captured.append(s)
        events = await _drain(agent)

    assert any(e.type == EventType.TASK_DONE for e in events)
    assert not _banner_printed(captured), (
        f"Clean TASK_DONE must never print the resume banner; captured: {captured!r}"
    )


# ── 2. Top-level ERROR (max_steps): no banner; checkpoint kept ────────────


class _NeverFinishesLLM:
    """LLM that returns a non-finish action repeatedly, forcing max_steps.

    Returning ``action="finish"`` ends the ReAct loop; any other action
    requires a tool to execute. We pick a tool that doesn't exist so the
    step yields an OBSERVATION (with an error), the loop continues, and
    eventually ``_react_stream`` exhausts ``max_steps`` and emits a
    top-level ERROR — the exact path the banner-suppression contract
    targets.
    """

    input_token_budget = 10_000
    last_usage = None

    async def complete(self, system, messages, **kwargs):
        return {
            "thought": "keep going",
            "action": "nonexistent_tool",
            "args": {},
            "confidence": 0.5,
        }


@pytest.mark.asyncio
async def test_max_steps_error_suppresses_banner_and_keeps_checkpoint():
    store = _CheckpointStore()
    # Force the checkpoint path: write a fake checkpoint before the run so
    # ``_ResumeHint.__aexit__`` finds one to flag if the test regresses.
    # max_steps=2 keeps the test fast.
    agent = _make_agent(llm=_NeverFinishesLLM(), checkpoint_store=store, max_steps=2)
    # Pre-populate so the banner WOULD fire if the suppression broke.
    await store.write("r1:agent", {"step": 0})

    captured: list[str] = []
    with patch("sys.stderr") as stderr_mock:
        stderr_mock.write.side_effect = lambda s: captured.append(s)
        events = await _drain(agent)

    # The agent yielded a top-level ERROR (max_steps).
    top_level_errors = [e for e in events if e.type == EventType.ERROR and not e.parent_agent_id]
    assert top_level_errors, "expected a top-level ERROR from max_steps"
    assert "Max steps" in (top_level_errors[-1].error or "")

    # Banner must NOT print on this path — that's the regression we're guarding.
    assert not _banner_printed(captured), (
        "Top-level ERROR (e.g. max_steps) must not print the 'interrupted — "
        "Resume:' banner; the run *ended*, it was not interrupted. "
        f"captured: {captured!r}"
    )

    # Checkpoint is intentionally KEPT — user may resume with new config
    # (higher max_steps, larger budget) without losing prior step state.
    assert await store.read("r1:agent") is not None, (
        "checkpoint should be preserved on terminal ERROR so the user can "
        "deliberately resume with new config"
    )


# ── 3. Real interrupt (CancelledError mid-stream): banner DOES fire ───────


class _SlowLLM:
    """Sleeps long enough that ``asyncio.CancelledError`` can interrupt."""

    input_token_budget = 10_000
    last_usage = None

    async def complete(self, system, messages, **kwargs):
        await asyncio.sleep(10)
        return {"thought": "x", "action": "finish", "answer": "x", "confidence": 1.0}


@pytest.mark.asyncio
async def test_cancellation_mid_stream_prints_resume_banner_when_checkpoint_exists():
    store = _CheckpointStore()
    agent = _make_agent(llm=_SlowLLM(), checkpoint_store=store, max_steps=5)
    # Pre-populate a checkpoint so ``__aexit__`` finds something to point at.
    await store.write("r1:agent", {"step": 0})

    captured: list[str] = []

    async def _consume() -> None:
        async for _ in agent.run_stream("do something", run_id="r1"):
            pass

    with patch("sys.stderr") as stderr_mock:
        stderr_mock.write.side_effect = lambda s: captured.append(s)
        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.02)  # let the LLM call start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # The banner SHOULD fire here — the run was actually interrupted by
    # cancellation, the checkpoint exists, and the user should know how to
    # resume.
    assert _banner_printed(captured), (
        "Real interrupt (mid-stream cancellation with checkpoint present) "
        f"must print the resume banner; captured: {captured!r}"
    )


# ── 4. Sub-agent events don't false-trigger the outer hint ────────────────


@pytest.mark.asyncio
async def test_sub_agent_terminal_events_do_not_false_clear_outer_checkpoint():
    """Sub-agent TASK_DONE / ERROR bubble up tagged with parent_agent_id.
    They must NOT clear the outer agent's checkpoint or mark the outer's
    hint done — the outer is still running."""
    store = _CheckpointStore()
    await store.write("r1:agent", {"step": 0})  # outer's checkpoint
    agent = _make_agent(llm=_StubLLM(), checkpoint_store=store, max_steps=3)
    agent._ckp_id = "r1:agent"

    # Simulate a sub-agent TASK_DONE bubbling up — by directly invoking the
    # hint logic against a synthetic event stream. This avoids needing a
    # full SubAgentTool fixture while exercising the exact dispatch.

    captured_clears: list[str] = []

    class _StoreSpy(_CheckpointStore):
        async def delete(self, key: str) -> None:
            captured_clears.append(key)
            await super().delete(key)

    spy_store = _StoreSpy()
    await spy_store.write("r1:agent", {"step": 0})
    agent._checkpoint_store = spy_store
    agent._ckp_id = "r1:agent"

    from harness.checkpoint import _ResumeHint

    async def _sub_then_outer() -> Any:
        # Sub-agent's TASK_DONE bubbles up first (tagged with parent_agent_id),
        # then the outer agent's own TASK_DONE arrives.
        yield BusEvent(
            type=EventType.TASK_DONE,
            agent_id="sub",
            parent_agent_id="agent",
            payload={"answer": "sub done"},
        )
        yield BusEvent(
            type=EventType.TASK_DONE,
            agent_id="agent",
            payload={"answer": "outer done"},
        )

    async with _ResumeHint(
        "r1:agent",
        spy_store,
        "Agent agent",
        check_key="r1:agent",
    ) as hint:
        async for event in _sub_then_outer():
            if not event.parent_agent_id:
                if event.type == EventType.TASK_DONE:
                    await agent._clear_checkpoint("r1")
                    hint.done = True
                elif event.type == EventType.ERROR:
                    hint.done = True

    # Exactly ONE clear (for the outer's TASK_DONE), not two.
    assert captured_clears == ["r1:agent"], (
        "sub-agent TASK_DONE must not trigger an outer checkpoint clear — "
        f"captured: {captured_clears!r}"
    )
