"""Unit tests for SessionMemoryController in isolation.

The controller is exercised end-to-end through PersistentAgent elsewhere; these
tests pin its decision logic and the reconcile-at-compaction dedup directly,
with a real InMemorySessionStore and fake memory/LLM.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.persistent import (
    InMemorySessionStore,
    PersistentAgentConfig,
    SessionMessage,
    SessionState,
)
from harness.session_memory import SessionMemoryController


class _FakeMemory:
    def __init__(self) -> None:
        self.run_writes: list[dict] = []

    async def build_context(self, *, goal: str, agent_id: str):  # pragma: no cover - unused here
        raise AssertionError("build_context not expected in these tests")

    async def write_run_end(self, *, goal, agent_results, trace) -> None:
        self.run_writes.append({"goal": goal, "agent_results": agent_results, "trace": trace})


class _FakeLLM:
    async def complete(self, *, system, messages, source):
        return {"text": "summary"}


def _controller(
    *,
    memory: _FakeMemory,
    store: InMemorySessionStore,
    config: PersistentAgentConfig | None = None,
    token_budget: int | None = 10_000,
) -> SessionMemoryController:
    return SessionMemoryController(
        memory=memory,
        session_store=store,
        config=config or PersistentAgentConfig(),
        coordinator_agent_id="coordinator",
        token_budget=lambda: token_budget,
        summarizer_llm=_FakeLLM,
    )


def _state(messages: list[SessionMessage], **kw) -> SessionState:
    return SessionState(session_id="s", messages=messages, **kw)


# ── should_reconcile ──────────────────────────────────────────────────────────


def test_should_reconcile_true_on_durable_signal():
    ctrl = _controller(memory=_FakeMemory(), store=InMemorySessionStore())
    assert ctrl.should_reconcile(
        message="please remember my timezone is UTC",
        state=_state([]),
        tools_used=set(),
        subagents_used=set(),
        errors=[],
    )


def test_should_reconcile_false_on_ordinary_message():
    ctrl = _controller(memory=_FakeMemory(), store=InMemorySessionStore())
    assert not ctrl.should_reconcile(
        message="what is the capital of France",
        state=_state([]),
        tools_used={"search"},
        subagents_used=set(),
        errors=["boom"],
    )


# ── should_compact ────────────────────────────────────────────────────────────


def test_should_compact_false_when_no_budget_advertised():
    ctrl = _controller(memory=_FakeMemory(), store=InMemorySessionStore(), token_budget=None)
    big = _state([SessionMessage(role="user", content="x" * 10_000)])
    assert ctrl.should_compact(big) is False


def test_should_compact_threshold():
    # budget 1000, fraction 0.5 → threshold 500 tokens (~2000 chars).
    cfg = PersistentAgentConfig(compact_at_context_fraction=0.5)
    ctrl = _controller(
        memory=_FakeMemory(), store=InMemorySessionStore(), config=cfg, token_budget=1000
    )
    small = _state([SessionMessage(role="user", content="x" * 100)])  # ~25 tokens
    assert ctrl.should_compact(small) is False
    big = _state([SessionMessage(role="user", content="x" * 4000)])  # ~1000 tokens
    assert ctrl.should_compact(big) is True


# ── compaction_split ──────────────────────────────────────────────────────────


def test_compaction_split_retains_newest_within_budget():
    # budget 1000, retain fraction 0.15 → retain ~150 tokens (~600 chars).
    cfg = PersistentAgentConfig(retain_context_fraction=0.15)
    ctrl = _controller(
        memory=_FakeMemory(), store=InMemorySessionStore(), config=cfg, token_budget=1000
    )
    msgs = [SessionMessage(role="user", content="x" * 400) for _ in range(5)]  # ~100 tokens each
    to_compact, keep_last = ctrl._compaction_split(_state(msgs))
    # Only the newest message fits within ~150 tokens; the rest compact.
    assert keep_last == 1
    assert len(to_compact) == 4


def test_compaction_split_full_transcript_when_no_budget():
    ctrl = _controller(memory=_FakeMemory(), store=InMemorySessionStore(), token_budget=None)
    msgs = [SessionMessage(role="user", content="hi")]
    to_compact, keep_last = ctrl._compaction_split(_state(msgs))
    assert to_compact == msgs
    assert keep_last == 0


# ── messages_since_reconcile ──────────────────────────────────────────────────


def test_messages_since_reconcile_windows_after_last_turn():
    msgs = [
        SessionMessage(role="user", content="q1"),
        SessionMessage(role="assistant", content="a1"),
        SessionMessage(role="user", content="q2"),
        SessionMessage(role="assistant", content="a2"),
    ]
    state = _state(msgs, turn_count=2, last_reconcile_turn=1)
    pending = SessionMemoryController.messages_since_reconcile(
        _controller(memory=_FakeMemory(), store=InMemorySessionStore()), state
    )
    # Turn 1 (q1/a1) already reconciled; only turn 2 is pending.
    assert [m.content for m in pending] == ["q2", "a2"]


# ── finalize_turn: reconcile-at-compaction dedup ──────────────────────────────


@pytest.mark.asyncio
async def test_finalize_turn_durable_signal_reconciles_once_even_when_compacting():
    memory = _FakeMemory()
    store = InMemorySessionStore()
    # Small budget so the seeded transcript also triggers compaction.
    cfg = PersistentAgentConfig(compact_at_context_fraction=0.1, retain_context_fraction=0.05)
    ctrl = _controller(memory=memory, store=store, config=cfg, token_budget=1000)

    state = await store.append_messages(
        "s",
        [
            SessionMessage(role="user", content="remember x " + "y" * 2000),
            SessionMessage(role="assistant", content="ok"),
        ],
    )
    assert ctrl.should_compact(state) is True  # precondition

    await ctrl.finalize_turn(
        "s",
        state=state,
        message="remember x",  # durable signal → sync reconcile
        final_result={"answer": "ok"},
        trace=[],
        tools_used=set(),
        subagents_used=set(),
        errors=[],
    )

    # Sync reconcile wrote once; the compaction branch must NOT write again.
    assert len(memory.run_writes) == 1


@pytest.mark.asyncio
async def test_finalize_turn_background_reconcile_fires_at_interval():
    memory = _FakeMemory()
    store = InMemorySessionStore()
    cfg = PersistentAgentConfig(async_reconcile_every_turns=2)
    # Large budget so compaction never fires; isolate the background path.
    ctrl = _controller(memory=memory, store=store, config=cfg, token_budget=10_000_000)

    state = None
    for i in range(2):
        state = await store.append_messages(
            "s",
            [
                SessionMessage(role="user", content=f"q{i}"),
                SessionMessage(role="assistant", content=f"a{i}"),
            ],
        )
    assert state.turn_count == 2  # interval boundary

    await ctrl.finalize_turn(
        "s",
        state=state,
        message="ordinary message",  # no durable signal
        final_result={"answer": "a"},
        trace=[],
        tools_used=set(),
        subagents_used=set(),
        errors=[],
    )
    await asyncio.sleep(0)  # let the fire-and-forget reconcile run

    assert len(memory.run_writes) == 1
    reloaded = await store.load("s")
    assert reloaded.last_reconcile_turn == 2
