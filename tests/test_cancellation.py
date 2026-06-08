from __future__ import annotations

import asyncio
import io

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.cancellation import consume_with_cancel, run_until_cancelled
from harness.console import ConsoleRenderer
from harness.events import BusEvent, EventType
from harness.persistent import InMemorySessionStore, PersistentAgent, PersistentAgentConfig
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── run_until_cancelled — generic asyncio behaviour ────────────────────────


@pytest.mark.asyncio
async def test_returns_result_when_task_finishes_first():
    trigger = asyncio.Event()

    async def quick() -> str:
        return "done"

    cancelled, result = await run_until_cancelled(quick, trigger=trigger)

    assert cancelled is False
    assert result == "done"


@pytest.mark.asyncio
async def test_cancels_task_when_trigger_fires_first():
    trigger = asyncio.Event()
    cleanup_ran = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cleanup_ran.set()
            raise

    async def fire_soon() -> None:
        await asyncio.sleep(0.01)
        trigger.set()

    fire_task = asyncio.create_task(fire_soon())
    cancelled, result = await run_until_cancelled(slow, trigger=trigger)
    await fire_task

    assert cancelled is True
    assert result is None
    assert cleanup_ran.is_set(), "cancelled task's finally block did not run"


@pytest.mark.asyncio
async def test_trigger_cleared_at_entry_so_event_is_reusable():
    """A trigger left set from a prior turn must not auto-cancel the next one."""
    trigger = asyncio.Event()
    trigger.set()  # left over from a previous turn

    async def quick() -> str:
        await asyncio.sleep(0)  # one event-loop tick so the clear-then-wait
        # path actually exercises
        return "ok"

    cancelled, result = await run_until_cancelled(quick, trigger=trigger)

    assert cancelled is False
    assert result == "ok"
    assert not trigger.is_set()


@pytest.mark.asyncio
async def test_task_exception_propagates_when_no_cancel():
    trigger = asyncio.Event()

    async def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        await run_until_cancelled(boom, trigger=trigger)


@pytest.mark.asyncio
async def test_outer_cancellation_propagates_to_inner_task():
    """If the caller is cancelled, the inner task must not leak."""
    trigger = asyncio.Event()
    inner_cleanup = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            inner_cleanup.set()
            raise

    async def caller() -> None:
        await run_until_cancelled(slow, trigger=trigger)

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    caller_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller_task
    # Give the inner task a tick to observe its own cancel and run its
    # finally block.
    await asyncio.sleep(0.01)
    assert inner_cleanup.is_set()


# ── Composition with PersistentAgent.chat — no session-store write on cancel


class _SlowLLM:
    """LLM stub that yields one ACTION event slowly so a test can cancel."""

    input_token_budget = 1000
    last_usage = None

    async def complete(self, system, messages, **kwargs):
        # Sleeps long enough for the trigger fire-then-cancel path to win.
        await asyncio.sleep(10)
        return {"thought": "x", "action": "finish", "answer": "x", "confidence": 1.0}


def _persistent_app() -> PersistentAgent:
    llm = _SlowLLM()
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    coordinator = BaseAgent(
        config=AgentConfig(
            agent_id="coordinator",
            role="coordinates",
            system_prompt="You coordinate.",
            allowed_tools=[],
            max_steps=2,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )
    return PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )


@pytest.mark.asyncio
async def test_cancelling_mid_chat_writes_nothing_to_session_store():
    """Mid-stream cancel must leave the transcript untouched — matches
    chat-UX semantic of 'cancelled turn = never happened'."""
    app = _persistent_app()
    trigger = asyncio.Event()
    events: list[BusEvent] = []

    async def _run_turn() -> None:
        async for event in app.chat("hello", session_id="s"):
            events.append(event)

    async def fire_after_first_event() -> None:
        # Wait until the agent has actually started before firing,
        # otherwise the trigger-already-set path masks the real test.
        for _ in range(100):
            await asyncio.sleep(0.005)
            if events or trigger.is_set():
                break
        trigger.set()

    fire_task = asyncio.create_task(fire_after_first_event())
    cancelled, _ = await run_until_cancelled(_run_turn, trigger=trigger)
    await fire_task

    assert cancelled is True

    state = await app.session_state("s")
    assert state.messages == [], (
        "session store should be untouched on cancel — finalize_turn "
        "only runs on TASK_DONE / ERROR / clean stream end"
    )
    assert state.turn_count == 0
    assert state.last_reconcile_turn == 0


# ── consume_with_cancel — lower-level event-iterator helper ────────────────


@pytest.mark.asyncio
async def test_consume_with_cancel_drains_stream_and_returns_false():
    """Stream completes naturally → cancelled is False; every event is delivered."""
    events_in = [
        BusEvent(type=EventType.THOUGHT, agent_id="a"),
        BusEvent(type=EventType.ACTION, agent_id="a"),
        BusEvent(type=EventType.TASK_DONE, agent_id="a", payload={"answer": "ok"}),
    ]

    async def _stream():
        for e in events_in:
            yield e

    received: list[BusEvent] = []
    cancelled = await consume_with_cancel(_stream(), on_event=received.append)

    assert cancelled is False
    assert [e.type for e in received] == [
        EventType.THOUGHT,
        EventType.ACTION,
        EventType.TASK_DONE,
    ]


# ── ConsoleRenderer.render_stream — the renderer-method form ───────────────


@pytest.mark.asyncio
async def test_render_stream_captures_top_level_terminal_event():
    """Sub-agent TASK_DONE has parent_agent_id and must NOT masquerade as the
    outer terminal — the FINAL ANSWER banner would otherwise fire on every
    delegation."""

    async def _stream():
        yield BusEvent(type=EventType.THOUGHT, agent_id="coordinator")
        yield BusEvent(
            type=EventType.TASK_DONE,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"answer": "inner"},
        )
        yield BusEvent(
            type=EventType.TASK_DONE,
            agent_id="coordinator",
            payload={"answer": "outer"},
        )

    renderer = ConsoleRenderer(out=io.StringIO())
    cancelled, terminal = await renderer.render_stream(
        _stream(),
        terminal_event_type=EventType.TASK_DONE,
    )

    assert cancelled is False
    assert terminal is not None
    assert terminal.payload["answer"] == "outer", (
        "render_stream must skip sub-agent terminals (parent_agent_id set) "
        "and return the outermost TASK_DONE"
    )


@pytest.mark.asyncio
async def test_render_stream_returns_none_when_no_terminal_seen():
    """Stream completes without the requested terminal type → terminal=None."""

    async def _stream():
        yield BusEvent(type=EventType.THOUGHT, agent_id="a")
        yield BusEvent(type=EventType.ACTION, agent_id="a")

    renderer = ConsoleRenderer(out=io.StringIO())
    cancelled, terminal = await renderer.render_stream(
        _stream(),
        terminal_event_type=EventType.DONE,
    )

    assert cancelled is False
    assert terminal is None


@pytest.mark.asyncio
async def test_render_stream_with_no_terminal_type_renders_everything():
    """terminal_event_type=None → render every event, return (False, None)."""

    async def _stream():
        yield BusEvent(type=EventType.THOUGHT, agent_id="a")
        yield BusEvent(type=EventType.TASK_DONE, agent_id="a", payload={"answer": "x"})

    renderer = ConsoleRenderer(out=io.StringIO())
    cancelled, terminal = await renderer.render_stream(_stream())

    assert cancelled is False
    assert terminal is None


@pytest.mark.asyncio
async def test_render_stream_top_level_only_false_captures_inner_terminal():
    """Opt-out switch: top_level_only=False captures any matching event,
    sub-agent or not. Useful for tests that want to inspect every terminal."""

    async def _stream():
        yield BusEvent(
            type=EventType.TASK_DONE,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"answer": "inner"},
        )

    renderer = ConsoleRenderer(out=io.StringIO())
    cancelled, terminal = await renderer.render_stream(
        _stream(),
        terminal_event_type=EventType.TASK_DONE,
        top_level_only=False,
    )

    assert cancelled is False
    assert terminal is not None
    assert terminal.payload["answer"] == "inner"
