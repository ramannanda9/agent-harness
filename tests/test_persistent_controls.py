from __future__ import annotations

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.persistent import (
    InMemorySessionStore,
    PersistentAgent,
    PersistentAgentConfig,
    SessionMessage,
)
from harness.persistent_controls import PersistentCommandHandler
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore


class _LLM:
    input_token_budget = 1000
    last_usage = None

    async def complete(self, system, messages, **kwargs):
        return {
            "thought": "done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }


class _Memory(MemoryManager):
    def __init__(self, llm):
        super().__init__(
            semantic_store=InMemorySemanticStore(),
            episodic_store=InMemoryEpisodicStore(),
            llm=llm,
        )
        self.writes = []

    async def write_run_end(self, goal: str, agent_results: list[dict], trace: list[dict]):
        self.writes.append({"goal": goal, "agent_results": agent_results, "trace": trace})
        return await super().write_run_end(goal, agent_results, trace)


def _app():
    llm = _LLM()
    memory = _Memory(llm)
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
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )
    return app, memory, llm


@pytest.mark.asyncio
async def test_command_handler_ignores_non_commands():
    app, _, llm = _app()
    handler = PersistentCommandHandler(app)

    result = await handler.handle("hello", session_id="default")

    assert result.handled is False
    assert result.session_id == "default"


@pytest.mark.asyncio
async def test_command_handler_switch_new_sessions_and_filters():
    app, _, llm = _app()
    handler = PersistentCommandHandler(app)

    created = await handler.handle("/new research", session_id="default")
    assert created.handled is True
    assert created.session_id == "research"
    assert "new session: research" in created.text

    duplicate = await handler.handle("/new research", session_id="default")
    assert duplicate.session_id == "default"
    assert "session already exists: research" in duplicate.text

    listed = await handler.handle("/sessions sea", session_id="research")
    assert "* research" in listed.text

    switched = await handler.handle("/switch planning", session_id="research")
    assert switched.session_id == "planning"
    assert "created session: planning" in switched.text


@pytest.mark.asyncio
async def test_command_handler_save_clear_delete_and_end():
    app, memory, llm = _app()
    handler = PersistentCommandHandler(app)
    await app.session_state("work")
    await app._session_store.append_messages(
        "work",
        [
            SessionMessage(role="user", content="remember x"),
            SessionMessage(role="assistant", content="ok"),
        ],
    )

    saved = await handler.handle("/save", session_id="work")
    assert "saved 2 pending" in saved.text
    assert memory.writes

    cleared = await handler.handle("/clear", session_id="work")
    assert "cleared session work" in cleared.text
    assert (await app.session_state("work")).messages == []

    blocked_delete = await handler.handle("/delete work", session_id="work")
    assert "usage:" in blocked_delete.text

    deleted = await handler.handle("/delete confirm", session_id="work")
    assert deleted.session_id == "default"
    assert "deleted session work" in deleted.text

    ended = await handler.handle("/end", session_id="default")
    assert ended.should_exit is True


@pytest.mark.asyncio
async def test_command_handler_usage_displays_session_totals():
    app, _, _llm = _app()
    handler = PersistentCommandHandler(app)
    await app.session_state("usage")
    await app._session_store.record_usage(
        "usage",
        tokens_in=123,
        tokens_out=45,
        usage={
            "tokens_in": 123,
            "tokens_out": 45,
            "breakdown": {"agent:coordinator": {"tokens_in": 123, "tokens_out": 45}},
        },
    )

    result = await handler.handle("/usage", session_id="usage")

    assert "total tokens: in=123 out=45" in result.text
    assert "agent:coordinator" in result.text
