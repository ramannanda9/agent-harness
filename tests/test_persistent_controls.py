from __future__ import annotations

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.persistent import (
    InMemorySessionStore,
    PersistentAgent,
    PersistentAgentConfig,
    SessionMessage,
)
from harness.persistent_controls import (
    PersistentCommandHandler,
    slash_command_specs,
)
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


class _NamedLLM(_LLM):
    def __init__(self, name: str) -> None:
        self.name = name

    async def complete(self, system, messages, **kwargs):
        return {
            "thought": f"using {self.name}",
            "action": "finish",
            "answer": self.name,
            "confidence": 1.0,
        }


class _BudgetLLM(_NamedLLM):
    def __init__(self, name: str, budget: int) -> None:
        super().__init__(name)
        self.input_token_budget = budget


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


def _model_app():
    llm = _NamedLLM("default")
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
        llm_registry={"fast": lambda: _NamedLLM("fast"), "deep": lambda: _NamedLLM("deep")},
        default_model="fast",
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


@pytest.mark.asyncio
async def test_command_handler_lists_and_switches_models():
    app, _, _ = _model_app()
    handler = PersistentCommandHandler(app)

    models = await handler.handle("/models", session_id="s")
    assert "fast" in models.text
    assert "deep" in models.text

    status = await handler.handle("/model", session_id="s")
    assert "coordinator: fast" in status.text

    switched = await handler.handle("/model coordinator fast", session_id="s")
    assert "model override set" in switched.text
    assert await app.model_overrides("s") == {"coordinator": "fast"}

    session = await handler.handle("/session", session_id="s")
    assert "model overrides: coordinator=fast" in session.text

    agent_status = await handler.handle("/model coordinator", session_id="s")
    assert "coordinator: fast (override)" in agent_status.text

    reset = await handler.handle("/model coordinator default", session_id="s")
    assert "model override cleared" in reset.text
    assert await app.model_overrides("s") == {}


@pytest.mark.asyncio
async def test_command_handler_session_uses_live_context_budget_after_model_switch():
    llm = _BudgetLLM("fast", 1_000)
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
        llm=_BudgetLLM("control", 10),
        llm_registry={"deep": lambda: _BudgetLLM("deep", 2_000)},
    )
    handler = PersistentCommandHandler(app)

    await handler.handle("/model coordinator deep", session_id="s")
    session = await handler.handle("/session", session_id="s")

    assert "/ 2,000 tokens" in session.text


def test_slash_command_specs_match_dispatch_keys():
    """Every spec.name (and alias) has a handler; no handler is unspecced."""
    app, _, _ = _app()
    handler = PersistentCommandHandler(app)
    spec_keys: set[str] = set()
    for spec in slash_command_specs():
        spec_keys.add(spec.name)
        spec_keys.update(spec.aliases)
    assert spec_keys == set(handler._dispatch)


@pytest.mark.asyncio
async def test_help_text_mentions_every_spec_name():
    """Help body is derived from specs, so each command must appear."""
    app, _, _ = _app()
    handler = PersistentCommandHandler(app)

    result = await handler.handle("/help", session_id="x")
    for spec in slash_command_specs():
        assert spec.name in result.text, f"missing {spec.name} in /help body"


@pytest.mark.asyncio
async def test_help_alias_question_mark_works():
    app, _, _ = _app()
    handler = PersistentCommandHandler(app)

    canonical = await handler.handle("/help", session_id="x")
    alias = await handler.handle("/?", session_id="x")

    assert canonical.text == alias.text
    assert canonical.handled is True
