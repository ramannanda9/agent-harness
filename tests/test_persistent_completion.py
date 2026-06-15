from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from prompt_toolkit.document import Document

from agents.base import AgentConfig, BaseAgent
from harness.persistent import InMemorySessionStore, PersistentAgent, PersistentAgentConfig
from harness.persistent_completion import SlashCommandCompleter
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool


class _LLM:
    input_token_budget = 1000
    last_usage = None

    async def complete(self, system, messages, **kwargs):
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0}


def _app(*, models: bool = False) -> PersistentAgent:
    llm = _LLM()
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
        llm_registry={"fast": lambda: _LLM(), "deep": lambda: _LLM()} if models else None,
        default_model="fast" if models else None,
    )


def _subagent_app() -> PersistentAgent:
    llm = _LLM()
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    sub = BaseAgent(
        config=AgentConfig(
            agent_id="researcher",
            role="researches",
            system_prompt="You research.",
            allowed_tools=[],
            max_steps=2,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )
    delegate = SubAgentTool(sub, name="delegate_researcher")
    coordinator = BaseAgent(
        config=AgentConfig(
            agent_id="coordinator",
            role="coordinates",
            system_prompt="You coordinate.",
            allowed_tools=[delegate.name],
            max_steps=2,
        ),
        tools={delegate.name: delegate},
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


async def _collect(completer: SlashCommandCompleter, text: str) -> list[str]:
    doc = Document(text=text, cursor_position=len(text))
    items: AsyncIterator = completer.get_completions_async(doc, complete_event=None)
    return [c.text async for c in items]


@pytest.mark.asyncio
async def test_completer_ignores_non_slash_input():
    completer = SlashCommandCompleter(_app())

    assert await _collect(completer, "hello") == []
    assert await _collect(completer, "") == []


@pytest.mark.asyncio
async def test_completer_completes_command_names_by_prefix():
    completer = SlashCommandCompleter(_app())

    all_slash = await _collect(completer, "/")
    assert "/help" in all_slash
    assert "/save" in all_slash
    assert "/switch" in all_slash

    sw = await _collect(completer, "/sw")
    assert sw == ["/switch"]


@pytest.mark.asyncio
async def test_completer_completes_command_names_case_insensitively():
    completer = SlashCommandCompleter(_app())

    assert "/sessions" in await _collect(completer, "/SE")


@pytest.mark.asyncio
async def test_completer_lists_session_ids_for_switch():
    app = _app()
    await app.session_state("research")
    await app.session_state("planning")
    completer = SlashCommandCompleter(app)

    completions = await _collect(completer, "/switch ")
    assert set(completions) == {"research", "planning"}


@pytest.mark.asyncio
async def test_completer_filters_session_ids_by_partial_word():
    app = _app()
    await app.session_state("research")
    await app.session_state("planning")
    completer = SlashCommandCompleter(app)

    completions = await _collect(completer, "/switch re")
    assert "research" in completions
    assert "planning" not in completions


@pytest.mark.asyncio
async def test_completer_offers_session_ids_then_confirm_for_delete():
    app = _app()
    await app.session_state("research")
    completer = SlashCommandCompleter(app)

    first_arg = await _collect(completer, "/delete ")
    assert first_arg == ["research"]

    after_id = await _collect(completer, "/delete research ")
    assert after_id == ["confirm"]


@pytest.mark.asyncio
async def test_completer_offers_agents_and_models_for_model_switch():
    completer = SlashCommandCompleter(_app(models=True))

    agents = await _collect(completer, "/model ")
    assert agents == ["coordinator"]

    models = await _collect(completer, "/model coordinator ")
    assert set(models) == {"fast", "deep", "default"}


@pytest.mark.asyncio
async def test_completer_offers_subagents_for_background_tasks():
    completer = SlashCommandCompleter(_subagent_app())

    agents = await _collect(completer, "/background ")

    assert agents == ["researcher"]


@pytest.mark.asyncio
async def test_completer_offers_background_task_actions_and_ids():
    app = _subagent_app()
    session_id = "s"
    task = await app.start_background_subagent(session_id, "researcher", "work")
    other_task = await app.start_background_subagent("other", "researcher", "work")
    completer = SlashCommandCompleter(app, session_id_provider=lambda: session_id)

    actions = await _collect(completer, "/tasks ")
    ids = await _collect(completer, "/tasks collect ")

    assert set(actions) == {"collect", "cancel"}
    assert task.task_id in ids
    assert other_task.task_id not in ids
    assert "all" in ids
    await app.cancel_background_task("s", task.task_id)
    await app.cancel_background_task("other", other_task.task_id)


@pytest.mark.asyncio
async def test_completer_yields_nothing_for_argless_command():
    completer = SlashCommandCompleter(_app())

    assert await _collect(completer, "/save ") == []
    assert await _collect(completer, "/compact more") == []


@pytest.mark.asyncio
async def test_completer_help_alias_dispatches_to_help_spec():
    completer = SlashCommandCompleter(_app())

    # "/?" is an alias of /help; the alias never appears in name-completion
    # (which is sourced from spec.name only), but if the user typed /? and
    # tried to add a second token, the completer should treat it as a
    # no-arg command (arg_kind="none" on /help spec).
    assert await _collect(completer, "/? more") == []
