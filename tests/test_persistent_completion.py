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


class _LLM:
    input_token_budget = 1000
    last_usage = None

    async def complete(self, system, messages, **kwargs):
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0}


def _app() -> PersistentAgent:
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
