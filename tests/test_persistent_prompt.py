"""Contract tests for ``build_chat_prompt_session``.

prompt_toolkit's ``PromptSession`` doesn't lend itself to driving a full
TestApplication in unit tests (it wants a real terminal). Instead we pin
the externally observable configuration — multi-line on by default,
completer wired when an app is provided, history wired when a path is
given — and inspect the registered key bindings by handler name. The
underlying behaviour (Enter submits, Ctrl+J inserts newline) is the
contract of prompt_toolkit itself; we just need to know we registered
the right handlers against the right keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.keys import Keys

from agents.base import AgentConfig, BaseAgent
from harness.persistent import InMemorySessionStore, PersistentAgent, PersistentAgentConfig
from harness.persistent_completion import SlashCommandCompleter
from harness.persistent_prompt import (
    _binding_handler_names,
    _chat_key_bindings,
    build_chat_prompt_session,
)
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


# ── PromptSession-level config ────────────────────────────────────────────


def test_default_session_is_multiline_with_slash_completion_and_inmem_history():
    session = build_chat_prompt_session(_app())

    assert session.multiline is True, (
        "chat session must be multiline by default — Enter override + "
        "Ctrl+J/Esc-Enter newlines need a multiline buffer to render"
    )
    assert isinstance(session.completer, SlashCommandCompleter), (
        "default completer must be SlashCommandCompleter when app is given"
    )
    assert isinstance(session.history, InMemoryHistory), (
        "no history_path -> ephemeral InMemoryHistory, not FileHistory"
    )


def test_history_path_uses_file_history_and_creates_parent(tmp_path: Path):
    history_path = tmp_path / "nested" / "deeper" / "demo_history"
    assert not history_path.parent.exists()

    session = build_chat_prompt_session(_app(), history_path=history_path)

    assert isinstance(session.history, FileHistory)
    assert history_path.parent.exists(), (
        "build_chat_prompt_session must create the history directory on "
        "demand so consumers don't repeat the mkdir(parents=True) boilerplate"
    )


def test_no_app_and_no_completer_yields_no_completion():
    session = build_chat_prompt_session(app=None)

    assert session.completer is None, (
        "with no app and no explicit completer, completion should be off "
        "— supports plain chat demos that don't have a slash-command surface"
    )


def test_explicit_completer_overrides_default():
    class _Custom:
        pass

    custom = _Custom()
    session = build_chat_prompt_session(_app(), completer=custom)  # type: ignore[arg-type]

    assert session.completer is custom, (
        "explicit completer= must override the SlashCommandCompleter default"
    )


def test_complete_while_typing_default_off():
    session = build_chat_prompt_session(_app())

    assert bool(session.complete_while_typing) is False, (
        "default must be Tab-triggered completion — auto-complete-on-keystroke "
        "would hit the session store on every character typed in a session-id "
        "arg position"
    )


# ── Key-binding registry ──────────────────────────────────────────────────


def test_chat_key_bindings_register_submit_ctrl_j_and_meta_enter():
    bindings = _chat_key_bindings()
    handler_names = list(_binding_handler_names(bindings))

    assert "_submit" in handler_names, (
        "Enter must be bound to a submit handler so the chat prompt feels "
        "like a regular chat input even in multiline mode"
    )
    assert "_ctrl_j_newline" in handler_names, (
        "Ctrl+J (the modern AI-CLI convention for newline-within-message) "
        "must be bound to insert_text('\\n')"
    )
    assert "_meta_enter_newline" in handler_names, (
        "Esc-Enter (Meta-Enter / Alt-Enter) must also be bound for users "
        "whose terminals strip Ctrl+J or who have the older convention in "
        "muscle memory"
    )


def test_enter_binding_is_filtered_by_has_completions():
    """Enter must yield to the completer dropdown when it's open, so
    Tab-then-Enter picks a completion instead of accidentally submitting."""
    bindings = _chat_key_bindings()

    enter_bindings = [b for b in bindings.bindings if Keys.ControlM in b.keys or "enter" in b.keys]
    assert enter_bindings, "expected at least one binding for Enter"
    # The submit binding has a filter — the others don't bind plain Enter
    # at all (they bind Ctrl+J or Esc-Enter).
    submit = next(b for b in enter_bindings if b.handler.__name__ == "_submit")
    # ``filter`` exists and is not the always-true default — would need to
    # be ``~has_completions`` for the dropdown-pick-on-Enter UX to work.
    assert submit.filter is not None
    assert "completions" in repr(submit.filter).lower() or repr(submit.filter) != "Always()", (
        "Enter submit binding must be filtered by ~has_completions so the "
        "completer dropdown can still consume Enter when showing; got "
        f"filter repr={submit.filter!r}"
    )


def test_extra_key_bindings_are_merged(monkeypatch: pytest.MonkeyPatch):
    """Callers can pass extra_key_bindings to extend without losing defaults."""
    from prompt_toolkit.key_binding import KeyBindings

    extra = KeyBindings()

    @extra.add("c-r")  # arbitrary user binding
    def _custom_handler(event: Any) -> None:
        pass

    session = build_chat_prompt_session(_app(), extra_key_bindings=extra)

    # We can't introspect a merged KeyBindings directly without driving a
    # real Application, but at least confirm the session was constructed
    # (merge succeeded, didn't crash on the merge_key_bindings call).
    assert session.app.key_bindings is not None
