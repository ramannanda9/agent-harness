"""prompt_toolkit ``PromptSession`` factory for ``PersistentAgent`` chat UIs.

The slash-command spec + completer live in ``persistent_controls`` /
``persistent_completion``. This module handles the third piece of the
input layer: assembling a ``PromptSession`` with the chat-style key
bindings (Enter to submit, Ctrl+J / Esc-Enter for newline), command
completion, and persistent history, so demos and downstream CLIs don't
re-derive the same wiring.

Why a helper, not a per-demo block: multi-line input, slash-command
completion, and command history are baseline expectations of a chat
CLI — not demo-specific UX choices. Encoding them in the library
means every consumer gets the same defaults and the bug surface stays
in one place.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import has_completions
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from harness.persistent_completion import SlashCommandCompleter

if TYPE_CHECKING:
    from prompt_toolkit.completion import Completer

    from harness.persistent import PersistentAgent


def build_chat_prompt_session(
    app: PersistentAgent | None = None,
    *,
    history_path: str | Path | None = None,
    completer: Completer | None = None,
    multiline: bool = True,
    complete_while_typing: bool = False,
    extra_key_bindings: KeyBindings | None = None,
    plan_mode_toggle: Callable[[], Awaitable[bool]] | None = None,
) -> PromptSession[str]:
    """Construct a ``PromptSession`` suitable for multi-turn chat input.

    Defaults (each one overridable via the corresponding kwarg):

    - ``multiline=True``: buffer can hold and render multiple lines.
    - **Enter** submits — overrides prompt_toolkit's default in multiline
      mode (which inserts a newline) so the prompt feels like a regular
      chat input. Yielded to the completer dropdown when one is open, so
      Enter still picks a Tab-completion candidate before being treated
      as submit.
    - **Ctrl+J** and **Esc-Enter** (Meta-Enter / Alt-Enter) insert a
      literal newline. Two bindings because some terminals strip Esc
      modifiers, and Ctrl+J is the convention modern AI CLIs use.
    - **Shift-Tab** toggles plan mode when ``plan_mode_toggle`` is given —
      matches the Claude Code convention. The callable is invoked async
      and should return the new bool state; the helper prints a
      ``[plan mode: on|off]`` confirmation via ``run_in_terminal`` so the
      prompt doesn't get torn up by the print. Pass ``plan_mode_toggle``
      a thunk that reads the *live* current session id (sessions change
      mid-demo via ``/switch`` / ``/new`` / ``/delete``); see the persistent
      demo for the canonical wiring.
    - ``complete_while_typing=False``: Tab-triggered completion. The
      session-id completer otherwise hits the store on every keystroke.
    - ``completer=SlashCommandCompleter(app)`` when ``app`` is given.
      Pass ``completer=...`` to override (e.g. a richer completer that
      composes the slash-command one with something else); pass
      ``completer=None`` *and* ``app=None`` to disable completion
      entirely (single-input demos that don't need slash commands).
    - History at ``history_path`` when provided
      (``FileHistory(history_path)``). Parent directory is created on
      demand. Pass ``history_path=None`` for ephemeral in-memory history.

    ``extra_key_bindings`` are merged after the chat defaults, so user
    bindings can override them.
    """
    bindings = _chat_key_bindings(plan_mode_toggle=plan_mode_toggle)
    if extra_key_bindings is not None:
        # KeyBindings.add_binding via merge: prompt_toolkit's
        # merge_key_bindings is the proper composition primitive when
        # the caller wants to extend without losing defaults.
        from prompt_toolkit.key_binding import merge_key_bindings  # noqa: PLC0415

        merged = merge_key_bindings([bindings, extra_key_bindings])
    else:
        merged = bindings

    if completer is None and app is not None:
        completer = SlashCommandCompleter(app)

    history = _resolve_history(history_path)

    return PromptSession(
        history=history,
        completer=completer,
        complete_while_typing=complete_while_typing,
        multiline=multiline,
        key_bindings=merged,
    )


def _chat_key_bindings(
    *,
    plan_mode_toggle: Callable[[], Awaitable[bool]] | None = None,
) -> KeyBindings:
    """The default keybinding set for chat-style multi-line input.

    Kept as a module-level factory so tests can introspect the bindings
    independently of constructing a full ``PromptSession``.
    """
    bindings = KeyBindings()

    @bindings.add("enter", filter=~has_completions)
    def _submit(event: Any) -> None:
        # In multiline mode prompt_toolkit's default for Enter is
        # "insert newline". Override to submit — but only when the
        # completer dropdown isn't open. ``~has_completions`` yields
        # Enter to the completer when it's showing, so Tab-then-Enter
        # picks a completion without accidentally submitting.
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _ctrl_j_newline(event: Any) -> None:
        # Ctrl+J is the modern-AI-CLI convention for "newline within
        # the current message." On Unix terminals Ctrl+J is literally
        # the line-feed byte; in single-line mode that's
        # indistinguishable from Enter, which is one of the reasons
        # this helper defaults to ``multiline=True``.
        event.current_buffer.insert_text("\n")

    @bindings.add("escape", "enter")
    def _meta_enter_newline(event: Any) -> None:
        # Alt-Enter / Meta-Enter / Esc-Enter — common convention from
        # editors and other REPLs. Bound alongside Ctrl+J because some
        # terminals strip the Esc modifier and some users have one or
        # the other in muscle memory.
        event.current_buffer.insert_text("\n")

    if plan_mode_toggle is not None:

        @bindings.add("s-tab")
        def _shift_tab_plan_toggle(event: Any) -> None:
            # Claude Code convention: Shift-Tab cycles plan mode. The
            # toggle hits the session store (async); prompt_toolkit
            # keybindings are sync, so we schedule the work on the
            # running event loop. Output goes through ``run_in_terminal``
            # so it doesn't tear up the active prompt rendering — the
            # prompt suspends, the line prints, the prompt re-renders.
            async def _do_toggle() -> None:
                try:
                    new_state = await plan_mode_toggle()
                except Exception as exc:  # noqa: BLE001 — keybinding must not crash the loop
                    # ``exc`` bound via default arg so the closure
                    # captures *this* exception, not whatever the
                    # variable might be after the except scope ends.
                    await run_in_terminal(
                        lambda exc=exc: print(f"\n[plan mode toggle failed: {exc}]")
                    )
                    return
                label = "on" if new_state else "off"
                await run_in_terminal(lambda: print(f"\n[plan mode: {label}]"))

            event.app.create_background_task(_do_toggle())

    return bindings


def _resolve_history(history_path: str | Path | None) -> History:
    if history_path is None:
        return InMemoryHistory()
    path = Path(history_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(path))


def _binding_handler_names(bindings: KeyBindings) -> Iterable[str]:
    """Return the registered handler function names — useful for tests
    asserting "Enter is bound to submit, Ctrl+J is bound to newline" without
    needing to spin up a full prompt_toolkit Application.

    Public for tests; not part of the supported API surface.
    """
    for binding in bindings.bindings:
        yield binding.handler.__name__
