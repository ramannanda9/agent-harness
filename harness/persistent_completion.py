"""prompt_toolkit tab-completion for ``PersistentCommandHandler`` slash commands.

Binds the UI-agnostic ``slash_command_specs()`` registry from
``harness.persistent_controls`` to prompt_toolkit's ``Completer`` interface.
Other UIs (web autocomplete, fzf, IDE plugins) consume the spec registry
directly and don't import this module.

Session-id argument completion calls ``app.list_sessions(query=...)``;
the SQLite store pushes the filter down to ``lower(session_id) LIKE ?``
so the per-keystroke cost stays bounded even with many sessions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from prompt_toolkit.completion import Completer, Completion

from harness.persistent_controls import SlashCommandSpec, slash_command_specs

if TYPE_CHECKING:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from harness.persistent import PersistentAgent


class SlashCommandCompleter(Completer):
    """Tab-complete ``/<command>`` names and their arguments.

    Name completion is keyed off ``slash_command_specs()``. Argument
    completion dispatches on each spec's ``arg_kind`` — session-id
    arguments query the store; ``confirm`` arguments propose the literal.
    """

    def __init__(self, app: PersistentAgent) -> None:
        self._app = app
        # Pre-expand aliases for O(1) lookup during arg completion.
        self._spec_by_name: dict[str, SlashCommandSpec] = {}
        for spec in slash_command_specs():
            self._spec_by_name[spec.name] = spec
            for alias in spec.aliases:
                self._spec_by_name[alias] = spec

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Any:
        # prompt_toolkit prefers ``get_completions_async`` when defined.
        # The sync method is required by the abstract base class but unused.
        return iter(())

    async def get_completions_async(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> AsyncIterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if " " not in text:
            # Still typing the command name (or just typed "/").
            lower = text.lower()
            for spec in slash_command_specs():
                if spec.name.startswith(lower):
                    yield Completion(
                        spec.name,
                        start_position=-len(text),
                        display_meta=spec.description,
                    )
            return
        # Command + at least one space → argument completion.
        command, _, rest = text.partition(" ")
        spec = self._spec_by_name.get(command.lower())
        if spec is None or spec.arg_kind == "none":
            return
        words = rest.split()
        if text.endswith(" "):
            current_word = ""
            arg_index = len(words)
        else:
            current_word = words[-1] if words else ""
            arg_index = max(0, len(words) - 1)
        async for completion in self._complete_arg(spec, current_word, arg_index):
            yield completion

    async def _complete_arg(
        self,
        spec: SlashCommandSpec,
        current_word: str,
        arg_index: int,
    ) -> AsyncIterator[Completion]:
        if spec.arg_kind in {"session_id", "optional_session_id"}:
            if arg_index != 0:
                return
            async for completion in self._session_id_completions(current_word):
                yield completion
        elif spec.arg_kind == "session_id_then_confirm":
            if arg_index == 0:
                async for completion in self._session_id_completions(current_word):
                    yield completion
            elif arg_index == 1 and "confirm".startswith(current_word):
                yield Completion(
                    "confirm",
                    start_position=-len(current_word),
                    display_meta="confirm deletion",
                )
        elif spec.arg_kind == "plan_toggle":
            if arg_index != 0:
                return
            for literal in ("on", "off"):
                if literal.startswith(current_word):
                    yield Completion(
                        literal,
                        start_position=-len(current_word),
                        display_meta="enable" if literal == "on" else "disable",
                    )
        elif spec.arg_kind == "model_switch":
            if arg_index == 0:
                for agent_id in self._agent_ids():
                    if agent_id.startswith(current_word):
                        yield Completion(
                            agent_id,
                            start_position=-len(current_word),
                            display_meta="agent",
                        )
            elif arg_index == 1:
                for model in [*self._app.available_models(), "default"]:
                    if model.startswith(current_word):
                        yield Completion(
                            model,
                            start_position=-len(current_word),
                            display_meta="model" if model != "default" else "clear override",
                        )
        elif spec.arg_kind == "query":
            # Free text — suggest session ids as a hint for /sessions filter.
            if arg_index == 0:
                async for completion in self._session_id_completions(current_word):
                    yield completion

    def _agent_ids(self) -> list[str]:
        try:
            caps = self._app.capabilities()
        except Exception:  # noqa: BLE001 — completer must never raise into the prompt loop
            return []
        agents = [caps.get("coordinator", {}), *caps.get("subagents", [])]
        return sorted(str(agent.get("agent_id")) for agent in agents if agent.get("agent_id"))

    async def _session_id_completions(
        self,
        prefix: str,
    ) -> AsyncIterator[Completion]:
        try:
            sessions = await self._app.list_sessions(query=prefix or None)
        except Exception:  # noqa: BLE001 — completer must never raise into the prompt loop
            return
        for state in sessions:
            yield Completion(
                state.session_id,
                start_position=-len(prefix),
                display_meta=f"turns={state.turn_count}",
            )
