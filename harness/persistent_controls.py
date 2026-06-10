"""Reusable slash-command controls for ``PersistentAgent`` demos and CLIs.

The command surface is defined once in ``SLASH_COMMAND_SPECS`` and consumed by:
- ``PersistentCommandHandler`` (this module) for dispatch
- ``harness.persistent_completion.SlashCommandCompleter`` for tab-completion
- ``_help_text`` for the ``/help`` body

A spec registry, not three independent lists, keeps those consumers from
drifting as commands are added. Tests assert the dispatch map matches the
spec set exactly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from harness.persistent import PersistentAgent, PersistentAgentConfig, SessionState


@dataclass
class PersistentCommandResult:
    handled: bool
    text: str = ""
    session_id: str | None = None
    should_exit: bool = False


@dataclass(frozen=True)
class SlashCommandSpec:
    """Metadata for one slash command — name, help text, argument shape.

    ``arg_kind`` is the data contract for UIs that complete arguments:

    - ``"none"`` — no arguments
    - ``"session_id"`` — one required session id (e.g. ``/switch``)
    - ``"optional_session_id"`` — one optional session id (e.g. ``/new``)
    - ``"session_id_then_confirm"`` — optional session id + required ``confirm`` (e.g. ``/delete``)
    - ``"query"`` — one optional free-text token (e.g. ``/sessions [query]``)
    - ``"plan_toggle"`` — one optional ``on`` / ``off`` literal (e.g. ``/plan``)
    - ``"model_switch"`` — agent id plus model name/default (e.g. ``/model researcher gpt-5``)
    """

    name: str
    description: str
    category: str = "Other"
    arg_hint: str = ""
    arg_kind: str = "none"
    aliases: tuple[str, ...] = field(default_factory=tuple)


SLASH_COMMAND_SPECS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec("/help", "Show command help", category="Help", aliases=("/?",)),
    SlashCommandSpec(
        "/capabilities",
        "Inspect wired coordinator, sub-agents, MCP tools",
        category="Inspect",
    ),
    SlashCommandSpec("/agents", "List coordinator and sub-agents", category="Inspect"),
    SlashCommandSpec("/mcp", "List wired MCP tools", category="Inspect"),
    SlashCommandSpec(
        "/session",
        "Show current session state + reconcile cadence",
        category="Inspect",
    ),
    SlashCommandSpec(
        "/memory",
        "Show cached memory context for this session",
        category="Inspect",
    ),
    SlashCommandSpec(
        "/usage",
        "Show token usage totals + last-run breakdown",
        category="Inspect",
    ),
    SlashCommandSpec("/models", "List switchable models", category="Inspect"),
    SlashCommandSpec(
        "/model",
        "Show or switch a session-scoped agent model",
        category="Session",
        arg_hint="[agent_id] [model|default]",
        arg_kind="model_switch",
    ),
    SlashCommandSpec(
        "/sessions",
        "List known sessions",
        category="Inspect",
        arg_hint="[query]",
        arg_kind="query",
    ),
    SlashCommandSpec(
        "/save",
        "Reconcile pending turns into long-term memory",
        category="Memory",
    ),
    SlashCommandSpec(
        "/compact",
        "Structural reorg: summary + trim + reconcile",
        category="Memory",
    ),
    SlashCommandSpec(
        "/forget",
        "Drop cached memory context for this session",
        category="Memory",
    ),
    SlashCommandSpec(
        "/plan",
        "Toggle plan-before-execute mode (on / off / status)",
        category="Memory",
        arg_hint="[on|off]",
        arg_kind="plan_toggle",
    ),
    SlashCommandSpec(
        "/switch",
        "Switch to an existing or new logical session",
        category="Session",
        arg_hint="<id>",
        arg_kind="session_id",
    ),
    SlashCommandSpec(
        "/new",
        "Create a new session; refuses if id exists",
        category="Session",
        arg_hint="[id]",
        arg_kind="optional_session_id",
    ),
    SlashCommandSpec(
        "/clear",
        "Clear current transcript; long-term memory retained",
        category="Session",
    ),
    SlashCommandSpec(
        "/delete",
        "Delete a transcript row; long-term memory retained",
        category="Session",
        arg_hint="[id] confirm",
        arg_kind="session_id_then_confirm",
    ),
    SlashCommandSpec(
        "/end",
        "Exit the demo; transcript stays in SQLite",
        category="Session",
    ),
)


def slash_command_specs() -> tuple[SlashCommandSpec, ...]:
    """Return the slash-command registry. Stable across imports."""
    return SLASH_COMMAND_SPECS


_CommandHandler = Callable[..., Awaitable[PersistentCommandResult]]


class PersistentCommandHandler:
    """Map slash commands to ``PersistentAgent`` operations.

    UI glue, not core runtime behavior. Terminal demos can print the
    returned ``text``; web/API frontends should generally call
    ``PersistentAgent`` methods directly.
    """

    def __init__(
        self,
        app: PersistentAgent,
        *,
        config: PersistentAgentConfig | None = None,
    ) -> None:
        self._app = app
        self._config = config or app.config
        self._dispatch: dict[str, _CommandHandler] = self._build_dispatch()

    def _build_dispatch(self) -> dict[str, _CommandHandler]:
        # Map every spec.name (and alias) to the method that handles it.
        # The spec → method link lives only here; tests pin that every
        # spec has a handler and every handler corresponds to a spec.
        method_for_name: dict[str, _CommandHandler] = {
            "/help": self._cmd_help,
            "/capabilities": self._cmd_capabilities,
            "/agents": self._cmd_agents,
            "/mcp": self._cmd_mcp,
            "/session": self._cmd_session,
            "/memory": self._cmd_memory,
            "/usage": self._cmd_usage,
            "/models": self._cmd_models,
            "/model": self._cmd_model,
            "/sessions": self._cmd_sessions,
            "/save": self._cmd_save,
            "/compact": self._cmd_compact,
            "/forget": self._cmd_forget,
            "/plan": self._cmd_plan,
            "/switch": self._cmd_switch,
            "/new": self._cmd_new,
            "/clear": self._cmd_clear,
            "/delete": self._cmd_delete,
            "/end": self._cmd_end,
        }
        dispatch: dict[str, _CommandHandler] = {}
        for spec in SLASH_COMMAND_SPECS:
            handler = method_for_name[spec.name]
            dispatch[spec.name] = handler
            for alias in spec.aliases:
                dispatch[alias] = handler
        return dispatch

    async def handle(self, message: str, *, session_id: str) -> PersistentCommandResult:
        if not message.startswith("/"):
            return PersistentCommandResult(handled=False, session_id=session_id)
        command = message.split(maxsplit=1)[0].lower()
        handler = self._dispatch.get(command)
        if handler is None:
            return self._result(session_id, f"Unknown command: {command}. Try /help.")
        return await handler(message=message, command=command, session_id=session_id)

    def _result(self, session_id: str, text: str) -> PersistentCommandResult:
        return PersistentCommandResult(handled=True, text=text, session_id=session_id)

    # ── Per-command handlers ──────────────────────────────────────────────

    async def _cmd_help(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, _help_text())

    async def _cmd_capabilities(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, _format_capabilities(self._app))

    async def _cmd_agents(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, _format_agents(self._app))

    async def _cmd_mcp(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, _format_mcp(self._app))

    async def _cmd_session(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, await self._format_session(session_id))

    async def _cmd_memory(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, _format_memory(self._app, session_id=session_id))

    async def _cmd_usage(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, await self._format_usage(session_id))

    async def _cmd_models(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return self._result(session_id, _format_models(self._app))

    async def _cmd_model(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        parts = message[len(command) :].strip().split()
        if not parts:
            return self._result(session_id, await _format_model_overrides(self._app, session_id))
        if len(parts) == 1:
            return self._result(
                session_id,
                await _format_model_overrides(self._app, session_id, agent_id=parts[0]),
            )
        if len(parts) != 2:
            return self._result(session_id, "usage: /model [agent_id] [model|default]")
        agent_id, model_name = parts
        try:
            if model_name.lower() in {"default", "reset", "clear"}:
                await self._app.clear_model_override(session_id, agent_id)
                return self._result(
                    session_id,
                    f"model override cleared for {agent_id} in session {session_id}",
                )
            await self._app.switch_model(session_id, agent_id, model_name)
            return self._result(
                session_id,
                f"model override set for {agent_id} in session {session_id}: {model_name}",
            )
        except ValueError as exc:
            return self._result(session_id, f"cannot switch model: {exc}")

    async def _cmd_sessions(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        query = message[len(command) :].strip() or None
        return self._result(
            session_id,
            await _format_sessions(self._app, current_session_id=session_id, query=query),
        )

    async def _cmd_save(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        count = await self._app.save_to_memory(session_id)
        if count:
            return self._result(
                session_id,
                f"saved {count} pending message(s) from {session_id} to long-term memory",
            )
        return self._result(session_id, f"session {session_id} has no pending messages to save")

    async def _cmd_compact(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        state = await self._app.force_compact(session_id)
        return self._result(
            session_id,
            f"compacted session {session_id}; last_compact_turn={state.last_compact_turn}",
        )

    async def _cmd_forget(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        self._app.forget_memory_cache(session_id)
        return self._result(session_id, f"forgot cached memory context for session {session_id}")

    async def _cmd_plan(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        # ``/plan`` → show current state; ``/plan on`` or ``/plan off`` →
        # toggle. Anything else is a usage error. The persistence /
        # SQLite side lives on ``PersistentAgent.set_plan_mode``; this
        # handler is just argument parsing and a friendly message.
        arg = message[len(command) :].strip().lower()
        if arg == "":
            enabled = await self._app.plan_mode_enabled(session_id)
            state = "on" if enabled else "off"
            return self._result(
                session_id,
                f"plan mode: {state}  (toggle with `/plan on` or `/plan off`)",
            )
        if arg in {"on", "true", "1", "yes"}:
            await self._app.set_plan_mode(session_id, True)
            return self._result(
                session_id,
                "plan mode: on — the agent will propose a plan and wait for "
                "approval before each turn",
            )
        if arg in {"off", "false", "0", "no"}:
            await self._app.set_plan_mode(session_id, False)
            return self._result(
                session_id, "plan mode: off — the agent will execute turns directly"
            )
        return self._result(
            session_id,
            f"unrecognised plan argument: {arg!r}. Use `/plan`, `/plan on`, or `/plan off`.",
        )

    async def _cmd_switch(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        try:
            next_session_id = _session_arg(command, message)
        except ValueError as exc:
            return self._result(session_id, f"cannot switch session: {exc}")
        if not next_session_id:
            return self._result(session_id, "usage: /switch <session_id>")
        existed = await self._app.session_exists(next_session_id)
        await self._app.session_state(next_session_id)
        return self._result(
            next_session_id,
            f"{'resumed' if existed else 'created'} session: {next_session_id}",
        )

    async def _cmd_new(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        try:
            requested_session_id = _session_arg(command, message)
        except ValueError as exc:
            return self._result(session_id, f"cannot create session: {exc}")
        next_session_id = requested_session_id or f"sess_{uuid4().hex[:12]}"
        if await self._app.session_exists(next_session_id):
            return self._result(
                session_id,
                f"session already exists: {next_session_id}\n"
                f"use /switch {next_session_id} to resume it",
            )
        await self._app.session_state(next_session_id)
        return self._result(
            next_session_id,
            f"new session: {next_session_id} (previous transcript preserved in SQLite)",
        )

    async def _cmd_clear(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        await self._app.clear_session(session_id)
        return self._result(
            session_id,
            f"cleared session {session_id} transcript; long-term memory retained",
        )

    async def _cmd_delete(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        raw = message[len(command) :].strip()
        parts = raw.split()
        if parts == ["confirm"]:
            delete_session_id = session_id
        elif len(parts) == 2 and parts[1] == "confirm" and not parts[0].startswith("/"):
            delete_session_id = parts[0]
        else:
            return self._result(
                session_id,
                "usage: /delete confirm OR /delete <session_id> confirm\n"
                "deletes transcript only; long-term memory is retained",
            )
        deleted = await self._app.delete_session(delete_session_id)
        if not deleted:
            return self._result(session_id, f"session not found: {delete_session_id}")
        text = f"deleted session {delete_session_id}; long-term memory retained"
        if delete_session_id == session_id:
            next_session_id = "default"
            await self._app.session_state(next_session_id)
            return self._result(next_session_id, f"{text}\nswitched to session: {next_session_id}")
        return self._result(session_id, text)

    async def _cmd_end(
        self, *, message: str, command: str, session_id: str
    ) -> PersistentCommandResult:
        return PersistentCommandResult(
            handled=True,
            text=f"exiting; session {session_id} transcript preserved for next run",
            session_id=session_id,
            should_exit=True,
        )

    # ── Formatters (per-instance because they reference llm/config) ───────

    async def _format_session(self, session_id: str) -> str:
        state = await self._app.session_state(session_id)
        tokens = _estimate_session_tokens(state)
        budget = self._app.context_token_budget()
        if isinstance(budget, int) and budget > 0:
            pct = (tokens / budget) * 100
            token_line = f"{tokens:,} / {budget:,} tokens ({pct:.1f}%)"
        else:
            token_line = f"{tokens:,} estimated tokens"
        interval = self._config.async_reconcile_every_turns
        if interval > 0:
            remaining = interval - (state.turn_count % interval)
            reconcile = f"turn {state.turn_count + remaining} ({remaining} turn(s))"
        else:
            reconcile = "disabled"
        model_line = (
            ", ".join(f"{agent}={model}" for agent, model in sorted(state.model_overrides.items()))
            if state.model_overrides
            else "(none)"
        )
        return "\n".join(
            [
                f"session_id: {state.session_id}",
                f"turns: {state.turn_count}",
                f"transcript: {token_line}",
                f"usage total: in={state.tokens_in_total:,} out={state.tokens_out_total:,}",
                f"last run: in={state.last_run_tokens_in:,} out={state.last_run_tokens_out:,}",
                f"model overrides: {model_line}",
                f"last reconcile turn: {state.last_reconcile_turn or 'never'}",
                f"next async reconcile: {reconcile}",
                f"last compaction turn: {state.last_compact_turn or 'never'}",
                f"summary: {state.summary or '(none)'}",
            ]
        )

    async def _format_usage(self, session_id: str) -> str:
        state = await self._app.session_state(session_id)
        lines = [
            f"session_id: {state.session_id}",
            f"total tokens: in={state.tokens_in_total:,} out={state.tokens_out_total:,}",
            f"last run: in={state.last_run_tokens_in:,} out={state.last_run_tokens_out:,}",
        ]
        breakdown = (
            state.last_usage.get("breakdown") if isinstance(state.last_usage, dict) else None
        )
        if isinstance(breakdown, dict) and breakdown:
            lines.append("last run breakdown:")
            width = max(len(str(name)) for name in breakdown)
            for source, stats in breakdown.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(
                    f"  {source:<{width}} in={int(stats.get('tokens_in') or 0):,} "
                    f"out={int(stats.get('tokens_out') or 0):,}"
                )
        return "\n".join(lines)


def _help_text() -> str:
    # Categorised grid derived from the spec registry so a new spec auto-
    # appears in /help. Order within category follows declaration order.
    by_category: dict[str, list[SlashCommandSpec]] = {}
    for spec in SLASH_COMMAND_SPECS:
        by_category.setdefault(spec.category, []).append(spec)
    order = ("Inspect", "Memory", "Session", "Help", "Other")
    label_width = max(
        len(spec.name + (f" {spec.arg_hint}" if spec.arg_hint else ""))
        for spec in SLASH_COMMAND_SPECS
    )
    lines: list[str] = []
    for category in order:
        specs = by_category.get(category)
        if not specs:
            continue
        for spec in specs:
            label = spec.name + (f" {spec.arg_hint}" if spec.arg_hint else "")
            lines.append(f"  {label:<{label_width}}  {spec.description}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _estimate_session_tokens(state: SessionState) -> int:
    total = max(0, len(state.summary) // 4) if state.summary else 0
    for message in state.messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        total += max(1, len(content) // 4)
    return total


def _format_capabilities(app: PersistentAgent) -> str:
    caps = app.capabilities()
    coordinator = caps["coordinator"]
    lines = [f"Coordinator: {coordinator['agent_id']}"]
    if coordinator.get("skills"):
        lines.append(f"  skills: {_format_skill_names(coordinator['skills'])}")
    for tool in coordinator.get("tools", []):
        sub = next((s for s in caps["subagents"] if s["tool_name"] == tool), None)
        if sub is None:
            lines.append(f"  - {tool}")
            continue
        lines.append(f"  - {tool} -> {sub['agent_id']}")
        if sub.get("skills"):
            lines.append(f"      skills: {_format_skill_names(sub['skills'])}")
        for sub_tool in sub.get("tools", []):
            lines.append(f"      - {sub_tool}")
    return "\n".join(lines)


def _format_agents(app: PersistentAgent) -> str:
    caps = app.capabilities()
    coordinator = caps["coordinator"]
    lines = [f"{coordinator['agent_id']}: {coordinator['role']}"]
    if coordinator.get("skills"):
        lines.append(f"  skills: {_format_skill_names(coordinator['skills'])}")
    for sub in caps["subagents"]:
        lines.append(f"{sub['agent_id']}: {sub['role']} (via {sub['tool_name']})")
        if sub.get("skills"):
            lines.append(f"  skills: {_format_skill_names(sub['skills'])}")
    return "\n".join(lines)


def _format_skill_names(skills: list[dict]) -> str:
    return ", ".join(str(skill["name"]) for skill in skills)


def _format_mcp(app: PersistentAgent) -> str:
    tools = app.capabilities()["mcp_tools"]
    if not tools:
        return "No MCP tools are wired."
    by_owner: dict[str, list[dict]] = {}
    for tool in tools:
        by_owner.setdefault(tool["owner_agent_id"], []).append(tool)
    lines: list[str] = []
    for owner, owned_tools in sorted(by_owner.items()):
        lines.append(f"{owner}: {len(owned_tools)} MCP tools")
        for tool in sorted(owned_tools, key=lambda t: t["name"]):
            schema = tool.get("input_schema") or {}
            props = schema.get("properties") if isinstance(schema, dict) else None
            args = ", ".join(sorted(props)) if isinstance(props, dict) else ""
            suffix = f"({args})" if args else "()"
            description = tool.get("description") or ""
            lines.append(
                f"  - {tool['name']}{suffix}" + (f" — {description}" if description else "")
            )
    return "\n".join(lines)


def _format_models(app: PersistentAgent) -> str:
    models = app.available_models()
    default = app.default_model()
    if not models:
        return "No model registry is configured for this PersistentAgent."
    lines = ["available models:"]
    for model in models:
        suffix = " (default)" if default and model == default else ""
        lines.append(f"  - {model}{suffix}")
    lines.append("switch with: /model <agent_id> <model>")
    return "\n".join(lines)


async def _format_model_overrides(
    app: PersistentAgent,
    session_id: str,
    *,
    agent_id: str | None = None,
) -> str:
    caps = app.capabilities()
    agents = [caps["coordinator"], *caps["subagents"]]
    known_ids = {str(agent["agent_id"]) for agent in agents}
    if agent_id is not None and agent_id not in known_ids:
        return f"unknown agent: {agent_id}"
    overrides = await app.model_overrides(session_id)
    default = app.default_model() or "(construction default)"
    selected = [agent for agent in agents if agent_id is None or agent["agent_id"] == agent_id]
    lines = [f"session model overrides for {session_id}:"]
    for agent in selected:
        aid = str(agent["agent_id"])
        model = overrides.get(aid)
        suffix = f"{model} (override)" if model else default
        lines.append(f"  - {aid}: {suffix}")
    if app.available_models():
        lines.append("available: " + ", ".join(app.available_models()))
    else:
        lines.append("available: (no model registry configured)")
    return "\n".join(lines)


def _format_memory(app: PersistentAgent, *, session_id: str) -> str:
    context = app.cached_memory_context(session_id)
    if context is None:
        return "memory cache: not loaded yet"
    if not context:
        return "memory cache: loaded, no relevant facts or episodes"
    return f"memory cache:\n{context}"


async def _format_sessions(
    app: PersistentAgent,
    *,
    current_session_id: str,
    query: str | None = None,
) -> str:
    sessions = await app.list_sessions(query=query)
    if not sessions:
        return f"No sessions found{f' matching {query!r}' if query else ''}."
    lines: list[str] = []
    for state in sessions:
        marker = "*" if state.session_id == current_session_id else " "
        summary = state.summary or "(no summary)"
        if len(summary) > 80:
            summary = summary[:77] + "..."
        lines.append(
            f"{marker} {state.session_id}  turns={state.turn_count}  updated={state.updated_at}"
        )
        lines.append(f"    {summary}")
    return "\n".join(lines)


def _session_arg(command: str, message: str) -> str:
    arg = message[len(command) :].strip()
    if not arg:
        return ""
    parts = arg.split()
    if len(parts) != 1:
        raise ValueError("session ids cannot contain whitespace")
    if parts[0].startswith("/"):
        raise ValueError("session ids cannot start with '/'")
    return parts[0]
