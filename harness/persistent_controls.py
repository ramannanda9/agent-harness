"""Reusable slash-command controls for ``PersistentAgent`` demos and CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from harness.persistent import PersistentAgent, PersistentAgentConfig, SessionState


@dataclass
class PersistentCommandResult:
    handled: bool
    text: str = ""
    session_id: str | None = None
    should_exit: bool = False


class PersistentCommandHandler:
    """Map simple slash commands to ``PersistentAgent`` operations.

    This is UI glue, not core runtime behavior. Terminal demos can print the
    returned text; web/API frontends should generally call ``PersistentAgent``
    methods directly.
    """

    def __init__(
        self,
        app: PersistentAgent,
        *,
        config: PersistentAgentConfig | None = None,
        llm: Any | None = None,
    ) -> None:
        self._app = app
        self._config = config or app.config
        self._llm = llm or app.llm

    async def handle(self, message: str, *, session_id: str) -> PersistentCommandResult:
        if not message.startswith("/"):
            return PersistentCommandResult(handled=False, session_id=session_id)
        command = message.split(maxsplit=1)[0].lower()

        if command in {"/help", "/?"}:
            return self._result(session_id, _help_text())
        if command == "/capabilities":
            return self._result(session_id, _format_capabilities(self._app))
        if command == "/agents":
            return self._result(session_id, _format_agents(self._app))
        if command == "/mcp":
            return self._result(session_id, _format_mcp(self._app))
        if command == "/session":
            return self._result(session_id, await self._format_session(session_id))
        if command == "/memory":
            return self._result(session_id, _format_memory(self._app, session_id=session_id))
        if command == "/usage":
            return self._result(session_id, await self._format_usage(session_id))
        if command == "/sessions":
            query = message[len(command) :].strip() or None
            return self._result(
                session_id,
                await _format_sessions(self._app, current_session_id=session_id, query=query),
            )
        if command == "/save":
            count = await self._app.save_to_memory(session_id)
            if count:
                return self._result(
                    session_id,
                    f"saved {count} pending message(s) from {session_id} to long-term memory",
                )
            return self._result(session_id, f"session {session_id} has no pending messages to save")
        if command == "/compact":
            state = await self._app.force_compact(session_id)
            return self._result(
                session_id,
                f"compacted session {session_id}; last_compact_turn={state.last_compact_turn}",
            )
        if command == "/forget":
            self._app.forget_memory_cache(session_id)
            return self._result(
                session_id, f"forgot cached memory context for session {session_id}"
            )
        if command == "/switch":
            return await self._switch(message, command=command, session_id=session_id)
        if command == "/new":
            return await self._new(message, command=command, session_id=session_id)
        if command == "/clear":
            await self._app.clear_session(session_id)
            return self._result(
                session_id,
                f"cleared session {session_id} transcript; long-term memory retained",
            )
        if command == "/delete":
            return await self._delete(message, command=command, session_id=session_id)
        if command == "/end":
            return PersistentCommandResult(
                handled=True,
                text=f"exiting; session {session_id} transcript preserved for next run",
                session_id=session_id,
                should_exit=True,
            )
        return self._result(session_id, f"Unknown command: {command}. Try /help.")

    def _result(self, session_id: str, text: str) -> PersistentCommandResult:
        return PersistentCommandResult(handled=True, text=text, session_id=session_id)

    async def _format_session(self, session_id: str) -> str:
        state = await self._app.session_state(session_id)
        tokens = _estimate_session_tokens(state)
        budget = getattr(self._llm, "input_token_budget", None)
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
        return "\n".join(
            [
                f"session_id: {state.session_id}",
                f"turns: {state.turn_count}",
                f"transcript: {token_line}",
                f"usage total: in={state.tokens_in_total:,} out={state.tokens_out_total:,}",
                f"last run: in={state.last_run_tokens_in:,} out={state.last_run_tokens_out:,}",
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

    async def _switch(
        self,
        message: str,
        *,
        command: str,
        session_id: str,
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

    async def _new(
        self,
        message: str,
        *,
        command: str,
        session_id: str,
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

    async def _delete(
        self,
        message: str,
        *,
        command: str,
        session_id: str,
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


def _help_text() -> str:
    return "\n".join(
        [
            "Inspect:  /capabilities, /agents, /mcp, /session, /memory, /usage",
            "Memory:   /save     reconcile recent turns into long-term memory",
            "          /compact  structural reorg: summary + trim + reconcile",
            "          /forget   drop this session's cached memory prior",
            "Session:  /sessions [query] list known sessions",
            "          /switch <id> switch to an existing or new logical session",
            "          /new [id] create a new session; refuses if id exists",
            "          /clear    clear current transcript; memory retained",
            "          /delete [id] confirm delete transcript; memory retained",
            "          /end      exit the demo; transcript stays in SQLite",
        ]
    )


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
    for tool in coordinator.get("tools", []):
        sub = next((s for s in caps["subagents"] if s["tool_name"] == tool), None)
        if sub is None:
            lines.append(f"  - {tool}")
            continue
        lines.append(f"  - {tool} -> {sub['agent_id']}")
        for sub_tool in sub.get("tools", []):
            lines.append(f"      - {sub_tool}")
    return "\n".join(lines)


def _format_agents(app: PersistentAgent) -> str:
    caps = app.capabilities()
    coordinator = caps["coordinator"]
    lines = [f"{coordinator['agent_id']}: {coordinator['role']}"]
    for sub in caps["subagents"]:
        lines.append(f"{sub['agent_id']}: {sub['role']} (via {sub['tool_name']})")
    return "\n".join(lines)


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
