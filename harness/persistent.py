"""Persistent chat wrapper for agent-harness agents.

The wrapper keeps chat/session state durable while leaving agent construction
to callers. A user can wire any ``BaseAgent`` as the coordinator, including
sub-agents and MCP tools, then use ``PersistentAgent.chat`` for one user turn
at a time.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from agents.base import BaseAgent
from harness.events import BusEvent, EventType
from memory.manager import MemoryManager


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMessage:
    role: str
    content: str
    created_at: str = field(default_factory=_now)


@dataclass
class SessionState:
    session_id: str
    summary: str = ""
    messages: list[SessionMessage] = field(default_factory=list)
    turn_count: int = 0
    last_reconcile_turn: int = 0
    last_compact_turn: int = 0
    updated_at: str = field(default_factory=_now)


class SessionStore(Protocol):
    async def load(self, session_id: str) -> SessionState: ...

    async def append_messages(
        self, session_id: str, messages: Sequence[SessionMessage]
    ) -> SessionState: ...

    async def update_summary(self, session_id: str, summary: str) -> SessionState: ...

    async def trim_messages(self, session_id: str, keep_last: int) -> SessionState: ...

    async def mark_reconciled(self, session_id: str, turn_count: int) -> SessionState: ...

    async def mark_compacted(self, session_id: str, turn_count: int) -> SessionState: ...


class InMemorySessionStore:
    """Small test/demo session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    async def load(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
        return _copy_state(state)

    async def append_messages(
        self, session_id: str, messages: Sequence[SessionMessage]
    ) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.messages.extend(messages)
        state.turn_count += sum(1 for m in messages if m.role == "user")
        state.updated_at = _now()
        return _copy_state(state)

    async def update_summary(self, session_id: str, summary: str) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.summary = summary
        state.updated_at = _now()
        return _copy_state(state)

    async def trim_messages(self, session_id: str, keep_last: int) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        if keep_last >= 0:
            state.messages = state.messages[-keep_last:] if keep_last else []
        state.updated_at = _now()
        return _copy_state(state)

    async def mark_reconciled(self, session_id: str, turn_count: int) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.last_reconcile_turn = turn_count
        state.updated_at = _now()
        return _copy_state(state)

    async def mark_compacted(self, session_id: str, turn_count: int) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.last_compact_turn = turn_count
        state.updated_at = _now()
        return _copy_state(state)


class SQLiteSessionStore:
    """Durable local session store using stdlib SQLite."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._ready = False

    async def load(self, session_id: str) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            return self._load_locked(conn, session_id)

    async def append_messages(
        self, session_id: str, messages: Sequence[SessionMessage]
    ) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            for message in messages:
                conn.execute(
                    """
                    INSERT INTO session_messages(session_id, role, content_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        message.role,
                        json.dumps(message.content),
                        message.created_at,
                    ),
                )
            user_turns = sum(1 for m in messages if m.role == "user")
            conn.execute(
                """
                UPDATE sessions
                SET turn_count = turn_count + ?, updated_at = ?
                WHERE session_id = ?
                """,
                (user_turns, _now(), session_id),
            )
            return self._load_locked(conn, session_id)

    async def update_summary(self, session_id: str, summary: str) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, _now(), session_id),
            )
            return self._load_locked(conn, session_id)

    async def trim_messages(self, session_id: str, keep_last: int) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            if keep_last <= 0:
                conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            else:
                conn.execute(
                    """
                    DELETE FROM session_messages
                    WHERE session_id = ?
                      AND id NOT IN (
                        SELECT id FROM session_messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                      )
                    """,
                    (session_id, session_id, keep_last),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (_now(), session_id),
            )
            return self._load_locked(conn, session_id)

    async def mark_reconciled(self, session_id: str, turn_count: int) -> SessionState:
        return await self._mark(session_id, "last_reconcile_turn", turn_count)

    async def mark_compacted(self, session_id: str, turn_count: int) -> SessionState:
        return await self._mark(session_id, "last_compact_turn", turn_count)

    async def _mark(self, session_id: str, column: str, turn_count: int) -> SessionState:
        self._ensure_schema()
        if column not in {"last_reconcile_turn", "last_compact_turn"}:
            raise ValueError(f"unsupported session marker {column!r}")
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                f"UPDATE sessions SET {column} = ?, updated_at = ? WHERE session_id = ?",
                (turn_count, _now(), session_id),
            )
            return self._load_locked(conn, session_id)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    last_reconcile_turn INTEGER NOT NULL DEFAULT 0,
                    last_compact_turn INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
                ON session_messages(session_id, id)
                """
            )
        self._ready = True

    def _ensure_session(self, conn: sqlite3.Connection, session_id: str) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions(session_id, updated_at)
            VALUES (?, ?)
            """,
            (session_id, _now()),
        )

    def _load_locked(self, conn: sqlite3.Connection, session_id: str) -> SessionState:
        row = conn.execute(
            """
            SELECT session_id, summary, turn_count, last_reconcile_turn,
                   last_compact_turn, updated_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        messages = [
            SessionMessage(
                role=msg["role"],
                content=json.loads(msg["content_json"]),
                created_at=msg["created_at"],
            )
            for msg in conn.execute(
                """
                SELECT role, content_json, created_at
                FROM session_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        ]
        return SessionState(
            session_id=row["session_id"],
            summary=row["summary"],
            messages=messages,
            turn_count=row["turn_count"],
            last_reconcile_turn=row["last_reconcile_turn"],
            last_compact_turn=row["last_compact_turn"],
            updated_at=row["updated_at"],
        )


@dataclass
class PersistentAgentConfig:
    # Number of most-recent transcript messages to keep VERBATIM through a
    # compaction event. The transcript itself accumulates across turns and
    # is sent to the coordinator as real user/assistant role messages —
    # not folded into one inline-rendered user-message blob — which keeps
    # the prefix byte-identical between turns and lets OpenAI's automatic
    # prefix cache (and Anthropic's ``cache_control``) hit on the entire
    # historical prefix.
    recent_messages: int = 8
    reconcile_every_turns: int = 6
    compact_every_turns: int = 12
    compact_message_threshold: int = 24
    durable_signal_terms: tuple[str, ...] = (
        "remember",
        "always",
        "never",
        "prefer",
        "don't do",
        "do not",
        "instead",
    )


class PersistentAgent:
    """One-turn-at-a-time persistent chat facade around a coordinator agent."""

    def __init__(
        self,
        *,
        coordinator: BaseAgent,
        session_store: SessionStore,
        memory: MemoryManager,
        llm: Any | None = None,
        guard_factory: Callable[[], Any] | None = None,
        config: PersistentAgentConfig | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._session_store = session_store
        self._memory = memory
        self._llm = llm or coordinator._llm
        self._guard_factory = guard_factory
        self._config = config or PersistentAgentConfig()

    async def chat(
        self,
        message: str,
        *,
        session_id: str = "default",
        run_id: str | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        """Run one user turn with fresh working memory and durable session context."""
        state = await self._session_store.load(session_id)
        run_id = run_id or str(uuid.uuid4())
        prior_messages, pinned_priors = self._build_prior_messages(state)
        if self._guard_factory is not None:
            self._assign_guard(self._guard_factory())

        final_result: dict | None = None
        trace: list[dict[str, Any]] = []
        tools_used: set[str] = set()
        subagents_used: set[str] = set()
        errors: list[str] = []
        finalized = False

        async for event in self._coordinator.run_stream(
            message,
            run_id=run_id,
            prior_messages=prior_messages,
            pinned_priors=pinned_priors,
        ):
            trace.append(_event_to_trace(event))
            if event.type == EventType.ACTION:
                tool = event.payload.get("tool")
                if tool:
                    tools_used.add(str(tool))
            elif event.type == EventType.TASK_DONE and not event.parent_agent_id:
                final_result = event.payload
            elif event.type == EventType.ERROR:
                errors.append(event.error)
            if event.parent_agent_id:
                subagents_used.add(event.agent_id)
            if not event.parent_agent_id and event.type in (EventType.TASK_DONE, EventType.ERROR):
                await self._finalize_turn(
                    session_id=session_id,
                    message=message,
                    final_result=final_result,
                    trace=trace,
                    tools_used=tools_used,
                    subagents_used=subagents_used,
                    errors=errors,
                )
                finalized = True
            yield event

        if not finalized:
            await self._finalize_turn(
                session_id=session_id,
                message=message,
                final_result=final_result,
                trace=trace,
                tools_used=tools_used,
                subagents_used=subagents_used,
                errors=errors,
            )

    def capabilities(self) -> dict[str, Any]:
        """Describe the coordinator, sub-agents, tools, and MCP tools wired in.

        This is introspection only. ``PersistentAgent`` does not create or
        authorize tools dynamically; callers remain responsible for wiring
        sub-agents, MCP adapters, and auth before wrapping the coordinator.
        """
        seen_agents: set[int] = set()
        coordinator = _describe_agent(self._coordinator)
        subagents: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []

        def visit_agent(agent: BaseAgent, *, parent_agent_id: str) -> None:
            for tool_name, tool in getattr(agent, "_tools", {}).items():
                if _is_mcp_tool(tool):
                    mcp_tools.append(_describe_mcp_tool(tool, owner_agent_id=parent_agent_id))
                sub = _subagent_tool_agent(tool)
                if sub is None:
                    continue
                sub_info = {
                    "agent_id": sub.config.agent_id,
                    "role": sub.role,
                    "tool_name": tool_name,
                    "parent_agent_id": parent_agent_id,
                    "tools": sorted(getattr(sub, "_tools", {}).keys()),
                    "mcp_tools": [
                        _describe_mcp_tool(t, owner_agent_id=sub.config.agent_id)
                        for t in getattr(sub, "_tools", {}).values()
                        if _is_mcp_tool(t)
                    ],
                }
                subagents.append(sub_info)
                if id(sub) not in seen_agents:
                    seen_agents.add(id(sub))
                    visit_agent(sub, parent_agent_id=sub.config.agent_id)

        seen_agents.add(id(self._coordinator))
        visit_agent(self._coordinator, parent_agent_id=self._coordinator.config.agent_id)
        return {
            "coordinator": coordinator,
            "subagents": subagents,
            "mcp_tools": mcp_tools,
        }

    def _assign_guard(self, guard: Any) -> None:
        seen_agents: set[int] = set()

        def visit(agent: BaseAgent) -> None:
            if id(agent) in seen_agents:
                return
            seen_agents.add(id(agent))
            agent._guard = guard
            for tool in getattr(agent, "_tools", {}).values():
                sub = _subagent_tool_agent(tool)
                if sub is not None:
                    visit(sub)

        visit(self._coordinator)

    async def _finalize_turn(
        self,
        *,
        session_id: str,
        message: str,
        final_result: dict | None,
        trace: list[dict[str, Any]],
        tools_used: set[str],
        subagents_used: set[str],
        errors: list[str],
    ) -> None:
        answer = ""
        if final_result is not None:
            answer = str(final_result.get("answer") or "")
        elif errors:
            answer = f"ERROR: {errors[-1]}"

        state = await self._session_store.append_messages(
            session_id,
            [
                SessionMessage(role="user", content=message),
                SessionMessage(role="assistant", content=answer),
            ],
        )

        if self._should_reconcile(
            message=message,
            state=state,
            tools_used=tools_used,
            subagents_used=subagents_used,
            errors=errors,
        ):
            await self._memory.write_run_end(
                goal=message,
                agent_results=[final_result or {"error": errors[-1] if errors else "no result"}],
                trace=trace,
            )
            state = await self._session_store.mark_reconciled(session_id, state.turn_count)

        if self._should_compact(state):
            # Compact only the OLDER portion — everything before the last
            # ``recent_messages`` stays verbatim — so the still-recent
            # turns remain byte-identical across the compaction boundary.
            # This keeps OpenAI's prefix cache warm through compaction
            # events (only the leading summary block changes; the verbatim
            # tail does not).
            keep = self._config.recent_messages
            if keep <= 0:
                to_compact = list(state.messages)
            else:
                to_compact = state.messages[:-keep] if len(state.messages) > keep else []
            if to_compact:
                summary = await self._summarize_session(state, messages_to_compact=to_compact)
                state = await self._session_store.update_summary(session_id, summary)
                await self._session_store.trim_messages(session_id, keep)
                await self._session_store.mark_compacted(session_id, state.turn_count)

    def _build_prior_messages(
        self, state: SessionState
    ) -> tuple[list[tuple[str, str | list]], int]:
        """Return ``(prior_messages, pinned_priors)`` to seed into the
        coordinator's WorkingMemory.

        Cache-friendly shape:
          - The rolling summary (if any) lives as a stable user/assistant
            priming pair at the START of priors. Pinned, so WM eviction
            within a turn can't drop it. Stays byte-identical between
            turns until the next compaction → caches cleanly.
          - All accumulated session messages follow as their own
            user/assistant entries (NOT collapsed into one rendered text
            blob). Between turns N and N+1, this prefix grows by exactly
            two messages (the previous turn's user+assistant pair).
            Everything older stays byte-identical → OpenAI's automatic
            prefix cache and Anthropic's ``cache_control`` both hit.

        Without this shape — i.e. the older "render summary+recent+
        current into one user message" approach — the sliding-window
        slice rotates every turn and the cache breaks immediately after
        the system prompt.
        """
        prior_messages: list[tuple[str, str | list]] = []
        pinned_priors = 0
        if state.summary:
            prior_messages.append(("user", f"[Earlier conversation, summarised]\n{state.summary}"))
            prior_messages.append(("assistant", "Acknowledged."))
            pinned_priors = 2
        for msg in state.messages:
            prior_messages.append((msg.role, msg.content))
        return prior_messages, pinned_priors

    def _should_reconcile(
        self,
        *,
        message: str,
        state: SessionState,
        tools_used: set[str],
        subagents_used: set[str],
        errors: list[str],
    ) -> bool:
        if tools_used or subagents_used or errors:
            return True
        lower = message.lower()
        if any(term in lower for term in self._config.durable_signal_terms):
            return True
        return state.turn_count - state.last_reconcile_turn >= self._config.reconcile_every_turns

    def _should_compact(self, state: SessionState) -> bool:
        if len(state.messages) > self._config.compact_message_threshold:
            return True
        return state.turn_count - state.last_compact_turn >= self._config.compact_every_turns

    async def _summarize_session(
        self,
        state: SessionState,
        *,
        messages_to_compact: list[SessionMessage] | None = None,
    ) -> str:
        """Summarise the older portion of the session, folding any prior
        summary in. ``messages_to_compact`` defaults to the full state
        transcript (legacy path); the new compaction flow passes only the
        messages being trimmed so the still-verbatim recent tail stays out
        of the summary."""
        targets = messages_to_compact if messages_to_compact is not None else list(state.messages)
        rendered = "\n".join(f"{m.role}: {m.content}" for m in targets)
        response = await self._llm.complete(
            system="Summarize a persistent agent chat session. Return plain text only.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Existing summary:\n"
                        f"{state.summary or '(none)'}\n\n"
                        "Messages to fold in:\n"
                        f"{rendered}\n\n"
                        "Write a compact summary preserving user preferences, decisions, "
                        "open threads, and concrete references needed for future turns. "
                        "Treat any 'Existing summary' as the canonical past — merge new "
                        "evidence into it rather than starting over."
                    ),
                }
            ],
            source="persistent_session",
        )
        if isinstance(response, dict):
            return str(response.get("text") or response.get("answer") or "").strip()
        return str(response).strip()


def _copy_state(state: SessionState) -> SessionState:
    return SessionState(
        session_id=state.session_id,
        summary=state.summary,
        messages=[
            SessionMessage(role=m.role, content=m.content, created_at=m.created_at)
            for m in state.messages
        ],
        turn_count=state.turn_count,
        last_reconcile_turn=state.last_reconcile_turn,
        last_compact_turn=state.last_compact_turn,
        updated_at=state.updated_at,
    )


def _event_to_trace(event: BusEvent) -> dict[str, Any]:
    return {
        "type": event.type.value,
        "agent_id": event.agent_id,
        "parent_agent_id": event.parent_agent_id,
        "payload": event.payload,
        "token": event.token,
        "error": event.error,
        "timestamp": event.timestamp,
    }


def _describe_agent(agent: BaseAgent) -> dict[str, Any]:
    tools = getattr(agent, "_tools", {})
    return {
        "agent_id": agent.config.agent_id,
        "role": agent.role,
        "tools": sorted(tools.keys()),
        "mcp_tools": [
            _describe_mcp_tool(tool, owner_agent_id=agent.config.agent_id)
            for tool in tools.values()
            if _is_mcp_tool(tool)
        ],
    }


def _subagent_tool_agent(tool: Any) -> BaseAgent | None:
    if tool.__class__.__name__ != "SubAgentTool":
        return None
    agent = getattr(tool, "_agent", None)
    return agent if isinstance(agent, BaseAgent) else None


def _is_mcp_tool(tool: Any) -> bool:
    cls = tool.__class__
    return cls.__name__ == "MCPToolAdapter" and cls.__module__.startswith("tools.mcp")


def _describe_mcp_tool(tool: Any, *, owner_agent_id: str) -> dict[str, Any]:
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "owner_agent_id": owner_agent_id,
        "source": "mcp",
        "input_schema": getattr(tool, "input_schema", {}),
    }
