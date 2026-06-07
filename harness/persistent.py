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
from harness.utils import fire
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
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    last_run_tokens_in: int = 0
    last_run_tokens_out: int = 0
    last_usage: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_now)


class SessionStore(Protocol):
    async def load(self, session_id: str) -> SessionState: ...

    async def exists(self, session_id: str) -> bool: ...

    async def list_sessions(self, *, query: str | None = None) -> list[SessionState]: ...

    async def append_messages(
        self, session_id: str, messages: Sequence[SessionMessage]
    ) -> SessionState: ...

    async def update_summary(self, session_id: str, summary: str) -> SessionState: ...

    async def trim_messages(self, session_id: str, keep_last: int) -> SessionState: ...

    async def mark_reconciled(self, session_id: str, turn_count: int) -> SessionState: ...

    async def mark_compacted(self, session_id: str, turn_count: int) -> SessionState: ...

    async def record_usage(
        self,
        session_id: str,
        *,
        tokens_in: int,
        tokens_out: int,
        usage: dict[str, Any],
    ) -> SessionState: ...

    async def clear(self, session_id: str) -> SessionState: ...

    async def delete(self, session_id: str) -> bool: ...


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

    async def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def list_sessions(self, *, query: str | None = None) -> list[SessionState]:
        needle = query.lower() if query else None
        states = self._sessions.values()
        if needle:
            states = [state for state in states if needle in state.session_id.lower()]
        return sorted(
            (_copy_state(state) for state in states),
            key=lambda state: state.updated_at,
            reverse=True,
        )

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

    async def record_usage(
        self,
        session_id: str,
        *,
        tokens_in: int,
        tokens_out: int,
        usage: dict[str, Any],
    ) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.tokens_in_total += int(tokens_in)
        state.tokens_out_total += int(tokens_out)
        state.last_run_tokens_in = int(tokens_in)
        state.last_run_tokens_out = int(tokens_out)
        state.last_usage = dict(usage)
        state.updated_at = _now()
        return _copy_state(state)

    async def clear(self, session_id: str) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.summary = ""
        state.messages = []
        state.turn_count = 0
        state.last_reconcile_turn = 0
        state.last_compact_turn = 0
        state.tokens_in_total = 0
        state.tokens_out_total = 0
        state.last_run_tokens_in = 0
        state.last_run_tokens_out = 0
        state.last_usage = {}
        state.updated_at = _now()
        return _copy_state(state)

    async def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


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

    async def exists(self, session_id: str) -> bool:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row is not None

    async def list_sessions(self, *, query: str | None = None) -> list[SessionState]:
        self._ensure_schema()
        with self._connect() as conn:
            if query:
                rows = conn.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    WHERE lower(session_id) LIKE ?
                    ORDER BY updated_at DESC, session_id ASC
                    """,
                    (f"%{query.lower()}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    ORDER BY updated_at DESC, session_id ASC
                    """
                ).fetchall()
            return [self._load_locked(conn, row["session_id"]) for row in rows]

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

    async def record_usage(
        self,
        session_id: str,
        *,
        tokens_in: int,
        tokens_out: int,
        usage: dict[str, Any],
    ) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                """
                UPDATE sessions
                SET tokens_in_total = tokens_in_total + ?,
                    tokens_out_total = tokens_out_total + ?,
                    last_run_tokens_in = ?,
                    last_run_tokens_out = ?,
                    last_usage_json = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    int(tokens_in),
                    int(tokens_out),
                    int(tokens_in),
                    int(tokens_out),
                    json.dumps(usage, default=str),
                    _now(),
                    session_id,
                ),
            )
            return self._load_locked(conn, session_id)

    async def clear(self, session_id: str) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            conn.execute(
                """
                UPDATE sessions
                SET summary = '',
                    turn_count = 0,
                    last_reconcile_turn = 0,
                    last_compact_turn = 0,
                    tokens_in_total = 0,
                    tokens_out_total = 0,
                    last_run_tokens_in = 0,
                    last_run_tokens_out = 0,
                    last_usage_json = '{}',
                    updated_at = ?
                WHERE session_id = ?
                """,
                (_now(), session_id),
            )
            return self._load_locked(conn, session_id)

    async def delete(self, session_id: str) -> bool:
        self._ensure_schema()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is None:
                return False
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return True

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
                    tokens_in_total INTEGER NOT NULL DEFAULT 0,
                    tokens_out_total INTEGER NOT NULL DEFAULT 0,
                    last_run_tokens_in INTEGER NOT NULL DEFAULT 0,
                    last_run_tokens_out INTEGER NOT NULL DEFAULT 0,
                    last_usage_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_usage_columns(conn)
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

    def _ensure_usage_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        additions = {
            "tokens_in_total": "INTEGER NOT NULL DEFAULT 0",
            "tokens_out_total": "INTEGER NOT NULL DEFAULT 0",
            "last_run_tokens_in": "INTEGER NOT NULL DEFAULT 0",
            "last_run_tokens_out": "INTEGER NOT NULL DEFAULT 0",
            "last_usage_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for column, ddl in additions.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {ddl}")

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
                   last_compact_turn, tokens_in_total, tokens_out_total,
                   last_run_tokens_in, last_run_tokens_out, last_usage_json,
                   updated_at
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
            tokens_in_total=row["tokens_in_total"],
            tokens_out_total=row["tokens_out_total"],
            last_run_tokens_in=row["last_run_tokens_in"],
            last_run_tokens_out=row["last_run_tokens_out"],
            last_usage=json.loads(row["last_usage_json"] or "{}"),
            updated_at=row["updated_at"],
        )


@dataclass
class PersistentAgentConfig:
    # Compact when the transcript token count crosses this fraction of
    # the coordinator's ``llm.input_token_budget``. Replaces the previous
    # turn-count / message-count triggers, which fired arbitrarily often
    # on chat-light sessions (wasting context budget) and too late on
    # tool-heavy sessions (risking input-window overrun). 0.5 leaves room
    # for the current turn's task, any tool observations it will produce,
    # and tokeniser variance.
    compact_at_context_fraction: float = 0.5
    # After compaction, retain the newest transcript messages whose
    # estimated token total fits within this fraction of the coordinator
    # input budget. The rest is folded into ``SessionState.summary``.
    # Token-based retention scales across models and avoids preserving a
    # huge browser/tool observation merely because it is one of the last
    # few messages.
    retain_context_fraction: float = 0.15
    # Background memory accumulation: every N turns, schedule a
    # non-blocking ``write_run_end`` over the last N turns of the durable
    # session transcript. The current session's per-session memory
    # context cache is **not** evicted, so the prompt prefix stays
    # byte-identical and OpenAI / Anthropic prefix caches keep hitting.
    # The reconciler's new facts are visible to OTHER sessions
    # immediately and to THIS session at the next compaction (where the
    # cache breaks anyway). Set to 0 to disable.
    async_reconcile_every_turns: int = 10
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
        # Per-session memory-context cache. Memory retrieval used to fire
        # on every turn (inside ``_build_system_prompt``), which made the
        # system prompt content-dependent on whatever ``build_context``
        # returned for the current goal — defeating prefix caching from
        # position 0. We now retrieve once per session, hold the rendered
        # string here, and evict only at compaction (when the cache is
        # already breaking anyway). Within a compaction window, memory
        # context is byte-identical across turns.
        self._session_memory_context: dict[str, str] = {}

    @property
    def config(self) -> PersistentAgentConfig:
        """Persistent session policy used by this wrapper."""
        return self._config

    @property
    def llm(self) -> Any:
        """LLM used for session summarization/control introspection."""
        return self._llm

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
        memory_context_text = await self._get_session_memory_context(session_id, message=message)
        prior_messages, pinned_priors = self._build_prior_messages(
            state, memory_context_text=memory_context_text
        )
        turn_guard = None
        if self._guard_factory is not None:
            turn_guard = self._guard_factory()
            self._assign_guard(turn_guard)
        else:
            turn_guard = getattr(self._coordinator, "_guard", None)
        usage_before = _guard_token_totals(turn_guard)

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
            # Skip the live ``build_context`` inside ``_build_system_prompt``
            # — memory is in a pinned prior instead, cached per session and
            # refreshed only at compaction. Keeps the system prompt
            # byte-identical across turns so the prefix cache holds.
            precomputed_memory_context="_skip_",
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
                    usage_guard=turn_guard,
                    usage_before=usage_before,
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
                usage_guard=turn_guard,
                usage_before=usage_before,
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

    async def session_state(self, session_id: str) -> SessionState:
        """Return a copy of the durable session state."""
        return await self._session_store.load(session_id)

    async def list_sessions(self, *, query: str | None = None) -> list[SessionState]:
        """Return known durable sessions, newest first when supported by the store."""
        return await self._session_store.list_sessions(query=query)

    async def session_exists(self, session_id: str) -> bool:
        """Return whether a session exists without creating it."""
        return await self._session_store.exists(session_id)

    def cached_memory_context(self, session_id: str) -> str | None:
        """Return the currently cached memory-context blob for a session, if any."""
        return self._session_memory_context.get(session_id)

    def forget_memory_cache(self, session_id: str) -> None:
        """Drop cached memory context so the next turn fetches fresh memory."""
        self._evict_session_memory_context(session_id)

    async def force_compact(self, session_id: str) -> SessionState:
        """Summarize, trim, and reconcile the older transcript portion.

        This uses the same compaction shape as automatic context-pressure
        compaction: keep the newest messages that fit inside
        ``retain_context_fraction`` of the input budget, fold older messages
        into the rolling summary, reconcile the folded window into long-term
        memory, then evict cached memory context so the next turn can pick
        up freshly reconciled facts.
        """
        state = await self._session_store.load(session_id)
        to_compact = self._messages_to_compact(state)
        state = await self._compact_session(session_id, state)
        if to_compact and state.last_compact_turn:
            await self._write_session_window_to_memory(
                session_id=session_id,
                messages=to_compact,
                goal_fallback="(force compact)",
            )
            state = await self._session_store.mark_reconciled(session_id, state.turn_count)
        self._evict_session_memory_context(session_id)
        return state

    async def save_to_memory(self, session_id: str) -> int:
        """Reconcile pending transcript messages into long-term memory now.

        Samples only messages after ``last_reconcile_turn``, then awaits the
        ``write_run_end`` call so the caller can confirm completion. Use when
        a user explicitly wants "save what we discussed" before leaving the
        session (e.g. demo ``/save`` command).

        Crucially does NOT evict the per-session memory cache: the active
        session keeps its warm prefix. New facts are visible to OTHER
        sessions immediately and to THIS session at the next compaction
        (where the cache breaks anyway for the summary refresh).

        Returns the number of transcript messages included in the reconcile
        payload — 0 if there's nothing pending to save.
        """
        state = await self._session_store.load(session_id)
        if not state.messages:
            return 0
        window = self._messages_since_reconcile(state)
        if not window:
            return 0
        await self._write_session_window_to_memory(
            session_id=session_id,
            messages=window,
            goal_fallback="(explicit save)",
        )
        await self._session_store.mark_reconciled(session_id, state.turn_count)
        return len(window)

    async def clear_session(self, session_id: str) -> SessionState:
        """Clear transcript/summary/counters for a session.

        Long-term semantic/episodic memory is intentionally retained; this
        only resets the durable chat workspace for ``session_id``.
        """
        state = await self._session_store.clear(session_id)
        self._evict_session_memory_context(session_id)
        return state

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session transcript/summary row.

        Long-term semantic/episodic memory is intentionally retained.
        Returns ``True`` when a session existed and was deleted.
        """
        deleted = await self._session_store.delete(session_id)
        self._evict_session_memory_context(session_id)
        return deleted

    def _assign_guard(self, guard: Any) -> None:
        seen_agents: set[int] = set()

        def visit(agent: BaseAgent) -> None:
            if id(agent) in seen_agents:
                return
            seen_agents.add(id(agent))
            agent._guard = guard
            llm = getattr(agent, "_llm", None)
            if hasattr(llm, "set_budget"):
                llm.set_budget(guard)
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
        usage_guard: Any | None = None,
        usage_before: tuple[int, int] = (0, 0),
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

        sync_reconciled = self._should_reconcile(
            message=message,
            state=state,
            tools_used=tools_used,
            subagents_used=subagents_used,
            errors=errors,
        )
        if sync_reconciled:
            await self._memory.write_run_end(
                goal=message,
                agent_results=[final_result or {"error": errors[-1] if errors else "no result"}],
                trace=trace,
            )
            state = await self._session_store.mark_reconciled(session_id, state.turn_count)
            # We just wrote new facts → next turn's memory context might
            # differ. Drop the cache so the refresh on turn N+1 picks them
            # up. This turn already paid a cache miss (tool / signal
            # caused the reconcile); the immediate next turn pays at
            # most one more if the new facts actually changed retrieval.
            self._evict_session_memory_context(session_id)

        compacted = self._should_compact(state)
        if not sync_reconciled and not compacted:
            # Background memory accumulation. The fire-and-forget reconcile
            # uses the durable transcript window — no buffer to maintain —
            # and intentionally does NOT evict the per-session memory
            # context cache. New facts land in the long-term store
            # immediately for OTHER sessions; THIS session sees them at
            # the next compaction.
            self._maybe_fire_async_reconcile(session_id, state)

        if compacted:
            to_compact = self._messages_to_compact(state)
            compacted_state = await self._compact_session(session_id, state)
            if compacted_state.last_compact_turn != state.last_compact_turn:
                state = compacted_state
                # Compaction is the natural moment to reconcile the older
                # transcript window into long-term memory: the reconciler
                # LLM call colocates with the summary write, both touching
                # the same cache-miss boundary so we don't pay twice for
                # cache invalidation. Also evict the cached memory context
                # so the next turn's first build_context call picks up any
                # facts the reconciler added/updated.
                if not self._should_reconcile(
                    message=message,
                    state=state,
                    tools_used=tools_used,
                    subagents_used=subagents_used,
                    errors=errors,
                ):
                    # High-signal events already triggered reconcile above;
                    # don't repeat. Only fire here when this turn didn't
                    # already write.
                    try:
                        await self._write_session_window_to_memory(
                            session_id=session_id,
                            messages=to_compact,
                            goal_fallback=message,
                        )
                        state = await self._session_store.mark_reconciled(
                            session_id, state.turn_count
                        )
                    except Exception:  # noqa: BLE001 — best-effort at compaction
                        pass
                self._evict_session_memory_context(session_id)

        await self._record_turn_usage(
            session_id=session_id,
            guard=usage_guard,
            usage_before=usage_before,
        )

    async def _record_turn_usage(
        self,
        *,
        session_id: str,
        guard: Any | None,
        usage_before: tuple[int, int],
    ) -> None:
        if guard is None or not hasattr(guard, "snapshot"):
            return
        snapshot = guard.snapshot()
        tokens_in = int(snapshot.get("tokens_in") or 0) - usage_before[0]
        tokens_out = int(snapshot.get("tokens_out") or 0) - usage_before[1]
        if tokens_in <= 0 and tokens_out <= 0:
            return
        usage = dict(snapshot)
        usage["tokens_in"] = max(tokens_in, 0)
        usage["tokens_out"] = max(tokens_out, 0)
        usage["total_tokens"] = usage["tokens_in"] + usage["tokens_out"]
        await self._session_store.record_usage(
            session_id,
            tokens_in=usage["tokens_in"],
            tokens_out=usage["tokens_out"],
            usage=usage,
        )

    async def _compact_session(self, session_id: str, state: SessionState) -> SessionState:
        to_compact, keep_last = self._compaction_split(state)
        if not to_compact:
            return state
        summary = await self._summarize_session(state, messages_to_compact=to_compact)
        state = await self._session_store.update_summary(session_id, summary)
        state = await self._session_store.trim_messages(session_id, keep_last)
        state = await self._session_store.mark_compacted(session_id, state.turn_count)
        return state

    def _messages_to_compact(self, state: SessionState) -> list[SessionMessage]:
        return self._compaction_split(state)[0]

    def _compaction_split(self, state: SessionState) -> tuple[list[SessionMessage], int]:
        """Return ``(messages_to_compact, keep_last_count)``.

        Retention is token-budget based: keep newest messages until adding
        another would exceed ``retain_context_fraction`` of the coordinator's
        input budget. If no budget is advertised, compact the full verbatim
        transcript into the summary.
        """
        if not state.messages:
            return [], 0
        budget = self._coordinator_input_token_budget()
        if budget is None:
            return list(state.messages), 0
        retain_tokens = int(budget * self._config.retain_context_fraction)
        if retain_tokens <= 0:
            return list(state.messages), 0

        kept_tokens = 0
        keep_last = 0
        for message in reversed(state.messages):
            tokens = self._message_token_count(message)
            if keep_last and kept_tokens + tokens > retain_tokens:
                break
            if not keep_last and tokens > retain_tokens:
                # Always keep the newest message, even if it alone exceeds
                # the retention target. Dropping the latest assistant reply
                # makes resumption feel broken and loses immediate context.
                keep_last = 1
                break
            kept_tokens += tokens
            keep_last += 1

        if keep_last >= len(state.messages):
            return [], len(state.messages)
        return state.messages[:-keep_last] if keep_last else list(state.messages), keep_last

    def _messages_since_reconcile(self, state: SessionState) -> list[SessionMessage]:
        if state.last_reconcile_turn <= 0:
            return list(state.messages)
        user_messages = sum(1 for message in state.messages if message.role == "user")
        current_turn = state.turn_count - user_messages
        pending: list[SessionMessage] = []
        for message in state.messages:
            if message.role == "user":
                current_turn += 1
            if current_turn > state.last_reconcile_turn:
                pending.append(message)
        return pending

    async def _write_session_window_to_memory(
        self,
        *,
        session_id: str,
        messages: Sequence[SessionMessage],
        goal_fallback: str,
    ) -> None:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        last_assistant = next((m.content for m in reversed(messages) if m.role == "assistant"), "")
        trace = [
            {
                "type": m.role,
                "content": m.content,
                "timestamp": m.created_at,
            }
            for m in messages
        ]
        await self._memory.write_run_end(
            goal=str(last_user) if last_user else goal_fallback,
            agent_results=[
                {
                    "agent_id": self._coordinator.config.agent_id,
                    "answer": str(last_assistant),
                    "confidence": 1.0,
                    "session_id": session_id,
                }
            ],
            trace=trace,
        )

    async def _get_session_memory_context(self, session_id: str, *, message: str) -> str:
        """Return the per-session memory-context blob (cached).

        First call for the session fetches via ``MemoryManager.build_context``,
        rendering whatever semantic + episodic context is relevant to the
        first goal. The result is cached on the PersistentAgent instance
        keyed by ``session_id`` and reused for every subsequent turn in
        the same compaction window — see ``_evict_session_memory_context``
        for the eviction path (compaction).
        """
        cached = self._session_memory_context.get(session_id)
        if cached is not None:
            return cached
        try:
            mem_context = await self._memory.build_context(
                goal=message,
                agent_id=self._coordinator.config.agent_id,
            )
        except Exception:  # noqa: BLE001 — memory backend hiccup shouldn't crash chat
            mem_context = None
        rendered = ""
        if mem_context is not None and not mem_context.is_empty():
            rendered = mem_context.render()
        self._session_memory_context[session_id] = rendered
        return rendered

    def _evict_session_memory_context(self, session_id: str) -> None:
        """Drop the cached memory context for ``session_id`` so the next
        turn re-fetches. Called at compaction, where the cache is already
        breaking from the summary refresh anyway."""
        self._session_memory_context.pop(session_id, None)

    def _build_prior_messages(
        self,
        state: SessionState,
        *,
        memory_context_text: str = "",
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
        # Memory context goes FIRST so it's the most-cached prefix block.
        # It was previously embedded in the system prompt, but
        # ``MemoryManager.build_context`` is goal-dependent and shifts
        # between turns — putting it in the system prompt broke the cache
        # at position 0. Moved here and cached per-session in
        # ``self._session_memory_context``.
        if memory_context_text:
            prior_messages.append(
                ("user", f"[Relevant memory for this session]\n{memory_context_text}")
            )
            prior_messages.append(("assistant", "Noted."))
            pinned_priors += 2
        if state.summary:
            prior_messages.append(("user", f"[Earlier conversation, summarised]\n{state.summary}"))
            prior_messages.append(("assistant", "Acknowledged."))
            pinned_priors += 2
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
        """User-intent-only immediate reconciliation policy.

        Compaction handles bulk reconciliation when context pressure
        forces a summary LLM call (the same boundary already pays a
        cache miss). Tool runs / sub-agent runs / errors used to also
        trigger reconciliation here, but their facts are mostly
        situational (this run's outputs) — the session transcript
        captures them within the session, and the next compaction
        boundary folds them into long-term memory. Bypassing
        compaction for every tool run was just trashing the cache for
        marginal cross-session benefit.

        What stays: user-explicit signals like "remember X", "always do
        Y", "I prefer Z" — these are cross-session by nature and should
        persist immediately even if it costs a cache miss.
        """
        lower = message.lower()
        return any(term in lower for term in self._config.durable_signal_terms)

    def _should_compact(self, state: SessionState) -> bool:
        """Context-pressure trigger — fires when the accumulated transcript
        (rolling summary + verbatim history) crosses
        ``compact_at_context_fraction`` of the coordinator's
        ``llm.input_token_budget``.

        Replaces the previous turn-count / message-count triggers, which
        fired every N turns regardless of actual size. Plain chat
        sessions (~20 tokens/turn) now go thousands of turns between
        compactions; browser-heavy research sessions (~3K tokens/turn)
        compact only when budget pressure forces it.
        """
        budget = self._coordinator_input_token_budget()
        if budget is None:
            # Adapter doesn't expose ``input_token_budget`` (custom stub
            # or older client) — never auto-compact; rely on explicit
            # signals only.
            return False
        threshold = int(budget * self._config.compact_at_context_fraction)
        if threshold <= 0:
            return False
        return self._transcript_token_count(state) >= threshold

    def _coordinator_input_token_budget(self) -> int | None:
        """Read the coordinator LLM's input budget. Falls back to ``None``
        when the adapter doesn't advertise one."""
        llm = getattr(self._coordinator, "_llm", None) or self._llm
        budget = getattr(llm, "input_token_budget", None)
        return int(budget) if isinstance(budget, int) and budget > 0 else None

    def _transcript_token_count(self, state: SessionState) -> int:
        """Cheap chars/4 token estimate over the rolling summary +
        accumulated transcript. Same heuristic ``WorkingMemory`` uses
        when no exact counter is wired, so the threshold maps coherently
        to the LLM's eviction behaviour."""
        total = 0
        if state.summary:
            total += max(1, len(state.summary) // 4)
        for msg in state.messages:
            total += self._message_token_count(msg)
        return total

    def _message_token_count(self, message: SessionMessage) -> int:
        content = message.content if isinstance(message.content, str) else str(message.content)
        return max(1, len(content) // 4)

    def _maybe_fire_async_reconcile(self, session_id: str, state: SessionState) -> None:
        """Background reconcile every ``async_reconcile_every_turns`` turns.

        Samples the durable session transcript directly — no separate
        evidence buffer to maintain. Fires ``write_run_end`` via
        ``fire()`` so the chat turn never blocks on the reconciler LLM
        call, and crucially does NOT evict the per-session memory
        context cache. The session's prompt prefix therefore stays
        byte-identical → prefix cache keeps hitting.

        New facts land in the long-term store immediately for other
        sessions; THIS session sees them at the next compaction (where
        the cache is already breaking for the summary refresh anyway).
        """
        interval = self._config.async_reconcile_every_turns
        if interval <= 0:
            return
        if state.turn_count == 0 or state.turn_count % interval != 0:
            return

        # Sample the last N turn-pairs (= 2N messages) from the durable
        # transcript. Each turn is one user message + one assistant
        # reply; the reconciler sees a coherent slice of conversation
        # and can make MERGE / NOOP / UPDATE decisions across it.
        window = state.messages[-(2 * interval) :]
        if not window:
            return
        fire(self._async_reconcile_window(session_id, turn_count=state.turn_count, window=window))

    async def _async_reconcile_window(
        self,
        session_id: str,
        *,
        turn_count: int,
        window: Sequence[SessionMessage],
    ) -> None:
        await self._write_session_window_to_memory(
            session_id=session_id,
            messages=window,
            goal_fallback="(background session reconcile)",
        )
        await self._session_store.mark_reconciled(session_id, turn_count)

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
        tokens_in_total=state.tokens_in_total,
        tokens_out_total=state.tokens_out_total,
        last_run_tokens_in=state.last_run_tokens_in,
        last_run_tokens_out=state.last_run_tokens_out,
        last_usage=dict(state.last_usage),
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


def _guard_token_totals(guard: Any | None) -> tuple[int, int]:
    if guard is None:
        return (0, 0)
    return (
        int(getattr(guard, "tokens_in", 0) or 0),
        int(getattr(guard, "tokens_out", 0) or 0),
    )


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
