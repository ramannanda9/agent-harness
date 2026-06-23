"""Persistent chat wrapper for agent-harness agents.

The wrapper keeps chat/session state durable while leaving agent construction
to callers. A user can wire any ``BaseAgent`` as the coordinator, including
sub-agents and MCP tools, then use ``PersistentAgent.chat`` for one user turn
at a time.
"""

from __future__ import annotations

import contextvars
import json
import sqlite3
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from agents.base import BaseAgent, _parse_action_json
from harness.agent_tree import iter_agents, subagent_tool_agent
from harness.background_tasks import BackgroundTaskManager, BackgroundTaskState
from harness.events import BusEvent, EventType
from harness.model_switching import ModelSwitcher
from harness.session_memory import SessionMemoryController
from memory.manager import MemoryManager


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Plan mode ────────────────────────────────────────────────────────────────
#
# When ``SessionState.plan_mode_enabled`` is True, ``PersistentAgent.chat``
# asks the coordinator's LLM for a structured plan before any tools run,
# yields a ``PLAN_PROPOSED`` event so renderers can display it, and gates
# execution on ``harness.hitl.request_plan_approval``. Plan approval handles
# Enter / y / n / correction only; there is no session-allow or persistent-
# allow shortcut because plan mode is intentionally per-turn.

_PLAN_SYSTEM_PROMPT = """You are in plan mode for an autonomous agent.

Output rules — read carefully, these are strict:
- Output ONE JSON object. Nothing else.
- No markdown code fences (no triple-backticks).
- No prose before or after the object.
- No ``<thinking>`` tags or other XML wrappers.
- The first character of your response MUST be ``{``.
- The last character of your response MUST be ``}``.

Do NOT execute any tools. Do NOT plan to execute tools yourself — the
plan is a description for a separate executor to follow.

The plan is the user's chance to approve INTENT, not arguments. Many
real tasks have args that can only be known at runtime — a URL
discovered by a search, a file path extracted from a directory listing,
a row id returned by a query. Be honest about which step args you can
fill in upfront and which depend on the previous step's observation.

Output schema:
{
  "summary": "<one-line summary of what the plan will accomplish>",
  "steps": [
    {
      "step": 1,
      "intent": "<what this step achieves>",
      "tool": "<tool name from the available list, or null if no tool>",
      "args": {<concrete arguments, OR null when args depend on prior observation>},
      "why": "<one-line justification>"
    },
    ...
  ]
}

Available tools: {tools}

Argument rules:
- Use concrete ``args`` only when the value was supplied by the user
  (e.g. they said "fetch https://x.com/y") or is otherwise knowable
  without running any tool first.
- Use ``null`` (or omit the field) when the value depends on an earlier
  step's tool output — e.g. "navigate to the most-cited paper's URL"
  after a search. Fabricating placeholder URLs / paths the agent will
  ignore is misleading.

If the user's request is conversational and needs no tool calls, return
a plan with a single step whose ``tool`` is null."""


_PLAN_CORRECTION_HINT = (
    "\n\nThe user reviewed your previous plan and asked for this revision:\n"
    "\n{correction}\n\n"
    "Output a revised JSON plan."
)


# Hard cap on revision loops to protect against pathological re-planning
# cycles. Each revision is one LLM call + one HITL prompt; users with
# stronger feedback can always reject (n) and start fresh.
_PLAN_REVISION_LIMIT = 3


def _coerce_plan(raw: Any) -> dict[str, Any] | None:
    """Best-effort plan extraction from whatever the LLM returned.

    Why this isn't a simple ``json.loads``: ``response_format={"type":
    "json_object"}`` is passed on every planner call, but **only the
    OpenAI adapter honours it** (``harness/llm/openai.py:178``). The
    Anthropic and Claude Code adapters silently drop the kwarg, so the
    model is free to wrap its JSON in markdown code fences or sandwich
    it inside prose — and routinely does.

    Recovery rides the existing extractor: ``_parse_action_json`` (in
    ``agents/base.py``) is the same helper the ReAct loop uses to pull
    JSON out of Anthropic / Claude Code responses every turn. It walks
    each ``{`` in the text and lets Python's ``JSONDecoder.raw_decode``
    handle bracket balancing, so fence-wrapped, prose-wrapped, and
    plain-JSON inputs all parse without bespoke logic here.
    """
    # Peel adapter wrapping. Every harness LLM adapter returns
    # ``{"text": <content>, "usage": <dict>}``; v0.9.x checked
    # ``len(raw) == 1`` and wrongly rejected the two-key shape.
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        raw = raw["text"]

    if isinstance(raw, str):
        raw = _parse_action_json(raw)
        if raw is None:
            return None

    if not isinstance(raw, dict):
        return None
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return None
    summary = raw.get("summary")
    return {
        "summary": str(summary) if summary is not None else "",
        "steps": [s for s in steps if isinstance(s, dict)],
    }


def _step_args_deferred(step: dict[str, Any]) -> bool:
    """Whether a step's args are deferred to runtime.

    A step is "deferred" when the planner could not commit to concrete
    args upfront — usually because the value depends on what an earlier
    step's tool observes (URL discovered by a search, file path from a
    directory listing, row id from a query, etc.).

    Convention:
    - ``"args"`` key missing → deferred
    - ``args is None`` → deferred (explicit null in the planner JSON)
    - ``args == {}`` → NOT deferred ("tool takes no arguments")
    - ``args == {...}`` → NOT deferred (concrete)

    Drives the "(resolved at runtime)" banner line and the
    ``dynamic_steps`` count in the HITL approval args.
    """
    if "args" not in step:
        return True
    return step["args"] is None


def _dynamic_step_count(plan: dict[str, Any]) -> int:
    """Number of plan steps whose args will be resolved at runtime."""
    steps = plan.get("steps") or []
    return sum(
        1
        for step in steps
        if isinstance(step, dict) and step.get("tool") and _step_args_deferred(step)
    )


def _render_plan_for_banner(plan: dict[str, Any]) -> str:
    """Multi-line text used as the HITL banner ``command`` field. Compact
    enough to read quickly but structured enough that a reviewer can spot
    a wrong tool / wrong URL before approving.

    Steps whose args are deferred to runtime render as
    ``args: (resolved at runtime)`` rather than fabricating placeholder
    JSON — the planner can't know e.g. a URL that a prior search step
    will return, and showing ``{}`` would imply otherwise.
    """
    lines: list[str] = []
    summary = plan.get("summary") or "(no summary)"
    lines.append(f"Plan: {summary}")
    steps = plan.get("steps") or []
    for step in steps:
        idx = step.get("step")
        intent = step.get("intent") or "(no intent)"
        tool = step.get("tool")
        why = step.get("why")
        prefix = f"  {idx}." if idx is not None else "  -"
        lines.append(f"{prefix} {intent}")
        if tool:
            if _step_args_deferred(step):
                lines.append(f"      tool: {tool}  args: (resolved at runtime)")
            else:
                args = step.get("args") or {}
                args_repr = json.dumps(args, ensure_ascii=False) if args else "{}"
                lines.append(f"      tool: {tool}  args: {args_repr}")
        if why:
            lines.append(f"      why:  {why}")
    return "\n".join(lines)


def _render_plan_for_priors(plan: dict[str, Any]) -> str:
    """Prior-message body shown to the coordinator after approval.

    Kept distinct from the banner format so the wording can drift
    independently (the LLM sees prose; the human sees a structured
    inspector view) and so we can mark the prior with an unambiguous
    ``[Approved plan]`` header the planner / executor system prompts can
    pattern-match against if needed.
    """
    body = _render_plan_for_banner(plan)
    return (
        "[Approved plan]\n"
        f"{body}\n\n"
        "The plan above is the agreed INTENT for this turn. Args shown "
        "are concrete only where the user supplied them or where they're "
        "knowable without running any tool first; everywhere else the "
        "value '(resolved at runtime)' means: derive the argument from "
        "your observations of the prior step, NOT from any placeholder "
        "in this prior. You may skip or merge steps when an observation "
        "makes them unnecessary; report any such deviation in your next "
        "thought."
    )


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
    # Session-scoped model overrides: agent_id -> model registry key.
    # Applied at chat-turn start so switching models does not mutate the
    # durable transcript.
    model_overrides: dict[str, str] = field(default_factory=dict)
    # When True, ``PersistentAgent.chat`` proposes a plan and gates
    # execution on HITL approval before running the ReAct loop. Per
    # session so plan-mode preference is sticky across process restarts
    # without bleeding between unrelated workspaces. Toggle via the
    # ``/plan`` slash command or ``PersistentAgent.set_plan_mode``.
    plan_mode_enabled: bool = False
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

    async def set_plan_mode(self, session_id: str, enabled: bool) -> SessionState: ...

    async def set_model_override(
        self, session_id: str, agent_id: str, model_name: str | None
    ) -> SessionState: ...


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

    async def set_plan_mode(self, session_id: str, enabled: bool) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        state.plan_mode_enabled = bool(enabled)
        state.updated_at = _now()
        return _copy_state(state)

    async def set_model_override(
        self, session_id: str, agent_id: str, model_name: str | None
    ) -> SessionState:
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        if model_name:
            state.model_overrides[agent_id] = model_name
        else:
            state.model_overrides.pop(agent_id, None)
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

    async def set_plan_mode(self, session_id: str, enabled: bool) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                """
                UPDATE sessions
                SET plan_mode_enabled = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (1 if enabled else 0, _now(), session_id),
            )
            return self._load_locked(conn, session_id)

    async def set_model_override(
        self, session_id: str, agent_id: str, model_name: str | None
    ) -> SessionState:
        self._ensure_schema()
        with self._connect() as conn:
            self._ensure_session(conn, session_id)
            state = self._load_locked(conn, session_id)
            overrides = dict(state.model_overrides)
            if model_name:
                overrides[agent_id] = model_name
            else:
                overrides.pop(agent_id, None)
            conn.execute(
                """
                UPDATE sessions
                SET model_overrides_json = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (json.dumps(overrides, sort_keys=True), _now(), session_id),
            )
            return self._load_locked(conn, session_id)

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
                    model_overrides_json TEXT NOT NULL DEFAULT '{}',
                    plan_mode_enabled INTEGER NOT NULL DEFAULT 0,
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
        # Additive column migrations for the ``sessions`` table. Each entry
        # is ``column → DDL fragment for ALTER TABLE ADD COLUMN``. Existing
        # databases get the new column with the declared default; fresh
        # databases pick the column up via the ``CREATE TABLE`` above.
        # Name kept for historical reasons — this function now handles all
        # additive sessions-table columns, not just usage ones.
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        additions = {
            "tokens_in_total": "INTEGER NOT NULL DEFAULT 0",
            "tokens_out_total": "INTEGER NOT NULL DEFAULT 0",
            "last_run_tokens_in": "INTEGER NOT NULL DEFAULT 0",
            "last_run_tokens_out": "INTEGER NOT NULL DEFAULT 0",
            "last_usage_json": "TEXT NOT NULL DEFAULT '{}'",
            "model_overrides_json": "TEXT NOT NULL DEFAULT '{}'",
            "plan_mode_enabled": "INTEGER NOT NULL DEFAULT 0",
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
                   model_overrides_json, plan_mode_enabled, updated_at
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
            model_overrides=json.loads(row["model_overrides_json"] or "{}"),
            plan_mode_enabled=bool(row["plan_mode_enabled"]),
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
        "don't forget",
        "keep in mind",
        "from now on",
        "going forward",
        "make note",
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
        llm_registry: dict[str, Callable[[], Any]] | None = None,
        default_model: str | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._session_store = session_store
        self._memory = memory
        # Explicit summarizer/control LLM, if the caller supplied one. When
        # absent, ``llm`` tracks the coordinator's current LLM (which model
        # switching may swap), so it stays correct without re-pointing.
        self._explicit_llm = llm
        self._guard_factory = guard_factory
        self._config = config or PersistentAgentConfig()
        self._models = ModelSwitcher(
            coordinator=coordinator,
            session_store=session_store,
            llm_registry=llm_registry,
            default_model=default_model,
        )
        self._active_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
            f"agent_harness_session_{id(self)}",
            default="",
        )
        self._background = BackgroundTaskManager(
            coordinator=coordinator,
            session_store=session_store,
            session_id_provider=self._active_session_id.get,
            apply_overrides=self._models.apply,
            session_message_factory=lambda content: SessionMessage(
                role="assistant",
                content=content,
            ),
        )
        self._background.install_tools()
        # All memory retrieval, reconciliation, and compaction policy lives in
        # the controller. It receives the coordinator token-budget accessor and
        # the session summarizer LLM as callables so model switching is picked
        # up at call time.
        self._mem = SessionMemoryController(
            memory=self._memory,
            session_store=self._session_store,
            config=self._config,
            coordinator_agent_id=self._coordinator.config.agent_id,
            token_budget=self._coordinator_input_token_budget,
            summarizer_llm=lambda: self.llm,
        )

    @property
    def config(self) -> PersistentAgentConfig:
        """Persistent session policy used by this wrapper."""
        return self._config

    @property
    def llm(self) -> Any:
        """LLM used for session summarization/control introspection.

        Defaults to the coordinator's current LLM (so model switching is
        reflected automatically) unless an explicit LLM was supplied.
        """
        return self._explicit_llm or self._coordinator._llm

    # ── Plan mode public surface ──────────────────────────────────────────

    async def set_plan_mode(self, session_id: str, enabled: bool) -> bool:
        """Toggle plan mode on or off for ``session_id``.

        When enabled, the next ``chat()`` call generates a plan, yields
        ``PLAN_PROPOSED``, and gates execution on HITL approval before
        running the ReAct loop. Returns the new value.
        """
        state = await self._session_store.set_plan_mode(session_id, enabled)
        return state.plan_mode_enabled

    async def plan_mode_enabled(self, session_id: str) -> bool:
        """Return the current plan-mode setting for ``session_id``."""
        state = await self._session_store.load(session_id)
        return state.plan_mode_enabled

    async def start_background_subagent(
        self,
        session_id: str,
        agent_id: str,
        instruction: str,
    ) -> BackgroundTaskState:
        """Launch ``agent_id`` on ``instruction`` without blocking the caller.

        Background tasks are process-local: completed metadata stays available
        through this ``PersistentAgent`` instance, and ``collect_background_task``
        can write the result back into the durable transcript.
        """
        return await self._background.start(session_id, agent_id, instruction)

    async def list_background_tasks(
        self, session_id: str | None = None
    ) -> list[BackgroundTaskState]:
        """Return process-local background tasks, newest first."""
        return await self._background.list(session_id)

    async def collect_background_task(
        self,
        session_id: str,
        task_id: str,
    ) -> BackgroundTaskState:
        """Append a finished background task result to the session transcript."""
        return await self._background.collect(session_id, task_id)

    async def cancel_background_task(
        self,
        session_id: str,
        task_id: str,
    ) -> BackgroundTaskState:
        """Cancel a running process-local background task."""
        return await self._background.cancel(session_id, task_id)

    def available_models(self) -> list[str]:
        """Return model names available for session-scoped switching."""
        return self._models.available_models()

    def default_model(self) -> str | None:
        """Return the label for the construction-time/default model, if supplied."""
        return self._models.default_model()

    def model_switching_enabled(self) -> bool:
        """True when an LLM registry was supplied at construction time."""
        return self._models.enabled()

    def context_token_budget(self) -> int | None:
        """Return the live coordinator input budget used for compaction."""
        return self._coordinator_input_token_budget()

    async def model_overrides(self, session_id: str) -> dict[str, str]:
        """Return session-scoped agent_id -> model_name overrides."""
        return await self._models.overrides(session_id)

    async def switch_model(
        self,
        session_id: str,
        agent_id: str,
        model_name: str,
    ) -> SessionState:
        """Persist and apply a session-scoped model override for one agent."""
        return await self._models.switch(session_id, agent_id, model_name)

    async def clear_model_override(self, session_id: str, agent_id: str) -> SessionState:
        """Remove and apply a session-scoped model override for one agent."""
        return await self._models.clear(session_id, agent_id)

    async def chat(
        self,
        message: str,
        *,
        session_id: str = "default",
        run_id: str | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        """Run one user turn with fresh working memory and durable session context.

        When plan mode is enabled for the session, generates a plan first,
        yields ``PLAN_PROPOSED``, and gates execution on HITL approval.
        Rejection or unrecoverable plan-generation failure yields an
        ``ERROR`` and returns without writing to the session store.
        """
        state = await self._session_store.load(session_id)
        self._models.apply(state)
        active_session_token = self._active_session_id.set(session_id)
        run_id = run_id or str(uuid.uuid4())
        try:
            memory_context_text = await self._mem.context(session_id, message=message)
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

            # Plan-mode gate. Inserts a pinned plan prior into the priors list
            # before the ReAct loop runs; or yields ERROR and returns if the
            # user rejected the plan. Plan generation + HITL approval both
            # propagate ``asyncio.CancelledError`` (Esc-cancel) cleanly without
            # writing to the session store.
            if state.plan_mode_enabled:
                # Inline the plan-and-approve loop so PLAN_PROPOSED events
                # yield to the consumer (renderer) BEFORE the approval
                # banner blocks on stdin. Yielding directly preserves the
                # order: plan first, then banner, then user input.
                #
                # Uses ``request_plan_approval`` (sibling of the per-tool
                # ``request_approval``) — same input plumbing, but no
                # ``a`` / ``A`` options. Session-allow / persistent-allow
                # make sense when the "tool" is something like
                # ``shell/grep`` that fires repeatedly; they're actively
                # wrong when the gate is the plan itself (``a`` would
                # silently turn plan mode off for the session).
                from harness.hitl import request_plan_approval  # noqa: PLC0415

                approved_plan: dict[str, Any] | None = None
                plan_rejected = False
                correction: str | None = None

                for revision in range(_PLAN_REVISION_LIMIT + 1):
                    try:
                        candidate_plan = await self._generate_plan(
                            message=message,
                            session_id=session_id,
                            correction=correction,
                        )
                    except Exception as exc:  # noqa: BLE001 — surface as ERROR
                        yield BusEvent.error_event(
                            self._coordinator.config.agent_id,
                            error=f"plan generation failed: {exc}",
                        )
                        plan_rejected = True
                        break

                    # Yield FIRST so the renderer prints the plan, then
                    # block for approval so the user is responding to a
                    # plan they've actually seen.
                    yield BusEvent.plan_proposed(
                        self._coordinator.config.agent_id,
                        plan=candidate_plan,
                        revision=revision,
                    )

                    response = await request_plan_approval(
                        summary=candidate_plan.get("summary", ""),
                        step_count=len(candidate_plan.get("steps", [])),
                        dynamic_step_count=_dynamic_step_count(candidate_plan),
                        agent_id=self._coordinator.config.agent_id,
                        guard=turn_guard,
                    )
                    if response.approved:
                        approved_plan = candidate_plan
                        break
                    if response.correction:
                        # Free-text revision — re-plan with this feedback.
                        correction = response.correction
                        continue
                    # Plain rejection (no correction text).
                    yield BusEvent.error_event(
                        self._coordinator.config.agent_id, error="plan rejected by user"
                    )
                    plan_rejected = True
                    break
                else:
                    # Revision budget exhausted — give the user a clean
                    # rejection rather than silently executing the last plan.
                    yield BusEvent.error_event(
                        self._coordinator.config.agent_id,
                        error=(
                            f"plan revision limit ({_PLAN_REVISION_LIMIT}) reached; "
                            "send 'y' to approve, 'n' to reject, or shorten the feedback"
                        ),
                    )
                    plan_rejected = True

                if plan_rejected:
                    return
                if approved_plan is not None:
                    # Inject as a pinned prior so the ReAct loop's system
                    # prompt + memory prior stays byte-identical across the
                    # turn, and the plan rides as a user/assistant pair the
                    # coordinator clearly sees as "approved guidance."
                    plan_priors = [
                        ("user", _render_plan_for_priors(approved_plan)),
                        ("assistant", "Acknowledged. Executing the approved plan."),
                    ]
                    prior_messages = list(prior_messages) + plan_priors
                    pinned_priors += len(plan_priors)

            final_result: dict | None = None
            trace: list[dict[str, Any]] = []
            tools_used: set[str] = set()
            subagents_used: set[str] = set()
            errors: list[str] = []
            finalized = False

            restore_hitl_context = self._disable_checkpoint_resume_for_persistent_hitl()
            try:
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
                    if not event.parent_agent_id and event.type in (
                        EventType.TASK_DONE,
                        EventType.ERROR,
                    ):
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
            finally:
                restore_hitl_context()

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
        finally:
            self._active_session_id.reset(active_session_token)

    async def _generate_plan(
        self,
        *,
        message: str,
        session_id: str,  # noqa: ARG002 — reserved for future per-session context
        correction: str | None,
    ) -> dict[str, Any]:
        """Call the coordinator's LLM with the planner system prompt.

        Returns the parsed plan dict. Raises ``ValueError`` if the LLM
        response can't be coerced into the expected ``{summary, steps}``
        shape.
        """
        tools = self._available_tool_names()
        system = _PLAN_SYSTEM_PROMPT.replace("{tools}", ", ".join(tools) or "(none)")
        if correction:
            system = system + _PLAN_CORRECTION_HINT.replace("{correction}", correction)

        raw = await self.llm.complete(
            system=system,
            messages=[{"role": "user", "content": message}],
            response_format={"type": "json_object"},
            source="planner",
        )
        plan = _coerce_plan(raw)
        if plan is None:
            raise ValueError(
                "planner returned a response that could not be parsed as "
                "a {summary, steps} JSON object"
            )
        return plan

    def _available_tool_names(self) -> list[str]:
        """Tools the coordinator (and any wired sub-agents at first level)
        can call. Surfaces this to the planner so it doesn't propose tools
        that don't exist."""
        names: set[str] = set()
        names.update(self._coordinator.config.allowed_tools or [])
        # Also surface sub-agent delegate-tool names so plan mode can
        # reference them — the SubAgentTool registers as a tool on the
        # coordinator with name ``delegate_<sub_agent_id>``.
        for tool_name in getattr(self._coordinator, "_tools", {}):
            names.add(str(tool_name))
        return sorted(names)

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
                sub = subagent_tool_agent(tool)
                if sub is None:
                    continue
                sub_info = {
                    "agent_id": sub.config.agent_id,
                    "role": sub.role,
                    "skills": _describe_skills(sub),
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
        return self._mem.cached(session_id)

    def forget_memory_cache(self, session_id: str) -> None:
        """Drop cached memory context so the next turn fetches fresh memory."""
        self._mem.evict(session_id)

    async def force_compact(self, session_id: str) -> SessionState:
        """Summarize, trim, and reconcile the older transcript portion."""
        return await self._mem.force_compact(session_id)

    async def save_to_memory(self, session_id: str) -> int:
        """Reconcile pending transcript messages into long-term memory now.

        Used when a user explicitly wants "save what we discussed" before
        leaving the session (e.g. demo ``/save`` command). Returns the number
        of transcript messages included in the reconcile payload.
        """
        return await self._mem.save_now(session_id)

    async def clear_session(self, session_id: str) -> SessionState:
        """Clear transcript/summary/counters for a session.

        Long-term semantic/episodic memory is intentionally retained; this
        only resets the durable chat workspace for ``session_id``.
        """
        state = await self._session_store.clear(session_id)
        self._mem.evict(session_id)
        return state

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session transcript/summary row.

        Long-term semantic/episodic memory is intentionally retained.
        Returns ``True`` when a session existed and was deleted.
        """
        deleted = await self._session_store.delete(session_id)
        self._mem.evict(session_id)
        return deleted

    def _iter_agents(self) -> list[BaseAgent]:
        return iter_agents(self._coordinator)

    def _disable_checkpoint_resume_for_persistent_hitl(self) -> Callable[[], None]:
        agents = self._iter_agents()
        originals = [
            (
                agent,
                getattr(agent, "_checkpoint_resume_enabled", True),
                getattr(agent, "_hitl_resume_hint", None),
            )
            for agent in agents
        ]
        hint = "Esc cancels this turn; completed session history is preserved."
        for agent in agents:
            agent._checkpoint_resume_enabled = False
            agent._hitl_resume_hint = hint

        def restore() -> None:
            for agent, resume_enabled, resume_hint in originals:
                agent._checkpoint_resume_enabled = resume_enabled
                agent._hitl_resume_hint = resume_hint

        return restore

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
                sub = subagent_tool_agent(tool)
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

        # Owns SessionMessage construction; the controller only reads the
        # resulting transcript to drive reconciliation/compaction.
        state = await self._session_store.append_messages(
            session_id,
            [
                SessionMessage(role="user", content=message),
                SessionMessage(role="assistant", content=answer),
            ],
        )

        await self._mem.finalize_turn(
            session_id,
            state=state,
            message=message,
            final_result=final_result,
            trace=trace,
            tools_used=tools_used,
            subagents_used=subagents_used,
            errors=errors,
        )

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
        # at position 0. It's now retrieved + cached per-session by
        # ``SessionMemoryController`` and passed in as ``memory_context_text``.
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

    def _coordinator_input_token_budget(self) -> int | None:
        """Read the coordinator LLM's input budget. Falls back to ``None``
        when the adapter doesn't advertise one."""
        llm = getattr(self._coordinator, "_llm", None) or self.llm
        budget = getattr(llm, "input_token_budget", None)
        return int(budget) if isinstance(budget, int) and budget > 0 else None


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
        model_overrides=dict(state.model_overrides),
        plan_mode_enabled=state.plan_mode_enabled,
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
        "skills": _describe_skills(agent),
        "tools": sorted(tools.keys()),
        "mcp_tools": [
            _describe_mcp_tool(tool, owner_agent_id=agent.config.agent_id)
            for tool in tools.values()
            if _is_mcp_tool(tool)
        ],
    }


def _describe_skills(agent: BaseAgent) -> list[dict[str, Any]]:
    return [skill.summary() for skill in getattr(agent.config, "skills", [])]


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
