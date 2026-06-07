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
    # Compact when the transcript token count crosses this fraction of
    # the coordinator's ``llm.input_token_budget``. Replaces the previous
    # turn-count / message-count triggers, which fired arbitrarily often
    # on chat-light sessions (wasting context budget) and too late on
    # tool-heavy sessions (risking input-window overrun). 0.7 leaves room
    # for the current turn's task, any tool observations it will produce,
    # and tokeniser variance.
    compact_at_context_fraction: float = 0.7
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
                        await self._memory.write_run_end(
                            goal=message,
                            agent_results=[
                                final_result or {"error": errors[-1] if errors else "no result"}
                            ],
                            trace=trace,
                        )
                    except Exception:  # noqa: BLE001 — best-effort at compaction
                        pass
                self._evict_session_memory_context(session_id)

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
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += max(1, len(content) // 4)
        return total

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
        last_user = next((m.content for m in reversed(window) if m.role == "user"), "")
        last_assistant = next((m.content for m in reversed(window) if m.role == "assistant"), "")
        trace = [
            {
                "type": m.role,
                "content": m.content,
                "timestamp": m.created_at,
            }
            for m in window
        ]
        fire(
            self._memory.write_run_end(
                goal=str(last_user) if last_user else "(background session reconcile)",
                agent_results=[
                    {
                        "agent_id": self._coordinator.config.agent_id,
                        "answer": str(last_assistant),
                        "confidence": 1.0,
                    }
                ],
                trace=trace,
            )
        )

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
