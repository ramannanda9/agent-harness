"""Session memory & compaction policy for ``PersistentAgent``.

``SessionMemoryController`` owns the full memory/compaction lifecycle that used
to be smeared across ``PersistentAgent``:

  - the per-session memory-context cache (retrieve once, reuse until eviction)
  - the reconciliation policy (when to fold transcript into long-term memory)
  - context-pressure compaction (summarise + trim the older transcript window)
  - background reconciliation scheduling
  - the reconcile-at-compaction dedup

It holds no back-reference to ``PersistentAgent``; everything it needs is
injected at construction (the same stateless-accessor pattern used elsewhere in
the codebase). The session record types are imported only for type-checking —
the controller never constructs a ``SessionMessage`` (the turn append stays in
``PersistentAgent``), so there is no runtime import cycle with ``persistent``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from harness.utils import fire

if TYPE_CHECKING:
    from harness.persistent import (
        PersistentAgentConfig,
        SessionMessage,
        SessionState,
        SessionStore,
    )
    from memory.manager import MemoryManager


class SessionMemoryController:
    """Encapsulates memory retrieval, reconciliation, and compaction for one
    ``PersistentAgent`` instance (covering all of its sessions)."""

    def __init__(
        self,
        *,
        memory: MemoryManager,
        session_store: SessionStore,
        config: PersistentAgentConfig,
        coordinator_agent_id: str,
        token_budget: Callable[[], int | None],
        summarizer_llm: Callable[[], Any],
    ) -> None:
        self._memory = memory
        self._session_store = session_store
        self._config = config
        self._coordinator_agent_id = coordinator_agent_id
        self._token_budget = token_budget
        self._summarizer_llm = summarizer_llm
        # Per-session memory-context cache. Memory retrieval used to fire on
        # every turn (inside ``_build_system_prompt``), which made the system
        # prompt content-dependent on whatever ``build_context`` returned for
        # the current goal — defeating prefix caching from position 0. We now
        # retrieve once per session, hold the rendered string here, and evict
        # only at compaction (when the cache is already breaking anyway).
        # Within a compaction window, memory context is byte-identical across
        # turns.
        self._cache: dict[str, str] = {}

    # ── Memory-context cache ──────────────────────────────────────────────────

    def cached(self, session_id: str) -> str | None:
        """Return the currently cached memory-context blob, if any."""
        return self._cache.get(session_id)

    def evict(self, session_id: str) -> None:
        """Drop the cached memory context so the next turn re-fetches. Called
        at compaction, where the cache is already breaking from the summary
        refresh anyway."""
        self._cache.pop(session_id, None)

    async def context(self, session_id: str, *, message: str) -> str:
        """Return the per-session memory-context blob (cached).

        First call for the session fetches via ``MemoryManager.build_context``,
        rendering whatever semantic + episodic context is relevant to the
        first goal. The result is cached keyed by ``session_id`` and reused for
        every subsequent turn in the same compaction window — see ``evict`` for
        the eviction path (compaction).
        """
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        try:
            mem_context = await self._memory.build_context(
                goal=message,
                agent_id=self._coordinator_agent_id,
            )
        except Exception:  # noqa: BLE001 — memory backend hiccup shouldn't crash chat
            mem_context = None
        rendered = ""
        if mem_context is not None and not mem_context.is_empty():
            rendered = mem_context.render()
        self._cache[session_id] = rendered
        return rendered

    # ── Turn finalization ─────────────────────────────────────────────────────

    async def finalize_turn(
        self,
        session_id: str,
        *,
        state: SessionState,
        message: str,
        final_result: dict[str, Any] | None,
        trace: list[dict[str, Any]],
        tools_used: set[str],
        subagents_used: set[str],
        errors: list[str],
    ) -> None:
        """Run the reconcile + compaction policy for a finished turn.

        ``state`` must already reflect the turn's appended messages (the caller
        owns ``SessionMessage`` construction). This method only reads the
        transcript and drives reconciliation/compaction via the session store
        and memory manager.
        """
        sync_reconciled = self.should_reconcile(
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
            # differ. Drop the cache so the refresh on turn N+1 picks them up.
            # This turn already paid a cache miss (tool / signal caused the
            # reconcile); the immediate next turn pays at most one more if the
            # new facts actually changed retrieval.
            self.evict(session_id)

        compacted = self.should_compact(state)
        if not sync_reconciled and not compacted:
            # Background memory accumulation. The fire-and-forget reconcile
            # uses the durable transcript window — no buffer to maintain — and
            # intentionally does NOT evict the per-session memory context
            # cache. New facts land in the long-term store immediately for
            # OTHER sessions; THIS session sees them at the next compaction.
            self._maybe_fire_async_reconcile(session_id, state)

        if compacted:
            to_compact = self.messages_to_compact(state)
            compacted_state = await self._compact_session(session_id, state)
            if compacted_state.last_compact_turn != state.last_compact_turn:
                state = compacted_state
                # Compaction is the natural moment to reconcile the older
                # transcript window into long-term memory: the reconciler LLM
                # call colocates with the summary write, both touching the same
                # cache-miss boundary so we don't pay twice for cache
                # invalidation. Also evict the cached memory context so the
                # next turn's first build_context call picks up any facts the
                # reconciler added/updated.
                if not self.should_reconcile(
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
                self.evict(session_id)

    # ── Explicit (user-triggered) operations ──────────────────────────────────

    async def save_now(self, session_id: str) -> int:
        """Reconcile pending transcript messages into long-term memory now.

        Samples only messages after ``last_reconcile_turn``, then awaits the
        ``write_run_end`` call so the caller can confirm completion. Use when a
        user explicitly wants "save what we discussed" before leaving the
        session (e.g. demo ``/save`` command).

        Crucially does NOT evict the per-session memory cache: the active
        session keeps its warm prefix. New facts are visible to OTHER sessions
        immediately and to THIS session at the next compaction (where the cache
        breaks anyway for the summary refresh).

        Returns the number of transcript messages included in the reconcile
        payload — 0 if there's nothing pending to save.
        """
        state = await self._session_store.load(session_id)
        if not state.messages:
            return 0
        window = self.messages_since_reconcile(state)
        if not window:
            return 0
        await self._write_session_window_to_memory(
            session_id=session_id,
            messages=window,
            goal_fallback="(explicit save)",
        )
        await self._session_store.mark_reconciled(session_id, state.turn_count)
        return len(window)

    async def force_compact(self, session_id: str) -> SessionState:
        """Summarize, trim, and reconcile the older transcript portion.

        This uses the same compaction shape as automatic context-pressure
        compaction: keep the newest messages that fit inside
        ``retain_context_fraction`` of the input budget, fold older messages
        into the rolling summary, reconcile the folded window into long-term
        memory, then evict cached memory context so the next turn can pick up
        freshly reconciled facts.
        """
        state = await self._session_store.load(session_id)
        to_compact = self.messages_to_compact(state)
        state = await self._compact_session(session_id, state)
        if to_compact and state.last_compact_turn:
            await self._write_session_window_to_memory(
                session_id=session_id,
                messages=to_compact,
                goal_fallback="(force compact)",
            )
            state = await self._session_store.mark_reconciled(session_id, state.turn_count)
        self.evict(session_id)
        return state

    # ── Decisions (pure given state + injected budget) ─────────────────────────

    def should_reconcile(
        self,
        *,
        message: str,
        state: SessionState,
        tools_used: set[str],
        subagents_used: set[str],
        errors: list[str],
    ) -> bool:
        """User-intent-only immediate reconciliation policy.

        Compaction handles bulk reconciliation when context pressure forces a
        summary LLM call (the same boundary already pays a cache miss). Tool
        runs / sub-agent runs / errors used to also trigger reconciliation
        here, but their facts are mostly situational (this run's outputs) — the
        session transcript captures them within the session, and the next
        compaction boundary folds them into long-term memory. Bypassing
        compaction for every tool run was just trashing the cache for marginal
        cross-session benefit.

        What stays: user-explicit signals like "remember X", "from now on Y" —
        these are cross-session by nature and should persist immediately even
        if it costs a cache miss.
        """
        lower = message.lower()
        return any(term in lower for term in self._config.durable_signal_terms)

    def should_compact(self, state: SessionState) -> bool:
        """Context-pressure trigger — fires when the accumulated transcript
        (rolling summary + verbatim history) crosses
        ``compact_at_context_fraction`` of the coordinator's
        ``llm.input_token_budget``.

        Plain chat sessions (~20 tokens/turn) go thousands of turns between
        compactions; browser-heavy research sessions (~3K tokens/turn) compact
        only when budget pressure forces it.
        """
        budget = self._token_budget()
        if budget is None:
            # Adapter doesn't expose ``input_token_budget`` (custom stub or
            # older client) — never auto-compact; rely on explicit signals.
            return False
        threshold = int(budget * self._config.compact_at_context_fraction)
        if threshold <= 0:
            return False
        return self._transcript_token_count(state) >= threshold

    def messages_to_compact(self, state: SessionState) -> list[SessionMessage]:
        return self._compaction_split(state)[0]

    def messages_since_reconcile(self, state: SessionState) -> list[SessionMessage]:
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

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _transcript_token_count(self, state: SessionState) -> int:
        """Cheap chars/4 token estimate over the rolling summary + accumulated
        transcript. Same heuristic ``WorkingMemory`` uses when no exact counter
        is wired, so the threshold maps coherently to the LLM's eviction
        behaviour."""
        total = 0
        if state.summary:
            total += max(1, len(state.summary) // 4)
        for msg in state.messages:
            total += self._message_token_count(msg)
        return total

    def _message_token_count(self, message: SessionMessage) -> int:
        content = message.content if isinstance(message.content, str) else str(message.content)
        return max(1, len(content) // 4)

    def _compaction_split(self, state: SessionState) -> tuple[list[SessionMessage], int]:
        """Return ``(messages_to_compact, keep_last_count)``.

        Retention is token-budget based: keep newest messages until adding
        another would exceed ``retain_context_fraction`` of the coordinator's
        input budget. If no budget is advertised, compact the full verbatim
        transcript into the summary.
        """
        if not state.messages:
            return [], 0
        budget = self._token_budget()
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
                # Always keep the newest message, even if it alone exceeds the
                # retention target. Dropping the latest assistant reply makes
                # resumption feel broken and loses immediate context.
                keep_last = 1
                break
            kept_tokens += tokens
            keep_last += 1

        if keep_last >= len(state.messages):
            return [], len(state.messages)
        return state.messages[:-keep_last] if keep_last else list(state.messages), keep_last

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
                    "agent_id": self._coordinator_agent_id,
                    "answer": str(last_assistant),
                    "confidence": 1.0,
                    "session_id": session_id,
                }
            ],
            trace=trace,
        )

    def _maybe_fire_async_reconcile(self, session_id: str, state: SessionState) -> None:
        """Background reconcile every ``async_reconcile_every_turns`` turns.

        Samples the durable session transcript directly — no separate evidence
        buffer to maintain. Fires ``write_run_end`` via ``fire()`` so the chat
        turn never blocks on the reconciler LLM call, and crucially does NOT
        evict the per-session memory context cache. The session's prompt prefix
        therefore stays byte-identical → prefix cache keeps hitting.

        New facts land in the long-term store immediately for other sessions;
        THIS session sees them at the next compaction (where the cache is
        already breaking for the summary refresh anyway).
        """
        interval = self._config.async_reconcile_every_turns
        if interval <= 0:
            return
        if state.turn_count == 0 or state.turn_count % interval != 0:
            return

        # Sample the last N turn-pairs (= 2N messages) from the durable
        # transcript. Each turn is one user message + one assistant reply; the
        # reconciler sees a coherent slice of conversation and can make MERGE /
        # NOOP / UPDATE decisions across it.
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

    async def _compact_session(self, session_id: str, state: SessionState) -> SessionState:
        to_compact, keep_last = self._compaction_split(state)
        if not to_compact:
            return state
        summary = await self._summarize_session(state, messages_to_compact=to_compact)
        state = await self._session_store.update_summary(session_id, summary)
        state = await self._session_store.trim_messages(session_id, keep_last)
        state = await self._session_store.mark_compacted(session_id, state.turn_count)
        return state

    async def _summarize_session(
        self,
        state: SessionState,
        *,
        messages_to_compact: list[SessionMessage] | None = None,
    ) -> str:
        """Summarise the older portion of the session, folding any prior
        summary in. ``messages_to_compact`` defaults to the full state
        transcript (legacy path); the new compaction flow passes only the
        messages being trimmed so the still-verbatim recent tail stays out of
        the summary."""
        targets = messages_to_compact if messages_to_compact is not None else list(state.messages)
        rendered = "\n".join(f"{m.role}: {m.content}" for m in targets)
        response = await self._summarizer_llm().complete(
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
