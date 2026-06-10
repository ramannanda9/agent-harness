"""
WorkingMemory — per-agent, per-run in-context memory.

Eviction strategy: rolling structured summary with a recency window.

When total tokens exceed `max_tokens`, the oldest unpinned messages outside
the recency window are folded into a single structured summary block
(sections: Facts / Tools used / Errors / Open questions). At most one
summary block exists at any time — subsequent evictions EXTEND the existing
summary (the LLM sees the prior structured summary plus the new batch and
returns an updated structured summary) instead of re-summarizing its own
paragraph output, avoiding fidelity decay across passes.

The recency window (last N non-pinned, non-summary messages) is protected
from eviction in normal operation and is only relaxed when budget pressure
forces it. The summary's role is always set opposite the next non-pinned
non-summary message so the ReAct user/assistant alternation invariant holds.

Token counting: chars/4 heuristic by default — stable across content types
(code, JSON, English, non-English) within ~10–20% of real BPE counts, with
zero dependencies. For exact counts, pass a custom `token_counter`:

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    wm = WorkingMemory(llm=..., token_counter=lambda s: len(enc.encode(s)))
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# ── Token counting ────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """
    Chars-per-4 heuristic — the standard "~4 chars per token" rule. Stable
    across JSON, code, English, and most non-English text within ~10–20% of
    real BPE counts. Override via WorkingMemory(token_counter=…) for exact.
    """
    return max(1, len(text) // 4) if text else 0


# Rough token cost for a single image block regardless of resolution.
# GPT-4o "auto" detail ≈ 85–1700 tokens depending on size; 500 is a conservative
# mid-point that avoids under-counting without being too aggressive.
_IMAGE_TOKEN_ESTIMATE = 500


def _count_content(content: str | list, counter: Callable[[str], int]) -> int:
    """Token count for a message whose content may be a string or a content-block list."""
    if isinstance(content, str):
        return counter(content)
    total = sum(
        counter(block.get("text", ""))
        if isinstance(block, dict) and block.get("type") == "text"
        else _IMAGE_TOKEN_ESTIMATE
        for block in content
    )
    return max(1, total) if total else 0


# ── LLM Protocol — injected, not imported ─────────────────────────────────────


class LLMClient(Protocol):
    async def complete(
        self,
        system: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict: ...


# ── Working Memory ────────────────────────────────────────────────────────────


def _format_for_summary(m: Message) -> str:
    """Render a message as plain text for the summarization LLM.

    Image content blocks become "[image]" so a text-only summarizer can still
    produce a useful summary that acknowledges the image was present.
    """
    if isinstance(m.content, str):
        return f"[{m.role.upper()}]: {m.content}"
    parts = []
    for block in m.content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append("[image]")
    return f"[{m.role.upper()}]: {''.join(parts)}"


# Marker emitted by new summaries. Legacy checkpoints from the previous
# implementation used "[Memory compressed]:" — both are recognized on load.
SUMMARY_HEADER = "[Memory summary]"
_LEGACY_SUMMARY_PREFIXES = ("[Memory summary]", "[Memory compressed]")


SUMMARIZE_SYSTEM = """
You are a memory compressor for an AI agent.
Produce a structured summary of the conversation messages below.

Use exactly this format, omitting any section that has no entries:

[Memory summary]
Facts:
- <one fact per bullet>
Tools used:
- <tool>: <one-line outcome>
Errors:
- <error or failed approach>
Open questions:
- <unresolved item or next step>

Rules:
- One short line per bullet. No multi-sentence bullets.
- Preserve concrete details (file paths, names, numbers, error messages).
- Discard pleasantries, restated context, and verbose tool output.
- Output ONLY the [Memory summary] block — no preamble, no closing remarks.
""".strip()


EXTEND_SUMMARY_SYSTEM = """
You are a memory compressor for an AI agent.
You are given an existing structured summary and a batch of new conversation
messages. Produce an UPDATED structured summary that merges the new
information into the existing one.

Rules:
- Use the same format and section headers as the existing summary.
- Merge or deduplicate bullets where the new messages elaborate on or
  resolve existing items (e.g. an open question becoming a fact).
- Add new bullets for genuinely new information.
- Keep bullets short, one line each. Preserve concrete details.
- Omit sections with no entries.
- Output ONLY the [Memory summary] block — no preamble, no closing remarks.
""".strip()


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str | list  # str for text; list of content blocks for multimodal
    token_count: int = 0  # set by WorkingMemory.append using its configured counter
    pinned: bool = False  # pinned messages are never evicted (e.g. system prompt)
    is_summary: bool = False  # marks the rolling-summary message (at most one)


class WorkingMemory:
    """
    Token-budget-aware in-context memory for a single agent run.

    Parameters:
        llm: client used for compression calls.
        max_tokens: budget; eviction fires when total exceeds this.
        summarize_ratio: fraction of *eligible* messages folded in per
            eviction call. Eligible = non-pinned, non-summary, outside the
            recency window.
        recency_window: number of trailing non-pinned, non-summary messages
            protected from eviction in normal operation. The window is
            relaxed (oldest-protected-first) if budget forces it. Default 4
            preserves the last two ReAct steps verbatim.
        token_counter: optional exact counter; defaults to chars/4 heuristic.

    Eviction:
        - If a prior summary exists, the new batch is folded into it
          (extend mode); otherwise a fresh summary is created.
        - The new summary occupies the slot of the oldest message in the
          replaced set, with its role set opposite the next non-pinned
          non-summary message to preserve ReAct alternation.
        - Up to two compaction passes fire per append() before falling
          back to a hard FIFO drop (which still protects the recency
          window until forced to relax it).
    """

    def __init__(
        self,
        llm: LLMClient,
        max_tokens: int | None = None,
        summarize_ratio: float = 0.5,  # summarize oldest 50% of eligible when evicting
        recency_window: int = 4,  # protect last N non-pinned non-summary messages
        token_counter: Callable[[str], int] | None = None,
        compact_at_fraction: float = 0.8,
    ) -> None:
        self._llm = llm
        # Auto-derive from the LLM's input budget when the caller didn't
        # set an explicit cap — 80% of available headroom leaves room for
        # system prompt + tool schemas + tokeniser variance, and adapts
        # automatically when the user swaps models. Falls back to 32K for
        # LLMs that don't expose ``input_token_budget`` (custom adapters,
        # test stubs).
        if max_tokens is None:
            budget = getattr(llm, "input_token_budget", None)
            max_tokens = (
                int(budget * compact_at_fraction)
                if isinstance(budget, int) and budget > 0
                else 32_000
            )
        self.max_tokens = max_tokens
        self.summarize_ratio = summarize_ratio
        self.recency_window = max(0, recency_window)
        self._count = token_counter or count_tokens
        self._messages: list[Message] = []
        self._token_total: int = 0
        self._summarization_count: int = 0
        self._eviction_lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def append(self, role: str, content: str | list, pinned: bool = False) -> None:
        msg = Message(
            role=role,
            content=content,
            pinned=pinned,
            token_count=_count_content(content, self._count),
        )
        self._messages.append(msg)
        self._token_total += msg.token_count

        if self._token_total > self.max_tokens:
            async with self._eviction_lock:
                if self._token_total > self.max_tokens:
                    await self._evict_unlocked()

    def get_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def token_count(self) -> int:
        return self._token_total

    def context_usage(self) -> dict:
        percent = self._token_total / self.max_tokens if self.max_tokens > 0 else 0.0
        if percent >= 0.95:
            level = "critical"
        elif percent >= 0.80:
            level = "warning"
        else:
            level = "normal"
        return {
            "tokens": self._token_total,
            "max_tokens": self.max_tokens,
            "percent": percent,
            "level": level,
            "messages": len(self._messages),
            "summarizations": self._summarization_count,
        }

    def clear(self) -> None:
        self._messages.clear()
        self._token_total = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for checkpoint storage."""
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "pinned": m.pinned,
                    "token_count": m.token_count,
                    "is_summary": m.is_summary,
                }
                for m in self._messages
            ],
            "summarization_count": self._summarization_count,
            "max_tokens": self.max_tokens,
            "summarize_ratio": self.summarize_ratio,
            "recency_window": self.recency_window,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        llm: LLMClient,
        token_counter: Callable[[str], int] | None = None,
    ) -> WorkingMemory:
        """Restore from a checkpoint dict. Stored token counts are reused as-is.

        Legacy checkpoints (no `is_summary` field, no `recency_window`) are
        backfilled: content prefixed with a known summary marker is treated
        as `is_summary=True`, and `recency_window` defaults to 4.
        """
        wm = cls(
            llm=llm,
            max_tokens=data["max_tokens"],
            summarize_ratio=data["summarize_ratio"],
            recency_window=data.get("recency_window", 4),
            token_counter=token_counter,
        )
        for m in data["messages"]:
            content = m["content"]
            is_summary = m.get("is_summary")
            if is_summary is None:
                is_summary = isinstance(content, str) and any(
                    content.startswith(p) for p in _LEGACY_SUMMARY_PREFIXES
                )
            wm._messages.append(
                Message(
                    role=m["role"],
                    content=content,
                    pinned=m["pinned"],
                    token_count=m["token_count"],
                    is_summary=bool(is_summary),
                )
            )
        wm._token_total = sum(msg.token_count for msg in wm._messages)
        wm._summarization_count = data["summarization_count"]
        return wm

    @property
    def summarization_count(self) -> int:
        return self._summarization_count

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _eligible_indices(self, relax_recency: int = 0) -> list[int]:
        """Indices of summarizable messages (non-pinned, non-summary), oldest first.

        The newest `recency_window - relax_recency` non-pinned non-summary
        messages are protected; everything older is eligible. Walks from
        newest backward so the protection count is robust to interleaved
        pinned/summary messages.
        """
        protect = max(0, self.recency_window - relax_recency)
        candidates_back: list[int] = []
        for i in range(len(self._messages) - 1, -1, -1):
            m = self._messages[i]
            if m.pinned or m.is_summary:
                continue
            candidates_back.append(i)
        # candidates_back is newest-first; protect the first `protect` of them.
        return sorted(candidates_back[protect:])

    async def _evict_unlocked(self, _depth: int = 0) -> None:
        # Find eligible messages, relaxing recency_window only if necessary.
        relax = 0
        eligible_idx = self._eligible_indices(relax)
        while not eligible_idx and relax <= self.recency_window:
            relax += 1
            eligible_idx = self._eligible_indices(relax)

        if eligible_idx:
            cutoff = max(1, int(len(eligible_idx) * self.summarize_ratio))
            chosen_idx = eligible_idx[:cutoff]
            to_summarize = [self._messages[i] for i in chosen_idx]

            prior_summary = next((m for m in self._messages if m.is_summary), None)
            prior_summary_idx = (
                next(
                    (i for i, m in enumerate(self._messages) if m is prior_summary),
                    None,
                )
                if prior_summary is not None
                else None
            )
            insert_idx = min(
                [*chosen_idx, *([] if prior_summary_idx is None else [prior_summary_idx])]
            )
            if prior_summary is None:
                summary_text = await self._summarize_initial(to_summarize)
            else:
                summary_text = await self._summarize_extend(prior_summary.content, to_summarize)

            summary_content = (
                summary_text
                if summary_text.startswith(SUMMARY_HEADER)
                else f"{SUMMARY_HEADER}\n{summary_text}"
            )

            # Remove the picked messages AND the prior summary (if any); the
            # new summary replaces both.
            removed_ids = {id(m) for m in to_summarize}
            if prior_summary is not None:
                removed_ids.add(id(prior_summary))

            if not any(id(m) in removed_ids for m in self._messages):
                # The summarizer is awaited above. If callers accidentally
                # mutate the same WorkingMemory concurrently, the selected
                # messages may already be gone. Do not reinsert a stale
                # summary or leak StopIteration out of this coroutine; fall
                # through to the second-pass / hard-drop safety logic.
                self._token_total = sum(m.token_count for m in self._messages)
            else:
                remaining = [m for m in self._messages if id(m) not in removed_ids]

                # Role = opposite of the next non-pinned non-summary message so
                # the ReAct alternating invariant holds. No such message → "user".
                first_after = next(
                    (m for m in remaining if not m.pinned and not m.is_summary),
                    None,
                )
                summary_role = (
                    "assistant" if (first_after and first_after.role == "user") else "user"
                )

                summary_msg = Message(
                    role=summary_role,
                    content=summary_content,
                    token_count=self._count(summary_content),
                    is_summary=True,
                )

                self._messages = remaining
                self._messages.insert(min(insert_idx, len(self._messages)), summary_msg)
                self._token_total = sum(m.token_count for m in self._messages)
                self._summarization_count += 1

        # Second pass before resorting to hard drops (max 2 passes).
        if self._token_total > self.max_tokens and _depth < 1:
            await self._evict_unlocked(_depth=_depth + 1)
            return

        # Safety valve: hard FIFO drop. Drops oldest non-pinned, non-summary,
        # non-recency-window messages first. The newest ReAct action /
        # observation pair is the provider-valid continuation point; dropping
        # only its user observation can leave the transcript ending with
        # assistant, which Bedrock-compatible providers reject as assistant
        # prefill. Only relax recency protection when there is no older
        # unpinned material left.
        hard_dropped = False
        while self._token_total > self.max_tokens:
            protected_recent = set(self._protected_recent_indices())
            drop_idx = next(
                (
                    i
                    for i, m in enumerate(self._messages)
                    if not m.pinned and not m.is_summary and i not in protected_recent
                ),
                None,
            )
            if drop_idx is None:
                drop_idx = next(
                    (i for i, m in enumerate(self._messages) if not m.pinned and m.is_summary),
                    None,
                )
            if drop_idx is None:
                break  # only pinned/protected-recent messages remain — accept overshoot
            self._token_total -= self._messages[drop_idx].token_count
            self._messages.pop(drop_idx)
            hard_dropped = True

        if hard_dropped:
            self._repair_trailing_assistant()

    def _protected_recent_indices(self) -> list[int]:
        """Newest non-pinned, non-summary indices protected by recency_window."""
        if self.recency_window <= 0:
            return []
        protected: list[int] = []
        for i in range(len(self._messages) - 1, -1, -1):
            m = self._messages[i]
            if m.pinned or m.is_summary:
                continue
            protected.append(i)
            if len(protected) >= self.recency_window:
                break
        return protected

    def _repair_trailing_assistant(self) -> None:
        """Never leave the transcript ending with an assistant message.

        This is a last-resort repair after hard drops. It removes trailing
        non-pinned assistant messages rather than fabricating a user message.
        """
        while self._messages and self._messages[-1].role == "assistant":
            last = self._messages[-1]
            if last.pinned:
                break
            self._token_total -= last.token_count
            self._messages.pop()

    # ── Summarization helpers ─────────────────────────────────────────────────

    async def _summarize_initial(self, messages: list[Message]) -> str:
        formatted = "\n".join(_format_for_summary(m) for m in messages)
        return await self._call_llm(SUMMARIZE_SYSTEM, formatted)

    async def _summarize_extend(
        self,
        prior_summary_content: str | list,
        new_messages: list[Message],
    ) -> str:
        # prior_summary_content is expected to be a string (summaries are
        # always text), but defensively handle the list-of-blocks case too.
        if isinstance(prior_summary_content, list):
            prior_text = "".join(
                b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else "[image]"
                for b in prior_summary_content
            )
        else:
            prior_text = prior_summary_content
        new_text = "\n".join(_format_for_summary(m) for m in new_messages)
        user_content = f"Existing summary:\n{prior_text}\n\nNew messages to fold in:\n{new_text}"
        return await self._call_llm(EXTEND_SUMMARY_SYSTEM, user_content)

    async def _call_llm(self, system: str, user_content: str) -> str:
        try:
            result = await self._llm.complete(
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            if isinstance(result, dict):
                return result.get("text") or result.get("answer") or str(result)
            return str(result)
        except Exception as e:
            # Fallback: truncated raw context — never let summarization break the agent.
            fallback = user_content[:500]
            return f"{SUMMARY_HEADER}\n[Summarization failed: {e}] Truncated context: {fallback}"
