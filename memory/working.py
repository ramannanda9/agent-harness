"""
WorkingMemory — per-agent, per-run in-context memory.

Eviction strategy: when token budget is hit, summarize the oldest
50% of messages via LLM into a single compressed message, then drop
the originals. Summarization cost is always cheaper than what it replaces.

Token counting: uses a chars-per-4 heuristic by default — stable across
content types (code, JSON, English, non-English) within ~10–20% of real
tokenizer counts, with zero dependencies. For exact counts, pass a custom
`token_counter` callable to WorkingMemory:

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    wm = WorkingMemory(llm=..., token_counter=lambda s: len(enc.encode(s)))

Anthropic users can wrap their `count_tokens` API call similarly; just
be aware that remote calls in the eviction hot path add latency.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# ── Token counting ────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """
    Chars-per-4 heuristic — the standard "~4 characters per token" rule.
    Stable across JSON, code, English, and most non-English text within
    ~10–20% of real BPE counts. Override via WorkingMemory(token_counter=…)
    for exact counts.
    """
    return max(1, len(text) // 4) if text else 0


# ── LLM Protocol — injected, not imported ─────────────────────────────────────

class LLMClient(Protocol):
    async def complete(
        self,
        system: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict: ...


# ── Working Memory ────────────────────────────────────────────────────────────

SUMMARIZE_SYSTEM = """
You are a memory compressor for an AI agent.
Summarize the provided conversation messages into a single, dense paragraph.
Preserve: key facts discovered, tool results, decisions made, errors encountered.
Discard: pleasantries, repeated information, verbose tool output.
Output ONLY the summary paragraph — no preamble, no labels.
"""


@dataclass
class Message:
    role: str         # system | user | assistant
    content: str
    token_count: int = 0   # set by WorkingMemory.append using its configured counter
    pinned: bool = False   # pinned messages are never evicted (e.g. system prompt)


class WorkingMemory:
    """
    Token-budget-aware in-context memory for a single agent run.

    Eviction:
        When total tokens exceed max_tokens, the oldest unpinned 50% of
        messages are passed to the LLM for summarization, replaced by a
        single compressed message, and the originals are dropped.

        Summarization fires at most once per append() to avoid recursive
        compression. If the summarized result still exceeds budget, a hard
        FIFO drop is applied as a safety valve.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_tokens: int = 8000,
        summarize_ratio: float = 0.5,   # summarize oldest 50% when evicting
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._llm = llm
        self.max_tokens = max_tokens
        self.summarize_ratio = summarize_ratio
        self._count = token_counter or count_tokens
        self._messages: list[Message] = []
        self._token_total: int = 0
        self._summarization_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def append(self, role: str, content: str, pinned: bool = False) -> None:
        msg = Message(
            role=role, content=content, pinned=pinned, token_count=self._count(content),
        )
        self._messages.append(msg)
        self._token_total += msg.token_count

        if self._token_total > self.max_tokens:
            await self._evict()

    def get_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def token_count(self) -> int:
        return self._token_total

    def clear(self) -> None:
        self._messages.clear()
        self._token_total = 0

    @property
    def summarization_count(self) -> int:
        return self._summarization_count

    # ── Eviction ──────────────────────────────────────────────────────────────

    async def _evict(self) -> None:
        evictable = [m for m in self._messages if not m.pinned]
        if not evictable:
            return  # all messages pinned — hard budget violation, accept it

        # take oldest summarize_ratio of evictable messages
        cutoff = max(1, int(len(evictable) * self.summarize_ratio))
        to_summarize = evictable[:cutoff]

        summary_text = await self._summarize(to_summarize)
        summary_content = f"[Memory summary]: {summary_text}"
        summary_msg = Message(
            role="user", content=summary_content, token_count=self._count(summary_content),
        )

        # remove summarized messages, insert summary in their place
        summarized_set = set(id(m) for m in to_summarize)
        insert_idx = next(
            i for i, m in enumerate(self._messages) if id(m) in summarized_set
        )
        self._messages = [m for m in self._messages if id(m) not in summarized_set]
        self._messages.insert(insert_idx, summary_msg)

        # recompute token total
        self._token_total = sum(m.token_count for m in self._messages)
        self._summarization_count += 1

        # safety valve: if still over budget, hard FIFO drop non-pinned
        while self._token_total > self.max_tokens:
            for i, m in enumerate(self._messages):
                if not m.pinned:
                    self._token_total -= m.token_count
                    self._messages.pop(i)
                    break
            else:
                break  # nothing left to drop

    async def _summarize(self, messages: list[Message]) -> str:
        formatted = "\n".join(
            f"[{m.role.upper()}]: {m.content}" for m in messages
        )
        try:
            result = await self._llm.complete(
                system=SUMMARIZE_SYSTEM,
                messages=[{"role": "user", "content": formatted}],
            )
            # handle both raw string and dict response
            if isinstance(result, dict):
                return result.get("text") or result.get("answer") or str(result)
            return str(result)
        except Exception as e:
            # fallback: truncated concatenation — never let summarization break the agent
            fallback = formatted[:500]
            return f"[Summarization failed: {e}] Truncated context: {fallback}"
