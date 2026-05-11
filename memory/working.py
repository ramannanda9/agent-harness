"""
WorkingMemory — per-agent, per-run in-context memory.

Eviction strategy: when token budget is hit, summarize the oldest
50% of messages via LLM into a single compressed message, then drop
the originals. Summarization cost is always cheaper than what it replaces.

Token counting: uses a word-based estimate by default.
Swap count_tokens() for tiktoken in production for exact counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Fast approximation: ~1.3 tokens per word.
    Replace with tiktoken for production accuracy:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    """
    return int(len(text.split()) * 1.3)


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
    token_count: int = field(init=False)
    pinned: bool = False   # pinned messages are never evicted (e.g. system prompt)

    def __post_init__(self) -> None:
        self.token_count = count_tokens(self.content)


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
        max_tokens: int = 2000,
        summarize_ratio: float = 0.5,   # summarize oldest 50% when evicting
    ) -> None:
        self._llm = llm
        self.max_tokens = max_tokens
        self.summarize_ratio = summarize_ratio
        self._messages: list[Message] = []
        self._token_total: int = 0
        self._summarization_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def append(self, role: str, content: str, pinned: bool = False) -> None:
        msg = Message(role=role, content=content, pinned=pinned)
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
        tokens_before = sum(m.token_count for m in to_summarize)

        summary_text = await self._summarize(to_summarize)
        summary_msg = Message(role="user", content=f"[Memory summary]: {summary_text}")

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
