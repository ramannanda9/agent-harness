"""WorkingMemory eviction tests — role invariant and second-pass summarization.

All tests use token_counter=len (1 char = 1 token) for exact budget control.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory.working import WorkingMemory

# Each content string is this many chars = this many tokens with token_counter=len.
MSG = "x" * 100  # 100 tokens


def _wm(max_tokens: int, summary_text: str = "ok") -> tuple[WorkingMemory, list[str]]:
    """Build a WorkingMemory with a scripted LLM and per-call tracking."""
    calls: list[str] = []

    class ScriptedLLM:
        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            calls.append(summary_text)
            return {"text": summary_text}

    wm = WorkingMemory(llm=ScriptedLLM(), max_tokens=max_tokens, token_counter=len)
    return wm, calls


# ── Role invariant ─────────────────────────────────────────────────────────────
#
# Budget math (token_counter=len, MSG=100 chars):
#   system "s" = 1 token (pinned)
#
# "assistant" role test — 4 evictable, cutoff=2, first remaining = user:
#   max_tokens=350 → eviction fires when total hits 401 (4 non-pinned appended)
#   summarize [user_A, asst_B], first_after = user_C → summary must be "assistant"
#   summary = "[Memory compressed]: ok" = 23 tokens
#   after eviction: 1 + 23 + 100 + 100 = 224 < 350 → no second pass
#
# "user" role test — 3 evictable, cutoff=1, first remaining = assistant:
#   max_tokens=250 → eviction fires when total hits 301 (3 non-pinned appended)
#   summarize [user_A], first_after = asst_B → summary must be "user"
#   after eviction: 1 + 23 + 100 + 100 = 224 < 250 → no second pass


@pytest.mark.asyncio
async def test_summary_role_is_assistant_when_first_after_is_user():
    wm, _ = _wm(max_tokens=350)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)  # user_A  — will be summarized
    await wm.append("assistant", MSG)  # asst_B  — will be summarized
    await wm.append("user", MSG)  # user_C  — survives, first_after
    await wm.append("assistant", MSG)  # triggers eviction

    msgs = wm.get_messages()
    summary = next((m for m in msgs if "[Memory compressed]" in m["content"]), None)
    assert summary is not None, "eviction should have produced a summary"
    assert summary["role"] == "assistant", (
        f"expected 'assistant' (opposite of first_after='user'), got '{summary['role']}'"
    )


@pytest.mark.asyncio
async def test_summary_role_is_user_when_first_after_is_assistant():
    wm, _ = _wm(max_tokens=250)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)  # user_A  — will be summarized (cutoff=1)
    await wm.append("assistant", MSG)  # asst_B  — survives, first_after
    await wm.append("user", MSG)  # triggers eviction

    msgs = wm.get_messages()
    summary = next((m for m in msgs if "[Memory compressed]" in m["content"]), None)
    assert summary is not None, "eviction should have produced a summary"
    assert summary["role"] == "user", (
        f"expected 'user' (opposite of first_after='assistant'), got '{summary['role']}'"
    )


@pytest.mark.asyncio
async def test_no_consecutive_same_roles_after_multiple_evictions():
    """After one or more evictions the user/assistant alternation must hold."""
    wm, _ = _wm(max_tokens=250)
    await wm.append("system", "system prompt", pinned=True)
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        await wm.append(role, MSG)

    msgs = wm.get_messages()
    non_system = [m for m in msgs if m["role"] != "system"]
    for prev, cur in zip(non_system, non_system[1:], strict=False):
        assert prev["role"] != cur["role"], (
            f"consecutive '{prev['role']}' messages: "
            f"[{prev['content'][:30]}] then [{cur['content'][:30]}]"
        )


# ── Second-pass summarization ──────────────────────────────────────────────────
#
# Budget: max_tokens=40, token_counter=len, messages = "y" * 15 (15 tokens each)
#   sys "s" = 1 (pinned)
#   append user(15)   → total 16
#   append asst(15)   → total 31
#   append user(15)   → total 46 > 40 → first eviction, 3 evictable, cutoff=1
#
# First LLM call returns "z" * 200 = 221-char summary → still > 40 → second pass
# Second LLM call returns "ok" (small) → summarization_count ends at 2


@pytest.mark.asyncio
async def test_second_pass_fires_when_first_summary_still_over_budget():
    call_count = 0
    responses = ["z" * 200, "ok"]  # first too large, second fits

    class TwoPassLLM:
        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return {"text": resp}

    wm = WorkingMemory(llm=TwoPassLLM(), max_tokens=40, token_counter=len)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", "y" * 15)
    await wm.append("assistant", "y" * 15)
    await wm.append("user", "y" * 15)  # triggers eviction

    assert call_count == 2, f"expected 2 LLM calls (two-pass), got {call_count}"
    assert wm.summarization_count == 2


@pytest.mark.asyncio
async def test_hard_drop_not_used_when_second_pass_fits():
    """When the second-pass summary fits in budget, no messages are hard-dropped.

    Budget math (token_counter=len, messages="y"*15 = 15 tokens):
      sys "s" = 1 (pinned)
      append × 4 non-pinned → total = 1 + 4*15 = 61 > max_tokens=60 → first eviction
      4 evictable, cutoff=2 → summarize first two (30 tokens)
      first summary = "z"*200 = 221 tokens → total = 1 + 221 + 15 + 15 = 252 > 60 → second pass
      second summary = "ok" = 23 tokens → total = 1 + 23 + 15 + 15 = 54 < 60 → no hard drop
      final messages: [sys, second_summary, user, asst] = 4
    """

    class TwoPassLLM:
        def __init__(self):
            self._call = 0

        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            self._call += 1
            return {"text": "z" * 200 if self._call == 1 else "ok"}

    wm = WorkingMemory(llm=TwoPassLLM(), max_tokens=60, token_counter=len)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", "y" * 15)
    await wm.append("assistant", "y" * 15)
    await wm.append("user", "y" * 15)
    await wm.append("assistant", "y" * 15)  # triggers eviction

    msgs = wm.get_messages()
    # System + second summary + two surviving messages = 4 messages.
    # Hard drop would have removed one of the surviving messages.
    assert len(msgs) == 4, (
        f"expected sys + summary + 2 survivors = 4 messages, got {len(msgs)}: "
        + str([m["role"] for m in msgs])
    )


@pytest.mark.asyncio
async def test_summarization_count_tracks_actual_llm_calls():
    """summarization_count must equal the number of LLM compaction calls made."""

    class CountingLLM:
        def __init__(self):
            self.count = 0

        async def complete(self, system, messages, **_):
            self.count += 1
            return {"text": "short"}

    llm = CountingLLM()
    wm = WorkingMemory(llm=llm, max_tokens=40, token_counter=len)
    await wm.append("system", "s", pinned=True)
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        await wm.append(role, "y" * 15)

    assert wm.summarization_count == llm.count
    assert wm.summarization_count >= 1
