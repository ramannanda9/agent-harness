"""WorkingMemory eviction tests — role invariant, rolling summary, recency window.

All tests use token_counter=len (1 char = 1 token) for exact budget control.

Existing role-invariant and two-pass tests pin recency_window=0 to keep their
budget math focused on summarization behavior; dedicated tests cover the
recency window separately.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory.working import (
    EXTEND_SUMMARY_SYSTEM,
    SUMMARIZE_SYSTEM,
    SUMMARY_HEADER,
    WorkingMemory,
)

# Each content string is this many chars = this many tokens with token_counter=len.
MSG = "x" * 100  # 100 tokens

# Summary content for a scripted LLM returning "ok":
#   f"{SUMMARY_HEADER}\nok" = "[Memory summary]\nok" → 19 chars / 19 tokens.
_OK_SUMMARY_LEN = len(f"{SUMMARY_HEADER}\nok")


def _wm(
    max_tokens: int,
    summary_text: str = "ok",
    recency_window: int = 0,
) -> tuple[WorkingMemory, list[dict]]:
    """Build a WorkingMemory with a scripted LLM that records each call."""
    calls: list[dict] = []

    class ScriptedLLM:
        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            calls.append({"system": system, "messages": messages})
            return {"text": summary_text}

    wm = WorkingMemory(
        llm=ScriptedLLM(),
        max_tokens=max_tokens,
        recency_window=recency_window,
        token_counter=len,
    )
    return wm, calls


# ── max_tokens auto-derivation ───────────────────────────────────────────────


class _LLMWithBudget:
    """Adapter stub that reports an ``input_token_budget`` — mirrors what
    the real ``OpenAILLM`` / ``AnthropicLLM`` expose for WM to read."""

    def __init__(self, budget: int) -> None:
        self.input_token_budget = budget

    async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
        return {"text": "ok"}


def test_max_tokens_derives_from_llm_input_budget():
    """Default compact_at_fraction=0.8 → 80K out of 100K budget."""
    wm = WorkingMemory(llm=_LLMWithBudget(100_000))
    assert wm.max_tokens == 80_000


def test_max_tokens_compact_at_fraction_is_configurable():
    wm = WorkingMemory(llm=_LLMWithBudget(100_000), compact_at_fraction=0.5)
    assert wm.max_tokens == 50_000


def test_explicit_max_tokens_overrides_derivation():
    wm = WorkingMemory(llm=_LLMWithBudget(100_000), max_tokens=4_000)
    assert wm.max_tokens == 4_000


def test_max_tokens_falls_back_to_32k_when_llm_lacks_budget():
    """Test stubs and custom adapters without ``input_token_budget`` get a
    conservative 32K default — bigger than the historical 8000 since
    modern models have plenty of context — but not catastrophic if the
    underlying model is actually smaller."""

    class _NoBudgetLLM:
        async def complete(self, system, messages, **_) -> dict:
            return {"text": "ok"}

    wm = WorkingMemory(llm=_NoBudgetLLM())
    assert wm.max_tokens == 32_000


def test_max_tokens_falls_back_to_32k_on_invalid_budget():
    """A garbage value on ``input_token_budget`` shouldn't translate into a
    zero or negative WM cap — fall back to 32K so the agent still runs."""

    class _BadBudgetLLM:
        input_token_budget = 0  # nonsensical

        async def complete(self, system, messages, **_) -> dict:
            return {"text": "ok"}

    wm = WorkingMemory(llm=_BadBudgetLLM())
    assert wm.max_tokens == 32_000


# ── Role invariant ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_role_is_assistant_when_first_after_is_user():
    """4 evictable, cutoff=2, first remaining = user → summary role = 'assistant'."""
    wm, _ = _wm(max_tokens=350)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)  # user_A  — summarized
    await wm.append("assistant", MSG)  # asst_B  — summarized
    await wm.append("user", MSG)  # user_C  — survives, first_after
    await wm.append("assistant", MSG)  # triggers eviction

    msgs = wm.get_messages()
    summary = next((m for m in msgs if SUMMARY_HEADER in m["content"]), None)
    assert summary is not None, "eviction should have produced a summary"
    assert summary["role"] == "assistant", (
        f"expected 'assistant' (opposite of first_after='user'), got '{summary['role']}'"
    )


@pytest.mark.asyncio
async def test_context_usage_reports_budget_level():
    wm, _ = _wm(max_tokens=100)
    await wm.append("user", "x" * 81)

    usage = wm.context_usage()

    assert usage["tokens"] == 81
    assert usage["max_tokens"] == 100
    assert usage["percent"] == 0.81
    assert usage["level"] == "warning"
    assert usage["messages"] == 1
    assert usage["summarizations"] == 0


@pytest.mark.asyncio
async def test_summary_role_is_user_when_first_after_is_assistant():
    """3 evictable, cutoff=1, first remaining = assistant → summary role = 'user'."""
    wm, _ = _wm(max_tokens=250)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)  # user_A  — summarized (cutoff=1)
    await wm.append("assistant", MSG)  # asst_B  — survives, first_after
    await wm.append("user", MSG)  # triggers eviction

    msgs = wm.get_messages()
    summary = next((m for m in msgs if SUMMARY_HEADER in m["content"]), None)
    assert summary is not None, "eviction should have produced a summary"
    assert summary["role"] == "user", (
        f"expected 'user' (opposite of first_after='assistant'), got '{summary['role']}'"
    )


@pytest.mark.asyncio
async def test_no_consecutive_same_roles_after_multiple_evictions():
    """After repeated rolling-summary evictions, user/assistant alternation holds."""
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


# ── Two-pass summarization ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_pass_fires_when_first_summary_still_over_budget():
    """When the first summary is still over budget, a second pass extends it."""
    call_count = 0
    responses = ["z" * 200, "ok"]  # first too large, second fits

    class TwoPassLLM:
        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return {"text": resp}

    wm = WorkingMemory(llm=TwoPassLLM(), max_tokens=40, recency_window=0, token_counter=len)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", "y" * 15)
    await wm.append("assistant", "y" * 15)
    await wm.append("user", "y" * 15)  # triggers eviction

    assert call_count == 2, f"expected 2 LLM calls (two-pass), got {call_count}"
    assert wm.summarization_count == 2


@pytest.mark.asyncio
async def test_hard_drop_not_used_when_second_pass_fits():
    """When the second-pass summary fits, no message is hard-dropped.

    With the rolling design, the second pass folds the prior summary AND
    another batch into a new summary, so the buffer size can shrink below
    what the original implementation produced. The assertion here is the
    invariant the test name guarantees: no message was hard-dropped
    (the surviving non-summary, non-pinned tail is intact).
    """

    class TwoPassLLM:
        def __init__(self):
            self._call = 0

        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            self._call += 1
            return {"text": "z" * 200 if self._call == 1 else "ok"}

    wm = WorkingMemory(llm=TwoPassLLM(), max_tokens=60, recency_window=0, token_counter=len)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", "y" * 15)
    await wm.append("assistant", "y" * 15)
    await wm.append("user", "y" * 15)
    await wm.append("assistant", "y" * 15)  # triggers eviction

    msgs = wm.get_messages()
    # Hard-drop would have removed every non-summary message under tight
    # budget. At least one verbatim "y"*15 message must remain.
    verbatim = [m for m in msgs if m["content"] == "y" * 15]
    assert verbatim, (
        f"second pass should have fit without hard drop; got messages "
        f"{[(m['role'], m['content'][:20]) for m in msgs]}"
    )
    assert wm.summarization_count == 2


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
    wm = WorkingMemory(llm=llm, max_tokens=40, recency_window=0, token_counter=len)
    await wm.append("system", "s", pinned=True)
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        await wm.append(role, "y" * 15)

    assert wm.summarization_count == llm.count
    assert wm.summarization_count >= 1


# ── Rolling summary (extend, not re-summarize) ────────────────────────────────


@pytest.mark.asyncio
async def test_first_eviction_uses_initial_summary_prompt():
    """The first compaction uses SUMMARIZE_SYSTEM (no prior summary exists)."""
    wm, calls = _wm(max_tokens=250)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)
    await wm.append("assistant", MSG)
    await wm.append("user", MSG)  # triggers first eviction

    assert len(calls) == 1
    assert calls[0]["system"] == SUMMARIZE_SYSTEM


@pytest.mark.asyncio
async def test_second_eviction_uses_extend_prompt_with_prior_summary():
    """The second compaction uses EXTEND_SUMMARY_SYSTEM and includes prior summary."""
    wm, calls = _wm(max_tokens=250)
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)
    await wm.append("assistant", MSG)
    await wm.append("user", MSG)  # eviction #1 — initial
    # Now force eviction #2 by appending more.
    await wm.append("assistant", MSG)  # may or may not push over; nudge if needed
    await wm.append("user", MSG)
    await wm.append("assistant", MSG)  # should have triggered at least one more eviction

    # At least two compactions happened.
    assert len(calls) >= 2

    # The first call uses SUMMARIZE_SYSTEM; subsequent calls use EXTEND_SUMMARY_SYSTEM.
    assert calls[0]["system"] == SUMMARIZE_SYSTEM
    for c in calls[1:]:
        assert c["system"] == EXTEND_SUMMARY_SYSTEM
        # And the user message passed to the extend prompt embeds the prior summary.
        user_content = c["messages"][0]["content"]
        assert "Existing summary:" in user_content
        assert SUMMARY_HEADER in user_content


@pytest.mark.asyncio
async def test_only_one_summary_message_exists_after_multiple_evictions():
    """The rolling design keeps at most one [Memory summary] message in the buffer."""
    wm, _ = _wm(max_tokens=250)
    await wm.append("system", "s", pinned=True)
    for i in range(12):
        role = "user" if i % 2 == 0 else "assistant"
        await wm.append(role, MSG)

    msgs = wm.get_messages()
    summary_count = sum(1 for m in msgs if m["content"].startswith(SUMMARY_HEADER))
    assert summary_count == 1, f"expected exactly one summary message, got {summary_count}: " + str(
        [m["content"][:30] for m in msgs]
    )


# ── Recency window ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recency_window_protects_last_n_messages():
    """With recency_window=2 and a budget that admits it, the two newest
    non-pinned messages survive verbatim across repeated evictions.

    Budget math (token_counter=len, MSG=100):
      sys (1) + summary (~19) + 2 recency × 100 = 220 verbatim floor.
      max_tokens=300 leaves headroom so each new append triggers an eviction
      that picks the OLDEST eligible (not the recency-window pair).
    """
    wm = WorkingMemory(
        llm=_scripted_llm("ok"),
        max_tokens=300,
        recency_window=2,
        token_counter=len,
    )
    await wm.append("system", "s", pinned=True)
    contents = [f"{'a' * 99}{i}" for i in range(5)]  # each 100 chars, distinguishable
    for i, c in enumerate(contents):
        role = "user" if i % 2 == 0 else "assistant"
        await wm.append(role, c)

    msgs = wm.get_messages()
    surviving_contents = {m["content"] for m in msgs if isinstance(m["content"], str)}
    assert contents[-1] in surviving_contents, "newest non-pinned message was evicted"
    assert contents[-2] in surviving_contents, "second-newest non-pinned message was evicted"


@pytest.mark.asyncio
async def test_recency_window_relaxes_when_budget_forces_it():
    """If everything is inside the recency window, eviction relaxes it to make progress."""
    # recency_window=4 but only 3 non-pinned messages exist when eviction fires:
    # the window must relax so something becomes eligible.
    wm = WorkingMemory(
        llm=_scripted_llm("ok"),
        max_tokens=250,
        recency_window=4,
        token_counter=len,
    )
    await wm.append("system", "s", pinned=True)
    await wm.append("user", MSG)
    await wm.append("assistant", MSG)
    await wm.append("user", MSG)  # total = 1 + 300 = 301 > 250 → eviction

    # Eviction must have happened — at least one summary message present.
    msgs = wm.get_messages()
    assert any(m["content"].startswith(SUMMARY_HEADER) for m in msgs), (
        "recency window failed to relax — no summary produced"
    )
    assert wm.summarization_count >= 1


@pytest.mark.asyncio
async def test_hard_drop_preserves_recent_user_observation_shape():
    """Hard-drop fallback must not leave a trailing assistant action.

    A large tool observation can remain over budget even after two
    summarization passes. In that case, preserve the newest ReAct suffix
    when possible instead of dropping the user observation and leaving
    Bedrock-style providers with an assistant-prefill shaped transcript.
    """
    wm = WorkingMemory(
        llm=_scripted_llm("ok"),
        max_tokens=120,
        recency_window=4,
        token_counter=len,
    )
    await wm.append("system", "s", pinned=True)
    await wm.append("user", "task")
    await wm.append("assistant", "old action")
    await wm.append("user", "old observation")
    await wm.append("assistant", "snapshot action")
    await wm.append("user", "huge snapshot observation " + ("x" * 500))

    msgs = wm.get_messages()
    assert msgs[-1]["role"] == "user", [m["role"] for m in msgs]
    assert "huge snapshot observation" in msgs[-1]["content"]
    assert not any(
        prev["role"] == cur["role"] == "assistant"
        for prev, cur in zip(msgs, msgs[1:], strict=False)
    ), [m["role"] for m in msgs]


# ── Checkpoint backward compatibility ─────────────────────────────────────────


def test_from_dict_legacy_checkpoint_backfills_is_summary():
    """A pre-rolling-summary checkpoint loads with is_summary inferred from content."""
    legacy_data = {
        "messages": [
            {
                "role": "system",
                "content": "sys",
                "pinned": True,
                "token_count": 3,
                # no is_summary field
            },
            {
                "role": "user",
                "content": "[Memory compressed]: old summary body",
                "pinned": False,
                "token_count": 40,
                # no is_summary field
            },
            {
                "role": "assistant",
                "content": "regular message",
                "pinned": False,
                "token_count": 15,
            },
        ],
        "summarization_count": 1,
        "max_tokens": 1000,
        "summarize_ratio": 0.5,
        # no recency_window field
    }

    class _LLM:
        async def complete(self, *_a, **_kw):
            return {"text": "ok"}

    wm = WorkingMemory.from_dict(legacy_data, llm=_LLM())
    assert wm.recency_window == 4  # default applied
    msgs_internal = wm._messages
    assert msgs_internal[0].is_summary is False  # system isn't a summary
    assert msgs_internal[1].is_summary is True  # legacy marker recognized
    assert msgs_internal[2].is_summary is False  # plain message


def test_to_dict_round_trip_preserves_new_fields():
    """to_dict → from_dict preserves is_summary and recency_window."""

    class _LLM:
        async def complete(self, *_a, **_kw):
            return {"text": "ok"}

    wm = WorkingMemory(llm=_LLM(), max_tokens=500, recency_window=6, token_counter=len)
    # Bypass append() to inject a synthetic summary message for round-trip test.
    from memory.working import Message

    wm._messages.append(Message(role="system", content="sys", pinned=True, token_count=3))
    wm._messages.append(
        Message(
            role="user",
            content="[Memory summary]\nFacts:\n- foo",
            token_count=24,
            is_summary=True,
        )
    )
    wm._messages.append(Message(role="assistant", content="hi", token_count=2))
    wm._token_total = 29
    wm._summarization_count = 1

    restored = WorkingMemory.from_dict(wm.to_dict(), llm=_LLM(), token_counter=len)
    assert restored.recency_window == 6
    assert restored._messages[0].is_summary is False
    assert restored._messages[1].is_summary is True
    assert restored._messages[2].is_summary is False
    assert restored.summarization_count == 1


# ── Helpers ───────────────────────────────────────────────────────────────────


def _scripted_llm(text: str):
    class _LLM:
        async def complete(self, *_a, **_kw):
            return {"text": text}

    return _LLM()
