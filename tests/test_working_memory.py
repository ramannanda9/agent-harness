"""Tests for WorkingMemory token-budget eviction and pluggable counter."""
from __future__ import annotations

from agents.base import AgentConfig
from memory.working import Message, WorkingMemory, count_tokens
from tests.conftest import ScriptedLLM

# ── Token counter ─────────────────────────────────────────────────────────────


def test_default_counter_is_chars_over_4():
    # 40 chars → ~10 tokens. Stable for code/JSON/text.
    assert count_tokens("a" * 40) == 10
    assert count_tokens("") == 0
    # short strings still count as at least 1
    assert count_tokens("hi") == 1


def test_default_counter_handles_json_better_than_words():
    """The old word-based estimator severely undercounted JSON. Verify the
    new heuristic is closer to a real tokenizer."""
    json_blob = '{"name": "alice", "age": 30, "roles": ["admin", "user"]}'
    # word-based would give: len(split()) * 1.3 ≈ 5 * 1.3 ≈ 6
    # chars/4 gives: 56 / 4 = 14 — much closer to real (~18)
    assert count_tokens(json_blob) >= 12


# ── WorkingMemory respects token_counter override ───────────────────────────


async def test_custom_counter_is_used():
    """A user-supplied counter (e.g. tiktoken / anthropic API) should be
    used in place of the default."""
    calls = []

    def fake_counter(text: str) -> int:
        calls.append(text)
        return 100  # every message is "expensive"

    wm = WorkingMemory(llm=ScriptedLLM(), max_tokens=10_000, token_counter=fake_counter)
    await wm.append("user", "hello")
    assert wm.token_count() == 100
    assert calls == ["hello"]


async def test_append_counts_per_message():
    wm = WorkingMemory(llm=ScriptedLLM(), max_tokens=10_000)
    await wm.append("user", "a" * 40)  # ~10 tokens
    await wm.append("assistant", "b" * 80)  # ~20 tokens
    assert wm.token_count() == 30


async def test_message_dataclass_no_longer_auto_counts():
    """Direct Message() construction has token_count=0 by default; counting
    is the WorkingMemory's responsibility now."""
    m = Message(role="user", content="this is some content here")
    assert m.token_count == 0


# ── Eviction respects custom budget ─────────────────────────────────────────


async def test_eviction_fires_when_over_budget():
    summaries = []

    def summarize(system, messages, kwargs):
        formatted = messages[0]["content"]
        summaries.append(formatted)
        return {"text": "compressed"}

    llm = ScriptedLLM(routes={"memory compressor": summarize})
    wm = WorkingMemory(llm=llm, max_tokens=50)  # tight budget

    # Push enough content to trigger eviction.
    await wm.append("system", "sys prompt", pinned=True)
    await wm.append("user", "a" * 200)  # 50 tokens — over budget → evict
    await wm.append("user", "b" * 200)

    assert wm.summarization_count >= 1
    assert len(summaries) >= 1


async def test_pinned_messages_not_evicted():
    def summarize(system, messages, kwargs):
        # the system-role pinned message should NOT appear in what we summarize
        assert "PINNED SYSTEM" not in messages[0]["content"]
        return {"text": "summary"}

    llm = ScriptedLLM(routes={"memory compressor": summarize})
    wm = WorkingMemory(llm=llm, max_tokens=50)
    await wm.append("system", "PINNED SYSTEM PROMPT " * 5, pinned=True)
    await wm.append("user", "x" * 400)
    await wm.append("user", "y" * 400)
    # at least one summarization fired
    assert wm.summarization_count >= 1


# ── AgentConfig.working_memory_max_tokens propagation ─────────────────────


def test_agent_config_default_is_8000():
    cfg = AgentConfig(
        agent_id="a", role="r", system_prompt="p", allowed_tools=[],
    )
    assert cfg.working_memory_max_tokens == 8000


async def test_agent_uses_configured_budget(agent_factory):
    """BaseAgent must read working_memory_max_tokens from AgentConfig, not
    hardcode it. We verify by inspecting the WorkingMemory instance the agent
    constructs at run-start."""
    cfg = AgentConfig(
        agent_id="a", role="r", system_prompt="finish.", allowed_tools=[],
        working_memory_max_tokens=42,
    )
    agent = agent_factory(cfg)
    await agent.run("hi")
    # _working_memory is private but stable; tests are allowed to peek.
    assert agent._working_memory.max_tokens == 42
