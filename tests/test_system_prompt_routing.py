"""Pin the LLM-call shape so the Anthropic adapter never silently drops
the system prompt again.

History: ``BaseAgent`` used to call ``llm.complete(system=None, messages=...)``
with the system prompt folded into ``messages`` as a ``role="system"`` entry.
That worked on OpenAI (chat API accepts inline system entries) but the
Anthropic adapter's ``_build_messages`` skips inline system entries —
"consumed by caller as the system param" — and the caller wasn't passing
it. Net effect: the entire system prompt was discarded on every Anthropic
call. The fix (``_split_system`` in ``agents/base.py``) pulls system
entries out of working memory at the LLM call boundary and routes them
through the official ``system=`` parameter for every adapter.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent, _split_system
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── Pure unit tests for _split_system ───────────────────────────────────


def test_split_system_extracts_single_system_entry():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    system, rest = _split_system(msgs)
    assert system == "you are helpful"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_joins_multiple_system_entries_with_blank_line():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": "also be concise"},
        {"role": "user", "content": "next"},
    ]
    system, rest = _split_system(msgs)
    assert system == "you are helpful\n\nalso be concise"
    assert [m["role"] for m in rest] == ["user", "assistant", "user"]


def test_split_system_returns_none_when_no_system_entries():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
    system, rest = _split_system(msgs)
    assert system is None
    assert rest == msgs


def test_split_system_skips_empty_system_contents():
    msgs = [
        {"role": "system", "content": ""},
        {"role": "system", "content": "real prompt"},
        {"role": "user", "content": "hi"},
    ]
    system, rest = _split_system(msgs)
    # Empty system entries contribute nothing; only the real one remains.
    assert system == "real prompt"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_handles_empty_input():
    system, rest = _split_system([])
    assert system is None
    assert rest == []


# ── Integration: BaseAgent passes system via system= parameter ──────────


class _SpyLLM:
    """Records every complete / stream_complete call's kwargs.

    The contract this test pins: ``system`` must never be ``None`` when
    ``BaseAgent.config.system_prompt`` is non-empty. The Anthropic-style
    adapters require the top-level ``system=`` to be set, otherwise the
    prompt is dropped.
    """

    input_token_budget = 10_000
    last_usage = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, system: str | None, messages: list[dict], **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        return {
            "thought": "all done",
            "action": "finish",
            "answer": "ok",
            "confidence": 1.0,
        }


def _make_agent(llm: _SpyLLM) -> BaseAgent:
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    return BaseAgent(
        config=AgentConfig(
            agent_id="sysagent",
            role="ReAct test agent",
            system_prompt="You are a test agent. Always answer concisely.",
            allowed_tools=[],
            max_steps=2,
            stream_tokens=False,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )


@pytest.mark.asyncio
async def test_base_agent_routes_system_prompt_via_top_level_parameter():
    """Regression guard for the Anthropic silent-drop bug.

    BaseAgent must call the LLM with the system prompt in the ``system=``
    kwarg (not folded inline as a ``role="system"`` message), so adapters
    that route system separately (Anthropic, claude_code) see it.
    """
    llm = _SpyLLM()
    agent = _make_agent(llm)

    async for _ in agent.run_stream("explain something briefly", run_id="r1"):
        pass

    assert llm.calls, "expected at least one LLM call"
    first_call = llm.calls[0]

    # Contract 1: system prompt arrives via the top-level parameter.
    assert first_call["system"] is not None
    assert "test agent" in first_call["system"], (
        f"system prompt must contain the configured agent identity; got: {first_call['system']!r}"
    )

    # Contract 2: no inline role=system in the messages list — that's the
    # exact path the Anthropic adapter silently drops.
    inline_system = [m for m in first_call["messages"] if m.get("role") == "system"]
    assert inline_system == [], (
        "messages list must not contain inline role=system entries — "
        "Anthropic's _build_messages drops them. Found: "
        f"{[m.get('content') for m in inline_system]}"
    )

    # Contract 3: the chat turn (user message) is still present.
    user_msgs = [m for m in first_call["messages"] if m.get("role") == "user"]
    assert user_msgs and "explain something briefly" in user_msgs[-1]["content"]
