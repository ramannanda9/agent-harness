"""Pin the trailing-user defensive normalisation in ``BaseAgent._think_stream``.

Some inference proxies (AWS Bedrock via OpenAI-compatible gateways) reject
any messages array that ends with an assistant role — they interpret
trailing assistant as an Anthropic-style "message prefill" request and
refuse with:

    bedrock error: This model does not support assistant message
    prefill. The conversation must end with a user message.

The ReAct loop is structurally ``assistant_response -> user_observation``
in lockstep at every exit point, so working memory *should* always end
with ``user`` before any LLM call. If something upstream leaves it
ending with assistant, ``_think_stream`` now:

1. Logs a WARNING with the role sequence (diagnostic).
2. Appends a terse ``(user, "Continue.")`` cue to the wire-only copy
   (does not mutate working memory).

Both behaviours are pinned here as a regression guard so the defensive
fix isn't quietly removed before the upstream cause is identified.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore


class _SpyLLM:
    """Captures every LLM call's messages so the test can inspect the
    final wire shape (especially the last message's role)."""

    input_token_budget = 10_000
    last_usage = None

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, system: str | None, messages: list[dict], **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        return {
            "thought": "ok",
            "action": "finish",
            "answer": "done",
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
            agent_id="trailing_user_agent",
            role="test agent",
            system_prompt="You answer briefly.",
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
async def test_defensive_continue_not_appended_when_working_memory_ends_with_user(
    caplog: pytest.LogCaptureFixture,
):
    """Happy path: priors + task → working memory ends with user.
    The defensive append must NOT fire, and no warning must be logged."""
    llm = _SpyLLM()
    agent = _make_agent(llm)

    caplog.set_level(logging.WARNING, logger="agents.base")

    async for _ in agent.run_stream("hi", run_id="r1"):
        pass

    assert llm.calls, "expected one LLM call"
    wire = llm.calls[0]["messages"]
    assert wire[-1]["role"] == "user", (
        "happy-path wire must end with the user task — no defensive "
        f"append needed; got role sequence: {[m['role'] for m in wire]}"
    )
    assert not any("defensive 'Continue.'" in record.message for record in caplog.records), (
        "warning must not fire when working memory already ends with user"
    )


@pytest.mark.asyncio
async def test_defensive_continue_appends_when_working_memory_ends_with_assistant(
    caplog: pytest.LogCaptureFixture,
):
    """Defensive path: when ``prior_messages`` leave the wire ending
    with an assistant role (and the agent is invoked WITHOUT a fresh
    user task), the wire-shape normaliser must append (user, "Continue.")
    and log a warning describing the role sequence.

    Construction: pass priors that end with assistant, then call
    ``run_stream`` with a non-empty task. BaseAgent's ``run_stream``
    *does* append the task — but if we instead exercise the lower-level
    ``_run_stream_internal`` after manually setting up working memory to
    end with assistant, we can pin the defensive logic directly.
    """
    import json

    from memory.working import WorkingMemory

    llm = _SpyLLM()
    agent = _make_agent(llm)

    # Hand-construct a working memory that ends with assistant. This
    # mimics the broken upstream state that would otherwise hit Bedrock's
    # rejection — we don't need to identify the upstream cause to pin
    # the defensive normalisation.
    agent._working_memory = WorkingMemory(llm=llm, max_tokens=10_000)
    await agent._working_memory.append("system", "You answer briefly.", pinned=True)
    await agent._working_memory.append("user", "earlier turn task")
    await agent._working_memory.append(
        "assistant",
        json.dumps(
            {"thought": "done", "action": "finish", "answer": "previous", "confidence": 1.0}
        ),
    )

    caplog.set_level(logging.WARNING, logger="agents.base")

    # Drive a single think step. ``_think_stream`` is the surface that
    # builds the wire and would-be-sends to the LLM.
    events = [event async for event in agent._think_stream()]
    assert events, "expected at least one event from _think_stream"
    assert llm.calls, "expected one LLM call from _think_stream"

    wire = llm.calls[0]["messages"]
    assert wire[-1]["role"] == "user", (
        "defensive normalisation must leave the wire ending in user; "
        f"got role sequence: {[m['role'] for m in wire]}"
    )
    assert wire[-1]["content"] == "Continue.", (
        "defensive cue must be the literal string 'Continue.' (the agent "
        "system prompt already enforces the ReAct JSON format, so a terse "
        "cue suffices); got content: " + repr(wire[-1]["content"])
    )

    warnings = [r for r in caplog.records if "defensive 'Continue.'" in r.message]
    assert len(warnings) == 1, (
        "exactly one diagnostic warning must fire so the upstream cause "
        f"can be identified from logs; got {len(warnings)} warnings"
    )
    assert "role_sequence" in warnings[0].message, (
        "warning must include the role sequence — that's how we'll "
        "diagnose what produced the trailing-assistant shape"
    )


@pytest.mark.asyncio
async def test_defensive_continue_does_not_mutate_working_memory(
    caplog: pytest.LogCaptureFixture,
):
    """The defensive cue must be wire-only — appending it to working
    memory would persist a phantom (user, "Continue.") entry that
    subsequent turns would see, and could create consecutive same-role
    sequences once a real user message arrives."""
    import json

    from memory.working import WorkingMemory

    llm = _SpyLLM()
    agent = _make_agent(llm)

    agent._working_memory = WorkingMemory(llm=llm, max_tokens=10_000)
    await agent._working_memory.append("system", "You answer briefly.", pinned=True)
    await agent._working_memory.append("user", "task")
    await agent._working_memory.append(
        "assistant",
        json.dumps({"thought": "done", "action": "finish", "answer": "prev", "confidence": 1.0}),
    )

    caplog.set_level(logging.WARNING, logger="agents.base")
    [event async for event in agent._think_stream()]

    wm_messages = agent._working_memory.get_messages()
    contents = [m["content"] for m in wm_messages]
    assert "Continue." not in contents, (
        "defensive 'Continue.' cue must be wire-only — finding it in "
        "working memory means a follow-up turn would re-send it as part "
        "of priors, polluting downstream context"
    )
