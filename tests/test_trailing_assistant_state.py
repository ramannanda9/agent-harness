"""Pin trailing-assistant handling at the LLM call boundary.

The ReAct loop should call the LLM after a user task or user observation.
If working memory ends with an assistant message, that is an invalid state
shape worth logging. The call boundary must not hide it by inventing a fake
user message such as ``Continue.``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from memory.working import WorkingMemory


class _SpyLLM:
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
            agent_id="trailing_assistant_agent",
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
async def test_no_warning_when_working_memory_ends_with_user(caplog: pytest.LogCaptureFixture):
    llm = _SpyLLM()
    agent = _make_agent(llm)

    caplog.set_level(logging.WARNING, logger="agents.base")

    async for _ in agent.run_stream("hi", run_id="r1"):
        pass

    assert llm.calls, "expected one LLM call"
    wire = llm.calls[0]["messages"]
    assert wire[-1]["role"] == "user"
    assert not any("messages end with assistant" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_trailing_assistant_logs_but_does_not_append_fake_user(
    caplog: pytest.LogCaptureFixture,
):
    llm = _SpyLLM()
    agent = _make_agent(llm)

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

    events = [event async for event in agent._think_stream()]

    assert events, "expected at least one event from _think_stream"
    assert llm.calls, "expected one LLM call from _think_stream"

    wire = llm.calls[0]["messages"]
    assert wire[-1]["role"] == "assistant"
    assert "Continue." not in [m.get("content") for m in wire], (
        "LLM call boundary must not fabricate a user continuation message"
    )

    warnings = [r for r in caplog.records if "messages end with assistant" in r.message]
    assert len(warnings) == 1
    assert "role_sequence" in warnings[0].message

    wm_contents = [m["content"] for m in agent._working_memory.get_messages()]
    assert "Continue." not in wm_contents
