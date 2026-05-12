"""Shared fixtures: scriptable mock LLM and basic tool."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore


class ScriptedLLM:
    """
    Deterministic LLM stub. Routes by system-prompt keyword to avoid hard-coding
    call counts — same routing strategy used by examples/quickstart.py.

    Each route is a callable that receives (system, messages, kwargs) and returns
    the response dict. Override per test by passing `routes={...}`.
    """

    def __init__(self, routes: dict[str, Callable[..., Any]] | None = None) -> None:
        self.calls: list[dict] = []
        self.routes = routes or {}

    async def complete(self, system: str | None, messages: list[dict], **kwargs: Any) -> Any:
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        # BaseAgent._think calls with system=None and puts the agent system prompt
        # in the messages list. Fall back to the first system-role message so routes
        # can match agent ReAct prompts as well as orchestrator/memory prompts.
        routing_text = system or next((m["content"] for m in messages if m["role"] == "system"), "")
        routing_text = routing_text.lower()
        for needle, handler in self.routes.items():
            if needle.lower() in routing_text:
                return handler(system, messages, kwargs)
        # default: agent ReAct — finish on first step
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return {
            "thought": "default finish",
            "action": "finish",
            "answer": f"done: {last_user[:60]}",
            "confidence": 0.9,
        }


class EchoTool:
    name = "echo"

    async def execute(self, message: str = "") -> dict:
        return {"echo": message}


class FailingTool:
    name = "fail"

    async def execute(self, **_: Any) -> Any:
        raise RuntimeError("boom")


class SlowTool:
    """Records start/end times to verify concurrent execution."""

    name = "slow"

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.starts: list[float] = []
        self.ends: list[float] = []

    async def execute(self, label: str = "") -> dict:
        import time

        self.starts.append(time.monotonic())
        await asyncio.sleep(self.delay)
        self.ends.append(time.monotonic())
        return {"label": label}


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def memory(llm: ScriptedLLM) -> MemoryManager:
    return MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )


@pytest.fixture
def tracer() -> Tracer:
    return Tracer()


@pytest.fixture
def guard() -> BudgetGuard:
    return BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60))


@pytest.fixture
def agent_factory(memory, tracer, guard, llm):
    """Factory: build a BaseAgent with the given config and tool dict."""

    def _build(config: AgentConfig, tools: dict[str, Any] | None = None) -> BaseAgent:
        return BaseAgent(
            config=config,
            tools=tools or {},
            memory=memory,
            tracer=tracer,
            guard=guard,
            llm=llm,
        )

    return _build
