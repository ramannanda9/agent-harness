"""Per-run tool-result memoization on ``BaseAgent._execute_tool``.

Opt-in via ``AgentConfig.cache_tool_results``. A tool can veto caching
for itself with ``cacheable = False`` on the instance (required for
side-effectful or time-dependent tools). Errors are never cached so a
transient failure doesn't poison the rest of the run.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import AgentConfig


class _CountingTool:
    """Records every invocation so tests can assert call count vs. arg uniqueness."""

    name = "counter"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return {"call_n": len(self.calls), "args": kwargs}


class _NonCacheableTool:
    """Tool that vetoes caching even when the agent has it on."""

    name = "now"
    cacheable = False

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **_kwargs: Any) -> dict:
        self.calls += 1
        return {"call_n": self.calls}


def _config(**overrides: Any) -> AgentConfig:
    base = {
        "agent_id": "alpha",
        "role": "test agent",
        "system_prompt": "you are alpha",
        "allowed_tools": ["counter"],
    }
    base.update(overrides)
    return AgentConfig(**base)


@pytest.mark.asyncio
async def test_tool_cache_off_by_default(agent_factory):
    tool = _CountingTool()
    agent = agent_factory(_config(allowed_tools=["counter"]), tools={"counter": tool})
    # Cache disabled — both calls hit the underlying tool even with identical args.
    r1 = await agent._execute_tool("counter", {"x": 1})
    r2 = await agent._execute_tool("counter", {"x": 1})
    assert r1["call_n"] == 1
    assert r2["call_n"] == 2
    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_tool_cache_reuses_result_for_identical_args(agent_factory):
    tool = _CountingTool()
    agent = agent_factory(
        _config(cache_tool_results=True, allowed_tools=["counter"]),
        tools={"counter": tool},
    )
    r1 = await agent._execute_tool("counter", {"x": 1})
    r2 = await agent._execute_tool("counter", {"x": 1})
    assert r1 == r2
    assert r1["call_n"] == 1
    assert len(tool.calls) == 1, "second identical call should hit the cache"


@pytest.mark.asyncio
async def test_tool_cache_keys_on_args(agent_factory):
    """Different args must NOT share a cache slot."""
    tool = _CountingTool()
    agent = agent_factory(
        _config(cache_tool_results=True, allowed_tools=["counter"]),
        tools={"counter": tool},
    )
    await agent._execute_tool("counter", {"x": 1})
    await agent._execute_tool("counter", {"x": 2})
    await agent._execute_tool("counter", {"x": 1})
    # First and third {x:1} share a hit; second {x:2} is its own miss.
    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_tool_cache_respects_tool_cacheable_false(agent_factory):
    """Even with caching on at the agent, a tool may opt out via cacheable=False."""
    tool = _NonCacheableTool()
    agent = agent_factory(
        _config(cache_tool_results=True, allowed_tools=["now"]),
        tools={"now": tool},
    )
    r1 = await agent._execute_tool("now", {})
    r2 = await agent._execute_tool("now", {})
    assert r1["call_n"] == 1
    assert r2["call_n"] == 2, "cacheable=False must veto the per-run cache"
