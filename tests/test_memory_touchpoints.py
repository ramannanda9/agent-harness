"""New memory touchpoints: tool-result cache, replan-lesson writes, human-correction writes.

The agent-level memory injection on the routed path was already in place
(``AgentConfig.memory_context_enabled = True`` + ``_build_system_prompt``).
These tests cover the three durable touchpoints added on top.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.base import AgentConfig
from orchestrator.planner import TaskResult, _replan_lesson_key

# ── Tool-result cache ────────────────────────────────────────────────────────


class _CountingTool:
    """Records every invocation so we can assert call count vs. arg uniqueness."""

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


# ── Replan lesson key ────────────────────────────────────────────────────────


def test_replan_lesson_key_is_stable_for_same_failure():
    a = TaskResult(
        task_id="t1",
        agent_id="shell",
        answer="",
        confidence=0.0,
        steps=1,
        success=False,
        error="timeout: kubectl get pods (30s)",
    )
    b = TaskResult(
        task_id="t99",  # different task id, same failure pattern
        agent_id="shell",
        answer="",
        confidence=0.0,
        steps=4,
        success=False,
        error="timeout: kubectl get pods (30s)",
    )
    assert _replan_lesson_key(a) == _replan_lesson_key(b), (
        "same (agent, error) must collapse to one key so lessons don't duplicate"
    )


def test_replan_lesson_key_differs_for_different_agents():
    error = "permission denied"
    a = TaskResult(
        task_id="t1",
        agent_id="shell",
        answer="",
        confidence=0.0,
        steps=1,
        success=False,
        error=error,
    )
    b = TaskResult(
        task_id="t1",
        agent_id="web",
        answer="",
        confidence=0.0,
        steps=1,
        success=False,
        error=error,
    )
    assert _replan_lesson_key(a) != _replan_lesson_key(b)


def test_replan_lesson_key_uses_global_prefix():
    """``build_context`` retrieves keys that don't start with run: or agent: ;
    the replan-lesson key uses ``replan:`` so it survives that filter."""
    result = TaskResult(
        task_id="t",
        agent_id="shell",
        answer="",
        confidence=0.0,
        steps=1,
        success=False,
        error="boom",
    )
    key = _replan_lesson_key(result)
    assert key.startswith("replan:")
    assert not key.startswith(("run:", "agent:"))


# ── Human correction → semantic write ───────────────────────────────────────


class _RecordingSemanticStore:
    """Captures every write so tests can assert the durable-fact path fires."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []
        self._data: dict[str, Any] = {}

    async def write(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        self.writes.append((key, value))
        self._data[key] = value

    async def read(self, key: str) -> Any | None:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def search_prefix(self, prefix: str) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if k.startswith(prefix)}


@pytest.mark.asyncio
async def test_steering_drains_to_semantic_memory(memory, agent_factory):
    """Async steering text should land in working memory AND as a durable
    semantic fact so future runs can read past corrections back."""
    semantic = _RecordingSemanticStore()
    memory._semantic = semantic

    agent = agent_factory(_config())
    # WorkingMemory must exist for _drain_steering to append.
    from memory.working import WorkingMemory

    agent._working_memory = WorkingMemory(llm=agent._llm, max_tokens=4000)
    agent._task = "investigate disk usage"
    agent.steer("Skip /tmp — symlinked to /var/tmp on this host.")

    events = [event async for event in agent._drain_steering(step=2)]
    # Give the fire-and-forget write a tick to land.
    await asyncio.sleep(0)

    assert any(e.type.value == "human_guidance" for e in events)
    keys = [k for k, _ in semantic.writes]
    correction_keys = [k for k in keys if k.startswith("human-correction:alpha:")]
    assert correction_keys, f"expected a human-correction:alpha:* write; got {keys!r}"
    _, value = next((k, v) for k, v in semantic.writes if k == correction_keys[0])
    assert value["guidance"].startswith("Skip /tmp")
    assert value["task"] == "investigate disk usage"
    assert value["step"] == 2


@pytest.mark.asyncio
async def test_steering_whitespace_only_text_does_not_persist(memory, agent_factory):
    """Empty/whitespace guidance is already discarded by ``steer`` — but if
    something else slips an empty string into the queue, the persistence
    path must not write a meaningless fact."""
    semantic = _RecordingSemanticStore()
    memory._semantic = semantic
    agent = agent_factory(_config())
    # _persist_human_correction is a sync helper — call it directly.
    agent._task = "x"
    agent._persist_human_correction("   ", step=0)
    await asyncio.sleep(0)
    assert semantic.writes == []
