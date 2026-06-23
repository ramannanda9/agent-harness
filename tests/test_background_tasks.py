from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.background_tasks import BackgroundTaskManager
from harness.persistent import InMemorySessionStore, SessionMessage
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool


class _LLM:
    async def complete(self, system, messages, **kwargs):
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return {
            "thought": "done",
            "action": "finish",
            "answer": f"answer: {last_user}",
            "confidence": 0.8,
        }


class _SlowLLM(_LLM):
    async def complete(self, system, messages, **kwargs):
        await asyncio.sleep(60)
        return await super().complete(system, messages, **kwargs)


def _memory(llm: Any) -> MemoryManager:
    return MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )


def _agent(agent_id: str, llm: Any, tools: dict[str, Any] | None = None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            role=f"{agent_id} role",
            system_prompt="Finish the task.",
            allowed_tools=list((tools or {}).keys()),
            max_steps=2,
        ),
        tools=tools or {},
        memory=_memory(llm),
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )


def _manager(
    *,
    coordinator: BaseAgent,
    store: InMemorySessionStore | None = None,
    session_id: str = "s",
) -> BackgroundTaskManager:
    return BackgroundTaskManager(
        coordinator=coordinator,
        session_store=store or InMemorySessionStore(),
        session_id_provider=lambda: session_id,
        apply_overrides=lambda _state: None,
        session_message_factory=lambda content: SessionMessage(
            role="assistant",
            content=content,
        ),
    )


async def _wait_done(
    manager: BackgroundTaskManager,
    session_id: str,
    task_id: str,
) -> None:
    for _ in range(20):
        task = next(task for task in await manager.list(session_id) if task.task_id == task_id)
        if task.status != "running":
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"background task still running: {task_id}")


@pytest.mark.asyncio
async def test_background_task_manager_starts_and_collects_result():
    sub = _agent("researcher", _LLM())
    coordinator = _agent(
        "coordinator",
        _LLM(),
        tools={"delegate_researcher": SubAgentTool(sub, name="delegate_researcher")},
    )
    store = InMemorySessionStore()
    manager = _manager(coordinator=coordinator, store=store)

    started = await manager.start("s", "researcher", "find a fact")
    await _wait_done(manager, "s", started.task_id)

    collected = await manager.collect("s", started.task_id)
    state = await store.load("s")

    assert collected.status == "done"
    assert collected.collected is True
    assert "answer: find a fact" in collected.answer
    assert state.messages[-1].role == "assistant"
    assert started.task_id in state.messages[-1].content


@pytest.mark.asyncio
async def test_background_task_manager_cancels_running_task():
    sub = _agent("researcher", _SlowLLM())
    coordinator = _agent(
        "coordinator",
        _LLM(),
        tools={"delegate_researcher": SubAgentTool(sub, name="delegate_researcher")},
    )
    manager = _manager(coordinator=coordinator)

    started = await manager.start("s", "researcher", "slow work")
    cancelled = await manager.cancel("s", started.task_id)

    assert cancelled.status == "cancelled"
    assert cancelled.error == "cancelled by user"


@pytest.mark.asyncio
async def test_background_task_manager_installs_llm_visible_tools():
    sub = _agent("researcher", _LLM())
    coordinator = _agent(
        "coordinator",
        _LLM(),
        tools={"delegate_researcher": SubAgentTool(sub, name="delegate_researcher")},
    )
    manager = _manager(coordinator=coordinator)

    manager.install_tools()

    assert "background_delegate_researcher" in coordinator._tools
    assert "check_background_task" in coordinator._tools
    assert "collect_background_task" in coordinator._tools
    assert "background_delegate_researcher" in coordinator.config.allowed_tools
