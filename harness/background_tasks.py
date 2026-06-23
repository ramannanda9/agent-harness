"""Process-local background sub-agent task management for ``PersistentAgent``."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from harness.agent_tree import find_agent, subagent_tool_agent
from harness.events import EventType
from tools.builtin.background import (
    BackgroundDelegateTool,
    CheckBackgroundTaskTool,
    CollectBackgroundTaskTool,
)

if TYPE_CHECKING:
    from agents.base import BaseAgent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BackgroundTaskState:
    """Process-local background sub-agent task metadata."""

    task_id: str
    session_id: str
    agent_id: str
    instruction: str
    status: str = "running"
    answer: str = ""
    confidence: float = 0.0
    steps: int = 0
    error: str = ""
    collected: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class BackgroundTaskManager:
    """Own background sub-agent task state and LLM-visible task tools."""

    def __init__(
        self,
        *,
        coordinator: BaseAgent,
        session_store: Any,
        session_id_provider: Callable[[], str],
        apply_overrides: Callable[[Any], None],
        session_message_factory: Callable[[str], Any],
    ) -> None:
        self._coordinator = coordinator
        self._session_store = session_store
        self._session_id_provider = session_id_provider
        self._apply_overrides = apply_overrides
        self._session_message_factory = session_message_factory
        self._tasks: dict[str, BackgroundTaskState] = {}
        self._handles: dict[str, asyncio.Task[None]] = {}

    async def start(
        self,
        session_id: str,
        agent_id: str,
        instruction: str,
    ) -> BackgroundTaskState:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("background task instruction is required")
        agent = find_agent(self._coordinator, agent_id)
        if agent is None or agent is self._coordinator:
            raise ValueError(f"unknown sub-agent {agent_id!r}")
        if self._session_id_provider() != session_id:
            state = await self._session_store.load(session_id)
            self._apply_overrides(state)
        task_id = f"bg_{uuid.uuid4().hex[:12]}"
        task = BackgroundTaskState(
            task_id=task_id,
            session_id=session_id,
            agent_id=agent_id,
            instruction=instruction,
        )
        self._tasks[task_id] = task
        self._handles[task_id] = asyncio.create_task(self._run(task, agent))
        return copy_background_task(task)

    async def list(self, session_id: str | None = None) -> list[BackgroundTaskState]:
        tasks = self._tasks.values()
        if session_id is not None:
            tasks = [task for task in tasks if task.session_id == session_id]
        return sorted(
            (copy_background_task(task) for task in tasks),
            key=lambda task: task.created_at,
            reverse=True,
        )

    async def collect(self, session_id: str, task_id: str) -> BackgroundTaskState:
        task = self._tasks.get(task_id)
        if task is None or task.session_id != session_id:
            raise ValueError(f"background task not found: {task_id}")
        if task.status == "running":
            raise ValueError(f"background task still running: {task_id}")
        if not task.collected:
            await self._session_store.append_messages(
                session_id,
                [self._session_message_factory(render_background_result(task))],
            )
            task.collected = True
            task.updated_at = _now()
        return copy_background_task(task)

    async def cancel(self, session_id: str, task_id: str) -> BackgroundTaskState:
        task = self._tasks.get(task_id)
        if task is None or task.session_id != session_id:
            raise ValueError(f"background task not found: {task_id}")
        handle = self._handles.get(task_id)
        if task.status == "running" and handle is not None and not handle.done():
            handle.cancel()
            task.status = "cancelled"
            task.error = "cancelled by user"
            task.updated_at = _now()
        return copy_background_task(task)

    def install_tools(self) -> None:
        tools = getattr(self._coordinator, "_tools", {})
        if not isinstance(tools, dict):
            return
        background_tools: dict[str, Any] = {}
        for tool in list(tools.values()):
            sub = subagent_tool_agent(tool)
            if sub is None:
                continue
            background = BackgroundDelegateTool(
                agent_id=sub.config.agent_id,
                start_task=self.start,
                session_id_provider=self._session_id_provider,
            )
            background_tools.setdefault(background.name, background)
        if background_tools:
            background_tools.setdefault(
                CheckBackgroundTaskTool.name,
                CheckBackgroundTaskTool(
                    list_tasks=self.list,
                    session_id_provider=self._session_id_provider,
                ),
            )
            background_tools.setdefault(
                CollectBackgroundTaskTool.name,
                CollectBackgroundTaskTool(
                    collect_task=self.collect,
                    session_id_provider=self._session_id_provider,
                ),
            )
        for name, tool in background_tools.items():
            tools.setdefault(name, tool)
            if name not in self._coordinator.config.allowed_tools:
                self._coordinator.config.allowed_tools.append(name)

    async def _run(self, task: BackgroundTaskState, agent: BaseAgent) -> None:
        last_done: dict[str, Any] | None = None
        last_error = ""
        try:
            async for event in agent.run_stream(
                task=task.instruction,
                run_id=task.task_id,
            ):
                if event.type == EventType.TASK_DONE:
                    last_done = event.payload
                elif event.type == EventType.ERROR:
                    last_error = event.error
        except asyncio.CancelledError:
            task.status = "cancelled"
            task.error = "cancelled by user"
            raise
        except Exception as exc:  # noqa: BLE001 — store for /tasks
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
        else:
            if last_done is not None:
                task.status = "done"
                task.answer = str(last_done.get("answer", ""))
                task.confidence = float(last_done.get("confidence", 0.0) or 0.0)
                task.steps = int(last_done.get("steps", 0) or 0)
            else:
                task.status = "failed"
                task.error = last_error or "sub-agent stream ended without TASK_DONE"
        finally:
            task.updated_at = _now()


def copy_background_task(task: BackgroundTaskState) -> BackgroundTaskState:
    return BackgroundTaskState(
        task_id=task.task_id,
        session_id=task.session_id,
        agent_id=task.agent_id,
        instruction=task.instruction,
        status=task.status,
        answer=task.answer,
        confidence=task.confidence,
        steps=task.steps,
        error=task.error,
        collected=task.collected,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def render_background_result(task: BackgroundTaskState) -> str:
    header = f"[Background task {task.task_id}] {task.agent_id} {task.status}: {task.instruction}"
    if task.status == "done":
        return f"{header}\n\n{task.answer}\n\nconfidence={task.confidence:.2f} steps={task.steps}"
    return f"{header}\n\nerror: {task.error or task.status}"
