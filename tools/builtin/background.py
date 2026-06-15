"""LLM-visible tools for PersistentAgent background sub-agent tasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


class BackgroundDelegateTool:
    """Start a persistent background task for one sub-agent."""

    cacheable = False

    def __init__(
        self,
        *,
        agent_id: str,
        start_task: Callable[[str, str, str], Awaitable[Any]],
        session_id_provider: Callable[[], str],
        name: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.name = name or f"background_delegate_{agent_id}"
        self.description = (
            f"Start {agent_id} on a background task and return immediately with a task_id."
        )
        self._start_task = start_task
        self._session_id_provider = session_id_provider

    async def execute(self, task: str) -> dict[str, Any]:
        started = await self._start_task(
            self._session_id_provider(),
            self.agent_id,
            str(task or ""),
        )
        return {
            "task_id": started.task_id,
            "agent_id": started.agent_id,
            "status": started.status,
            "message": (
                f"Background task {started.task_id} started for {started.agent_id}. "
                "Use check_background_task or collect_background_task with this task_id."
            ),
        }


class CheckBackgroundTaskTool:
    """Inspect process-local background task status."""

    cacheable = False
    name = "check_background_task"
    description = "Check status and preview result for a background task in this session."

    def __init__(
        self,
        *,
        list_tasks: Callable[[str | None], Awaitable[list[Any]]],
        session_id_provider: Callable[[], str],
    ) -> None:
        self._list_tasks = list_tasks
        self._session_id_provider = session_id_provider

    async def execute(self, task_id: str) -> dict[str, Any]:
        task = await _find_task(
            self._list_tasks,
            self._session_id_provider(),
            task_id,
        )
        if task is None:
            return {"task_id": task_id, "status": "missing", "error": "task not found"}
        return _task_payload(task, include_answer=True)


class CollectBackgroundTaskTool:
    """Collect a finished background task and return its answer to the LLM."""

    cacheable = False
    name = "collect_background_task"
    description = (
        "Collect a completed background task result for this session and return its answer."
    )

    def __init__(
        self,
        *,
        collect_task: Callable[[str, str], Awaitable[Any]],
        session_id_provider: Callable[[], str],
    ) -> None:
        self._collect_task = collect_task
        self._session_id_provider = session_id_provider

    async def execute(self, task_id: str) -> dict[str, Any]:
        try:
            task = await self._collect_task(self._session_id_provider(), task_id)
        except ValueError as exc:
            return {"task_id": task_id, "status": "error", "error": str(exc)}
        return _task_payload(task, include_answer=True)


async def _find_task(
    list_tasks: Callable[[str | None], Awaitable[list[Any]]],
    session_id: str,
    task_id: str,
) -> Any | None:
    tasks = await list_tasks(session_id)
    return next((task for task in tasks if task.task_id == task_id), None)


def _task_payload(task: Any, *, include_answer: bool) -> dict[str, Any]:
    payload = {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "agent_id": task.agent_id,
        "instruction": task.instruction,
        "status": task.status,
        "confidence": task.confidence,
        "steps": task.steps,
        "error": task.error,
        "collected": task.collected,
    }
    if include_answer:
        payload["answer"] = task.answer
    return payload
