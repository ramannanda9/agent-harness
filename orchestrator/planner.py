"""
Orchestrator — hybrid planning with replan trigger. Streaming-primary.

Planning style: Hybrid
  - Plan upfront as a static DAG
  - After each agent completes, evaluate result quality
  - Replan trigger: confidence < threshold OR agent failed
  - Replanning is an LLM call — guarded by max_replan_count
  - Replan produces a new DAG from the current state forward

DAG execution:
  - Tasks with no unmet dependencies run in parallel; their event streams
    are fanned-in to the orchestrator's event stream via an asyncio.Queue.
  - Dependency graph is re-evaluated after each batch.
  - Partial failure: failed tasks can be retried, skipped, or trigger replan
    depending on task.on_failure setting.

Synthesizer:
  - After all tasks complete (or max replans hit), synthesizer LLM
    merges all agent results into a final answer.
  - Conflicting agent conclusions are surfaced explicitly.

Public API:
  - run_stream(goal)  — canonical. AsyncGenerator[BusEvent, None]. Live events.
  - run(goal)         — thin drain that returns the final result dict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness.events import BusEvent, EventType
from harness.utils import parse_llm_json

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────


class OnFailure(str, Enum):
    RETRY = "retry"  # retry the task once
    SKIP = "skip"  # skip and continue
    REPLAN = "replan"  # trigger replanning
    ABORT = "abort"  # abort entire run


@dataclass
class Task:
    id: str
    agent_id: str
    instruction: str
    depends_on: list[str] = field(default_factory=list)
    on_failure: OnFailure = OnFailure.REPLAN
    max_retries: int = 1
    _retry_count: int = field(default=0, init=False, repr=False)


@dataclass
class TaskResult:
    task_id: str
    agent_id: str
    answer: str
    confidence: float  # 0.0 – 1.0
    steps: int
    success: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Plan:
    tasks: list[Task]
    rationale: str = ""


# ── Prompts ───────────────────────────────────────────────────────────────────

PLAN_SYSTEM = """
You are a planning agent that decomposes goals into tasks for specialist agents.

Available agents:
{agent_descriptions}

Return a JSON plan with this exact shape:
{{
  "tasks": [
    {{
      "id": "t1",
      "agent_id": "<agent_id from above>",
      "instruction": "<specific, actionable instruction for this agent>",
      "depends_on": [],
      "on_failure": "replan"
    }},
    ...
  ],
  "rationale": "<one sentence explaining the decomposition>"
}}

Rules:
- Use only agent_ids from the available agents list
- Prefer the minimum number of tasks — use a single task if one agent can handle the entire goal
- Only split into multiple tasks when different agents are needed OR when true parallelism provides clear value
- Parallelize tasks that don't depend on each other
- on_failure options: retry | skip | replan | abort
- depends_on: list of task ids that must complete before this task starts
- Keep instructions specific and actionable — not vague
- Return JSON only, no markdown fences
"""

REPLAN_SYSTEM = """
You are a replanning agent. A previous plan partially executed but hit a failure or low-confidence result.

Available agents:
{agent_descriptions}

Completed tasks (do NOT re-run these):
{completed}

Failed/low-confidence task:
{failed_task}

Remaining original tasks (may need revision):
{remaining}

Produce a revised plan for the REMAINING work only. Do not include already-completed tasks.
Return the same JSON shape as the original plan.
Return JSON only, no markdown fences.
"""

SYNTHESIZE_SYSTEM = """
You are a synthesis agent. Multiple specialist agents have completed their work.
Produce a final, coherent answer to the original goal.

If agents produced conflicting conclusions, explicitly note the conflict and reason through it.
If some agents failed, note what is unknown as a result.

Return a JSON object:
{{
  "answer": "<comprehensive final answer>",
  "confidence": <0.0-1.0>,
  "conflicts": ["<conflict description>", ...],
  "unknowns": ["<what we couldn't determine>", ...]
}}
Return JSON only.
"""


# ── Evaluator ─────────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    confidence_threshold: float = 0.6  # below this → replan trigger
    max_replan_count: int = 2  # hard limit on replanning iterations


def should_replan(result: TaskResult, config: EvalConfig) -> bool:
    """Replan trigger: task failed OR confidence below threshold."""
    return not result.success or result.confidence < config.confidence_threshold


# ── Orchestrator ──────────────────────────────────────────────────────────────


class Orchestrator:
    """
    Hybrid orchestrator: static DAG planning with replan-on-failure.

    Lifecycle per run (driven by run_stream; run() is just a drain):
      1. plan(goal)              → initial DAG → PLAN event
      2. execute_dag(plan)       → run tasks in dependency order, parallel where possible
                                   forward each agent's events into the orchestrator stream
                                   after each task:
                                     evaluate result, yield TASK_DONE
                                     if replan trigger → replan → yield REPLAN
      3. synthesize(all_results) → yield SYNTHESIS
      4. memory.write_run_end()  → durable memory write
      5. yield DONE with the final result dict
    """

    def __init__(
        self,
        agents: dict[str, Any],  # agent_id → BaseAgent
        memory,  # MemoryManager
        tracer,  # Tracer
        guard,  # BudgetGuard
        llm,
        eval_config: EvalConfig | None = None,
    ) -> None:
        self._agents = agents
        self._memory = memory
        self._tracer = tracer
        self._guard = guard
        self._llm = llm
        self._eval = eval_config or EvalConfig()
        self._run_id = str(uuid.uuid4())

    # ── Streaming entry point (canonical) ─────────────────────────────────────

    async def run_stream(self, goal: str) -> AsyncGenerator[BusEvent, None]:
        logger.info("Orchestrator run_id=%s goal=%r", self._run_id, goal[:80])
        self._tracer.start_run(self._run_id, goal)

        # ── 1. Plan ────────────────────────────────────────────────────────────
        plan = await self._plan(goal)
        plan_dict = _plan_to_dict(plan)
        self._tracer.log("plan", "orchestrator", {"plan": plan_dict})
        yield BusEvent(
            type=EventType.PLAN,
            agent_id="orchestrator",
            payload={"plan": plan_dict},
        )

        # ── 2. Execute with hybrid replan ──────────────────────────────────────
        completed: dict[str, TaskResult] = {}
        pending: list[Task] = list(plan.tasks)
        replan_count = 0
        aborted = False

        while pending and not aborted:
            self._guard.check()
            ready = [t for t in pending if all(dep in completed for dep in t.depends_on)]
            if not ready:
                logger.warning(
                    "Dependency deadlock — remaining tasks: %s",
                    [t.id for t in pending],
                )
                break

            batch_results: dict[str, TaskResult] = {}
            async for event in self._run_batch(ready, batch_results, completed):
                yield event

            for t in ready:
                if t in pending:
                    pending.remove(t)
            completed.update(batch_results)

            # Per-task replan / on_failure decisions
            for task in ready:
                result = batch_results.get(task.id)
                if result is None:
                    continue

                self._tracer.log(
                    "task_result",
                    task.agent_id,
                    {
                        "task_id": task.id,
                        "success": result.success,
                        "confidence": result.confidence,
                    },
                )
                yield BusEvent(
                    type=EventType.TASK_DONE,
                    agent_id=task.agent_id,
                    payload={
                        "task_id": task.id,
                        "success": result.success,
                        "confidence": result.confidence,
                        "answer": result.answer,
                        "error": result.error,
                    },
                )

                if not should_replan(result, self._eval):
                    continue
                if replan_count >= self._eval.max_replan_count:
                    logger.warning(
                        "Max replans (%d) reached — continuing with low-confidence result",
                        self._eval.max_replan_count,
                    )
                    continue
                if task.on_failure == OnFailure.ABORT:
                    logger.error("Task %s failed with on_failure=abort", task.id)
                    aborted = True
                    break
                if task.on_failure == OnFailure.SKIP:
                    logger.info("Task %s skipped (on_failure=skip)", task.id)
                    continue
                if task.on_failure == OnFailure.RETRY and task._retry_count < task.max_retries:
                    task._retry_count += 1
                    pending.insert(0, task)
                    logger.info("Retrying task %s (attempt %d)", task.id, task._retry_count)
                    continue

                # REPLAN — rebuild remaining DAG from current state
                new_plan = await self._replan(
                    goal=goal,
                    completed=list(completed.values()),
                    failed_result=result,
                    remaining_tasks=pending,
                )
                pending = list(new_plan.tasks)
                replan_count += 1
                self._tracer.log(
                    "replan",
                    "orchestrator",
                    {
                        "replan_count": replan_count,
                        "trigger_task": task.id,
                        "new_task_count": len(pending),
                    },
                )
                yield BusEvent(
                    type=EventType.REPLAN,
                    agent_id="orchestrator",
                    payload={
                        "replan_count": replan_count,
                        "trigger_task": task.id,
                        "new_task_count": len(pending),
                    },
                )

        all_results = list(completed.values())

        # ── 3. Synthesize ──────────────────────────────────────────────────────
        synthesis = await self._synthesize(goal, all_results)
        self._tracer.log("synthesis", "orchestrator", synthesis)
        yield BusEvent(
            type=EventType.SYNTHESIS,
            agent_id="orchestrator",
            payload=synthesis,
        )

        # ── 4. Run-end memory write ────────────────────────────────────────────
        await self._memory.write_run_end(
            goal=goal,
            agent_results=[
                {
                    "agent_id": r.agent_id,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "success": r.success,
                }
                for r in all_results
            ],
            trace=self._tracer.dump(),
        )

        # ── 5. Final DONE ──────────────────────────────────────────────────────
        self._tracer.end_run()
        yield BusEvent(
            type=EventType.DONE,
            agent_id="orchestrator",
            payload={
                "run_id": self._run_id,
                "goal": goal,
                "answer": synthesis.get("answer", ""),
                "confidence": synthesis.get("confidence", 0.0),
                "conflicts": synthesis.get("conflicts", []),
                "unknowns": synthesis.get("unknowns", []),
                "replan_count": replan_count,
                # Budget surface — populated by adapters that report usage/cost.
                # cost_usd may be 0.0 if no cost source is wired up (still useful
                # to know zero was billed; tokens flow via TASK_DONE/OTEL).
                "cost_usd": self._guard.cost,
                "elapsed_seconds": self._guard.elapsed,
                "task_results": [
                    {
                        "task_id": r.task_id,
                        "agent_id": r.agent_id,
                        "success": r.success,
                        "confidence": r.confidence,
                    }
                    for r in all_results
                ],
            },
        )

    # ── Blocking entry point (thin drain) ─────────────────────────────────────

    async def run(self, goal: str) -> dict:
        result: dict = {}
        async for event in self.run_stream(goal):
            if event.type == EventType.DONE:
                result = event.payload
        return result

    # ── Planning ──────────────────────────────────────────────────────────────

    async def _plan(self, goal: str, context: str = "") -> Plan:
        agent_descriptions = "\n".join(
            f"  {aid}: {getattr(agent, 'role', 'no description')}"
            for aid, agent in self._agents.items()
        )
        prompt = f"Goal: {goal}"

        mem_context = await self._memory.build_context(goal)
        if not mem_context.is_empty():
            prompt += f"\n\nRelevant context from memory:\n{mem_context.render()}"

        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        response = await self._llm.complete(
            system=PLAN_SYSTEM.format(agent_descriptions=agent_descriptions),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return _parse_plan(response)

    async def _replan(
        self,
        goal: str,
        completed: list[TaskResult],
        failed_result: TaskResult,
        remaining_tasks: list[Task],
    ) -> Plan:
        agent_descriptions = "\n".join(
            f"  {aid}: {getattr(agent, 'role', 'no description')}"
            for aid, agent in self._agents.items()
        )
        replan_prompt = f"Replan for goal: {goal}"
        mem_context = await self._memory.build_context(goal)
        if not mem_context.is_empty():
            replan_prompt += f"\n\nRelevant context from memory:\n{mem_context.render()}"

        response = await self._llm.complete(
            system=REPLAN_SYSTEM.format(
                agent_descriptions=agent_descriptions,
                completed=json.dumps(
                    [{"task_id": r.task_id, "answer": r.answer} for r in completed],
                    indent=2,
                ),
                failed_task=json.dumps(
                    {
                        "task_id": failed_result.task_id,
                        "answer": failed_result.answer,
                        "confidence": failed_result.confidence,
                        "error": failed_result.error,
                    },
                    indent=2,
                ),
                remaining=json.dumps(
                    [
                        {"id": t.id, "agent_id": t.agent_id, "instruction": t.instruction}
                        for t in remaining_tasks
                    ],
                    indent=2,
                ),
            ),
            messages=[{"role": "user", "content": replan_prompt}],
            response_format={"type": "json_object"},
        )
        return _parse_plan(response)

    # ── Execution: parallel batch with fan-in ─────────────────────────────────

    async def _run_batch(
        self,
        ready: list[Task],
        results_out: dict[str, TaskResult],
        completed_results: dict[str, TaskResult] | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        """
        Run a batch of ready tasks in parallel, forwarding each agent's events
        upstream. When all tasks finish, results_out is populated with one
        TaskResult per task id.

        completed_results is used to inject upstream dependency outputs into
        each task's instruction so the agent has the context it needs.
        """
        bus: asyncio.Queue = asyncio.Queue()
        DRIVER_DONE = object()

        async def drive(task: Task) -> None:
            agent = self._agents.get(task.agent_id)
            if agent is None:
                await bus.put(
                    (
                        task,
                        TaskResult(
                            task_id=task.id,
                            agent_id=task.agent_id,
                            answer="",
                            confidence=0.0,
                            steps=0,
                            success=False,
                            error=f"Agent '{task.agent_id}' not found in registry",
                        ),
                    )
                )
                await bus.put((task, DRIVER_DONE))
                return

            # Inject upstream dependency results into the instruction
            instruction = task.instruction
            if completed_results and task.depends_on:
                dep_parts = []
                for dep_id in task.depends_on:
                    dep_result = completed_results.get(dep_id)
                    if dep_result is not None and dep_result.answer:
                        dep_parts.append(
                            f"[Result from task {dep_id} "
                            f"(agent: {dep_result.agent_id})]:\n"
                            f"{dep_result.answer}"
                        )
                if dep_parts:
                    instruction = (
                        f"{task.instruction}\n\n"
                        f"--- Context from completed upstream tasks ---\n" + "\n\n".join(dep_parts)
                    )

            last_done: dict | None = None
            last_error: str | None = None
            try:
                async for event in agent.run_stream(
                    task=instruction,
                    run_id=self._run_id,
                ):
                    if event.type == EventType.TASK_DONE:
                        last_done = event.payload
                    elif event.type == EventType.ERROR:
                        last_error = event.error
                    await bus.put((task, event))
            except Exception as e:
                logger.error(
                    "Task %s agent %s crashed: %s",
                    task.id,
                    task.agent_id,
                    e,
                )
                last_error = str(e)

            if last_done is not None:
                result = TaskResult(
                    task_id=task.id,
                    agent_id=task.agent_id,
                    answer=last_done.get("answer", ""),
                    confidence=last_done.get("confidence", 1.0),
                    steps=last_done.get("steps", 0),
                    success=True,
                    metadata=last_done.get("metadata", {}),
                )
            else:
                result = TaskResult(
                    task_id=task.id,
                    agent_id=task.agent_id,
                    answer="",
                    confidence=0.0,
                    steps=0,
                    success=False,
                    error=last_error or "agent stream ended without TASK_DONE",
                )
            await bus.put((task, result))
            await bus.put((task, DRIVER_DONE))

        drivers = [asyncio.create_task(drive(t)) for t in ready]
        finished = 0
        try:
            while finished < len(ready):
                task, payload = await bus.get()
                if payload is DRIVER_DONE:
                    finished += 1
                elif isinstance(payload, BusEvent):
                    # Skip per-agent TASK_DONE; orchestrator re-emits a richer one
                    # in run_stream after building the TaskResult and applying
                    # replan / on_failure logic.
                    if payload.type != EventType.TASK_DONE:
                        yield payload
                elif isinstance(payload, TaskResult):
                    results_out[task.id] = payload
        finally:
            await asyncio.gather(*drivers, return_exceptions=True)

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def _synthesize(self, goal: str, results: list[TaskResult]) -> dict:
        results_text = json.dumps(
            [
                {
                    "agent_id": r.agent_id,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        )
        try:
            response = await self._llm.complete(
                system=SYNTHESIZE_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"Goal: {goal}\n\nAgent results:\n{results_text}",
                    }
                ],
                response_format={"type": "json_object"},
            )
            return parse_llm_json(response)
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            return {
                "answer": "\n".join(r.answer for r in results if r.answer),
                "confidence": 0.5,
                "conflicts": [],
                "unknowns": [f"Synthesis failed: {e}"],
            }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_plan(response: Any) -> Plan:
    data = parse_llm_json(response)

    tasks = [
        Task(
            id=t["id"],
            agent_id=t["agent_id"],
            instruction=t["instruction"],
            depends_on=t.get("depends_on", []),
            on_failure=OnFailure(t.get("on_failure", "replan")),
        )
        for t in data.get("tasks", [])
    ]
    return Plan(tasks=tasks, rationale=data.get("rationale", ""))


def _plan_to_dict(plan: Plan) -> dict:
    return {
        "rationale": plan.rationale,
        "tasks": [
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "instruction": t.instruction,
                "depends_on": t.depends_on,
                "on_failure": t.on_failure.value,
            }
            for t in plan.tasks
        ],
    }
