"""
Orchestrator — hybrid planning with replan trigger.

Planning style (per design decision): Hybrid
  - Plan upfront as a static DAG
  - After each agent completes, evaluate result quality
  - Replan trigger: confidence < threshold OR agent failed
  - Replanning is an LLM call — guarded by max_replan_count
  - Replan produces a new DAG from the current state forward

DAG execution:
  - Tasks with no unmet dependencies run in parallel via asyncio.gather
  - Dependency graph is re-evaluated after each batch
  - Partial failure: failed tasks can be retried, skipped, or trigger replan
    depending on task.on_failure setting

Synthesizer:
  - After all tasks complete (or max replans hit), synthesizer LLM
    merges all agent results into a final answer
  - Conflicting agent conclusions are surfaced explicitly
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

class OnFailure(str, Enum):
    RETRY   = "retry"    # retry the task once
    SKIP    = "skip"     # skip and continue
    REPLAN  = "replan"   # trigger replanning
    ABORT   = "abort"    # abort entire run


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
    confidence: float        # 0.0 – 1.0
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
    confidence_threshold: float = 0.6    # below this → replan trigger
    max_replan_count: int = 2            # hard limit on replanning iterations


def should_replan(result: TaskResult, config: EvalConfig) -> bool:
    """Replan trigger: task failed OR confidence below threshold."""
    return not result.success or result.confidence < config.confidence_threshold


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Hybrid orchestrator: static DAG planning with replan-on-failure.

    Lifecycle per run:
      1. plan(goal)              → initial DAG
      2. execute_dag(plan)       → run tasks in dependency order, parallel where possible
           after each task:
             evaluate result
             if replan trigger → replan(remaining tasks) → new DAG → continue
      3. synthesize(all_results) → final answer
      4. memory.write_run_end()  → durable memory write
    """

    def __init__(
        self,
        agents: dict[str, Any],      # agent_id → BaseAgent
        memory,                       # MemoryManager
        tracer,                       # Tracer
        guard,                        # BudgetGuard
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

    async def run(self, goal: str) -> dict:
        logger.info("Orchestrator run_id=%s goal=%r", self._run_id, goal[:80])

        # ── 1. Plan ────────────────────────────────────────────────────────────
        plan = await self._plan(goal)
        self._tracer.log("plan", "orchestrator", {"plan": _plan_to_dict(plan)})

        # ── 2. Execute with hybrid replan ──────────────────────────────────────
        all_results, replan_count = await self._execute_hybrid(goal, plan)

        # ── 3. Synthesize ──────────────────────────────────────────────────────
        synthesis = await self._synthesize(goal, all_results)
        self._tracer.log("synthesis", "orchestrator", synthesis)

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

        return {
            "run_id": self._run_id,
            "goal": goal,
            "answer": synthesis.get("answer", ""),
            "confidence": synthesis.get("confidence", 0.0),
            "conflicts": synthesis.get("conflicts", []),
            "unknowns": synthesis.get("unknowns", []),
            "replan_count": replan_count,
            "task_results": [
                {"task_id": r.task_id, "agent_id": r.agent_id,
                 "success": r.success, "confidence": r.confidence}
                for r in all_results
            ],
        }

    # ── Planning ──────────────────────────────────────────────────────────────

    async def _plan(self, goal: str, context: str = "") -> Plan:
        agent_descriptions = "\n".join(
            f"  {aid}: {getattr(agent, 'role', 'no description')}"
            for aid, agent in self._agents.items()
        )
        prompt = f"Goal: {goal}"
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
            messages=[{"role": "user", "content": f"Replan for goal: {goal}"}],
            response_format={"type": "json_object"},
        )
        return _parse_plan(response)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_hybrid(
        self, goal: str, plan: Plan
    ) -> tuple[list[TaskResult], int]:
        """
        Execute plan as DAG. After each task, evaluate and optionally replan.
        Returns (all_results, replan_count).
        """
        completed: dict[str, TaskResult] = {}   # task_id → result
        pending: list[Task] = list(plan.tasks)
        replan_count = 0

        while pending:
            self._guard.check()

            # find tasks whose dependencies are all satisfied
            ready = [
                t for t in pending
                if all(dep in completed for dep in t.depends_on)
            ]

            if not ready:
                logger.warning("Dependency deadlock — remaining tasks: %s",
                               [t.id for t in pending])
                break

            # run ready tasks in parallel
            batch_results = await asyncio.gather(
                *[self._run_task(t) for t in ready],
                return_exceptions=True,
            )

            for task, result in zip(ready, batch_results, strict=True):
                pending.remove(task)

                # handle asyncio exceptions
                if isinstance(result, Exception):
                    result = TaskResult(
                        task_id=task.id,
                        agent_id=task.agent_id,
                        answer="",
                        confidence=0.0,
                        steps=0,
                        success=False,
                        error=str(result),
                    )

                completed[task.id] = result
                self._tracer.log("task_result", task.agent_id, {
                    "task_id": task.id,
                    "success": result.success,
                    "confidence": result.confidence,
                })

                # ── Replan evaluation ───────────────────────────────────────
                if should_replan(result, self._eval):
                    if replan_count >= self._eval.max_replan_count:
                        logger.warning(
                            "Max replans (%d) reached — continuing with low-confidence result",
                            self._eval.max_replan_count,
                        )
                        continue

                    if task.on_failure == OnFailure.ABORT:
                        logger.error("Task %s failed with on_failure=abort", task.id)
                        return list(completed.values()), replan_count

                    if task.on_failure == OnFailure.SKIP:
                        logger.info("Task %s skipped (on_failure=skip)", task.id)
                        continue

                    if task.on_failure == OnFailure.RETRY and task._retry_count < task.max_retries:
                        task._retry_count += 1
                        pending.insert(0, task)   # re-queue at front
                        logger.info("Retrying task %s (attempt %d)", task.id, task._retry_count)
                        continue

                    # REPLAN — rebuild remaining DAG from current state
                    logger.info(
                        "Replan triggered by task=%s confidence=%.2f",
                        task.id, result.confidence,
                    )
                    new_plan = await self._replan(
                        goal=goal,
                        completed=list(completed.values()),
                        failed_result=result,
                        remaining_tasks=pending,
                    )
                    pending = list(new_plan.tasks)
                    replan_count += 1
                    self._tracer.log("replan", "orchestrator", {
                        "replan_count": replan_count,
                        "trigger_task": task.id,
                        "new_task_count": len(pending),
                    })

        return list(completed.values()), replan_count

    async def _run_task(self, task: Task) -> TaskResult:
        """Run a single task on the assigned agent."""
        agent = self._agents.get(task.agent_id)
        if agent is None:
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                answer="",
                confidence=0.0,
                steps=0,
                success=False,
                error=f"Agent '{task.agent_id}' not found in registry",
            )
        try:
            result = await agent.run(
                task=task.instruction,
                run_id=self._run_id,
            )
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                answer=result.get("answer", ""),
                confidence=result.get("confidence", 1.0),
                steps=result.get("steps", 0),
                success=True,
                metadata=result.get("metadata", {}),
            )
        except Exception as e:
            logger.error("Task %s agent %s failed: %s", task.id, task.agent_id, e)
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                answer="",
                confidence=0.0,
                steps=0,
                success=False,
                error=str(e),
            )

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
                messages=[{
                    "role": "user",
                    "content": f"Goal: {goal}\n\nAgent results:\n{results_text}",
                }],
                response_format={"type": "json_object"},
            )
            if isinstance(response, str):
                return json.loads(response)
            return response
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
    if isinstance(response, str):
        data = json.loads(response)
    elif isinstance(response, dict) and "text" in response:
        data = json.loads(response["text"])
    else:
        data = response

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
