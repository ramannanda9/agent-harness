"""
Canonical event types for the unified streaming/blocking execution model.

Every run flows through the same event sequence — callers using `run()` see
the same execution; they just collect the final `DONE` event's payload
instead of iterating events live.

Event lifecycle within a single goal:

  Dispatch path (dispatch / dispatch_stream)  ← recommended default entry point:
    DISPATCH              — classifier chose a path; payload has complexity + path
    then either routed or orchestrated path below

  Routed path (run_routed / run_routed_stream):
    ROUTE                 — router picked an agent; payload has agent_id + rationale
    THOUGHT / TOKEN / ACTION / OBSERVATION / TASK_DONE  (single ReAct loop)

  Orchestrated path (run / run_stream):
    PLAN                  — orchestrator emitted a static DAG
    (per task in DAG)
        HUMAN_GUIDANCE?   — async steering drained at top of step
        CONTEXT           — working-memory context budget estimate
        MEMORY            — working-memory compaction/summarization marker
        SUBAGENT_START?   — SubAgentTool delegation began (before sub's first event)
        THOUGHT           — agent's next-step reasoning
        TOKEN*            — partial LLM output (only when client streams)
        ACTION            — agent chose a tool + args
        OBSERVATION       — tool returned a result
        ... (loop until)
        TASK_DONE         — agent finished a task; carries result payload
        SUBAGENT_DONE?    — SubAgentTool delegation finished (after sub's TASK_DONE)
    REPLAN?               — replan fired (low confidence or failure)
    SYNTHESIS             — synthesizer merged task results
    DONE                  — orchestrator finished; payload is the final result dict
    ERROR                 — terminal failure (replaces DONE if it fires)

Typed payloads
--------------
``BusEvent.payload`` is a plain ``dict[str, Any]`` on the wire (so trace
serialization and renderers stay unchanged), but producers should NOT author
payload dicts by hand. Construct events through the typed factory classmethods
(``BusEvent.action(...)``, ``BusEvent.subagent_done(...)``, …). Each factory's
signature is the authoritative schema for that event's payload — there is no
longer a place to type ``steps=`` where ``step=`` was meant.

A few sites legitimately pass through an externally-shaped dict rather than
authoring keys (the SubAgentTool event re-tagging, the orchestrator SYNTHESIS
payload from the synthesizer LLM, the CONTEXT pre-think working-memory snapshot,
and trace reconstruction). Those keep the raw ``BusEvent(...)`` constructor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    DISPATCH = "dispatch"
    ROUTE = "route"
    PLAN = "plan"
    PLAN_PROPOSED = "plan_proposed"  # PersistentAgent plan-mode: pre-HITL plan
    THOUGHT = "thought"
    TOKEN = "token"
    ACTION = "action"
    OBSERVATION = "observation"
    CONTEXT = "context"
    MEMORY = "memory"
    HUMAN_GUIDANCE = "human_guidance"  # async steering injected at step boundary
    SUBAGENT_START = "subagent_start"  # SubAgentTool begins; payload: task, invocation_id
    SUBAGENT_DONE = "subagent_done"  # SubAgentTool finished; payload: success, steps, confidence
    TASK_DONE = "task_done"
    REPLAN = "replan"
    SYNTHESIS = "synthesis"
    DONE = "done"
    ERROR = "error"


# ── Typed payload schemas ─────────────────────────────────────────────────────
#
# Frozen dataclasses define the exact payload key set for each event. ``to_dict``
# returns the on-the-wire shape (shallow — nested dicts/lists are passed through
# by reference, matching the prior hand-authored construction). The ``BusEvent``
# factory classmethods below build payloads through these so the schema lives in
# exactly one place.


@dataclass(frozen=True)
class DispatchPayload:
    complexity: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {"complexity": self.complexity, "path": self.path}


@dataclass(frozen=True)
class RoutePayload:
    agent_id: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "rationale": self.rationale}


@dataclass(frozen=True)
class ActionPayload:
    step: int
    tool: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "tool": self.tool, "args": self.args}


@dataclass(frozen=True)
class ObservationPayload:
    step: int
    tool: str
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "tool": self.tool, "observation": self.observation}


@dataclass(frozen=True)
class MemoryPayload:
    before: dict[str, Any]
    after: dict[str, Any]
    summarizations: int
    event: str = "summarized"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "before": self.before,
            "after": self.after,
            "summarizations": self.summarizations,
        }


@dataclass(frozen=True)
class HumanGuidancePayload:
    step: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "text": self.text}


@dataclass(frozen=True)
class SubagentStartPayload:
    task: str
    invocation_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "invocation_id": self.invocation_id}


@dataclass(frozen=True)
class SubagentDonePayload:
    success: bool
    steps: int
    confidence: float
    answer: str
    error: str
    invocation_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": self.steps,
            "confidence": self.confidence,
            "answer": self.answer,
            "error": self.error,
            "invocation_id": self.invocation_id,
        }


@dataclass(frozen=True)
class ReplanPayload:
    replan_count: int
    trigger_task: str
    new_task_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "replan_count": self.replan_count,
            "trigger_task": self.trigger_task,
            "new_task_count": self.new_task_count,
        }


@dataclass
class BusEvent:
    type: EventType
    agent_id: str = ""  # empty for orchestrator-level events
    payload: dict[str, Any] = field(default_factory=dict)
    token: str = ""  # set on TOKEN events
    error: str = ""  # set on ERROR events
    timestamp: float = field(default_factory=time.time)
    # Set on events emitted by a sub-agent run launched via SubAgentTool.
    # Lets renderers and tracers indent/group nested agent work without
    # losing the originating agent_id. Empty for top-level events.
    parent_agent_id: str = ""

    # ── Typed factories ───────────────────────────────────────────────────────
    # Construct events through these instead of authoring ``payload`` dicts.

    @classmethod
    def dispatch(cls, agent_id: str, *, complexity: str, path: str) -> BusEvent:
        return cls(
            type=EventType.DISPATCH,
            agent_id=agent_id,
            payload=DispatchPayload(complexity=complexity, path=path).to_dict(),
        )

    @classmethod
    def route(cls, agent_id: str, *, rationale: str) -> BusEvent:
        # ``agent_id`` is both the event's agent and a payload key (the routed
        # agent), matching the prior shape.
        return cls(
            type=EventType.ROUTE,
            agent_id=agent_id,
            payload=RoutePayload(agent_id=agent_id, rationale=rationale).to_dict(),
        )

    @classmethod
    def plan(
        cls,
        agent_id: str,
        *,
        plan: dict[str, Any],
        resumed: bool = False,
        pre_built: bool = False,
    ) -> BusEvent:
        payload: dict[str, Any] = {"plan": plan}
        if resumed:
            payload["resumed"] = True
        if pre_built:
            payload["pre_built"] = True
        return cls(type=EventType.PLAN, agent_id=agent_id, payload=payload)

    @classmethod
    def plan_proposed(cls, agent_id: str, *, plan: dict[str, Any], revision: int) -> BusEvent:
        return cls(
            type=EventType.PLAN_PROPOSED,
            agent_id=agent_id,
            payload={"plan": plan, "revision": revision},
        )

    @classmethod
    def thought(cls, agent_id: str, *, response: dict[str, Any] | None) -> BusEvent:
        # Derive the display fields from the parsed LLM response, matching the
        # prior inline construction (empty/None when there is no response).
        return cls(
            type=EventType.THOUGHT,
            agent_id=agent_id,
            payload={
                "response": response,
                "thought": response.get("thought", "") if response else "",
                "action": response.get("action") if response else None,
            },
        )

    @classmethod
    def token_event(cls, agent_id: str, *, token: str) -> BusEvent:
        # Named ``token_event`` (not ``token``) to avoid colliding with the
        # ``token`` dataclass field — a classmethod of the same name would
        # overwrite the field's default at class-body execution.
        return cls(type=EventType.TOKEN, agent_id=agent_id, token=token)

    @classmethod
    def action(
        cls,
        agent_id: str,
        *,
        step: int,
        tool: str,
        args: dict[str, Any],
    ) -> BusEvent:
        return cls(
            type=EventType.ACTION,
            agent_id=agent_id,
            payload=ActionPayload(step=step, tool=tool, args=args).to_dict(),
        )

    @classmethod
    def observation(cls, agent_id: str, *, step: int, tool: str, observation: str) -> BusEvent:
        return cls(
            type=EventType.OBSERVATION,
            agent_id=agent_id,
            payload=ObservationPayload(step=step, tool=tool, observation=observation).to_dict(),
        )

    @classmethod
    def context_usage(
        cls,
        agent_id: str,
        *,
        usage: dict[str, Any],
        tokens_in: int | None,
        tokens_out: int | None,
        cache_read_tokens: int | None,
        cache_creation_tokens: int | None,
    ) -> BusEvent:
        # Post-think CONTEXT: working-memory usage merged with the LLM's
        # per-call token counts (the four keys are always present, possibly
        # ``None``, matching the prior shape). The pre-think snapshot uses the
        # raw constructor since it is a pure passthrough of ``context_usage()``.
        return cls(
            type=EventType.CONTEXT,
            agent_id=agent_id,
            payload={
                **usage,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
            },
        )

    @classmethod
    def memory(
        cls,
        agent_id: str,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        summarizations: int,
    ) -> BusEvent:
        return cls(
            type=EventType.MEMORY,
            agent_id=agent_id,
            payload=MemoryPayload(
                before=before, after=after, summarizations=summarizations
            ).to_dict(),
        )

    @classmethod
    def human_guidance(cls, agent_id: str, *, step: int, text: str) -> BusEvent:
        return cls(
            type=EventType.HUMAN_GUIDANCE,
            agent_id=agent_id,
            payload=HumanGuidancePayload(step=step, text=text).to_dict(),
        )

    @classmethod
    def subagent_start(
        cls,
        agent_id: str,
        *,
        task: str,
        invocation_id: str,
        parent_agent_id: str = "",
    ) -> BusEvent:
        return cls(
            type=EventType.SUBAGENT_START,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            payload=SubagentStartPayload(task=task, invocation_id=invocation_id).to_dict(),
        )

    @classmethod
    def subagent_done(
        cls,
        agent_id: str,
        *,
        success: bool,
        steps: int,
        confidence: float,
        answer: str,
        error: str,
        invocation_id: str,
        parent_agent_id: str = "",
    ) -> BusEvent:
        return cls(
            type=EventType.SUBAGENT_DONE,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            payload=SubagentDonePayload(
                success=success,
                steps=steps,
                confidence=confidence,
                answer=answer,
                error=error,
                invocation_id=invocation_id,
            ).to_dict(),
        )

    @classmethod
    def task_done_agent(cls, agent_id: str, *, result: dict[str, Any]) -> BusEvent:
        # The agent ReAct loop's terminal result is a controlled dict (agent_id,
        # answer, confidence, steps, metadata, optional budget) that doubles as
        # the run()'s return value, so it is passed through as-is.
        return cls(type=EventType.TASK_DONE, agent_id=agent_id, payload=result)

    @classmethod
    def task_done_task(
        cls,
        agent_id: str,
        *,
        task_id: str,
        success: bool,
        confidence: float,
        answer: str,
        error: str | None,
    ) -> BusEvent:
        return cls(
            type=EventType.TASK_DONE,
            agent_id=agent_id,
            payload={
                "task_id": task_id,
                "success": success,
                "confidence": confidence,
                "answer": answer,
                "error": error,
            },
        )

    @classmethod
    def replan(
        cls,
        agent_id: str,
        *,
        replan_count: int,
        trigger_task: str,
        new_task_count: int,
    ) -> BusEvent:
        return cls(
            type=EventType.REPLAN,
            agent_id=agent_id,
            payload=ReplanPayload(
                replan_count=replan_count,
                trigger_task=trigger_task,
                new_task_count=new_task_count,
            ).to_dict(),
        )

    @classmethod
    def done(cls, agent_id: str, *, payload: dict[str, Any]) -> BusEvent:
        # The orchestrator's terminal payload is wide (run_id, goal, answer,
        # confidence, conflicts, unknowns, replan_count, cost/elapsed/budget,
        # task_results) and assembled at the call site; passed through as-is.
        return cls(type=EventType.DONE, agent_id=agent_id, payload=payload)

    @classmethod
    def error_event(cls, agent_id: str, *, error: str, steps: int | None = None) -> BusEvent:
        # Named ``error_event`` (not ``error``) to avoid colliding with the
        # ``error`` dataclass field.
        payload: dict[str, Any] = {} if steps is None else {"steps": steps}
        return cls(type=EventType.ERROR, agent_id=agent_id, error=error, payload=payload)
