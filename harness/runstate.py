"""
harness/runstate.py — explicit, serializable state for a run in progress.

The ReAct loop used to keep its position implicitly, in the Python coroutine
stack: ``for step in range(...)`` plus whatever local variables happened to be
live. That works perfectly while the process is alive — a HITL pause is just an
``await`` — but it means a run's position is unreadable from the outside and
unrecoverable after the process dies. Resumption had to be a second, partial
re-implementation of the loop, and the two drifted.

This module makes the position a value. ``RunState`` is everything needed to
continue a run: where it is (``step`` + ``phase``), what it was doing
(``response`` + ``actions``), and what it has spent (``memory`` + ``budget``).
Driving a run and resuming one are then the same operation — load the state and
keep advancing it.

Phases
------
Each ReAct step walks THINK → APPROVE → ACT → OBSERVE and back to THINK, or
exits to a terminal phase::

    THINK ──finish──> DONE
      │
      v                    (correction re-enters THINK at step+1)
    APPROVE ──> ACT ──> OBSERVE ──> THINK
      │
      └── max_steps / budget / unparseable response ──> FAILED

Actions carry their own status so a batch is resumable *element-wise*: a crash
partway through gating or executing three parallel tools resumes knowing exactly
which were approved and which observations already landed, instead of replaying
one and dropping the rest.

Orchestration
-------------
``OrchestratorState`` is the same idea one level up: per-task ``TaskState``
replaces "membership in a completed dict", so a task interrupted mid-flight is
distinguishable from one never started, and carries the checkpoint key of the
agent that was running it.

Both records carry a ``kind`` discriminator so stored state is identified by a
field rather than by sniffing which keys happen to be present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Bumped when the on-disk shape changes incompatibly. ``load_state`` refuses
# anything it does not recognise rather than guessing at a migration.
STATE_VERSION = 1


class Phase(str, Enum):
    """Where a single ReAct step currently is."""

    THINK = "think"  # about to call the LLM
    APPROVE = "approve"  # actions parsed; resolving HITL for each
    ACT = "act"  # approvals resolved; executing tools
    OBSERVE = "observe"  # tools finished; recording observations
    DONE = "done"  # terminal: finished successfully
    FAILED = "failed"  # terminal: ran to completion but failed


TERMINAL_PHASES = (Phase.DONE, Phase.FAILED)


class ActionStatus(str, Enum):
    """Where one tool call within a step currently is."""

    PENDING = "pending"  # not yet gated
    APPROVED = "approved"  # gate passed (or not gated); not yet executed
    REJECTED = "rejected"  # human declined; will not execute
    EXECUTED = "executed"  # observation captured


@dataclass
class ActionState:
    """One tool call and how far it has got.

    ``observation`` is whatever the tool returned — a string, a parsed JSON
    value, or an image content block. It is written once the action reaches
    ``EXECUTED`` so that a resumed run does not re-run a tool whose side
    effects already happened.
    """

    tool: str
    args: dict = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    observation: Any = None
    approval_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "status": self.status.value,
            "observation": self.observation,
            "approval_id": self.approval_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActionState:
        return cls(
            tool=d["tool"],
            args=d.get("args") or {},
            status=ActionStatus(d.get("status", ActionStatus.PENDING.value)),
            observation=d.get("observation"),
            approval_id=d.get("approval_id"),
        )


@dataclass
class RunState:
    """Everything needed to continue one agent's ReAct loop."""

    run_id: str
    agent_id: str
    task: str
    memory: dict = field(default_factory=dict)  # WorkingMemory.to_dict()
    step: int = 0
    phase: Phase = Phase.THINK
    response: dict | None = None  # parsed LLM action JSON for this step
    actions: list[ActionState] = field(default_factory=list)
    budget: dict | None = None  # BudgetGuard.snapshot()
    result: dict | None = None
    error: str | None = None
    kind: str = "agent"
    version: int = STATE_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "task": self.task,
            "step": self.step,
            "phase": self.phase.value,
            "memory": self.memory,
            "response": self.response,
            "actions": [a.to_dict() for a in self.actions],
            "budget": self.budget,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunState:
        return cls(
            run_id=d["run_id"],
            agent_id=d["agent_id"],
            task=d["task"],
            memory=d.get("memory") or {},
            step=d.get("step", 0),
            phase=Phase(d.get("phase", Phase.THINK.value)),
            response=d.get("response"),
            actions=[ActionState.from_dict(a) for a in d.get("actions") or []],
            budget=d.get("budget"),
            result=d.get("result"),
            error=d.get("error"),
            version=d.get("version", STATE_VERSION),
        )


class TaskStatus(str, Enum):
    """Where one planned task currently is.

    ``RUNNING`` is the state the old ``completed``-dict model could not
    express: a task that started and did not finish was indistinguishable
    from one that never started, so it always restarted from zero.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskState:
    """One planned task and how far it has got.

    ``attempt`` is durable, so ``on_failure=retry`` cannot silently hand a
    task a fresh retry budget every time the run resumes.

    ``agent_ckp_id`` is the checkpoint key of the agent that was running this
    task, which is what lets an interrupted task resume from its own ReAct
    position rather than from the top.
    """

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    result: dict | None = None  # TaskResult, as a dict
    agent_ckp_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "attempt": self.attempt,
            "result": self.result,
            "agent_ckp_id": self.agent_ckp_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskState:
        return cls(
            task_id=d["task_id"],
            status=TaskStatus(d.get("status", TaskStatus.PENDING.value)),
            attempt=d.get("attempt", 0),
            result=d.get("result"),
            agent_ckp_id=d.get("agent_ckp_id"),
        )


@dataclass
class OrchestratorState:
    """Everything needed to continue a multi-agent orchestration.

    ``plan`` is the *current* plan. Replanning replaces it here, so resuming
    after a replan continues with the DAG the run was actually executing
    rather than the one it started with.
    """

    run_id: str
    goal: str
    plan: dict = field(default_factory=dict)  # Plan, as a dict
    tasks: dict[str, TaskState] = field(default_factory=dict)
    replan_count: int = 0
    budget: dict | None = None
    kind: str = "orchestrator"
    version: int = STATE_VERSION

    def completed_results(self) -> dict[str, dict]:
        """Results of tasks that reached a terminal, successful state."""
        return {
            tid: ts.result
            for tid, ts in self.tasks.items()
            if ts.status is TaskStatus.DONE and ts.result is not None
        }

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "run_id": self.run_id,
            "goal": self.goal,
            "plan": self.plan,
            "tasks": {tid: ts.to_dict() for tid, ts in self.tasks.items()},
            "replan_count": self.replan_count,
            "budget": self.budget,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OrchestratorState:
        return cls(
            run_id=d["run_id"],
            goal=d["goal"],
            plan=d.get("plan") or {},
            tasks={tid: TaskState.from_dict(ts) for tid, ts in (d.get("tasks") or {}).items()},
            replan_count=d.get("replan_count", 0),
            budget=d.get("budget"),
            version=d.get("version", STATE_VERSION),
        )


class LegacyCheckpointError(ValueError):
    """Raised for checkpoints written before run state became explicit.

    A ``ValueError`` subclass so callers that only care that decoding failed
    keep working, while callers that want to say something specific about
    upgrading can catch this.
    """


def load_state(d: dict) -> RunState | OrchestratorState:
    """Decode a stored checkpoint into its state object.

    Dispatches on the ``kind`` discriminator rather than sniffing for keys.
    Checkpoints written before this module existed carry no ``kind`` and are
    refused: they predate per-action and per-task status, so there is no
    honest way to reconstruct where such a run actually was.
    """
    if not isinstance(d, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(d).__name__}")

    kind = d.get("kind")
    if kind is None:
        raise LegacyCheckpointError(
            "This checkpoint predates the explicit run-state format "
            "(no 'kind' field) and cannot be resumed. Re-run the task; "
            "checkpoints are short-lived by design."
        )

    version = d.get("version", STATE_VERSION)
    if version > STATE_VERSION:
        raise ValueError(
            f"Checkpoint version {version} is newer than this harness "
            f"supports (version {STATE_VERSION}). Upgrade the harness to resume it."
        )

    if kind == "agent":
        return RunState.from_dict(d)
    if kind == "orchestrator":
        return OrchestratorState.from_dict(d)
    raise ValueError(f"Unknown checkpoint kind {kind!r}; expected 'agent' or 'orchestrator'.")
