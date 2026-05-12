"""
Canonical event types for the unified streaming/blocking execution model.

Every run flows through the same event sequence — callers using `run()` see
the same execution; they just collect the final `DONE` event's payload
instead of iterating events live.

Event lifecycle within a single goal:

  Routed path (run_routed / run_routed_stream):
    ROUTE                 — router picked an agent; payload has agent_id + rationale
    THOUGHT / TOKEN / ACTION / OBSERVATION / TASK_DONE  (single ReAct loop)

  Orchestrated path (run / run_stream):
    PLAN                  — orchestrator emitted a static DAG
    (per task in DAG)
        THOUGHT           — agent's next-step reasoning
        TOKEN*            — partial LLM output (only when client streams)
        ACTION            — agent chose a tool + args
        OBSERVATION       — tool returned a result
        ... (loop until)
        TASK_DONE         — agent finished a task; carries result payload
    REPLAN?               — replan fired (low confidence or failure)
    SYNTHESIS             — synthesizer merged task results
    DONE                  — orchestrator finished; payload is the final result dict
    ERROR                 — terminal failure (replaces DONE if it fires)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    ROUTE       = "route"
    PLAN        = "plan"
    THOUGHT     = "thought"
    TOKEN       = "token"
    ACTION      = "action"
    OBSERVATION = "observation"
    TASK_DONE   = "task_done"
    REPLAN      = "replan"
    SYNTHESIS   = "synthesis"
    DONE        = "done"
    ERROR       = "error"


@dataclass
class BusEvent:
    type: EventType
    agent_id: str = ""                            # empty for orchestrator-level events
    payload: dict[str, Any] = field(default_factory=dict)
    token: str = ""                               # set on TOKEN events
    error: str = ""                               # set on ERROR events
    timestamp: float = field(default_factory=time.time)
