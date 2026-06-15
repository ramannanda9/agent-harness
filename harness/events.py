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
