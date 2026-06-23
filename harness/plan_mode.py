"""Plan-mode proposal, approval, and rendering helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from agents.base import BaseAgent, _parse_action_json
from harness.events import BusEvent

# When ``SessionState.plan_mode_enabled`` is True, ``PersistentAgent.chat``
# asks the coordinator's LLM for a structured plan before any tools run,
# yields a ``PLAN_PROPOSED`` event so renderers can display it, and gates
# execution on ``harness.hitl.request_plan_approval``. Plan approval handles
# Enter / y / n / correction only; there is no session-allow or persistent-
# allow shortcut because plan mode is intentionally per-turn.

_PLAN_SYSTEM_PROMPT = """You are in plan mode for an autonomous agent.

Output rules — read carefully, these are strict:
- Output ONE JSON object. Nothing else.
- No markdown code fences (no triple-backticks).
- No prose before or after the object.
- No ``<thinking>`` tags or other XML wrappers.
- The first character of your response MUST be ``{``.
- The last character of your response MUST be ``}``.

Do NOT execute any tools. Do NOT plan to execute tools yourself — the
plan is a description for a separate executor to follow.

The plan is the user's chance to approve INTENT, not arguments. Many
real tasks have args that can only be known at runtime — a URL
discovered by a search, a file path extracted from a directory listing,
a row id returned by a query. Be honest about which step args you can
fill in upfront and which depend on the previous step's observation.

Output schema:
{
  "summary": "<one-line summary of what the plan will accomplish>",
  "steps": [
    {
      "step": 1,
      "intent": "<what this step achieves>",
      "tool": "<tool name from the available list, or null if no tool>",
      "args": {<concrete arguments, OR null when args depend on prior observation>},
      "why": "<one-line justification>"
    },
    ...
  ]
}

Available tools: {tools}

Argument rules:
- Use concrete ``args`` only when the value was supplied by the user
  (e.g. they said "fetch https://x.com/y") or is otherwise knowable
  without running any tool first.
- Use ``null`` (or omit the field) when the value depends on an earlier
  step's tool output — e.g. "navigate to the most-cited paper's URL"
  after a search. Fabricating placeholder URLs / paths the agent will
  ignore is misleading.

If the user's request is conversational and needs no tool calls, return
a plan with a single step whose ``tool`` is null."""


_PLAN_CORRECTION_HINT = (
    "\n\nThe user reviewed your previous plan and asked for this revision:\n"
    "\n{correction}\n\n"
    "Output a revised JSON plan."
)


# Hard cap on revision loops to protect against pathological re-planning
# cycles. Each revision is one LLM call + one HITL prompt; users with
# stronger feedback can always reject (n) and start fresh.
_PLAN_REVISION_LIMIT = 3


@dataclass(frozen=True)
class PlanDecision:
    approved_plan: dict[str, Any] | None
    rejected: bool


class PlanModeController:
    """Own plan generation plus HITL approval/revision flow."""

    def __init__(
        self,
        *,
        coordinator: BaseAgent,
        llm: Callable[[], Any],
        tool_names: Callable[[], list[str]],
    ) -> None:
        self._coordinator = coordinator
        self._llm = llm
        self._tool_names = tool_names

    async def run(
        self,
        *,
        message: str,
        session_id: str,
        guard: Any,
    ) -> AsyncIterator[BusEvent | PlanDecision]:
        """Yield proposal/error events, then exactly one terminal decision."""
        from harness.hitl import request_plan_approval  # noqa: PLC0415

        correction: str | None = None
        for revision in range(_PLAN_REVISION_LIMIT + 1):
            try:
                candidate_plan = await self.generate_plan(
                    message=message,
                    session_id=session_id,
                    correction=correction,
                )
            except Exception as exc:  # noqa: BLE001 — surface as ERROR
                yield BusEvent.error_event(
                    self._coordinator.config.agent_id,
                    error=f"plan generation failed: {exc}",
                )
                yield PlanDecision(approved_plan=None, rejected=True)
                return

            # Yield FIRST so the renderer prints the plan, then block for
            # approval so the user is responding to a plan they've seen.
            yield BusEvent.plan_proposed(
                self._coordinator.config.agent_id,
                plan=candidate_plan,
                revision=revision,
            )

            response = await request_plan_approval(
                summary=candidate_plan.get("summary", ""),
                step_count=len(candidate_plan.get("steps", [])),
                dynamic_step_count=_dynamic_step_count(candidate_plan),
                agent_id=self._coordinator.config.agent_id,
                guard=guard,
            )
            if response.approved:
                yield PlanDecision(approved_plan=candidate_plan, rejected=False)
                return
            if response.correction:
                correction = response.correction
                continue
            yield BusEvent.error_event(
                self._coordinator.config.agent_id,
                error="plan rejected by user",
            )
            yield PlanDecision(approved_plan=None, rejected=True)
            return

        yield BusEvent.error_event(
            self._coordinator.config.agent_id,
            error=(
                f"plan revision limit ({_PLAN_REVISION_LIMIT}) reached; "
                "send 'y' to approve, 'n' to reject, or shorten the feedback"
            ),
        )
        yield PlanDecision(approved_plan=None, rejected=True)

    async def generate_plan(
        self,
        *,
        message: str,
        session_id: str,  # noqa: ARG002 — reserved for future per-session context
        correction: str | None,
    ) -> dict[str, Any]:
        """Call the coordinator's LLM with the planner system prompt."""
        tools = self._tool_names()
        system = _PLAN_SYSTEM_PROMPT.replace("{tools}", ", ".join(tools) or "(none)")
        if correction:
            system = system + _PLAN_CORRECTION_HINT.replace("{correction}", correction)

        raw = await self._llm().complete(
            system=system,
            messages=[{"role": "user", "content": message}],
            response_format={"type": "json_object"},
            source="planner",
        )
        plan = _coerce_plan(raw)
        if plan is None:
            raise ValueError(
                "planner returned a response that could not be parsed as "
                "a {summary, steps} JSON object"
            )
        return plan


def _coerce_plan(raw: Any) -> dict[str, Any] | None:
    """Best-effort plan extraction from whatever the LLM returned.

    Why this isn't a simple ``json.loads``: ``response_format={"type":
    "json_object"}`` is passed on every planner call, but only adapters
    that support structured output honour it. Other adapters may wrap JSON
    in markdown or prose.

    Recovery rides the existing extractor: ``_parse_action_json`` is the
    same helper the ReAct loop uses to pull JSON out of model responses.
    """
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        raw = raw["text"]

    if isinstance(raw, str):
        raw = _parse_action_json(raw)
        if raw is None:
            return None

    if not isinstance(raw, dict):
        return None
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return None
    summary = raw.get("summary")
    return {
        "summary": str(summary) if summary is not None else "",
        "steps": [s for s in steps if isinstance(s, dict)],
    }


def _step_args_deferred(step: dict[str, Any]) -> bool:
    """Whether a step's args are deferred to runtime."""
    if "args" not in step:
        return True
    return step["args"] is None


def _dynamic_step_count(plan: dict[str, Any]) -> int:
    """Number of plan steps whose args will be resolved at runtime."""
    steps = plan.get("steps") or []
    return sum(
        1
        for step in steps
        if isinstance(step, dict) and step.get("tool") and _step_args_deferred(step)
    )


def _render_plan_for_banner(plan: dict[str, Any]) -> str:
    """Multi-line text used as the HITL banner ``command`` field."""
    lines: list[str] = []
    summary = plan.get("summary") or "(no summary)"
    lines.append(f"Plan: {summary}")
    steps = plan.get("steps") or []
    for step in steps:
        idx = step.get("step")
        intent = step.get("intent") or "(no intent)"
        tool = step.get("tool")
        why = step.get("why")
        prefix = f"  {idx}." if idx is not None else "  -"
        lines.append(f"{prefix} {intent}")
        if tool:
            if _step_args_deferred(step):
                lines.append(f"      tool: {tool}  args: (resolved at runtime)")
            else:
                args = step.get("args") or {}
                args_repr = json.dumps(args, ensure_ascii=False) if args else "{}"
                lines.append(f"      tool: {tool}  args: {args_repr}")
        if why:
            lines.append(f"      why:  {why}")
    return "\n".join(lines)


def _render_plan_for_priors(plan: dict[str, Any]) -> str:
    """Prior-message body shown to the coordinator after approval."""
    body = _render_plan_for_banner(plan)
    return (
        "[Approved plan]\n"
        f"{body}\n\n"
        "The plan above is the agreed INTENT for this turn. Args shown "
        "are concrete only where the user supplied them or where they're "
        "knowable without running any tool first; everywhere else the "
        "value '(resolved at runtime)' means: derive the argument from "
        "your observations of the prior step, NOT from any placeholder "
        "in this prior. You may skip or merge steps when an observation "
        "makes them unnecessary; report any such deviation in your next "
        "thought."
    )
