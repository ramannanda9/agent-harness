"""
harness/hitl.py — Human-in-the-Loop approval gate.

Opt-in per agent via AgentConfig.hitl_tools.  Zero overhead when unused.

Same-session flow:
  1. Agent hits a gated tool call.
  2. A checkpoint is written to the CheckpointStore (step + WorkingMemory +
     pending tool).  BudgetGuard clock suspends.
  3. Approval banner is printed to the terminal.
  4. Human types  y / n / a / A / <correction>  in the terminal.
  5. Guard resumes; agent continues (or injects correction and skips the tool).

Crash / Ctrl-C / kill flow:
  1-3 as above — checkpoint is already durable before stdin blocks.
  4. Process dies.
  5. Banner printed "Resume: python your_script.py --resume <run_id>".
  6. Human re-runs the same script with --resume <run_id>.
  7. maybe_resume(runtime) detects the flag, restores checkpoint, re-prompts.
  8. Human responds; run continues from the saved step.

The UUID printed at the prompt is an audit reference only.

Correction steering:
  Any text that isn't y/yes/a/allow/A/always/n/no is treated as a correction.
  The gated tool is skipped and the text is injected into WorkingMemory
  as a user message, so the LLM sees it on the next think step.

Session allow:
  Typing  a  or  allow  approves the current call AND adds a (tool, prefix)
  key to a process-scoped allow-list.  For shell-like tools the prefix is the
  first word of the command arg (e.g. 'git'), so allowing 'git' doesn't also
  allow 'rm'.  Subsequent calls matching the key skip checkpoint + banner.
  Use is_session_allowed(tool, args) to query the list from outside.

Persistent allow:
  Typing  A  or  always  approves the current call and writes a user-scoped
  allow rule to ~/.agent-harness/policies/tool_policy.json. Rules are narrow:
  shell-like tools are scoped by first command word, other tools by tool name.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SEP = "─" * 60

# Held by request_approval for the duration of the banner+stdin interaction.
# Concurrent agents acquire-and-release this before printing events so their
# output doesn't interleave with the approval prompt.
# asyncio.Lock() is safe to create at import time in Python ≥ 3.10.
stdout_lock: asyncio.Lock = asyncio.Lock()

# Process-scoped session allow-list.  Entries are (tool_name, prefix) tuples
# where prefix is the first word of the primary command arg for shell-like
# tools, or None for tools with no meaningful sub-command (allow any args).
_session_allowed: set[tuple[str, str | None]] = set()


def _session_key(tool: str, args: dict) -> tuple[str, str | None]:
    """Return the (tool, prefix) key used for session-allow scoping."""
    if tool in ("shell", "bash", "run", "exec"):
        cmd = (args.get("cmd") or args.get("command") or "").strip()
        prefix = cmd.split()[0] if cmd else None
        return (tool, prefix)
    return (tool, None)


def is_session_allowed(tool: str, args: dict) -> bool:
    """True if this tool+args combination has been session-allowed by the human."""
    return _session_key(tool, args) in _session_allowed


def is_persistently_allowed(tool: str, args: dict) -> bool:
    """True if this tool+args combination is allowed by the user policy file."""
    from harness.tool_policy import ToolPolicyStore

    return ToolPolicyStore().is_allowed(tool, args)


def is_allowed(tool: str, args: dict) -> bool:
    """True if this tool+args combination is session- or user-policy allowed."""
    return is_session_allowed(tool, args) or is_persistently_allowed(tool, args)


def _session_label(tool: str, args: dict) -> str:
    """Human-readable description of what 'a' would allow."""
    _, prefix = _session_key(tool, args)
    return f"{tool} {prefix}" if prefix else tool


# ── Resume helper ─────────────────────────────────────────────────────────────


async def maybe_resume(runtime: Any) -> dict | None:
    """
    Check sys.argv for --resume <ckp_id>. If present, restore the checkpoint
    and continue the run; return its result dict.
    Return None if --resume is not in argv so the caller's normal path runs.

    For most scripts, prefer the automatic resume built into dispatch_stream /
    run_stream — no explicit call needed.

    The approval banner prints the exact command to paste, e.g.:
        python my_script.py --resume 3f7a1b2c-...:researcher
    """
    from harness.checkpoint import maybe_resume_key

    ckp_id = maybe_resume_key()
    if ckp_id is None:
        return None
    return await runtime.resume(ckp_id)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ApprovalRequest:
    approval_id: str
    run_id: str
    agent_id: str
    tool: str
    args: dict
    step: int
    timestamp: str


@dataclass
class ApprovalResponse:
    approval_id: str
    approved: bool
    correction: str | None = None  # non-None → steering; tool is skipped
    session_allow: bool = False  # True → add (tool, prefix) to _session_allowed
    persistent_allow: bool = False  # True → write a user-scoped allow rule


# ── CLI gate ──────────────────────────────────────────────────────────────────


def _print_banner(req: ApprovalRequest) -> None:
    script = sys.argv[0] if sys.argv else "your_script.py"
    label = _session_label(req.tool, req.args)
    print(f"\n{_SEP}")
    print("  HITL Approval Required")
    print(_SEP)
    print(f"  Tool:  {req.tool}")
    print(f"  Args:  {json.dumps(req.args, default=str)}")
    # ``Agent: X  step=N`` is the only context line — Run/ID UUIDs used
    # to print here but the run_id is already echoed in the resume hint
    # below (so the top line was duplicate) and the approval_id is a
    # purely internal correlation token that's never user-actionable in
    # the synchronous CLI flow. Both fields remain on ``ApprovalRequest``
    # for callers that need them (tests, future async-approval systems).
    print(f"  Agent: {req.agent_id}  step={req.step}")
    print(_SEP)
    print(
        "  y = approve once  |  "
        f"a = allow '{label}' for session  |  "
        f"A = always allow '{label}'  |  "
        "n = reject  |  <text> = steer"
    )
    print(f"  Ctrl-C to pause. Resume: python {script} --resume {req.run_id}")
    print(_SEP)


def _parse_stdin(approval_id: str, raw: str) -> ApprovalResponse:
    stripped = raw.strip()
    if stripped == "A":
        return ApprovalResponse(approval_id=approval_id, approved=True, persistent_allow=True)
    lo = stripped.lower()
    if lo in ("y", "yes"):
        return ApprovalResponse(approval_id=approval_id, approved=True)
    if lo in ("a", "allow"):
        return ApprovalResponse(approval_id=approval_id, approved=True, session_allow=True)
    if lo in ("always", "allow always"):
        return ApprovalResponse(approval_id=approval_id, approved=True, persistent_allow=True)
    if lo in ("n", "no"):
        return ApprovalResponse(approval_id=approval_id, approved=False)
    return ApprovalResponse(approval_id=approval_id, approved=True, correction=stripped or None)


@dataclass
class PlanApprovalResponse:
    """Sibling of ``ApprovalResponse`` for ``request_plan_approval``.

    Plan mode is per-turn, not per-tool — there's no a / A here because
    "allow plans for the session" silently turns plan mode off (defeats
    the whole feature), and "always allow plans" persistently always-
    allows the gate, which is the same thing forever.

    Three outcomes:
    - ``approved=True``: run the plan.
    - ``approved=False, correction=None``: user rejected — abort the turn.
    - ``approved=False, correction=<text>``: user wants a revision; the
      planner re-runs with the correction folded into its system prompt.
    """

    approved: bool
    correction: str | None = None


async def request_plan_approval(
    *,
    summary: str,
    step_count: int,
    dynamic_step_count: int,
    agent_id: str,
    guard: Any,
) -> PlanApprovalResponse:
    """Plan-mode approval prompt: y / n / revision text.

    Sibling of :func:`request_approval` that drops the session-allow and
    persistent-allow options. They make sense for per-tool gating
    (approve ``shell/grep`` for the session and stop nagging) but they're
    actively wrong for plan-mode approval: ``a`` turns plan mode off
    silently, ``A`` does it permanently. Both contradict the user's
    intent in turning plan mode on at all.

    Shares ``stdout_lock``, the steering router interop, and the
    BudgetGuard suspend/resume cycle with :func:`request_approval`, so
    the input plumbing stays one place.

    The full plan was already rendered by the consumer when the
    ``PLAN_PROPOSED`` event fired upstream of this call; the banner here
    is a short confirmation prompt with just enough context to re-anchor
    a reviewer who scrolled.
    """
    from harness.steering import get_active_router

    async with stdout_lock:
        router = get_active_router()
        # Capital ``Y`` signals that Enter alone approves. Plan mode is
        # opt-in and the full plan has already been rendered above this
        # banner, so the reviewer's "I read it, proceed" gesture should
        # be the cheapest possible — matches the apt / pip convention.
        approve_prompt = "  Approve plan? [Y/n/revision]: "
        hitl_future: Any = (
            router.claim_next_line(prompt=approve_prompt) if router is not None else None
        )

        _print_plan_banner(
            summary=summary,
            step_count=step_count,
            dynamic_step_count=dynamic_step_count,
            agent_id=agent_id,
        )

        guard.suspend()
        try:
            if hitl_future is not None:
                raw = await hitl_future
            else:
                from prompt_toolkit import PromptSession

                from harness.steering import StdinRouter

                session: PromptSession = PromptSession()
                raw = await session.prompt_async(
                    approve_prompt,
                    multiline=True,
                    key_bindings=StdinRouter._build_key_bindings(),
                )
        finally:
            guard.resume()

        print()
        return _parse_plan_stdin(raw)


def _print_plan_banner(
    *,
    summary: str,
    step_count: int,
    dynamic_step_count: int,
    agent_id: str,
) -> None:
    """Compact banner for plan approval. The full plan already streamed
    via PLAN_PROPOSED — this banner is just a confirmation prompt with
    summary / step counts as scroll anchors.

    No ``--resume <run_id>`` hint here because PersistentAgent's
    resumption model is session-id-based — the next ``chat()`` call on
    the same session_id picks up state automatically. The
    orchestrator-style ``--resume`` flag this banner used to print
    doesn't apply.
    """
    print(f"\n{_SEP}")
    print(f"  Plan ready for {agent_id}")
    print(_SEP)
    if summary:
        print(f"  Summary: {summary}")
    detail = f"{step_count} step{'' if step_count == 1 else 's'}"
    if dynamic_step_count:
        detail += f"  ({dynamic_step_count} with args resolved at runtime)"
    print(f"  Steps:   {detail}")
    print(_SEP)
    print("  Enter or y = approve and run  |  n = reject  |  any other text = revise the plan")
    print(_SEP)


def _parse_plan_stdin(raw: str) -> PlanApprovalResponse:
    """Parse a plan-approval response.

    No a / A handling — plan mode gates per turn, not per tool. Any
    text that isn't ``y/yes/n/no`` is treated as a revision request.

    Empty input (just Enter) approves — the plan has already been
    rendered above the prompt and the capital ``Y`` in the prompt
    text signals the default. Mirrors the apt / pip ``[Y/n]``
    convention.
    """
    stripped = (raw or "").strip()
    lo = stripped.lower()
    if lo in ("", "y", "yes"):
        return PlanApprovalResponse(approved=True)
    if lo in ("n", "no"):
        return PlanApprovalResponse(approved=False)
    return PlanApprovalResponse(approved=False, correction=stripped or None)


async def request_approval(
    req: ApprovalRequest,
    guard: Any,  # BudgetGuard — suspend/resume during wait
) -> ApprovalResponse:
    """
    Show the approval prompt and read a response from stdin.

    Called both on the first encounter and again after crash/resume —
    the prompt is identical either way, and the UUID printed lets the
    human correlate with earlier output if they step away and come back.

    Prompt semantics:
      y / yes     → approved, tool runs
      n / no      → rejected, tool skipped (error observation returned)
      a / allow   → approved + session-allow registered; tool runs
      A / always  → approved + user policy allow registered; tool runs
      <any text>  → correction injected into WorkingMemory; tool skipped

    Holds stdout_lock for the duration so concurrent agent events don't
    interleave with the banner or the input prompt.

    Input always goes through prompt_toolkit:
      - If a steering router is active, HITL claims the next stdin read
        via the router. Text submitted at the active steering prompt is
        routed to HITL instead of subscribers; if the router reaches a
        pending claim between steering prompt cycles, it shows HITL's
        approval prompt directly.
      - If no router is active, HITL spins up a one-shot PromptSession
        for the approval prompt. Same UX either way.
    """
    from harness.steering import get_active_router

    async with stdout_lock:
        router = get_active_router()
        approve_prompt = "  Approve? [y/n/a/A/correction]: "
        # If a router is active, reserve the next stdin read BEFORE printing
        # the banner so the user's typed answer routes to HITL (not steering).
        hitl_future: Any = (
            router.claim_next_line(prompt=approve_prompt) if router is not None else None
        )

        _print_banner(req)

        guard.suspend()
        try:
            if hitl_future is not None:
                raw = await hitl_future
            else:
                # Standalone: one-shot prompt_toolkit session with the same
                # Enter-submits / Ctrl+J-newline bindings as steering so
                # single-token answers (y/n/a) and multi-line corrections
                # both compose naturally.
                from prompt_toolkit import PromptSession

                from harness.steering import StdinRouter

                session: PromptSession = PromptSession()
                raw = await session.prompt_async(
                    approve_prompt,
                    multiline=True,
                    key_bindings=StdinRouter._build_key_bindings(),
                )
        finally:
            guard.resume()

        print()
        resp = _parse_stdin(req.approval_id, raw)
        if resp.session_allow:
            _session_allowed.add(_session_key(req.tool, req.args))
            print(f"  ✓ '{_session_label(req.tool, req.args)}' allowed for this session\n")
        if resp.persistent_allow:
            from harness.tool_policy import ToolPolicyStore

            rule = ToolPolicyStore().add_allow_rule(tool=req.tool, args=req.args)
            print(f"  ✓ '{_session_label(req.tool, req.args)}' always allowed ({rule.id})\n")
        return resp
