"""
harness/hitl.py — Human-in-the-Loop approval gate.

Opt-in per agent via AgentConfig.hitl_tools.  Zero overhead when unused.

Same-session flow:
  1. Agent hits a gated tool call.
  2. A checkpoint is written to the CheckpointStore (step + WorkingMemory +
     pending tool).  BudgetGuard clock suspends.
  3. Approval banner is printed to the terminal.
  4. Human types  y / n / a / <correction>  in the terminal.
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
  Any text that isn't y/yes/a/allow/n/no is treated as a correction.
  The gated tool is skipped and the text is injected into WorkingMemory
  as a user message, so the LLM sees it on the next think step.

Session allow:
  Typing  a  or  allow  approves the current call AND adds a (tool, prefix)
  key to a process-scoped allow-list.  For shell-like tools the prefix is the
  first word of the command arg (e.g. 'git'), so allowing 'git' doesn't also
  allow 'rm'.  Subsequent calls matching the key skip checkpoint + banner.
  Use is_session_allowed(tool, args) to query the list from outside.
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

    Usage in any script:

        result = await maybe_resume(runtime) or await runtime.run_agent(...)

    The approval banner prints the exact command to paste, e.g.:
        python my_script.py --resume 3f7a1b2c-...:researcher
    """
    args = sys.argv[1:]
    if "--resume" not in args:
        return None
    idx = args.index("--resume")
    if idx + 1 >= len(args):
        print("Usage: --resume <ckp_id>", file=sys.stderr)
        sys.exit(1)
    ckp_id = args[idx + 1]
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


# ── CLI gate ──────────────────────────────────────────────────────────────────


def _print_banner(req: ApprovalRequest) -> None:
    script = sys.argv[0] if sys.argv else "your_script.py"
    label = _session_label(req.tool, req.args)
    print(f"\n{_SEP}")
    print("  HITL Approval Required")
    print(_SEP)
    print(f"  Tool:  {req.tool}")
    print(f"  Args:  {json.dumps(req.args, default=str)}")
    print(f"  Agent: {req.agent_id}  step={req.step}")
    print(f"  Run:   {req.run_id}")
    print(f"  ID:    {req.approval_id}")
    print(_SEP)
    print(
        f"  y = approve once  |  a = allow '{label}' for session  |  n = reject  |  <text> = steer"
    )
    print(f"  Ctrl-C to pause. Resume: python {script} --resume {req.run_id}")
    print(_SEP)


def _parse_stdin(approval_id: str, raw: str) -> ApprovalResponse:
    lo = raw.strip().lower()
    if lo in ("y", "yes"):
        return ApprovalResponse(approval_id=approval_id, approved=True)
    if lo in ("a", "allow"):
        return ApprovalResponse(approval_id=approval_id, approved=True, session_allow=True)
    if lo in ("n", "no"):
        return ApprovalResponse(approval_id=approval_id, approved=False)
    return ApprovalResponse(approval_id=approval_id, approved=True, correction=raw.strip() or None)


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
      <any text>  → correction injected into WorkingMemory; tool skipped

    Holds stdout_lock for the duration so concurrent agent events don't
    interleave with the banner or the input prompt.
    """
    async with stdout_lock:
        _print_banner(req)

        guard.suspend()
        try:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, input, "  Approve? [y/n/a/correction]: ")
        finally:
            guard.resume()

        print()
        resp = _parse_stdin(req.approval_id, raw)
        if resp.session_allow:
            _session_allowed.add(_session_key(req.tool, req.args))
            print(f"  ✓ '{_session_label(req.tool, req.args)}' allowed for this session\n")
        return resp
