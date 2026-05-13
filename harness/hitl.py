"""
harness/hitl.py — Human-in-the-Loop approval gate with Redis checkpointing.

Opt-in per agent via AgentConfig.hitl_tools.  Zero overhead when unused.

Same-session flow:
  1. Agent hits a gated tool call.
  2. Run checkpoint is written to Redis (step + WorkingMemory + pending tool).
  3. Approval request is printed to the terminal; BudgetGuard clock suspends.
  4. Human types  y / n / <correction>  in the terminal.
  5. Guard resumes; agent continues (or injects correction and skips the tool).

Crash / Ctrl-C / kill flow:
  1-3 as above — checkpoint is already durable in Redis.
  4. Process dies.
  5. Human calls  await runtime.resume_agent(run_id).
  6. Checkpoint is restored; the same approval prompt is shown again.
  7. Human responds; run continues from the saved step.

The UUID printed at the prompt is an audit reference only.

Correction steering:
  Any text that isn't y/yes/n/no is treated as a correction.
  The gated tool is skipped and the text is injected into WorkingMemory
  as a user message, so the LLM sees it on the next think step.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SEP = "─" * 60


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


# ── Redis store ───────────────────────────────────────────────────────────────


class RedisApprovalStore:
    """
    Durable store for approval requests and run checkpoints.

    Keys are prefixed and expire after ttl_seconds (default 24 h) so stale
    state doesn't accumulate.

    Usage:
        import redis.asyncio as redis
        client = redis.Redis(host="localhost", decode_responses=True)
        store = RedisApprovalStore(client)
    """

    _REQ = "hitl:req:{}"
    _CKP = "hitl:ckp:{}"

    def __init__(self, client: Any, ttl_seconds: int = 86_400) -> None:
        self._r = client
        self._ttl = ttl_seconds

    async def write_request(self, req: ApprovalRequest) -> None:
        await self._r.set(self._REQ.format(req.approval_id), json.dumps(asdict(req)), ex=self._ttl)

    async def get_request(self, approval_id: str) -> ApprovalRequest | None:
        raw = await self._r.get(self._REQ.format(approval_id))
        return ApprovalRequest(**json.loads(raw)) if raw else None

    async def write_checkpoint(self, run_id: str, checkpoint: dict) -> None:
        await self._r.set(
            self._CKP.format(run_id),
            json.dumps(checkpoint, default=str),
            ex=self._ttl,
        )

    async def get_checkpoint(self, run_id: str) -> dict | None:
        raw = await self._r.get(self._CKP.format(run_id))
        return json.loads(raw) if raw else None

    async def clear_checkpoint(self, run_id: str) -> None:
        await self._r.delete(self._CKP.format(run_id))


# ── CLI gate ──────────────────────────────────────────────────────────────────


def _print_banner(req: ApprovalRequest) -> None:
    print(f"\n{_SEP}")
    print("  HITL Approval Required")
    print(_SEP)
    print(f"  Tool:  {req.tool}")
    print(f"  Args:  {json.dumps(req.args, default=str)}")
    print(f"  Agent: {req.agent_id}  step={req.step}")
    print(f"  Run:   {req.run_id}")
    print(f"  ID:    {req.approval_id}")
    print(_SEP)


def _parse_stdin(approval_id: str, raw: str) -> ApprovalResponse:
    lo = raw.strip().lower()
    if lo in ("y", "yes"):
        return ApprovalResponse(approval_id=approval_id, approved=True)
    if lo in ("n", "no"):
        return ApprovalResponse(approval_id=approval_id, approved=False)
    return ApprovalResponse(approval_id=approval_id, approved=True, correction=raw.strip() or None)


async def request_approval(
    req: ApprovalRequest,
    store: RedisApprovalStore,
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
      <any text>  → correction injected into WorkingMemory; tool skipped
    """
    await store.write_request(req)
    _print_banner(req)

    guard.suspend()
    try:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, input, "  Approve? [y/n/correction]: ")
    finally:
        guard.resume()

    print()
    return _parse_stdin(req.approval_id, raw)
