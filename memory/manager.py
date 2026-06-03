"""
MemoryManager — unified interface over three memory tiers.

Write timing (per design decision):
  1. Agent-level (during run):
       agent.write_working_facts(key, value)
       → lightweight, no LLM, namespaced to run_id:agent_id
       → stored in semantic store with short TTL

  2. Run-end (after full orchestration):
       manager.write_run_end(goal, results, trace)
       → LLM extracts structured KV facts → global semantic store
       → LLM produces natural language summary → episodic vector store

Read path:
  manager.build_context(goal)
  → semantic: key lookups by prefix
  → episodic: vector similarity search
  → combined into memory context string injected into agent system prompt

Memory conflict resolution (two agents write same key):
  → last-write-wins by default
  → conflict logged in trace for post-hoc analysis
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from harness.utils import parse_llm_json

logger = logging.getLogger(__name__)


# ── Store Protocols ───────────────────────────────────────────────────────────


@runtime_checkable
class SemanticStore(Protocol):
    async def write(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...
    async def read(self, key: str) -> Any | None: ...
    async def delete(self, key: str) -> None: ...
    async def search_prefix(self, prefix: str) -> dict[str, Any]: ...


@runtime_checkable
class EpisodicStore(Protocol):
    async def write(self, text: str, metadata: dict, agent_id: str = "") -> str: ...
    async def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        memory_scope: str | None = None,
        agent_id: str | None = None,
        include_shared: bool = True,
        include_legacy: bool = True,
    ) -> list[dict]: ...
    async def get(self, episode_id: str) -> dict | None: ...
    async def invalidate(self, memory_key: str) -> int:
        """Soft-delete all episodes matching ``memory_key``. Returns count.

        Distinct from writing a ``latest``-policy replacement: ``invalidate``
        removes without adding. Used by the reconciler's DELETE action so a
        contradicted episode can be removed without inventing a placeholder.
        """
        ...


# ── Data contracts ────────────────────────────────────────────────────────────


@dataclass
class MemoryWriteRequest:
    """
    Structured output from LLM extraction at run end.
    All fields must be concrete — no vague observations.
    """

    semantic_facts: dict[str, Any]  # deterministic KV for global semantic store
    episodic_summary: str  # natural language paragraph for vector store
    metadata: dict = field(default_factory=dict)
    ttl_seconds: int | None = None  # None = no expiry


@dataclass
class MemoryContext:
    """Injected into agent system prompt at run start."""

    semantic_facts: dict[str, Any]
    episodes: list[dict]

    def render(self) -> str:
        """Render as a compact string for prompt injection."""
        lines: list[str] = []

        if self.semantic_facts:
            lines.append("## Known facts (from memory)")
            for k, v in self.semantic_facts.items():
                lines.append(f"  {k}: {v}")

        if self.episodes:
            lines.append("\n## Relevant past experience")
            for ep in self.episodes:
                ts = ep.get("metadata", {}).get("timestamp", "unknown")
                summary = ep.get("text", "")
                lines.append(f"  [{ts}] {summary}")

        return "\n".join(lines) if lines else ""

    def is_empty(self) -> bool:
        return not self.semantic_facts and not self.episodes


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
An agent run just completed. Extract structured memory for future runs.

Goal: {goal}

Agent results:
{results}

Recent trace (last 10 events):
{trace}

Return a JSON object with exactly these fields:
{{
  "semantic_facts": {{
    "<descriptive_key>": "<concrete_value>",
    ...
  }},
  "episodic_summary": "<one dense paragraph: what happened, what was found, what was resolved, what failed>",
  "metadata": {{}},
  "ttl_seconds": null
}}

Rules for semantic_facts keys:
- Must be reusable across future runs (e.g. "gpu_worker_07:failure_cause")
- Use colon-separated namespacing: "<entity>:<attribute>"
- Values must be concrete facts, not process descriptions
- Omit facts that are run-specific and won't generalize

Rules for episodic_summary:
- Include: goal, key findings, resolution, failure modes encountered
- Exclude: tool call details, intermediate steps
- Max 3 sentences

Return JSON only. No preamble, no markdown fences.
"""


RECONCILE_PROMPT = """
You are a memory reconciler. You see relevant existing memory plus optional
new evidence from a just-completed run. Decide what to ADD, UPDATE, MERGE,
DELETE, or NOOP so memory reflects current knowledge without duplicates or
contradictions.

Goal: {goal}

Existing semantic facts (key → value):
{existing_facts}

Existing episodic memories (memory_key → summary):
{existing_episodes}

New evidence (agent results + recent trace) — may be empty for compaction-only runs:
{evidence}

Return JSON with exactly this shape:
{{
  "semantic_actions": [
    {{"action": "add",    "key": "<key>", "value": "<value>"}},
    {{"action": "update", "key": "<key>", "value": "<new value>", "rationale": "<why>"}},
    {{"action": "merge",  "key": "<key>", "value": "<combined value>", "rationale": "<why>"}},
    {{"action": "delete", "key": "<key>", "rationale": "<contradicted by>"}},
    {{"action": "noop",   "key": "<key>", "rationale": "<duplicate or redundant>"}}
  ],
  "episodic_action": {{
    "action": "add" | "update" | "merge" | "delete" | "noop",
    "memory_key": "<key>",
    "text": "<new or merged summary>",   // omit for delete/noop
    "rationale": "<why this action>"
  }}
}}

Action semantics:
- ADD   — no existing entry covers this; write fresh.
- UPDATE — existing entry superseded by new evidence; replace value/text.
- MERGE — combine existing + new into one richer entry; emit the merged value.
- DELETE — existing entry is contradicted by new evidence; remove it.
- NOOP  — duplicate or redundant; nothing to do.

Rules:
- Prefer NOOP over redundant UPDATE — don't write something already true.
- Use DELETE only when new evidence directly contradicts; otherwise prefer UPDATE/MERGE.
- For semantic keys, use colon-separated namespacing (e.g. "redis:port").
- Episodic memory_key should be stable per subject (e.g. "run_summary:<topic>").
- Return JSON only. No preamble, no markdown fences.
"""


# ── Reconcile plan ────────────────────────────────────────────────────────────


class ReconcileAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    NOOP = "noop"


@dataclass
class SemanticAction:
    action: ReconcileAction
    key: str
    value: Any = None
    rationale: str = ""


@dataclass
class EpisodicAction:
    action: ReconcileAction
    memory_key: str
    text: str = ""
    rationale: str = ""


@dataclass
class ReconcilePlan:
    """LLM-emitted plan applied by ``MemoryManager._apply_reconcile_plan``.

    Empty plan = no writes. Parse failures upstream materialise as an empty
    plan plus a flag so callers can fall back to the legacy extract path.
    """

    semantic_actions: list[SemanticAction] = field(default_factory=list)
    episodic_action: EpisodicAction | None = None
    parse_failed: bool = False


# ── Reconciler ────────────────────────────────────────────────────────────────


class MemoryReconciler:
    """LLM-arbitrated memory reconciliation.

    One call decides ADD/UPDATE/MERGE/DELETE/NOOP per existing+new entry.
    Used by ``MemoryManager`` at ``write_run_end`` (with evidence) and at
    ``compact`` (without evidence — pure cleanup pass).

    On parse failure the reconciler returns ``ReconcilePlan(parse_failed=True)``
    so callers can fall back to the legacy flat-extract path without crashing
    a run.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def reconcile(
        self,
        *,
        goal: str,
        existing_facts: dict[str, Any],
        existing_episodes: list[dict],
        evidence: str | None,
    ) -> ReconcilePlan:
        prompt = RECONCILE_PROMPT.format(
            goal=goal or "(compaction — no new goal)",
            existing_facts=json.dumps(existing_facts, default=str, indent=2),
            existing_episodes=json.dumps(
                [
                    {
                        "memory_key": (ep.get("metadata") or {}).get("memory_key", ""),
                        "text": ep.get("text", ""),
                    }
                    for ep in existing_episodes
                ],
                default=str,
                indent=2,
            ),
            evidence=evidence or "(none — cleanup pass)",
        )
        try:
            response = await self._llm.complete(
                system="You are a memory reconciliation agent. Return JSON only.",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                source="reconciler",
            )
            data = parse_llm_json(response)
            return _parse_reconcile_plan(data)
        except Exception as e:
            logger.warning("Reconciler failed (%s) — falling back to legacy extract", e)
            return ReconcilePlan(parse_failed=True)


def _parse_reconcile_plan(data: Any) -> ReconcilePlan:
    if not isinstance(data, dict):
        return ReconcilePlan(parse_failed=True)
    semantic_actions: list[SemanticAction] = []
    for entry in data.get("semantic_actions") or []:
        if not isinstance(entry, dict):
            continue
        # Require an explicit action — silently defaulting to NOOP would
        # mask real LLM-output bugs.
        raw_action = entry.get("action")
        if not isinstance(raw_action, str):
            continue
        try:
            action = ReconcileAction(raw_action)
        except ValueError:
            continue
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            continue
        semantic_actions.append(
            SemanticAction(
                action=action,
                key=key,
                value=entry.get("value"),
                rationale=str(entry.get("rationale") or ""),
            )
        )
    episodic_action: EpisodicAction | None = None
    raw_ep = data.get("episodic_action")
    if isinstance(raw_ep, dict):
        try:
            ep_action = ReconcileAction(raw_ep.get("action", "noop"))
            ep_key = raw_ep.get("memory_key")
            if isinstance(ep_key, str) and ep_key:
                episodic_action = EpisodicAction(
                    action=ep_action,
                    memory_key=ep_key,
                    text=str(raw_ep.get("text") or ""),
                    rationale=str(raw_ep.get("rationale") or ""),
                )
        except ValueError:
            pass
    # If the response didn't carry either an actions list or an episodic
    # action, treat it as a parse failure so callers can fall back to the
    # legacy flat-extract path. This is the safety net for callers wiring an
    # LLM that doesn't follow the reconcile schema (older models, plain
    # extractors, or any adapter that returns ``{semantic_facts, summary}``
    # instead of ``{semantic_actions, episodic_action}``).
    if not semantic_actions and episodic_action is None:
        return ReconcilePlan(parse_failed=True)
    return ReconcilePlan(
        semantic_actions=semantic_actions,
        episodic_action=episodic_action,
    )


# ── Memory Manager ────────────────────────────────────────────────────────────


class MemoryManager:
    """
    Unified interface over semantic (Redis KV) and episodic (vector) stores.

    Two write paths:
      write_working_facts()  — called by agent during run, no LLM
      write_run_end()        — called by orchestrator after run, uses LLM

    One read path:
      build_context()        — called by agent before run starts
    """

    def __init__(
        self,
        semantic_store: SemanticStore,
        episodic_store: EpisodicStore,
        llm,
        working_facts_ttl: int = 3600,  # agent working facts expire in 1 hour
        context_max_episodes: int = 3,
        context_max_semantic_keys: int = 20,
        memory_scope: str | None = None,
        memory_subject: str | None = None,
        # ── Reconciler ────────────────────────────────────────────────────────
        # Default-on: ``write_run_end`` uses the LLM-arbitrated reconcile path
        # instead of naive extract-and-overwrite. The reconcile prompt
        # subsumes today's extraction (one LLM call, not two) and degenerates
        # to pure ADD when existing context is empty, so cost is bounded.
        reconcile_on_write: bool = True,
        # DELETE is the only destructive action; demoted to NOOP unless the
        # caller explicitly opts in. UPDATE/MERGE/NOOP fire by default.
        allow_destructive_reconcile: bool = False,
        # Per-``memory_key`` count thresholds that auto-trigger compaction in
        # ``write_agent_task_end``. E.g. ``{"agent_task": 20}`` runs a
        # compaction reconcile when an agent accumulates 20 task episodes.
        auto_compact_threshold: dict[str, int] | None = None,
        reconciler: MemoryReconciler | None = None,
    ) -> None:
        self._semantic = semantic_store
        self._episodic = episodic_store
        self._llm = llm
        self._working_facts_ttl = working_facts_ttl
        self._context_max_episodes = context_max_episodes
        self._context_max_semantic_keys = context_max_semantic_keys
        self._memory_scope = memory_scope
        self._memory_subject = memory_subject
        self._reconcile_on_write = reconcile_on_write
        self._allow_destructive_reconcile = allow_destructive_reconcile
        self._auto_compact_threshold = dict(auto_compact_threshold or {})
        self._reconciler = reconciler or MemoryReconciler(llm)
        self._conflict_log: list[dict] = []

    # ── Agent-level write (during run, no LLM) ────────────────────────────────

    async def write_working_fact(
        self,
        run_id: str,
        agent_id: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Lightweight fact write during agent execution.
        Namespaced to run_id:agent_id to avoid polluting global store.
        Short TTL — these are transient working facts, not durable knowledge.
        """
        namespaced_key = f"run:{run_id}:agent:{agent_id}:{key}"
        existing = await self._semantic.read(namespaced_key)

        if existing is not None and existing != value:
            self._conflict_log.append(
                {
                    "key": namespaced_key,
                    "old": existing,
                    "new": value,
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.debug(
                "Memory conflict on key=%s agent=%s old=%r new=%r",
                namespaced_key,
                agent_id,
                existing,
                value,
            )

        await self._semantic.write(namespaced_key, value, ttl_seconds=self._working_facts_ttl)

    async def read_working_facts(self, run_id: str) -> dict[str, Any]:
        """Read all working facts written during this run (across all agents)."""
        return await self._semantic.search_prefix(f"run:{run_id}:")

    # ── Run-end write (after full orchestration, uses LLM) ───────────────────

    async def write_run_end(
        self,
        goal: str,
        agent_results: list[dict],
        trace: list[dict],
    ) -> MemoryWriteRequest:
        """
        Called by orchestrator after all agents complete.

        Two write paths:
          * Reconcile (``reconcile_on_write=True``, default) — LLM emits an
            ADD/UPDATE/MERGE/DELETE/NOOP plan informed by relevant existing
            memory. Falls back to the legacy path if the plan fails to parse.
          * Legacy extract (``reconcile_on_write=False``) — LLM extracts new
            facts/summary and overwrites the store.

        Returns a ``MemoryWriteRequest`` representing what was applied
        (extracted facts + summary), regardless of which path ran. Useful
        for logging/debugging.
        """
        if self._reconcile_on_write:
            applied = await self._reconcile_run_end(goal, agent_results, trace)
            if applied is not None:
                return applied
            # Reconcile parse failed — fall through to the legacy path so the
            # run still gets a durable write.

        extracted = await self._extract_memory(goal, agent_results, trace)
        # write semantic facts globally (no run_id namespace)
        await self._write_semantic_global(extracted)
        # write episodic summary to vector store
        episode_id = await self._write_episodic(goal, agent_results, extracted)
        logger.info(
            "Run-end memory write complete (legacy): %d semantic facts, episode_id=%s",
            len(extracted.semantic_facts),
            episode_id,
        )
        return extracted

    async def _reconcile_run_end(
        self,
        goal: str,
        agent_results: list[dict],
        trace: list[dict],
    ) -> MemoryWriteRequest | None:
        """Run a reconcile pass at run-end. Returns ``None`` if parse failed."""
        context = await self.build_context(goal)
        evidence = json.dumps(
            {
                "agent_results": agent_results,
                "recent_trace": trace[-10:],
            },
            default=str,
            indent=2,
        )
        plan = await self._reconciler.reconcile(
            goal=goal,
            existing_facts=context.semantic_facts,
            existing_episodes=context.episodes,
            evidence=evidence,
        )
        if plan.parse_failed:
            return None
        applied = await self._apply_reconcile_plan(
            plan,
            goal=goal,
            agent_results=agent_results,
        )
        logger.info(
            "Run-end reconcile applied: %d semantic actions, episodic=%s",
            len(plan.semantic_actions),
            plan.episodic_action.action.value if plan.episodic_action else "none",
        )
        return applied

    async def _apply_reconcile_plan(
        self,
        plan: ReconcilePlan,
        *,
        goal: str,
        agent_results: list[dict] | None = None,
    ) -> MemoryWriteRequest:
        """Dispatch the plan to the stores. DELETE demoted to NOOP when
        ``allow_destructive_reconcile=False``; the demoted decision is logged
        to ``conflict_log`` so users can audit and decide to flip the flag.
        """
        applied_facts: dict[str, Any] = {}
        for action in plan.semantic_actions:
            storage_key = self._semantic_storage_key(action.key)
            if action.action is ReconcileAction.DELETE:
                if not self._allow_destructive_reconcile:
                    self._conflict_log.append(
                        {
                            "key": storage_key,
                            "action": "delete",
                            "rationale": action.rationale,
                            "outcome": "demoted_to_noop",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    continue
                await self._semantic.delete(storage_key)
                self._conflict_log.append(
                    {
                        "key": storage_key,
                        "action": "delete",
                        "rationale": action.rationale,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue
            if action.action is ReconcileAction.NOOP:
                continue
            # ADD / UPDATE / MERGE all resolve to a write of action.value.
            existing = await self._semantic.read(storage_key)
            if existing is not None and existing != action.value:
                self._conflict_log.append(
                    {
                        "key": storage_key,
                        "old": existing,
                        "new": action.value,
                        "action": action.action.value,
                        "rationale": action.rationale,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            await self._semantic.write(storage_key, action.value)
            applied_facts[action.key] = action.value

        episodic_summary = ""
        ep = plan.episodic_action
        if ep is not None and ep.action is not ReconcileAction.NOOP:
            if ep.action is ReconcileAction.DELETE:
                if self._allow_destructive_reconcile:
                    await self._episodic.invalidate(ep.memory_key)
                # else: demoted; conflict_log entry covers it.
                self._conflict_log.append(
                    {
                        "memory_key": ep.memory_key,
                        "action": "delete",
                        "scope": "episodic",
                        "rationale": ep.rationale,
                        "outcome": "applied"
                        if self._allow_destructive_reconcile
                        else "demoted_to_noop",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            elif ep.text:
                # ADD / UPDATE / MERGE — write the new (or merged) text under
                # the same memory_key using the ``latest`` policy so existing
                # rows with that key are hard-deleted.
                episodic_summary = ep.text
                metadata = {
                    "goal": goal,
                    "memory_key": ep.memory_key,
                    "memory_policy": "latest"
                    if ep.action in (ReconcileAction.UPDATE, ReconcileAction.MERGE)
                    else "append",
                    "memory_kind": "run_summary",
                    "shared": True,
                    "agent_ids": [r.get("agent_id") for r in (agent_results or [])],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reconcile_action": ep.action.value,
                }
                if self._memory_scope is not None:
                    metadata["memory_scope"] = self._memory_scope
                await self._episodic.write(text=ep.text, metadata=metadata, agent_id="")

        return MemoryWriteRequest(
            semantic_facts=applied_facts,
            episodic_summary=episodic_summary,
        )

    async def compact(
        self,
        *,
        goal: str = "",
        agent_id: str | None = None,
    ) -> ReconcilePlan:
        """Run a reconcile pass with no new evidence — pure cleanup.

        Pulls relevant existing memory and asks the reconciler to consolidate,
        deduplicate, and prune. Honours ``allow_destructive_reconcile`` the
        same way ``write_run_end`` does: DELETE demoted to NOOP by default.
        """
        context = await self.build_context(goal or "", agent_id=agent_id)
        plan = await self._reconciler.reconcile(
            goal=goal,
            existing_facts=context.semantic_facts,
            existing_episodes=context.episodes,
            evidence=None,
        )
        if not plan.parse_failed:
            await self._apply_reconcile_plan(plan, goal=goal)
        return plan

    async def write_agent_task_end(
        self,
        *,
        goal: str,
        task_id: str,
        agent_id: str,
        instruction: str,
        result: dict,
    ) -> None:
        """Write a compact per-agent episode after one task finishes.

        This complements the run-level ``write_run_end`` summary. Agent
        episodes are retrieved back only for that agent (within the same
        memory scope), while the run-level episode is shared.
        """
        answer = str(result.get("answer") or "")
        error = result.get("error")
        confidence = result.get("confidence")
        status = "failed" if error else "completed"
        text = (
            f"Agent {agent_id} {status} task {task_id}. "
            f"Instruction: {instruction}. "
            f"Answer: {answer[:1200]}"
        )
        if error:
            text += f" Error: {error}"
        metadata = {
            "goal": goal,
            "task_id": task_id,
            "agent_id": agent_id,
            "agent_ids": [agent_id],
            "memory_kind": "agent_task",
            "memory_policy": "append",
            "memory_key": f"agent_task:{agent_id}",
            "shared": False,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._memory_scope is not None:
            metadata["memory_scope"] = self._memory_scope
        await self._episodic.write(text=text, metadata=metadata, agent_id=agent_id)
        await self._maybe_auto_compact(goal=goal, agent_id=agent_id)

    async def _maybe_auto_compact(self, *, goal: str, agent_id: str) -> None:
        """Trigger a background compaction when this agent's task-episode
        count crosses the configured threshold.

        Fire-and-forget — the agent never blocks on the compaction LLM call.
        """
        threshold = (
            self._auto_compact_threshold.get("agent_task") if self._auto_compact_threshold else None
        )
        if not threshold or threshold <= 0:
            return
        # Sample current count for this agent. Cheap enough — the same query
        # the threshold is configured against.
        recent = await self._episodic.search(
            "",
            top_k=threshold + 1,
            memory_scope=self._memory_scope,
            agent_id=agent_id,
            include_shared=False,
            include_legacy=False,
        )
        if (
            len(
                [
                    ep
                    for ep in recent
                    if (ep.get("metadata") or {}).get("memory_kind") == "agent_task"
                ]
            )
            >= threshold
        ):
            from harness.utils import fire

            fire(self.compact(goal=goal, agent_id=agent_id))

    async def _extract_memory(
        self,
        goal: str,
        agent_results: list[dict],
        trace: list[dict],
    ) -> MemoryWriteRequest:
        prompt = EXTRACTION_PROMPT.format(
            goal=goal,
            results=json.dumps(agent_results, default=str, indent=2),
            trace=json.dumps(trace[-10:], default=str, indent=2),
        )
        try:
            response = await self._llm.complete(
                system="You are a memory extraction agent. Return JSON only.",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = parse_llm_json(response)

            return MemoryWriteRequest(
                semantic_facts=data.get("semantic_facts", {}),
                episodic_summary=data.get("episodic_summary", ""),
                metadata=data.get("metadata", {}),
                ttl_seconds=data.get("ttl_seconds"),
            )
        except Exception as e:
            logger.error("Memory extraction failed: %s", e)
            # graceful degradation — write empty memory rather than crash
            return MemoryWriteRequest(
                semantic_facts={},
                episodic_summary=f"Run completed for goal: {goal}. Extraction failed: {e}",
            )

    async def write_semantic_fact(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Write a single global semantic fact — persists across runs, no run/agent prefix."""
        storage_key = self._semantic_storage_key(key)
        existing = await self._semantic.read(storage_key)
        if existing is not None and existing != value:
            self._conflict_log.append(
                {
                    "key": storage_key,
                    "old": existing,
                    "new": value,
                    "scope": "global",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        await self._semantic.write(storage_key, value, ttl_seconds=ttl_seconds)

    async def _write_semantic_global(self, req: MemoryWriteRequest) -> None:
        for key, value in req.semantic_facts.items():
            storage_key = self._semantic_storage_key(key)
            existing = await self._semantic.read(storage_key)
            if existing is not None and existing != value:
                self._conflict_log.append(
                    {
                        "key": storage_key,
                        "old": existing,
                        "new": value,
                        "scope": "global",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            await self._semantic.write(storage_key, value, ttl_seconds=req.ttl_seconds)

    async def _write_episodic(
        self, goal: str, agent_results: list[dict], req: MemoryWriteRequest
    ) -> str:
        metadata = {
            "goal": goal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_ids": [r.get("agent_id") for r in agent_results],
            "memory_kind": "run_summary",
            "memory_policy": "latest",
            "memory_key": f"run_summary:{self._memory_subject or _memory_key_fragment(goal)}",
            "shared": True,
            "outcome": "success"
            if any(r.get("confidence", 0) > 0.5 for r in agent_results)
            else "uncertain",
            **req.metadata,
        }
        if self._memory_scope is not None:
            metadata["memory_scope"] = self._memory_scope
        return await self._episodic.write(
            text=req.episodic_summary,
            metadata=metadata,
            agent_id="",
        )

    # ── Read path ─────────────────────────────────────────────────────────────

    async def build_context(self, goal: str, agent_id: str | None = None) -> MemoryContext:
        """
        Build memory context for injection into agent system prompt.

        Reads:
          - Episodic: top-k similar past episodes via vector search
          - Semantic: agent-specific keys first, then global facts extracted at
            run-end, up to context_max_semantic_keys total.
        """
        episodes = await self._search_scoped_episodes(goal, agent_id=agent_id)

        semantic_facts: dict[str, Any] = {}

        # agent-scoped facts (written as "agent:{id}:..." during previous runs)
        if agent_id:
            agent_facts = await self._semantic.search_prefix(f"agent:{agent_id}:")
            semantic_facts.update(
                dict(list(agent_facts.items())[: self._context_max_semantic_keys])
            )

        # global facts extracted at run-end (no run: or agent: prefix)
        remaining = self._context_max_semantic_keys - len(semantic_facts)
        if remaining > 0:
            if self._memory_scope is not None:
                prefix = self._semantic_scope_prefix()
                scoped_facts = await self._semantic.search_prefix(prefix)
                global_facts = {
                    k.removeprefix(prefix): v
                    for k, v in scoped_facts.items()
                    if not k.removeprefix(prefix).startswith("orchestrator:")
                }
            else:
                all_facts = await self._semantic.search_prefix("")
                global_facts = {
                    k: v
                    for k, v in all_facts.items()
                    if not k.startswith("run:")
                    and not k.startswith("agent:")
                    and not k.startswith("orchestrator:")
                    and not k.startswith("scope:")
                }
            semantic_facts.update(dict(list(global_facts.items())[:remaining]))

        return MemoryContext(
            semantic_facts=semantic_facts,
            episodes=episodes,
        )

    async def _search_scoped_episodes(self, goal: str, agent_id: str | None) -> list[dict]:
        """Search episodic memory, filtering to scope and agent when configured."""
        return await self._episodic.search(
            goal,
            top_k=self._context_max_episodes,
            memory_scope=self._memory_scope,
            agent_id=agent_id,
            include_shared=True,
            include_legacy=self._memory_scope is None,
        )

    def _semantic_scope_prefix(self) -> str:
        assert self._memory_scope is not None
        return f"scope:{self._memory_scope}:"

    def _semantic_storage_key(self, key: str) -> str:
        if self._memory_scope is None:
            return key
        if key.startswith(("run:", "agent:", "scope:")):
            return key
        return f"{self._semantic_scope_prefix()}{key}"

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_conflict_log(self) -> list[dict]:
        return list(self._conflict_log)

    async def lookup(self, key: str) -> Any | None:
        """Direct semantic lookup — used by memory_lookup tool in ReAct loop."""
        return await self._semantic.read(key)


def _memory_key_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value.strip().lower()).strip("-")[:120] or "default"
