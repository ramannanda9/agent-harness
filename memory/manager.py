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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

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
    async def write(self, text: str, metadata: dict) -> str: ...   # returns episode_id
    async def search(self, query: str, top_k: int = 3) -> list[dict]: ...
    async def get(self, episode_id: str) -> dict | None: ...


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class MemoryWriteRequest:
    """
    Structured output from LLM extraction at run end.
    All fields must be concrete — no vague observations.
    """
    semantic_facts: dict[str, Any]   # deterministic KV for global semantic store
    episodic_summary: str            # natural language paragraph for vector store
    metadata: dict = field(default_factory=dict)
    ttl_seconds: int | None = None   # None = no expiry


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
        working_facts_ttl: int = 3600,     # agent working facts expire in 1 hour
        context_max_episodes: int = 3,
        context_max_semantic_keys: int = 20,
    ) -> None:
        self._semantic = semantic_store
        self._episodic = episodic_store
        self._llm = llm
        self._working_facts_ttl = working_facts_ttl
        self._context_max_episodes = context_max_episodes
        self._context_max_semantic_keys = context_max_semantic_keys
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
            self._conflict_log.append({
                "key": namespaced_key,
                "old": existing,
                "new": value,
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.debug(
                "Memory conflict on key=%s agent=%s old=%r new=%r",
                namespaced_key, agent_id, existing, value,
            )

        await self._semantic.write(
            namespaced_key, value, ttl_seconds=self._working_facts_ttl
        )

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
        LLM extracts global semantic facts + episodic summary.
        Returns the extracted request for logging/debugging.
        """
        extracted = await self._extract_memory(goal, agent_results, trace)

        # write semantic facts globally (no run_id namespace)
        await self._write_semantic_global(extracted)

        # write episodic summary to vector store
        episode_id = await self._write_episodic(goal, agent_results, extracted)

        logger.info(
            "Run-end memory write complete: %d semantic facts, episode_id=%s",
            len(extracted.semantic_facts), episode_id,
        )
        return extracted

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
            # normalize response — handle raw dict or string
            if isinstance(response, str):
                data = json.loads(response)
            elif isinstance(response, dict) and "text" in response:
                data = json.loads(response["text"])
            else:
                data = response

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

    async def _write_semantic_global(self, req: MemoryWriteRequest) -> None:
        for key, value in req.semantic_facts.items():
            existing = await self._semantic.read(key)
            if existing is not None and existing != value:
                self._conflict_log.append({
                    "key": key, "old": existing, "new": value,
                    "scope": "global",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            await self._semantic.write(key, value, ttl_seconds=req.ttl_seconds)

    async def _write_episodic(
        self, goal: str, agent_results: list[dict], req: MemoryWriteRequest
    ) -> str:
        metadata = {
            "goal": goal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_ids": [r.get("agent_id") for r in agent_results],
            "outcome": "success" if any(
                r.get("confidence", 0) > 0.5 for r in agent_results
            ) else "uncertain",
            **req.metadata,
        }
        return await self._episodic.write(
            text=req.episodic_summary,
            metadata=metadata,
        )

    # ── Read path ─────────────────────────────────────────────────────────────

    async def build_context(self, goal: str, agent_id: str | None = None) -> MemoryContext:
        """
        Build memory context for injection into agent system prompt.

        Reads:
          - Episodic: top-k similar past episodes via vector search
          - Semantic: agent-specific keys if agent_id provided, else empty
            (global semantic facts are too noisy to inject wholesale —
             agents should request specific keys via tools in the ReAct loop)
        """
        episodes = await self._episodic.search(goal, top_k=self._context_max_episodes)

        # only inject agent-specific semantic facts at context-build time
        # global facts are fetched on-demand during ReAct via memory_lookup tool
        semantic_facts: dict[str, Any] = {}
        if agent_id:
            all_facts = await self._semantic.search_prefix(f"agent:{agent_id}:")
            # take most recent N keys
            semantic_facts = dict(
                list(all_facts.items())[:self._context_max_semantic_keys]
            )

        return MemoryContext(
            semantic_facts=semantic_facts,
            episodes=episodes,
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_conflict_log(self) -> list[dict]:
        return list(self._conflict_log)

    async def lookup(self, key: str) -> Any | None:
        """Direct semantic lookup — used by memory_lookup tool in ReAct loop."""
        return await self._semantic.read(key)
