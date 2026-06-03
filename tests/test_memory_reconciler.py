"""``MemoryReconciler`` + ``MemoryManager`` reconcile path.

Covers:
  - parse layer: each action shape round-trips through ``_parse_reconcile_plan``
  - apply layer: ADD / UPDATE / MERGE / NOOP write the right things; DELETE
    is demoted to NOOP by default and applies when opted in
  - dispatch: ``write_run_end`` uses the reconcile path when the LLM returns
    plan-shaped JSON, and falls back to the legacy extract path on garbage
  - hard-delete: ``invalidate`` removes records rather than tombstoning
  - ``compact`` runs a reconcile pass without new evidence
"""

from __future__ import annotations

from typing import Any

import pytest

from memory.manager import (
    MemoryManager,
    ReconcileAction,
    ReconcilePlan,
    _parse_reconcile_plan,
)
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── Parse layer ──────────────────────────────────────────────────────────────


def test_parse_plan_recognises_each_action():
    raw = {
        "semantic_actions": [
            {"action": "add", "key": "k1", "value": "v1"},
            {"action": "update", "key": "k2", "value": "v2", "rationale": "r"},
            {"action": "merge", "key": "k3", "value": "v3", "rationale": "r"},
            {"action": "delete", "key": "k4", "rationale": "r"},
            {"action": "noop", "key": "k5", "rationale": "duplicate"},
        ],
        "episodic_action": {"action": "update", "memory_key": "ep", "text": "..."},
    }
    plan = _parse_reconcile_plan(raw)
    assert not plan.parse_failed
    assert [a.action for a in plan.semantic_actions] == [
        ReconcileAction.ADD,
        ReconcileAction.UPDATE,
        ReconcileAction.MERGE,
        ReconcileAction.DELETE,
        ReconcileAction.NOOP,
    ]
    assert plan.episodic_action is not None
    assert plan.episodic_action.action is ReconcileAction.UPDATE


def test_parse_plan_flags_empty_response_as_parse_failure():
    """An empty / non-conforming response must signal fallback to legacy."""
    assert _parse_reconcile_plan({}).parse_failed is True
    assert _parse_reconcile_plan({"semantic_facts": {"x": 1}}).parse_failed is True
    assert _parse_reconcile_plan("not a dict").parse_failed is True


def test_parse_plan_skips_malformed_action_entries():
    """Garbage entries are skipped, not crashed on."""
    raw = {
        "semantic_actions": [
            {"action": "add", "key": "good", "value": "ok"},
            {"action": "invalid_action", "key": "x"},
            "not a dict",
            {"action": "delete"},  # missing key
            {"key": "no_action"},  # missing action
        ],
        "episodic_action": None,
    }
    plan = _parse_reconcile_plan(raw)
    keys = [a.key for a in plan.semantic_actions]
    assert keys == ["good"], f"only the valid entry should survive; got {keys!r}"


# ── Apply layer ──────────────────────────────────────────────────────────────


class _PlanLLM:
    """LLM stub that always returns the same canned plan as a JSON string."""

    def __init__(self, plan: dict) -> None:
        import json

        self._text = json.dumps(plan)
        self.last_usage: dict | None = None

    async def complete(self, system, messages, **kwargs) -> dict:
        return {"text": self._text, "usage": {}}


def _manager_with_plan(
    plan: dict, **manager_kwargs
) -> tuple[MemoryManager, InMemorySemanticStore, InMemoryEpisodicStore]:
    semantic = InMemorySemanticStore()
    episodic = InMemoryEpisodicStore()
    llm = _PlanLLM(plan)
    manager = MemoryManager(
        semantic_store=semantic,
        episodic_store=episodic,
        llm=llm,
        **manager_kwargs,
    )
    return manager, semantic, episodic


@pytest.mark.asyncio
async def test_apply_add_writes_new_fact():
    manager, sem, _ = _manager_with_plan(
        {
            "semantic_actions": [{"action": "add", "key": "redis:port", "value": 6379}],
            "episodic_action": {"action": "noop", "memory_key": "k"},
        }
    )
    await manager.write_run_end(goal="audit", agent_results=[], trace=[])
    assert await sem.read("redis:port") == 6379


@pytest.mark.asyncio
async def test_apply_update_overwrites_existing_and_logs_conflict():
    manager, sem, _ = _manager_with_plan(
        {
            "semantic_actions": [
                {
                    "action": "update",
                    "key": "redis:status",
                    "value": "healthy",
                    "rationale": "passed health check",
                }
            ],
            "episodic_action": {"action": "noop", "memory_key": "k"},
        }
    )
    await sem.write("redis:status", "degraded")
    await manager.write_run_end(goal="audit", agent_results=[], trace=[])
    assert await sem.read("redis:status") == "healthy"
    log = manager.get_conflict_log()
    assert any(e.get("action") == "update" and e.get("rationale") for e in log)


@pytest.mark.asyncio
async def test_apply_delete_is_demoted_by_default():
    """DELETE must not fire without explicit opt-in — the canary against
    surprise data loss the moment reconcile defaults to on."""
    manager, sem, _ = _manager_with_plan(
        {
            "semantic_actions": [{"action": "delete", "key": "stale", "rationale": "contradicted"}],
            "episodic_action": {"action": "noop", "memory_key": "k"},
        }
    )
    await sem.write("stale", "old value")
    await manager.write_run_end(goal="audit", agent_results=[], trace=[])
    assert await sem.read("stale") == "old value", "DELETE should have been demoted to NOOP"
    log = manager.get_conflict_log()
    assert any(e.get("outcome") == "demoted_to_noop" for e in log), (
        "demoted DELETE must be logged for audit"
    )


@pytest.mark.asyncio
async def test_apply_delete_fires_when_opted_in():
    manager, sem, _ = _manager_with_plan(
        {
            "semantic_actions": [{"action": "delete", "key": "stale", "rationale": "contradicted"}],
            "episodic_action": {"action": "noop", "memory_key": "k"},
        },
        allow_destructive_reconcile=True,
    )
    await sem.write("stale", "old value")
    await manager.write_run_end(goal="audit", agent_results=[], trace=[])
    assert await sem.read("stale") is None


@pytest.mark.asyncio
async def test_apply_noop_does_not_touch_existing_value():
    manager, sem, _ = _manager_with_plan(
        {
            "semantic_actions": [{"action": "noop", "key": "kept", "rationale": "duplicate"}],
            "episodic_action": {"action": "noop", "memory_key": "k"},
        }
    )
    await sem.write("kept", "original")
    await manager.write_run_end(goal="audit", agent_results=[], trace=[])
    assert await sem.read("kept") == "original"


@pytest.mark.asyncio
async def test_apply_episodic_update_uses_latest_policy_hard_delete():
    """Episodic UPDATE writes the merged text under the same memory_key with
    ``latest`` policy — which now hard-deletes the prior episode (no
    tombstone accumulation)."""
    manager, _, ep = _manager_with_plan(
        {
            "semantic_actions": [],
            "episodic_action": {
                "action": "update",
                "memory_key": "run_summary:audit",
                "text": "merged summary v2",
            },
        }
    )
    # Seed a prior episode with the same memory_key.
    await ep.write(
        text="summary v1",
        metadata={
            "memory_key": "run_summary:audit",
            "memory_policy": "latest",
            "memory_kind": "run_summary",
            "shared": True,
        },
    )
    assert ep.count() == 1

    await manager.write_run_end(goal="audit", agent_results=[], trace=[])

    # Hard-delete: store holds exactly one episode (the merged update), not
    # one active + one inactive tombstone.
    assert ep.count() == 1
    remaining = ep._episodes[0]
    assert remaining["text"] == "merged summary v2"


# ── Dispatch / fallback ───────────────────────────────────────────────────────


class _LegacyShapeLLM:
    """Returns the old ``{semantic_facts, episodic_summary}`` shape.

    Verifies that ``write_run_end`` falls back to the legacy extract path
    when the LLM doesn't speak the reconcile schema.
    """

    last_usage = None

    async def complete(self, system, messages, **kwargs) -> dict:
        import json

        return {
            "text": json.dumps(
                {
                    "semantic_facts": {"legacy:fact": "v"},
                    "episodic_summary": "legacy summary",
                    "metadata": {},
                    "ttl_seconds": None,
                }
            ),
            "usage": {},
        }


@pytest.mark.asyncio
async def test_write_run_end_falls_back_to_legacy_when_reconcile_unsupported():
    semantic = InMemorySemanticStore()
    episodic = InMemoryEpisodicStore()
    manager = MemoryManager(
        semantic_store=semantic,
        episodic_store=episodic,
        llm=_LegacyShapeLLM(),
    )
    await manager.write_run_end(goal="x", agent_results=[{"agent_id": "a"}], trace=[])
    # The legacy fallback wrote the flat-extract fact.
    assert await semantic.read("legacy:fact") == "v"


# ── compact() ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_runs_reconcile_without_evidence():
    """``compact`` is reconcile with ``evidence=None`` — used for cleanup
    passes triggered outside a run."""
    seen_evidence: list[Any] = []

    class _RecordingLLM:
        last_usage = None

        async def complete(self, system, messages, **kwargs) -> dict:
            seen_evidence.append(messages[0]["content"])
            return {
                "text": '{"semantic_actions": [{"action": "noop", "key": "x", "rationale": "ok"}], "episodic_action": null}',
                "usage": {},
            }

    manager = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=_RecordingLLM(),
    )
    plan = await manager.compact(goal="cleanup")
    assert not plan.parse_failed
    assert seen_evidence, "compact() must invoke the reconciler"
    assert "(none — cleanup pass)" in seen_evidence[0], (
        "compact() must signal to the LLM that no new evidence is provided"
    )


# ── Hard-delete behaviour on the InMemory store ──────────────────────────────


@pytest.mark.asyncio
async def test_inmemory_invalidate_hard_deletes_episodes():
    """invalidate() must shrink the store, not just flip a flag."""
    store = InMemoryEpisodicStore()
    await store.write("a", {"memory_key": "k", "memory_policy": "append"})
    await store.write("b", {"memory_key": "k", "memory_policy": "append"})
    await store.write("c", {"memory_key": "other", "memory_policy": "append"})
    assert store.count() == 3

    removed = await store.invalidate("k")
    assert removed == 2
    assert store.count() == 1, "matching records should be removed, not tombstoned"


@pytest.mark.asyncio
async def test_inmemory_latest_policy_hard_deletes_prior_episodes():
    """``latest`` writes must collapse prior episodes for the same memory_key
    to a single live row — no accumulating tombstones."""
    store = InMemoryEpisodicStore()
    await store.write("v1", {"memory_key": "k", "memory_policy": "latest"})
    await store.write("v2", {"memory_key": "k", "memory_policy": "latest"})
    await store.write("v3", {"memory_key": "k", "memory_policy": "latest"})
    assert store.count() == 1
    assert store._episodes[0]["text"] == "v3"


# ── Auto-compact threshold ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_compact_fires_when_threshold_exceeded(monkeypatch):
    """When an agent accumulates ``agent_task`` episodes past the threshold,
    ``write_agent_task_end`` schedules a ``compact()`` via ``fire``."""

    class _PlanOnlyLLM:
        last_usage = None

        async def complete(self, system, messages, **kwargs) -> dict:
            return {
                "text": '{"semantic_actions": [{"action": "noop", "key": "x"}], "episodic_action": null}',
                "usage": {},
            }

    compact_calls: list[dict] = []

    manager = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=_PlanOnlyLLM(),
        auto_compact_threshold={"agent_task": 3},
    )

    async def fake_compact(*args, **kwargs):
        compact_calls.append(kwargs)
        return ReconcilePlan()

    monkeypatch.setattr(manager, "compact", fake_compact)

    for i in range(3):
        await manager.write_agent_task_end(
            goal="g",
            task_id=f"t{i}",
            agent_id="alpha",
            instruction="do thing",
            result={"answer": f"a{i}", "confidence": 1.0},
        )
        # Drain any scheduled background coroutines.
        import asyncio

        await asyncio.sleep(0)

    assert compact_calls, "compaction should have fired at threshold"
    assert compact_calls[-1].get("agent_id") == "alpha"
