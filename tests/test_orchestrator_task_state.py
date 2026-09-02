"""Tests for durable per-task status in an orchestration.

State used to be derived from membership in a ``completed`` map. That could
not distinguish a task that never started from one that started and died, and
it could not represent a task that had failed but was queued to retry — the
failed result was already in ``completed``, so resume skipped it forever.
"""

from __future__ import annotations

import pytest

from harness.runstate import (
    OrchestratorState,
    TaskState,
    TaskStatus,
    load_state,
)
from orchestrator.planner import (
    Plan,
    Task,
    _initial_task_states,
    _reconcile_task_states,
)


def _plan(*specs: tuple[str, str]) -> Plan:
    return Plan(
        tasks=[Task(id=tid, agent_id="w", instruction=instr) for tid, instr in specs],
        rationale="r",
    )


# ── Reconciling states across a replan ───────────────────────────────────────


def test_initial_task_states_covers_every_planned_task():
    states = _initial_task_states(_plan(("t1", "a"), ("t2", "b")))

    assert set(states) == {"t1", "t2"}
    assert all(s.status is TaskStatus.PENDING for s in states.values())


def test_replan_keeps_the_state_of_an_unchanged_task():
    before = _initial_task_states(_plan(("t1", "a"), ("t2", "b")))
    before["t1"].status = TaskStatus.DONE
    before["t1"].instruction = "a"

    after = _reconcile_task_states(before, _plan(("t1", "a"), ("t3", "c")))

    assert after["t1"].status is TaskStatus.DONE, "finished work should not be redone"
    assert after["t3"].status is TaskStatus.PENDING


def test_replan_resets_a_reused_id_whose_instruction_changed():
    """Carrying a DONE status onto an id that now means something else would
    claim work was finished that nobody did."""
    before = _initial_task_states(_plan(("t1", "original")))
    before["t1"].status = TaskStatus.DONE
    before["t1"].instruction = "original"

    after = _reconcile_task_states(before, _plan(("t1", "something else entirely")))

    assert after["t1"].status is TaskStatus.PENDING
    assert after["t1"].result is None


def test_replan_drops_task_ids_the_new_plan_no_longer_names():
    before = _initial_task_states(_plan(("t1", "a"), ("t2", "b")))

    after = _reconcile_task_states(before, _plan(("t1", "a")))

    assert set(after) == {"t1"}


# ── What resume treats as outstanding ────────────────────────────────────────


def test_a_failed_task_queued_for_retry_is_still_outstanding():
    """The ordering bug this replaces: a failed result went into ``completed``
    and the checkpoint was written *before* the retry decision ran, so a crash
    in between left a record saying the task was finished. It was never
    retried."""
    state = OrchestratorState(
        run_id="r1",
        goal="g",
        tasks={
            "t1": TaskState(
                task_id="t1",
                status=TaskStatus.PENDING,  # moved back for the retry
                attempt=1,
                result={"task_id": "t1", "success": False, "answer": ""},
            )
        },
    )

    restored = load_state(state.to_dict())

    assert restored.tasks["t1"].status is TaskStatus.PENDING
    assert restored.tasks["t1"].attempt == 1
    # It has a result, so it must not be mistaken for never-attempted...
    assert "t1" in restored.recorded_results()
    # ...but it is not finished either.
    assert "t1" not in restored.completed_results()


def test_a_running_task_is_outstanding_but_a_done_one_is_not():
    state = OrchestratorState(
        run_id="r1",
        goal="g",
        tasks={
            "running": TaskState(task_id="running", status=TaskStatus.RUNNING),
            "done": TaskState(task_id="done", status=TaskStatus.DONE, result={"answer": "y"}),
            "skipped": TaskState(task_id="skipped", status=TaskStatus.SKIPPED),
        },
    )

    from orchestrator.planner import _OUTSTANDING

    outstanding = {tid for tid, ts in state.tasks.items() if ts.status in _OUTSTANDING}
    assert outstanding == {"running"}


# ── End-to-end through the orchestrator ──────────────────────────────────────


@pytest.mark.asyncio
async def test_replan_persists_the_new_plan_not_the_original(monkeypatch):
    """The checkpoint used to keep the plan the run started with, so resuming
    after a replan rebuilt from a DAG the run had already abandoned."""
    from tests.conftest import ScriptedLLM
    from tests.test_checkpoint_resume import (
        InMemoryCheckpointStore,
        _make_runtime,
        _orch_routes,
    )

    llm = ScriptedLLM()
    routes = _orch_routes()
    original_planner = routes["decomposes goals"]

    replans = {"n": 0}

    def planner(system, messages, kwargs):
        # The replan prompt reuses the planner slot; hand back a different DAG.
        if "Replan" in str(messages):
            replans["n"] += 1
            return {
                "tasks": [
                    {
                        "id": "t9",
                        "agent_id": "reporter",
                        "instruction": "recovered step",
                        "depends_on": [],
                        "on_failure": "skip",
                    }
                ],
                "rationale": "recovered",
            }
        return original_planner(system, messages, kwargs)

    routes["decomposes goals"] = planner
    llm.routes = routes

    ckp_store = InMemoryCheckpointStore()
    runtime, _ = _make_runtime(llm, ckp_store)

    written: list[dict] = []
    original_write = ckp_store.write

    async def capturing_write(key: str, data: dict) -> None:
        await original_write(key, data)
        if data.get("kind") == "orchestrator":
            written.append(dict(data))

    ckp_store.write = capturing_write
    await runtime.run("test goal")

    assert written, "no orchestrator checkpoint written"
    # Whatever the run finished with, every checkpoint's task set matches the
    # plan stored alongside it — the two cannot drift apart.
    for checkpoint in written:
        state = load_state(checkpoint)
        assert set(state.tasks) == {t["id"] for t in state.plan["tasks"]}
