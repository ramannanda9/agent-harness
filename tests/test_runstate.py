"""Tests for the explicit run-state records in ``harness/runstate.py``.

These are pure data: no agent, no event loop. What matters is that a state
survives a JSON round-trip byte-for-byte in meaning, that the discriminator
routes to the right record, and that checkpoints from before this format
are refused loudly instead of being silently misread.
"""

from __future__ import annotations

import json

import pytest

from harness.runstate import (
    STATE_VERSION,
    ActionState,
    ActionStatus,
    LegacyCheckpointError,
    OrchestratorState,
    Phase,
    RunState,
    TaskState,
    TaskStatus,
    load_state,
)


def _roundtrip(state):
    """Round-trip through real JSON, the way a checkpoint store would."""
    return load_state(json.loads(json.dumps(state.to_dict(), default=str)))


# ── RunState ─────────────────────────────────────────────────────────────────


def test_run_state_defaults_to_the_start_of_a_run():
    state = RunState(run_id="r1", agent_id="researcher", task="find things")

    assert state.step == 0
    assert state.phase is Phase.THINK
    assert state.actions == []
    assert state.kind == "agent"
    assert state.version == STATE_VERSION
    assert not state.is_terminal


@pytest.mark.parametrize("phase", [Phase.DONE, Phase.FAILED])
def test_run_state_terminal_phases(phase):
    state = RunState(run_id="r1", agent_id="a", task="t", phase=phase)
    assert state.is_terminal


@pytest.mark.parametrize("phase", [Phase.THINK, Phase.APPROVE, Phase.ACT, Phase.OBSERVE])
def test_run_state_non_terminal_phases(phase):
    state = RunState(run_id="r1", agent_id="a", task="t", phase=phase)
    assert not state.is_terminal


def test_run_state_survives_a_json_round_trip():
    state = RunState(
        run_id="r1",
        agent_id="researcher",
        task="audit the system",
        memory={"messages": [{"role": "user", "content": "hi"}], "summarization_count": 2},
        step=4,
        phase=Phase.ACT,
        response={"thought": "look around", "action": "shell", "args": {"cmd": "ls"}},
        actions=[
            ActionState(
                tool="shell",
                args={"cmd": "ls"},
                status=ActionStatus.EXECUTED,
                observation="a\nb",
                approval_id="ap-1",
            ),
            ActionState(tool="shell", args={"cmd": "ps"}, status=ActionStatus.PENDING),
        ],
        budget={
            "cost_usd": 0.25,
            "elapsed_seconds": 12.5,
            "tokens_in": 100,
            "tokens_out": 20,
            "breakdown": {},
        },
    )

    restored = _roundtrip(state)

    assert isinstance(restored, RunState)
    assert restored.to_dict() == state.to_dict()
    assert restored.phase is Phase.ACT
    assert restored.step == 4
    assert restored.budget["cost_usd"] == 0.25


def test_run_state_preserves_per_action_status_and_observations():
    """The whole point of per-action state: a half-finished parallel batch
    resumes knowing which tools already ran, so their side effects are not
    repeated and their observations are not lost."""
    state = RunState(
        run_id="r1",
        agent_id="a",
        task="t",
        phase=Phase.ACT,
        actions=[
            ActionState(tool="one", status=ActionStatus.EXECUTED, observation="done"),
            ActionState(tool="two", status=ActionStatus.APPROVED),
            ActionState(tool="three", status=ActionStatus.REJECTED),
            ActionState(tool="four", status=ActionStatus.PENDING),
        ],
    )

    restored = _roundtrip(state)

    assert [a.status for a in restored.actions] == [
        ActionStatus.EXECUTED,
        ActionStatus.APPROVED,
        ActionStatus.REJECTED,
        ActionStatus.PENDING,
    ]
    assert restored.actions[0].observation == "done"


def test_action_state_tolerates_a_minimal_dict():
    action = ActionState.from_dict({"tool": "shell"})

    assert action.args == {}
    assert action.status is ActionStatus.PENDING
    assert action.observation is None


def test_run_state_carries_a_terminal_result():
    state = RunState(
        run_id="r1",
        agent_id="a",
        task="t",
        phase=Phase.DONE,
        result={"answer": "42", "confidence": 0.9, "steps": 3},
    )

    restored = _roundtrip(state)

    assert restored.result["answer"] == "42"
    assert restored.is_terminal


# ── OrchestratorState ────────────────────────────────────────────────────────


def test_orchestrator_state_survives_a_json_round_trip():
    state = OrchestratorState(
        run_id="r1",
        goal="audit everything",
        plan={"tasks": [{"id": "t0", "agent_id": "a", "instruction": "go"}], "rationale": "why"},
        tasks={
            "t0": TaskState(
                task_id="t0", status=TaskStatus.DONE, result={"task_id": "t0", "answer": "ok"}
            ),
            "t1": TaskState(
                task_id="t1", status=TaskStatus.RUNNING, attempt=1, agent_ckp_id="r1:worker"
            ),
        },
        replan_count=2,
    )

    restored = _roundtrip(state)

    assert isinstance(restored, OrchestratorState)
    assert restored.to_dict() == state.to_dict()
    assert restored.tasks["t1"].status is TaskStatus.RUNNING
    assert restored.tasks["t1"].agent_ckp_id == "r1:worker"
    assert restored.replan_count == 2


def test_orchestrator_state_preserves_attempt_across_resume():
    """A durable attempt counter is what stops ``on_failure=retry`` handing
    a task a fresh retry budget every time the run is resumed."""
    state = OrchestratorState(
        run_id="r1",
        goal="g",
        tasks={"t0": TaskState(task_id="t0", status=TaskStatus.FAILED, attempt=2)},
    )

    assert _roundtrip(state).tasks["t0"].attempt == 2


def test_completed_results_returns_only_successful_terminal_tasks():
    state = OrchestratorState(
        run_id="r1",
        goal="g",
        tasks={
            "done": TaskState(task_id="done", status=TaskStatus.DONE, result={"answer": "y"}),
            "running": TaskState(task_id="running", status=TaskStatus.RUNNING),
            "failed": TaskState(task_id="failed", status=TaskStatus.FAILED, result={"answer": ""}),
            "skipped": TaskState(task_id="skipped", status=TaskStatus.SKIPPED),
        },
    )

    assert set(state.completed_results()) == {"done"}


def test_a_running_task_is_distinguishable_from_one_never_started():
    """The old model derived state from membership in a ``completed`` dict,
    which collapsed these two cases and restarted both from zero."""
    state = OrchestratorState(
        run_id="r1",
        goal="g",
        tasks={
            "started": TaskState(task_id="started", status=TaskStatus.RUNNING, agent_ckp_id="r1:w"),
            "untouched": TaskState(task_id="untouched"),
        },
    )

    restored = _roundtrip(state)

    assert restored.tasks["started"].status is TaskStatus.RUNNING
    assert restored.tasks["untouched"].status is TaskStatus.PENDING
    assert restored.tasks["untouched"].agent_ckp_id is None


# ── load_state dispatch ──────────────────────────────────────────────────────


def test_load_state_dispatches_on_kind_not_on_key_sniffing():
    """An agent state that happens to mention a plan must still decode as an
    agent state — the old ``"plan" in checkpoint`` sniff could not promise that."""
    agent = RunState(run_id="r1", agent_id="a", task="write a plan for the migration")
    agent.memory = {"messages": [{"role": "user", "content": "plan"}]}

    assert isinstance(load_state(agent.to_dict()), RunState)


def test_load_state_rejects_legacy_checkpoints_with_a_clear_message():
    legacy = {
        "run_id": "r1",
        "agent_id": "researcher",
        "task": "t",
        "step": 3,
        "memory": {"messages": []},
        "pending": {"approval_id": "x", "tool": "shell", "args": {}, "step": 3},
    }

    with pytest.raises(LegacyCheckpointError, match="cannot be resumed"):
        load_state(legacy)


def test_legacy_checkpoint_error_is_a_value_error():
    with pytest.raises(ValueError):
        load_state({"run_id": "r1", "step": 0})


def test_load_state_rejects_a_newer_state_version():
    future = RunState(run_id="r1", agent_id="a", task="t").to_dict()
    future["version"] = STATE_VERSION + 1

    with pytest.raises(ValueError, match="newer than this harness"):
        load_state(future)


def test_load_state_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown checkpoint kind"):
        load_state({"kind": "wat", "version": STATE_VERSION})


def test_load_state_rejects_a_non_dict():
    with pytest.raises(TypeError):
        load_state(["not", "a", "checkpoint"])
