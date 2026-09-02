"""Tests for carrying a budget across a resume.

``BudgetGuard.snapshot`` has always existed as an output shape attached to
terminal events. Until it had an inverse, resuming a checkpointed run built a
fresh guard, so spend from before the crash was forgotten and the wall clock
restarted at zero — a run could be resumed indefinitely, each time with the
full allowance.
"""

from __future__ import annotations

import time

import pytest

from harness.runtime import BudgetGuard, GuardrailConfig


def _guard(**overrides) -> BudgetGuard:
    config = GuardrailConfig(
        max_total_cost_usd=10.0,
        max_wall_time_seconds=600,
        **overrides,
    )
    return BudgetGuard(config)


def test_snapshot_round_trips_through_restore():
    original = _guard()
    original.add_cost(1.25, source="planner")
    original.add_tokens(400, 90, source="planner")
    original.add_cost(0.75)
    original.add_tokens(100, 10)

    resumed = _guard()
    resumed.restore(original.snapshot())

    assert resumed.cost == pytest.approx(2.0)
    assert resumed.tokens_in == 500
    assert resumed.tokens_out == 100
    assert resumed.breakdown["planner"]["cost_usd"] == pytest.approx(1.25)
    assert resumed.breakdown["planner"]["tokens_in"] == 400


def test_restored_spending_counts_against_the_limit():
    """The point of the exercise: a resumed run cannot spend the budget twice."""
    spent = _guard()
    spent.add_cost(9.5)

    resumed = _guard()
    resumed.restore(spent.snapshot())
    resumed.check()  # 9.5 of 10.0 — still inside

    resumed.add_cost(1.0)
    with pytest.raises(RuntimeError, match="Cost budget exceeded"):
        resumed.check()


def test_restore_backdates_the_wall_clock():
    resumed = _guard()
    resumed.restore({"elapsed_seconds": 120.0})

    assert resumed.elapsed == pytest.approx(120.0, abs=1.0)


def test_restore_excludes_time_the_process_was_dead():
    """Wall-time budgets bound how long the agent worked, not how long the
    clock ran — the same reason HITL waits are suspended rather than billed."""
    original = _guard()
    original.add_cost(0.1)
    snapshot = original.snapshot()
    worked = snapshot["elapsed_seconds"]

    time.sleep(0.05)  # stands in for arbitrary downtime between processes

    resumed = _guard()
    resumed.restore(snapshot)

    assert resumed.elapsed == pytest.approx(worked, abs=0.05)


def test_restored_elapsed_counts_against_the_time_limit():
    resumed = BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60))
    resumed.restore({"elapsed_seconds": 61.0})

    with pytest.raises(RuntimeError, match="Time budget exceeded"):
        resumed.check()


def test_restore_clears_any_suspend_in_progress():
    """A snapshot's elapsed already excludes suspended time; carrying a live
    suspend across would subtract it a second time."""
    resumed = _guard()
    resumed.suspend()
    resumed.restore({"elapsed_seconds": 5.0})

    assert resumed._suspend_at is None
    assert resumed.elapsed == pytest.approx(5.0, abs=1.0)


def test_restore_tolerates_a_partial_snapshot():
    resumed = _guard()
    resumed.restore({})

    assert resumed.cost == 0.0
    assert resumed.tokens_in == 0
    assert resumed.breakdown == {}


def test_restore_does_not_alias_the_snapshot_breakdown():
    snapshot = {
        "cost_usd": 1.0,
        "breakdown": {"planner": {"cost_usd": 1.0, "tokens_in": 0, "tokens_out": 0}},
    }
    resumed = _guard()
    resumed.restore(snapshot)
    resumed.add_cost(2.0, source="planner")

    assert snapshot["breakdown"]["planner"]["cost_usd"] == 1.0


def test_token_limits_survive_a_restore():
    guard = BudgetGuard(
        GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=600, max_input_tokens=1000)
    )
    guard.restore({"tokens_in": 1200})

    with pytest.raises(RuntimeError, match="Input token budget exceeded"):
        guard.check()
