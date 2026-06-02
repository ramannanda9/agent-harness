"""Tests for ``BudgetGuard`` — token limits, per-source breakdown, back-compat.

Cost and wall-time enforcement were already covered indirectly by older
runtime tests; this file pins down the new dimensions (token limits +
per-source attribution) that the adapter / runtime wiring depends on.
"""

from __future__ import annotations

import pytest

from harness.runtime import BudgetGuard, GuardrailConfig

# ── add_cost / add_tokens accumulate independently ──────────────────────────


def test_add_cost_accumulates_into_total():
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=1.0))
    guard.add_cost(0.10)
    guard.add_cost(0.05)
    assert guard.cost == pytest.approx(0.15)


def test_add_tokens_accumulates_into_totals():
    guard = BudgetGuard(GuardrailConfig())
    guard.add_tokens(100, 50)
    guard.add_tokens(200, 25)
    assert guard.tokens_in == 300
    assert guard.tokens_out == 75


# ── Back-compat: add_cost without source still works ───────────────────────


def test_add_cost_without_source_does_not_create_breakdown_entry():
    guard = BudgetGuard(GuardrailConfig())
    guard.add_cost(0.42)
    assert guard.cost == pytest.approx(0.42)
    assert guard.breakdown == {}, "untagged cost should not appear in breakdown"


# ── Token limits enforce on check() ────────────────────────────────────────


def test_check_raises_when_input_tokens_exceed_limit():
    guard = BudgetGuard(GuardrailConfig(max_input_tokens=1000))
    guard.add_tokens(500, 100)
    guard.check()  # under limit — no raise
    guard.add_tokens(600, 50)  # cumulative 1100
    with pytest.raises(RuntimeError, match="Input token budget exceeded"):
        guard.check()


def test_check_raises_when_output_tokens_exceed_limit():
    guard = BudgetGuard(GuardrailConfig(max_output_tokens=200))
    guard.add_tokens(100, 100)
    guard.check()
    guard.add_tokens(50, 150)  # cumulative 250
    with pytest.raises(RuntimeError, match="Output token budget exceeded"):
        guard.check()


def test_token_limits_default_to_unlimited():
    """``None`` (the default) means the dimension isn't enforced — back-compat
    for callers that haven't opted in to token limits."""
    guard = BudgetGuard(GuardrailConfig())  # no token caps
    guard.add_tokens(10_000_000, 10_000_000)
    guard.check()  # still passes — only cost and time are enforced


def test_cost_limit_still_enforced_alongside_token_limit():
    """The new dimensions don't supersede the old ones; both gates fire."""
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=0.01, max_input_tokens=10_000))
    guard.add_cost(0.05)
    with pytest.raises(RuntimeError, match="Cost budget exceeded"):
        guard.check()


# ── Breakdown: per-source attribution ──────────────────────────────────────


def test_breakdown_tags_cost_by_source():
    guard = BudgetGuard(GuardrailConfig())
    guard.add_cost(0.10, source="classifier")
    guard.add_cost(0.20, source="planner")
    guard.add_cost(0.05, source="classifier")
    assert guard.breakdown["classifier"]["cost_usd"] == pytest.approx(0.15)
    assert guard.breakdown["planner"]["cost_usd"] == pytest.approx(0.20)


def test_breakdown_tags_tokens_by_source():
    guard = BudgetGuard(GuardrailConfig())
    guard.add_tokens(100, 50, source="classifier")
    guard.add_tokens(2_000, 800, source="planner")
    assert guard.breakdown["classifier"]["tokens_in"] == 100
    assert guard.breakdown["classifier"]["tokens_out"] == 50
    assert guard.breakdown["planner"]["tokens_in"] == 2_000
    assert guard.breakdown["planner"]["tokens_out"] == 800


def test_breakdown_snapshot_is_a_copy():
    """Mutating the returned snapshot must not affect the guard's state."""
    guard = BudgetGuard(GuardrailConfig())
    guard.add_cost(0.10, source="planner")
    snap = guard.breakdown
    snap["planner"]["cost_usd"] = 999.0
    assert guard.breakdown["planner"]["cost_usd"] == pytest.approx(0.10)


def test_totals_and_breakdown_stay_consistent():
    """Tagged additions land in BOTH the global totals and the breakdown."""
    guard = BudgetGuard(GuardrailConfig())
    guard.add_tokens(100, 50, source="classifier")
    guard.add_tokens(500, 200, source="planner")
    guard.add_cost(0.10, source="classifier")
    guard.add_cost(0.20, source="planner")
    assert guard.tokens_in == 600
    assert guard.tokens_out == 250
    assert guard.cost == pytest.approx(0.30)
    by_source = guard.breakdown
    assert sum(v["tokens_in"] for v in by_source.values()) == guard.tokens_in
    assert sum(v["tokens_out"] for v in by_source.values()) == guard.tokens_out
    assert sum(v["cost_usd"] for v in by_source.values()) == pytest.approx(guard.cost)


def test_untagged_additions_appear_in_totals_but_not_breakdown():
    """ReAct calls don't tag — their spending should still count toward the
    totals, but not pollute the per-call-site breakdown."""
    guard = BudgetGuard(GuardrailConfig())
    guard.add_tokens(100, 50)  # untagged (e.g. BaseAgent ReAct call)
    guard.add_tokens(200, 100, source="planner")
    assert guard.tokens_in == 300
    assert guard.tokens_out == 150
    assert set(guard.breakdown.keys()) == {"planner"}
