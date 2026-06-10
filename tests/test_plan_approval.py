"""Pin ``harness.hitl.request_plan_approval`` semantics.

Plan mode is per-turn gating, not per-tool. The general
``request_approval`` UX (y / n / a / A / correction) makes sense when
the same tool fires repeatedly and the user wants to allow
``shell/grep`` for the session. Those options are conceptually wrong
for plan-mode approval:

- ``a`` = "allow plans for the session" silently turns plan mode off
  until restart.
- ``A`` = "always allow plans" permanently always-allows the gate,
  i.e. the same thing forever.

Both contradict the user's intent in turning plan mode on.
``request_plan_approval`` is the focused sibling that exposes only
y / n / revision.
"""

from __future__ import annotations

import pytest

from harness.hitl import PlanApprovalResponse, _parse_plan_stdin


def test_y_is_approved_without_correction():
    resp = _parse_plan_stdin("y")
    assert resp == PlanApprovalResponse(approved=True, correction=None)


def test_yes_is_approved_without_correction():
    resp = _parse_plan_stdin("yes")
    assert resp == PlanApprovalResponse(approved=True, correction=None)


def test_n_is_rejected_without_correction():
    resp = _parse_plan_stdin("n")
    assert resp == PlanApprovalResponse(approved=False, correction=None)


def test_no_is_rejected_without_correction():
    resp = _parse_plan_stdin("no")
    assert resp == PlanApprovalResponse(approved=False, correction=None)


def test_a_is_treated_as_revision_text_not_session_allow():
    """In ``request_approval``, ``a`` registers a session-allow policy.
    In ``request_plan_approval``, the same letter is just a one-character
    revision request — there is no policy concept for plan approvals."""
    resp = _parse_plan_stdin("a")
    assert resp.approved is False
    assert resp.correction == "a"


def test_uppercase_A_is_treated_as_revision_text_not_persistent_allow():
    """In ``request_approval``, ``A`` registers a persistent-allow rule.
    In ``request_plan_approval``, no such rule exists — it's revision text."""
    resp = _parse_plan_stdin("A")
    assert resp.approved is False
    assert resp.correction == "A"


def test_free_text_becomes_revision():
    resp = _parse_plan_stdin("use step 3 instead of step 2")
    assert resp.approved is False
    assert resp.correction == "use step 3 instead of step 2"


def test_empty_input_is_plain_rejection_no_correction():
    """Empty input distinguishes plain rejection (``correction=None``)
    from a revision request — chat() treats them differently (plain
    rejection ends the turn; revision triggers a re-plan)."""
    resp = _parse_plan_stdin("")
    assert resp.approved is False
    assert resp.correction is None


def test_whitespace_input_is_plain_rejection_no_correction():
    resp = _parse_plan_stdin("   \n  \n  ")
    assert resp.approved is False
    assert resp.correction is None


def test_case_insensitive_y_n_parsing():
    assert _parse_plan_stdin("Y").approved is True
    assert _parse_plan_stdin("YES").approved is True
    assert _parse_plan_stdin("N").approved is False
    assert _parse_plan_stdin("NO").approved is False


def test_leading_trailing_whitespace_stripped_from_revision():
    """Revision text is stripped before being passed to the planner —
    avoids spurious newlines / spaces in the system prompt revision."""
    resp = _parse_plan_stdin("\n  use step 3  \n")
    assert resp.approved is False
    assert resp.correction == "use step 3"


@pytest.mark.parametrize(
    "text",
    [
        "n",
        "N",
        "no",
        "NO",
        "  n  ",
        "\nn\n",
    ],
)
def test_n_variants_all_parse_as_plain_rejection(text):
    """A reviewer typing any plain ``n`` variant gets a clean rejection
    (correction=None), not "n" treated as a single-letter revision."""
    resp = _parse_plan_stdin(text)
    assert resp.approved is False
    assert resp.correction is None
