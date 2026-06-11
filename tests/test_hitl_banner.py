from __future__ import annotations

from harness.hitl import ApprovalRequest, _print_banner


def _request(*, resume_hint: str | None = None) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="approval-1",
        run_id="run-1:agent",
        agent_id="agent",
        tool="shell",
        args={"cmd": "echo hi"},
        step=3,
        timestamp="2026-06-11T00:00:00Z",
        resume_hint=resume_hint,
    )


def test_hitl_banner_defaults_to_resume_hint(capsys):
    _print_banner(_request())

    out = capsys.readouterr().out
    assert "Ctrl-C to pause. Resume: python" in out
    assert "--resume run-1:agent" in out


def test_hitl_banner_uses_custom_resume_hint(capsys):
    _print_banner(
        _request(resume_hint="Esc cancels this turn; completed session history is preserved.")
    )

    out = capsys.readouterr().out
    assert "Esc cancels this turn; completed session history is preserved." in out
    assert "--resume" not in out
