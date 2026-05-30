from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from harness import cli
from harness.llm.auth import OAuthCredential


def test_cli_supports_claude_code_auth_status(tmp_path, monkeypatch, capsys):
    path = tmp_path / "auth.json"
    expires = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)
    path.write_text(
        json.dumps(
            {
                "claude-code": {
                    "type": "oauth",
                    "access": "access",
                    "refresh": "refresh",
                    "expires": expires,
                }
            }
        )
    )
    path.chmod(0o600)
    monkeypatch.setattr(
        "sys.argv",
        ["agent-harness", "auth", "status", "claude-code", "--auth-file", str(path)],
    )

    assert cli.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["provider"] == "claude-code"
    assert out["expired"] is False


def test_cli_writes_claude_code_login(monkeypatch, tmp_path):
    path = tmp_path / "auth.json"

    class _Client:
        def begin_login(self):
            return type(
                "Login",
                (),
                {"url": "https://claude.ai/oauth/authorize", "state": "state"},
            )()

        async def finish_login(self, _login, callback_input):
            assert callback_input == "code#state"
            return OAuthCredential(
                provider="claude-code",
                access="access",
                refresh="refresh",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(cli, "AnthropicClaudeCodeOAuthClient", _Client)
    monkeypatch.setattr("builtins.input", lambda _prompt: "code#state")
    monkeypatch.setattr(
        "sys.argv",
        ["agent-harness", "login", "claude-code", "--auth-file", str(path)],
    )

    assert cli.main() == 0
    data = json.loads(path.read_text())
    assert data["claude-code"]["access"] == "access"


def test_cli_policy_list_revoke_and_clear(monkeypatch, tmp_path, capsys):
    path = tmp_path / "tool_policy.json"
    store = cli.ToolPolicyStore(path)
    rule = store.add_allow_rule(tool="shell", args={"cmd": "git status"})

    monkeypatch.setattr(
        "sys.argv",
        ["agent-harness", "policy", "list", "--policy-file", str(path)],
    )
    assert cli.main() == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["rules"][0]["id"] == rule.id

    monkeypatch.setattr(
        "sys.argv",
        ["agent-harness", "policy", "revoke", rule.id, "--policy-file", str(path)],
    )
    assert cli.main() == 0
    assert "Removed policy rule" in capsys.readouterr().out

    store.add_allow_rule(tool="shell", args={"cmd": "git status"})
    monkeypatch.setattr(
        "sys.argv",
        ["agent-harness", "policy", "clear", "--policy-file", str(path)],
    )
    assert cli.main() == 0
    assert "Removed 1 policy rule" in capsys.readouterr().out
