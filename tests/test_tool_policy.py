from __future__ import annotations

import json
import os

import pytest

from harness.hitl import _parse_stdin, is_allowed
from harness.tool_policy import ToolPolicyStore, match_for_tool_args


def test_match_for_shell_uses_first_command_word():
    assert match_for_tool_args("shell", {"cmd": "git status --short"}) == {
        "command_prefix": ["git"]
    }


def test_policy_store_adds_and_matches_shell_prefix(tmp_path):
    path = tmp_path / "tool_policy.json"
    store = ToolPolicyStore(path)

    rule = store.add_allow_rule(tool="shell", args={"cmd": "git status"})

    assert rule.tool == "shell"
    assert rule.match == {"command_prefix": ["git"]}
    assert store.is_allowed("shell", {"cmd": "git diff"})
    assert not store.is_allowed("shell", {"cmd": "rm -rf /tmp/x"})
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_policy_store_adds_and_matches_tool_name(tmp_path):
    store = ToolPolicyStore(tmp_path / "tool_policy.json")

    rule = store.add_allow_rule(tool="mcp.datadog.query_logs", args={"query": "error"})

    assert rule.match == {}
    assert store.is_allowed("mcp.datadog.query_logs", {"query": "anything"})
    assert not store.is_allowed("mcp.datadog.delete_monitor", {})


def test_policy_store_deduplicates_same_match(tmp_path):
    store = ToolPolicyStore(tmp_path / "tool_policy.json")

    first = store.add_allow_rule(tool="shell", args={"cmd": "git status"})
    second = store.add_allow_rule(tool="shell", args={"cmd": "git diff"})

    assert first.id == second.id
    assert len(store.list_rules()) == 1


def test_policy_store_revoke_and_clear(tmp_path):
    store = ToolPolicyStore(tmp_path / "tool_policy.json")
    rule = store.add_allow_rule(tool="shell", args={"cmd": "git status"})

    assert store.revoke(rule.id)
    assert not store.revoke(rule.id)
    assert store.list_rules() == []

    store.add_allow_rule(tool="shell", args={"cmd": "git status"})
    assert store.clear() == 1
    assert store.list_rules() == []


def test_policy_store_rejects_public_file_permissions(tmp_path):
    path = tmp_path / "tool_policy.json"
    path.write_text(json.dumps({"version": 1, "rules": []}))
    if os.name == "nt":
        pytest.skip("POSIX permissions only")
    path.chmod(0o644)

    with pytest.raises(PermissionError):
        ToolPolicyStore(path).list_rules()


def test_hitl_parse_stdin_supports_persistent_allow():
    response = _parse_stdin("approval-1", "A")

    assert response.approved is True
    assert response.persistent_allow is True
    assert response.session_allow is False


def test_hitl_is_allowed_uses_policy_file(tmp_path, monkeypatch):
    path = tmp_path / "tool_policy.json"
    monkeypatch.setenv("AGENT_HARNESS_TOOL_POLICY_FILE", str(path))
    ToolPolicyStore(path).add_allow_rule(tool="shell", args={"cmd": "git status"})

    assert is_allowed("shell", {"cmd": "git log"})
