"""Persistent tool policy rules for HITL approvals."""

from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POLICY_VERSION = 1


@dataclass(frozen=True)
class ToolPolicyRule:
    id: str
    effect: str
    tool: str
    match: dict[str, Any]
    scope: str = "user"
    created_at: str = ""
    created_by: str = "hitl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "effect": self.effect,
            "tool": self.tool,
            "match": self.match,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }


def default_policy_file() -> Path:
    configured = os.environ.get("AGENT_HARNESS_TOOL_POLICY_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path("~/.agent-harness/policies/tool_policy.json").expanduser()


class ToolPolicyStore:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_policy_file()

    def list_rules(self) -> list[ToolPolicyRule]:
        data = self._read()
        rules = data.get("rules", [])
        if not isinstance(rules, list):
            return []
        return [rule for item in rules if (rule := _rule_from_dict(item)) is not None]

    def is_allowed(self, tool: str, args: dict[str, Any]) -> bool:
        return any(rule_matches(rule, tool, args) for rule in self.list_rules())

    def add_allow_rule(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        created_by: str = "hitl",
    ) -> ToolPolicyRule:
        match = match_for_tool_args(tool, args)
        existing = self.find_allow_rule(tool=tool, match=match)
        if existing is not None:
            return existing

        rule = ToolPolicyRule(
            id=_rule_id(tool, match),
            scope="user",
            effect="allow",
            tool=tool,
            match=match,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
        )
        data = self._read()
        rules = data.get("rules")
        if not isinstance(rules, list):
            rules = []
        rules.append(rule.to_dict())
        self._write({"version": POLICY_VERSION, "rules": rules})
        return rule

    def find_allow_rule(self, *, tool: str, match: dict[str, Any]) -> ToolPolicyRule | None:
        for rule in self.list_rules():
            if rule.effect == "allow" and rule.tool == tool and rule.match == match:
                return rule
        return None

    def revoke(self, rule_id: str) -> bool:
        data = self._read()
        rules = data.get("rules", [])
        if not isinstance(rules, list):
            return False
        kept = [rule for rule in rules if not isinstance(rule, dict) or rule.get("id") != rule_id]
        if len(kept) == len(rules):
            return False
        self._write({"version": POLICY_VERSION, "rules": kept})
        return True

    def clear(self) -> int:
        count = len(self.list_rules())
        self._write({"version": POLICY_VERSION, "rules": []})
        return count

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": POLICY_VERSION, "rules": []}
        _validate_private_file(self.path)
        data = json.loads(self.path.read_text())
        return data if isinstance(data, dict) else {"version": POLICY_VERSION, "rules": []}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        if os.name != "nt":
            self.path.chmod(0o600)


def match_for_tool_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool in ("shell", "bash", "run", "exec"):
        cmd = (args.get("cmd") or args.get("command") or "").strip()
        prefix = cmd.split()[0] if cmd else None
        return {"command_prefix": [prefix]} if prefix else {}
    return {}


def rule_matches(rule: ToolPolicyRule, tool: str, args: dict[str, Any]) -> bool:
    if rule.scope != "user" or rule.effect != "allow" or rule.tool != tool:
        return False
    expected = rule.match.get("command_prefix")
    if expected is not None:
        if not isinstance(expected, list) or not expected:
            return False
        cmd = (args.get("cmd") or args.get("command") or "").strip()
        actual = cmd.split()[0] if cmd else None
        return actual == expected[0]
    return True


def _rule_id(tool: str, match: dict[str, Any]) -> str:
    if "command_prefix" in match:
        prefix = str(match["command_prefix"][0]).replace("/", "-")
        return f"{tool}-{prefix}-{uuid.uuid4().hex[:8]}"
    return f"{tool}-{uuid.uuid4().hex[:8]}"


def _rule_from_dict(item: Any) -> ToolPolicyRule | None:
    if not isinstance(item, dict):
        return None
    rule_id = item.get("id")
    effect = item.get("effect")
    tool = item.get("tool")
    match = item.get("match")
    if not all(isinstance(value, str) and value for value in (rule_id, effect, tool)):
        return None
    if not isinstance(match, dict):
        return None
    scope = item.get("scope") if isinstance(item.get("scope"), str) else "user"
    created_at = item.get("created_at") if isinstance(item.get("created_at"), str) else ""
    created_by = item.get("created_by") if isinstance(item.get("created_by"), str) else "unknown"
    return ToolPolicyRule(
        id=rule_id,
        scope=scope,
        effect=effect,
        tool=tool,
        match=match,
        created_at=created_at,
        created_by=created_by,
    )


def _validate_private_file(path: Path) -> None:
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"{path} must not be readable or writable by group/other")
