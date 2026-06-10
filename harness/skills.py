"""Reusable prompt/context bundles for agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SKILLS_DIR = Path("~/.agent-harness/skills")


@dataclass(frozen=True)
class Skill:
    """Reusable instructions attached explicitly to an agent.

    Skills shape the agent prompt but do not grant tools. Tool access still
    comes only from ``AgentConfig.allowed_tools`` and the actual tool map.
    """

    name: str
    description: str
    instructions: str
    tool_hints: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"### {self.name}", f"Description: {self.description.strip()}"]
        if self.instructions.strip():
            lines.extend(["Instructions:", self.instructions.strip()])
        if self.tool_hints:
            lines.append("Useful tools: " + ", ".join(self.tool_hints))
        return "\n".join(line for line in lines if line)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "tool_hints": list(self.tool_hints),
        }


def load_skill(path: str | Path) -> Skill:
    """Load one Agent Skills-style directory or ``SKILL.md`` file.

    The loader intentionally treats tool declarations as hints only. A
    skill cannot grant permissions; agents still need tools wired into
    their tool map.
    """

    skill_path = Path(path).expanduser()
    if skill_path.is_dir():
        skill_path = skill_path / "SKILL.md"
    if skill_path.name != "SKILL.md":
        raise ValueError(f"skill path must be a skill directory or SKILL.md: {path}")
    if not skill_path.is_file():
        raise FileNotFoundError(skill_path)

    frontmatter, body = _parse_skill_markdown(skill_path.read_text())
    name = _required_string(frontmatter, "name", skill_path)
    description = _required_string(frontmatter, "description", skill_path)
    tool_hints = _string_list(
        frontmatter.get("tool_hints")
        or frontmatter.get("tool-hints")
        or frontmatter.get("allowed_tools")
        or frontmatter.get("allowed-tools")
    )
    return Skill(
        name=name,
        description=description,
        instructions=body.strip(),
        tool_hints=tool_hints,
    )


def default_skills_dir() -> Path:
    """Return the conventional user skill directory."""

    return DEFAULT_SKILLS_DIR.expanduser()


def load_skills(path: str | Path | None = None) -> list[Skill]:
    """Load every immediate child skill directory under ``path``.

    When ``path`` is omitted, loads from ``~/.agent-harness/skills`` and
    returns ``[]`` if that default directory is absent. Explicit paths
    still fail loudly when missing.
    """

    default_path = path is None
    root = default_skills_dir() if default_path else Path(path).expanduser()
    if root.is_file():
        return [load_skill(root)]
    if not root.is_dir():
        if default_path:
            return []
        raise FileNotFoundError(root)
    skills: list[Skill] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        skill_file = child / "SKILL.md"
        if child.is_dir() and skill_file.is_file():
            skills.append(load_skill(skill_file))
    return skills


def _parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    end = next((idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        raise ValueError("SKILL.md frontmatter is missing closing ---")
    return _parse_frontmatter(lines[1:end]), "\n".join(lines[end + 1 :])


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith((" ", "\t")) and raw.strip().startswith("- "):
            if current_key is None:
                raise ValueError("frontmatter list item has no key")
            data.setdefault(current_key, []).append(_unquote(raw.strip()[2:].strip()))
            continue
        if ":" not in raw:
            raise ValueError(f"invalid frontmatter line: {raw}")
        key, value = raw.split(":", 1)
        current_key = key.strip()
        value = value.strip()
        if not value:
            data[current_key] = []
        elif value.startswith("[") and value.endswith("]"):
            data[current_key] = [
                _unquote(item.strip()) for item in value[1:-1].split(",") if item.strip()
            ]
        else:
            data[current_key] = _unquote(value)
    return data


def _required_string(frontmatter: dict[str, Any], key: str, path: Path) -> str:
    value = frontmatter.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} frontmatter requires non-empty {key!r}")
    return value.strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("tool hints must be a string or list of strings")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
