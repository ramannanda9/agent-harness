from __future__ import annotations

import pytest

from harness import skills as skills_module
from harness.skills import Skill, load_skill, load_skills


def test_load_skill_from_directory_parses_frontmatter_and_body(tmp_path):
    skill_dir = tmp_path / "web-research"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: web-research
description: Research current information from primary sources.
allowed-tools:
  - browser_navigate
  - browser_snapshot
---

Prefer primary sources.
Capture dates.
"""
    )

    skill = load_skill(skill_dir)

    assert skill == Skill(
        name="web-research",
        description="Research current information from primary sources.",
        instructions="Prefer primary sources.\nCapture dates.",
        tool_hints=["browser_navigate", "browser_snapshot"],
    )


def test_load_skill_accepts_inline_tool_hints_and_quoted_values(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        """---
name: "code-review"
description: "Review code for regressions."
tool_hints: ["read_file", "pytest"]
---
Prioritize bugs and missing tests.
"""
    )

    skill = load_skill(skill_file)

    assert skill.name == "code-review"
    assert skill.description == "Review code for regressions."
    assert skill.tool_hints == ["read_file", "pytest"]
    assert skill.instructions == "Prioritize bugs and missing tests."


def test_load_skills_discovers_immediate_skill_directories(tmp_path):
    for name in ["b-skill", "a-skill"]:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"""---
name: {name}
description: {name} description.
---
Use {name}.
"""
        )
    ignored = tmp_path / "not-a-skill"
    ignored.mkdir()

    skills = load_skills(tmp_path)

    assert [skill.name for skill in skills] == ["a-skill", "b-skill"]


def test_load_skills_without_path_uses_default_user_directory(tmp_path, monkeypatch):
    root = tmp_path / "home-skills"
    skill_dir = root / "default-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: default-skill
description: Loaded from the default user skill directory.
---
Default instructions.
"""
    )
    monkeypatch.setattr(skills_module, "DEFAULT_SKILLS_DIR", root)

    skills = load_skills()

    assert [skill.name for skill in skills] == ["default-skill"]


def test_load_skills_without_path_returns_empty_when_default_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_module, "DEFAULT_SKILLS_DIR", tmp_path / "missing")

    assert load_skills() == []


def test_load_skill_requires_name_and_description(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        """---
name: incomplete
---
Body.
"""
    )

    with pytest.raises(ValueError, match="description"):
        load_skill(skill_file)


def test_load_skill_requires_frontmatter(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("No frontmatter.")

    with pytest.raises(ValueError, match="frontmatter"):
        load_skill(skill_file)
