"""Provider-neutral reasoning-effort configuration."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

ReasoningEffort: TypeAlias = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
OPENAI_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
ANTHROPIC_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def validate_reasoning_effort(
    effort: str | None,
    *,
    provider: str = "harness",
    allowed: frozenset[str] = REASONING_EFFORTS,
) -> ReasoningEffort | None:
    """Validate and normalize one reasoning-effort value."""
    if effort is None:
        return None
    normalized = effort.strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"unsupported reasoning_effort {effort!r} for {provider}; expected one of: {choices}"
        )
    return cast(ReasoningEffort, normalized)
