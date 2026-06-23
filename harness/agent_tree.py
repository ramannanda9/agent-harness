"""Agent-tree traversal shared across the persistent-agent concerns.

A coordinator ``BaseAgent`` may have ``SubAgentTool``s wired in as tools, each
wrapping another ``BaseAgent`` (which may itself delegate further). Model
switching, background tasks, and guard assignment all need to walk this tree.
These pure helpers are the single place that knows its shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.base import BaseAgent


def subagent_tool_agent(tool: Any) -> BaseAgent | None:
    """Return the ``BaseAgent`` wrapped by a ``SubAgentTool``, else ``None``."""
    from agents.base import BaseAgent  # noqa: PLC0415 — avoid import cycle at module load

    if tool.__class__.__name__ != "SubAgentTool":
        return None
    agent = getattr(tool, "_agent", None)
    return agent if isinstance(agent, BaseAgent) else None


def iter_agents(coordinator: BaseAgent) -> list[BaseAgent]:
    """Return the coordinator and every sub-agent reachable through its tools,
    depth-first, each visited once."""
    agents: list[BaseAgent] = []
    seen: set[int] = set()

    def visit(agent: BaseAgent) -> None:
        if id(agent) in seen:
            return
        seen.add(id(agent))
        agents.append(agent)
        for tool in getattr(agent, "_tools", {}).values():
            sub = subagent_tool_agent(tool)
            if sub is not None:
                visit(sub)

    visit(coordinator)
    return agents


def find_agent(coordinator: BaseAgent, agent_id: str) -> BaseAgent | None:
    """Return the reachable agent whose ``config.agent_id`` matches, else ``None``."""
    for agent in iter_agents(coordinator):
        if agent.config.agent_id == agent_id:
            return agent
    return None
