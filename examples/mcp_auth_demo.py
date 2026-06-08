"""
examples/mcp_auth_demo.py — authenticated remote MCP server tools.

Connects to a remote MCP server and injects auth through MCPServerConnection.

Usage:
    # Remote MCP with a bearer token
    OPENAI_API_KEY=sk-... \
      MCP_URL="https://example.com/mcp/sse" MCP_BEARER_TOKEN=... \
      python examples/mcp_auth_demo.py

    # Remote MCP with OAuth bearer token read from auth.json
    OPENAI_API_KEY=sk-... \
      MCP_URL="https://example.com/mcp/sse" MCP_AUTH_PROVIDER="datadog-mcp" \
      python examples/mcp_auth_demo.py

    # Override the goal
    OPENAI_API_KEY=sk-... MCP_GOAL="list available tools" python examples/mcp_auth_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents.base import AgentConfig
from harness.console import ConsoleRenderer
from harness.events import EventType
from harness.llm.openai import OpenAILLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.mcp import MCPServerConnection, OAuthMCPAuth, StaticMCPAuth

DEFAULT_GOAL = "List the available MCP tools and summarise what each one can do."

_renderer = ConsoleRenderer()


def _build_mcp_auth():
    if os.environ.get("MCP_AUTH_PROVIDER"):
        return OAuthMCPAuth.from_auth_file(
            os.environ.get("MCP_AUTH_FILE", "~/.agent-harness/auth/auth.json"),
            provider=os.environ["MCP_AUTH_PROVIDER"],
        )

    if os.environ.get("MCP_BEARER_TOKEN"):
        return StaticMCPAuth(headers={"Authorization": f"Bearer {os.environ['MCP_BEARER_TOKEN']}"})

    print(
        "ERROR: set MCP_BEARER_TOKEN or MCP_AUTH_PROVIDER for remote MCP auth.",
        file=sys.stderr,
    )
    sys.exit(2)


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running this demo.", file=sys.stderr)
        sys.exit(2)
    if not os.environ.get("MCP_URL"):
        print("ERROR: set MCP_URL to a remote MCP server URL.", file=sys.stderr)
        sys.exit(2)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    goal = os.environ.get("MCP_GOAL", DEFAULT_GOAL)
    mcp_url = os.environ["MCP_URL"]
    mcp_auth = _build_mcp_auth()

    print(f"Model:      {model}")
    print(f"MCP server: {mcp_url}")
    print("MCP auth:   configured")
    print(f"Goal:       {goal}")
    _renderer.sep()

    async with MCPServerConnection({"url": mcp_url}, server_name="remote", auth=mcp_auth) as conn:
        print(f"MCP tools:  {conn.tool_names}")
        _renderer.sep()

        llm = OpenAILLM(model=model)
        tools = ToolRegistry()
        conn.register_tools(tools)

        agents = AgentRegistry().register(
            AgentConfig(
                agent_id="remote_mcp_agent",
                role="uses authenticated remote MCP tools",
                system_prompt=(
                    "You use remote MCP tools to answer the user's question. "
                    "Always use the ReAct JSON format and never reply in plain prose."
                ),
                allowed_tools=conn.tool_names,
                max_steps=6,
                working_memory_max_tokens=8000,
            ),
        )

        memory = MemoryManager(
            semantic_store=InMemorySemanticStore(),
            episodic_store=InMemoryEpisodicStore(),
            llm=llm,
        )

        runtime = AgentRuntime(
            agent_registry=agents,
            tool_registry=tools,
            memory=memory,
            llm=llm,
            guardrail_config=GuardrailConfig(
                max_total_cost_usd=1.0,
                max_wall_time_seconds=90,
                max_replan_count=1,
                confidence_threshold=0.5,
            ),
        )

        # Press Esc during the run to cancel cleanly.
        cancelled, terminal = await _renderer.render_stream(
            runtime.run_stream(goal),
            terminal_event_type=EventType.DONE,
        )
        if cancelled:
            return
        final: dict = terminal.payload if terminal else {}
        _renderer.sep()
        print(f"Final answer:\n{final.get('answer', '(no answer)')}")
        print(f"Confidence:  {final.get('confidence')}")


if __name__ == "__main__":
    asyncio.run(main())
