"""
examples/mcp_demo.py — agent-harness + MCP server tools.

Connects to an MCP server (e.g. the filesystem server) and lets an agent
use MCP-provided tools alongside harness builtins.

Usage:
    # Filesystem server (requires npx / Node.js)
    OPENAI_API_KEY=sk-... python examples/mcp_demo.py

    # Custom MCP server
    OPENAI_API_KEY=sk-... MCP_COMMAND="python my_server.py" python examples/mcp_demo.py

    # Override the goal
    OPENAI_API_KEY=sk-... MCP_GOAL="list files in /tmp" python examples/mcp_demo.py
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
from tools.mcp import MCPServerConnection

DEFAULT_GOAL = "List the files in the current directory and summarise what you find."

_renderer = ConsoleRenderer()


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running this demo.", file=sys.stderr)
        sys.exit(2)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    goal = os.environ.get("MCP_GOAL", DEFAULT_GOAL)

    # MCP server command — default to the filesystem server
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mcp_command = os.environ.get("MCP_COMMAND", "npx")
    mcp_args = os.environ.get(
        "MCP_ARGS",
        f"-y @modelcontextprotocol/server-filesystem {project_root}",
    ).split()

    print(f"Model:      {model}")
    print(f"MCP server: {mcp_command} {' '.join(mcp_args)}")
    print(f"Goal:       {goal}")
    print("─" * 60)

    # ── Connect to MCP server ─────────────────────────────────────────────
    from mcp import StdioServerParameters

    server_params = StdioServerParameters(
        command=mcp_command,
        args=mcp_args,
    )

    async with MCPServerConnection(server_params, server_name="demo") as conn:
        print(f"MCP tools:  {conn.tool_names}")
        print("─" * 60)

        # ── Wire up the harness ───────────────────────────────────────────
        llm = OpenAILLM(model=model)

        tools = ToolRegistry()
        conn.register_tools(tools)

        # Build allowed_tools list from discovered MCP tools
        agents = AgentRegistry().register(
            AgentConfig(
                agent_id="explorer",
                role="explores a filesystem or data source using MCP tools",
                system_prompt=(
                    "You are a filesystem explorer. Use the tools available to you "
                    "to investigate the filesystem and answer the user's question. "
                    "Always use the ReAct JSON format — never reply in plain prose."
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
            enable_otel=True,
        )

        # ── Run ───────────────────────────────────────────────────────────
        # Press Esc during the run to cancel cleanly.
        cancelled, terminal = await _renderer.render_stream(
            runtime.run_stream(goal),
            terminal_event_type=EventType.DONE,
        )
        if cancelled:
            return
        final: dict = terminal.payload if terminal else {}

        print("─" * 60)
        print(f"Final answer:\n{final.get('answer', '(no answer)')}")
        print(f"Confidence:  {final.get('confidence')}")


if __name__ == "__main__":
    asyncio.run(main())
