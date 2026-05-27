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
import json
import os
import sys

from agents.base import AgentConfig
from harness.events import EventType
from harness.llm.openai import OpenAILLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from harness.utils import stream_tokens_inline
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.mcp import MCPServerConnection

DEFAULT_GOAL = "List the files in the current directory and summarise what you find."


def _truncate(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[:n] + "…"


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
        final: dict = {}
        async for event in stream_tokens_inline(runtime.run_stream(goal), show_agent_id=True):
            if event.type == EventType.PLAN:
                for t in event.payload["plan"].get("tasks", []):
                    print(f"[plan]      {t['id']}@{t['agent_id']}: {_truncate(t['instruction'])}")
            elif event.type == EventType.THOUGHT:
                thought = event.payload.get("thought", "")
                if thought:
                    print(f"[thought]   {event.agent_id}: {_truncate(thought)}")
            elif event.type == EventType.ACTION:
                args = json.dumps(event.payload["args"], default=str)
                print(f"[action]    {event.agent_id}: {event.payload['tool']}({_truncate(args)})")
            elif event.type == EventType.OBSERVATION:
                obs = event.payload.get("observation", "")
                print(f"[observe]   {event.agent_id}: {_truncate(obs)}")
            elif event.type == EventType.TASK_DONE:
                print(
                    f"[task_done] {event.agent_id} "
                    f"success={event.payload['success']} "
                    f"confidence={event.payload['confidence']}"
                )
            elif event.type == EventType.DONE:
                final = event.payload
            elif event.type == EventType.ERROR:
                print(f"[error]     {event.agent_id}: {event.error}", file=sys.stderr)

        print("─" * 60)
        print(f"Final answer:\n{final.get('answer', '(no answer)')}")
        print(f"Confidence:  {final.get('confidence')}")


if __name__ == "__main__":
    asyncio.run(main())
