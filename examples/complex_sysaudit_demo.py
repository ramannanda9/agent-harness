"""
examples/complex_sysaudit_demo.py — heterogeneous multi-agent project + system audit.

Three specialist agents with genuinely different tool sets work in parallel:

  shell_agent      — ah-executor shell tool: system resources, git status, process info
  filesystem_agent — MCP filesystem server: reads README, pyproject.toml, recent logs
  web_agent        — HTTPFetch: fetches PyPI metadata for the package

The dispatch classifier routes to the orchestrated path (planner → DAG → synthesis)
because the goal spans three heterogeneous tool domains. Tasks run in parallel where
dependencies allow; the synthesiser produces a combined project health report.

    OPENAI_API_KEY=sk-... python examples/complex_sysaudit_demo.py

Requires:
  cargo install --path executor          # shell tool
  pip install -e ".[openai,http,mcp]"    # adapters

Optional:
  OTEL_ENABLED=1   — send traces to Jaeger on localhost:4318
  PROJECT_DIR=...  — project root to audit (defaults to repo root)
  PYPI_PACKAGE=... — PyPI package name to fetch (defaults to agent-harness)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from agents.base import AgentConfig
from harness.events import EventType
from harness.executor_bridge import ExecutorBridge, ExecutorConfig, ExecutorTool, find_executor
from harness.llm.openai import OpenAILLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.http_fetch import HTTPFetch

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.5")
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).parent.parent)).resolve()
PYPI_PACKAGE = os.environ.get("PYPI_PACKAGE", "agent-harness")

GOAL = f"""Audit this project and machine. Investigate all three areas in parallel:

1. SYSTEM & GIT (shell_agent):
   - OS, kernel, CPU model, total/free memory, uptime
   - Git branch, last 5 commit messages, any uncommitted changes
   - Top 5 processes by memory usage

2. PROJECT FILES (filesystem_agent):
   - Read README.md: summarise what the project does in 2-3 sentences
   - Read pyproject.toml: list the package name, version, and all extras
   - Check if there are any .py files modified in the last 24 hours

3. PACKAGE REGISTRY (web_agent):
   - Fetch https://pypi.org/pypi/{PYPI_PACKAGE}/json and report the latest
     published version and its release date

Combine findings into a project health report. Flag anything noteworthy:
version drift between pyproject.toml and PyPI, uncommitted changes, high
memory pressure, or files changed recently."""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sep(char: str = "─", w: int = 72) -> str:
    return char * w


def _trunc(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[:n] + "…"


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(2)

    executor = find_executor()
    if not executor:
        print(
            "ERROR: ah-executor not found.\n  cargo install --path executor",
            file=sys.stderr,
        )
        sys.exit(2)

    # MCP filesystem server — npx required
    try:
        from mcp import StdioServerParameters

        from tools.mcp import MCPServerConnection
    except ImportError:
        print(
            "ERROR: MCP adapter not installed.\n  pip install -e '.[mcp]'",
            file=sys.stderr,
        )
        sys.exit(2)

    enable_otel = bool(os.environ.get("OTEL_ENABLED"))

    print(_sep("═"))
    print(f"Model:      {MODEL}")
    print(f"Project:    {PROJECT_DIR}")
    print(f"PyPI pkg:   {PYPI_PACKAGE}")
    print(f"OTEL:       {'enabled' if enable_otel else 'disabled'}")
    print(f"\nGoal:\n{GOAL}")
    print(_sep("═"))

    llm = OpenAILLM(model=MODEL)

    # ── Tools ─────────────────────────────────────────────────────────────────

    bridge = ExecutorBridge(ExecutorConfig(allowed_tools=("shell",)))
    shell_tool = ExecutorTool("shell", "shell", bridge)

    http_tool = HTTPFetch()

    mcp_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(PROJECT_DIR)],
    )

    async with MCPServerConnection(mcp_params, server_name="filesystem") as fs_conn:
        tools = ToolRegistry().register(shell_tool).register(http_tool)
        fs_conn.register_tools(tools)

        fs_tool_names = fs_conn.tool_names

        # ── Agents ────────────────────────────────────────────────────────────

        _react_suffix = (
            "Answer based on actual tool output — never guess. "
            "Use the ReAct JSON format — never reply in plain prose."
        )

        agents = (
            AgentRegistry()
            .register(
                AgentConfig(
                    agent_id="shell_agent",
                    role="gathers system info, git status, and process data using shell commands",
                    system_prompt=(
                        "You are a system analyst. Use the `shell` tool (takes `cmd`) "
                        "to run shell commands. Keep commands non-interactive and fast. "
                        + _react_suffix
                    ),
                    allowed_tools=["shell"],
                    max_steps=8,
                )
            )
            .register(
                AgentConfig(
                    agent_id="filesystem_agent",
                    role="reads and summarises local project files using MCP filesystem tools",
                    system_prompt=(
                        "You are a project analyst. Use the MCP filesystem tools to read "
                        "local files. Prefer read_file for specific files and search_files "
                        "for discovery. Stay within the allowed project directory. " + _react_suffix
                    ),
                    allowed_tools=fs_tool_names,
                    max_steps=8,
                )
            )
            .register(
                AgentConfig(
                    agent_id="web_agent",
                    role="fetches package metadata and external data from URLs",
                    system_prompt=(
                        "You are a web researcher. Use the `http_fetch` tool (takes `url`) "
                        "to retrieve remote content. Only fetch the URLs you are given. "
                        + _react_suffix
                    ),
                    allowed_tools=["http_fetch"],
                    max_steps=4,
                )
            )
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
                max_total_cost_usd=3.0,
                max_wall_time_seconds=180,
                max_replan_count=1,
                confidence_threshold=0.5,
            ),
            enable_otel=enable_otel,
        )

        # ── Stream ────────────────────────────────────────────────────────────

        task_results: list[dict] = []

        async for event in runtime.dispatch_stream(GOAL):
            if event.type == EventType.DISPATCH:
                print(
                    f"\n[dispatch]   complexity={event.payload['complexity']}"
                    f"  path={event.payload['path']}"
                )

            elif event.type == EventType.PLAN:
                tasks = event.payload.get("plan", {}).get("tasks", [])
                print(f"\n[plan]       {len(tasks)} tasks")
                for t in tasks:
                    deps = f"  ← {t['depends_on']}" if t.get("depends_on") else ""
                    print(
                        f"             {t['id']}@{t['agent_id']}: "
                        f"{_trunc(t['instruction'], 70)}{deps}"
                    )

            elif event.type == EventType.THOUGHT:
                thought = event.payload.get("thought", "")
                if thought:
                    print(f"[{event.agent_id:<16}] think   {_trunc(thought, 110)}")

            elif event.type == EventType.ACTION:
                args = json.dumps(event.payload["args"], default=str)
                print(f"[{event.agent_id:<16}] action  {event.payload['tool']}({_trunc(args, 90)})")

            elif event.type == EventType.OBSERVATION:
                obs = event.payload.get("observation", "")
                print(f"[{event.agent_id:<16}] obs     {_trunc(obs, 110)}")

            elif event.type == EventType.TASK_DONE:
                p = event.payload
                task_results.append(p)
                print(
                    f"[{event.agent_id:<16}] ✓ done  "
                    f"confidence={p.get('confidence', 0):.2f}  "
                    f"steps={p.get('steps', '?')}"
                )

            elif event.type == EventType.REPLAN:
                print(
                    f"\n[replan]     #{event.payload.get('replan_count')} — "
                    f"trigger={event.payload.get('trigger_task', '?')}"
                )

            elif event.type == EventType.SYNTHESIS:
                print(f"\n[synthesis]  confidence={event.payload.get('confidence', 0):.2f}")

            elif event.type == EventType.DONE:
                p = event.payload
                print("\n" + _sep("═"))
                print("PROJECT HEALTH REPORT")
                print(_sep("═"))
                print(p.get("answer", "(no answer)"))
                print(_sep())
                print(
                    f"Confidence: {p.get('confidence', 0):.2f}  |  "
                    f"Tasks: {len(task_results)}  |  "
                    f"Replans: {p.get('replan_count', 0)}  |  "
                    f"Cost: ${p.get('cost_usd', 0):.4f}  |  "
                    f"Time: {p.get('elapsed_seconds', 0):.1f}s"
                )

            elif event.type == EventType.ERROR:
                print(f"\n[error]      {event.error}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
