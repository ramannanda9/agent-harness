"""
examples/complex_sysaudit_demo.py — heterogeneous multi-agent project + system audit.

Three specialist agents with genuinely different tool sets work in parallel:

  shell_agent      — ah-executor shell tool: system resources, git status, process info
  filesystem_agent — MCP filesystem server: reads README, pyproject.toml, recent logs
  web_agent        — HTTPFetch: fetches PyPI metadata for the package

The dispatch classifier routes to the orchestrated path (planner → DAG → synthesis)
because the goal spans three heterogeneous tool domains. Tasks run in parallel where
dependencies allow; the synthesiser produces a combined project health report.

HITL: the `shell` tool on shell_agent requires human approval before each command
runs. Type y to approve, n to reject, or any text to steer the agent instead.
If the process is interrupted, the banner prints the exact command to resume:

    python examples/complex_sysaudit_demo.py --resume <run_id>

    OPENAI_API_KEY=sk-... python examples/complex_sysaudit_demo.py

Steering: while the audit runs, type guidance for any agent using the
prefix `<agent_id>: <text>` (or `*: <text>` to broadcast). Examples:

    shell_agent: skip the process list, just do uptime
    filesystem_agent: also peek at .github/workflows
    *: wrap up and synthesise what you have

Guidance lands as a `Human guidance:` user message at the next step
boundary of the targeted agent. HITL prompts still take precedence when
they fire — the next stdin line after a banner is consumed by HITL.

Requires:
  cargo install --path executor          # shell tool
  pip install -e ".[openai,http,mcp]"    # adapters

Optional:
  OTEL_ENABLED=1          — send traces to Jaeger on localhost:4318
  PROJECT_DIR=...         — project root to audit (defaults to repo root)
  PYPI_PACKAGE=...        — PyPI package name to fetch (defaults to agent-harness)
  HITL_CHECKPOINT_DIR=... — override checkpoint directory (default ~/.agent-harness/checkpoints)
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
from harness.steering import stdin_steering_factory
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.http_fetch import HTTPFetch

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).parent.parent)).resolve()
PYPI_PACKAGE = os.environ.get("PYPI_PACKAGE", "agent-harness")

# Persistent store config — set these to enable durable memory across runs.
# Without them the demo falls back to in-memory stores (lost on exit).
REDIS_URL = os.environ.get("REDIS_URL")  # e.g. redis://localhost:6379
LANCE_PATH = os.environ.get("LANCE_PATH")  # e.g. ./lance_episodic

AUDIT_GOAL = f"""Audit this project and machine. Investigate all three areas in parallel:

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

FOLLOWUP_GOAL = (
    "Synthesise the recent project audit findings into a prioritised action list. "
    "No new data collection needed — reason from what is already known."
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sep(char: str = "─", w: int = 72) -> str:
    return char * w


def _trunc(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[:n] + "…"


async def _build_stores(llm):
    """
    Build semantic + episodic stores.

    Tries Redis (REDIS_URL) and LanceDB (LANCE_PATH) for durable memory.
    Falls back to in-memory stores when env vars are not set or deps missing.
    Returns (semantic_store, episodic_store, description_string).
    """
    semantic_store = None
    episodic_store = None
    labels = []

    if REDIS_URL:
        try:
            import redis.asyncio as redis

            from memory.redis_store import RedisSemanticStore

            client = redis.from_url(REDIS_URL, decode_responses=True)
            await client.ping()
            semantic_store = RedisSemanticStore(client, key_prefix="sysaudit:")
            labels.append(f"semantic=redis ({REDIS_URL})")
        except Exception as e:
            print(f"[memory]     Redis unavailable ({e}) — falling back to in-memory")

    if LANCE_PATH:
        try:
            from memory.episodic_lance import LanceDBEpisodicStore, LocalEmbedder

            store = LanceDBEpisodicStore(uri=LANCE_PATH, embedder=LocalEmbedder())
            await store.initialize()
            episodic_store = store
            labels.append(f"episodic=lancedb ({LANCE_PATH})")
        except Exception as e:
            print(f"[memory]     LanceDB unavailable ({e}) — falling back to in-memory")

    if semantic_store is None:
        semantic_store = InMemorySemanticStore()
        labels.append("semantic=in-memory")
    if episodic_store is None:
        episodic_store = InMemoryEpisodicStore()
        labels.append("episodic=in-memory")

    return semantic_store, episodic_store, "  ".join(labels)


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
    semantic_store, episodic_store, store_label = await _build_stores(None)

    from harness.checkpoint import FileCheckpointStore

    checkpoint_dir = FileCheckpointStore()._dir

    print(_sep("═"))
    print(f"Model:      {MODEL}")
    print(f"Project:    {PROJECT_DIR}")
    print(f"PyPI pkg:   {PYPI_PACKAGE}")
    print(f"OTEL:       {'enabled' if enable_otel else 'disabled'}")
    print(f"Memory:     {store_label}")
    print(f"HITL:       shell commands gated  (checkpoints → {checkpoint_dir})")
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

        _analyst_config = AgentConfig(
            agent_id="analyst_agent",
            role="synthesises findings and prioritises actions from memory — no tool calls needed",
            system_prompt=(
                "You are a technical analyst. You have no tools. "
                "Reason over the facts and past experience already in your context "
                "and produce a clear, prioritised answer. "
                "Use the ReAct JSON format — finish in one step."
            ),
            allowed_tools=[],
            max_steps=2,
        )

        # Pass 1 registry: data-collection agents only — analyst_agent is hidden
        # so the planner cannot assign it a synthesis task before memory exists.
        audit_agents = (
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
                    hitl_tools=["shell"],  # every shell command requires human approval
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

        # Pass 2 registry: analyst_agent only — routes directly, no planner needed.
        followup_agents = AgentRegistry().register(_analyst_config)

        memory = MemoryManager(
            semantic_store=semantic_store,
            episodic_store=episodic_store,
            llm=llm,
        )

        # Steering: one factory shared across both runtimes. The factory is
        # both a per-agent source factory AND an async context manager that
        # owns the shared StdinRouter — the AgentRuntime enters/exits it
        # automatically around dispatch_stream, so we don't manage the
        # router lifecycle here.
        steering_factory = stdin_steering_factory()

        def _make_runtime(registry: AgentRegistry) -> AgentRuntime:
            return AgentRuntime(
                agent_registry=registry,
                tool_registry=tools,
                memory=memory,
                llm=llm,
                guardrail_config=GuardrailConfig(
                    max_total_cost_usd=5.0,
                    max_wall_time_seconds=300,
                    max_replan_count=1,
                    confidence_threshold=0.5,
                ),
                enable_otel=enable_otel,
                steering_source_factory=steering_factory,
            )

        audit_runtime = _make_runtime(audit_agents)
        followup_runtime = _make_runtime(followup_agents)

        active_prefixes = ", ".join(audit_agents.all_ids())
        print("\n" + _sep("─"))
        print("STEERING — type `<agent_id>: <text>` or `*: <text>` at any time")
        print(f"Active agents: {active_prefixes}")
        print(_sep("─"))

        # ── Pass 1: Full audit (or transparent resume) ────────────────────────
        # dispatch_stream automatically resumes from --resume <key> when a
        # checkpoint store is configured — no special handling needed here.
        # The runtime auto-wraps dispatch_stream in the steering factory's
        # lifecycle, so the StdinRouter starts/stops without explicit
        # management here. HITL prompts take precedence — the next stdin
        # line after a banner is consumed by HITL; otherwise lines route
        # to the matching agent's steer queue.

        print(f"\nPASS 1 — full audit\nGoal: {_trunc(AUDIT_GOAL, 120)}")
        print(_sep("═"))

        task_results: list[dict] = []

        async for event in audit_runtime.dispatch_stream(AUDIT_GOAL):
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

            elif event.type == EventType.HUMAN_GUIDANCE:
                p = event.payload
                print(f"\n[{event.agent_id:<16}] ▶ steered  step={p['step']}  text={p['text']!r}")

            elif event.type == EventType.ERROR:
                print(f"\n[error]      {event.error}", file=sys.stderr)

        # ── Memory inspection ─────────────────────────────────────────────────
        # Show exactly what was extracted and stored so the second pass is
        # transparent — you can see what the agent will read from memory.

        print("\n" + _sep("─"))
        print("MEMORY WRITTEN AFTER PASS 1")
        print(_sep("─"))

        all_semantic = await semantic_store.search_prefix("")
        global_facts = {
            k: v
            for k, v in all_semantic.items()
            if not k.startswith("run:") and not k.startswith("agent:")
        }
        print(f"Semantic facts ({len(global_facts)}):")
        for k, v in global_facts.items():
            print(f"  {k}: {_trunc(str(v), 100)}")

        # Episodic search works across both in-memory and LanceDB stores.
        recent_episodes = await episodic_store.search(AUDIT_GOAL, top_k=5)
        print(f"\nEpisodic summaries ({len(recent_episodes)}):")
        for ep in recent_episodes:
            ts = ep.get("metadata", {}).get("timestamp", "?")
            print(f"  [{ts}]")
            print(f"  {_trunc(ep['text'], 200)}")

        # ── Pass 2: Follow-up from memory ─────────────────────────────────────
        # Memory is injected into the agent's system prompt via build_context.
        # Watch the thought: the agent should reason from stored facts rather
        # than calling tools again.

        print("\n" + _sep("═"))
        print(f"PASS 2 — follow-up (memory recall)\nGoal: {FOLLOWUP_GOAL}")
        print(_sep("═"))

        async for event in followup_runtime.dispatch_stream(FOLLOWUP_GOAL):
            if event.type == EventType.DISPATCH:
                print(
                    f"\n[dispatch]   complexity={event.payload['complexity']}"
                    f"  path={event.payload['path']}"
                )

            elif event.type == EventType.ROUTE:
                print(
                    f"[route]      → {event.payload['agent_id']}: "
                    f"{_trunc(event.payload['rationale'], 90)}"
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
                print("\n" + _sep("═"))
                print("PRIORITISED ACTION LIST (from memory)")
                print(_sep("═"))
                print(p.get("answer", "(no answer)"))
                print(_sep())
                print(
                    f"Confidence: {p.get('confidence', 0):.2f}  |  "
                    f"Steps: {p.get('steps', '?')}  "
                    f"(fewer steps = agent answered from memory, not tools)"
                )

            elif event.type == EventType.HUMAN_GUIDANCE:
                p = event.payload
                print(f"\n[{event.agent_id:<16}] ▶ steered  step={p['step']}  text={p['text']!r}")

            elif event.type == EventType.ERROR:
                print(f"\n[error]      {event.error}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
