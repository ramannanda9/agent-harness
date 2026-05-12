"""
examples/openai_demo.py — end-to-end against OpenAI + HTTP tool + ah-executor.

Wires up:
  - OpenAILLM (reads OPENAI_API_KEY from env; override with OPENAI_MODEL)
  - HTTPFetch builtin tool
  - shell tool via ah-executor (native backend) — skipped gracefully if not installed
  - A single-agent registry that uses both tools
  - In-memory semantic + episodic stores

The agent fetches a remote URL and runs a shell command in the same run,
demonstrating the LLM orchestrating two heterogeneous tool types without
any glue code.

Streams live to stdout: plan → action → observation → synthesis.

    OPENAI_API_KEY=sk-... python examples/openai_demo.py

Install ah-executor to enable the shell tool (optional):
    cargo install --path executor
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from agents.base import AgentConfig
from harness.events import EventType
from harness.executor_bridge import ExecutorBridge, ExecutorConfig, ExecutorTool, find_executor
from harness.llm.openai import OpenAILLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.http_fetch import HTTPFetch

_EXECUTOR = find_executor()

GOAL = (
    "What OS and kernel version is this machine running, and what UUID does "
    "https://httpbin.org/uuid return right now? Use shell for the OS check and "
    "http_fetch for the UUID."
    if _EXECUTOR else
    "Fetch https://httpbin.org/json and report the slideshow title and author "
    "from the JSON response."
)


def _truncate(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[:n] + "…"


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running this demo.", file=sys.stderr)
        sys.exit(2)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"Model: {model}\nGoal:  {GOAL}\n" + "─" * 60)

    # Demo cost_fn — these are gpt-4o-mini per-token rates as of mid-2026.
    # Users would point at a gateway via base_url=... in production for
    # gateway-reported authoritative cost; this is the no-gateway fallback.
    PRICING_PER_TOKEN = {
        "gpt-4o-mini": (0.15e-6, 0.60e-6),   # (input, output) USD / token
    }

    def cost_fn(usage: dict) -> float:
        # match dated variants by prefix: "gpt-4o-mini-2024-07-18" → "gpt-4o-mini"
        served = usage.get("model", "")
        for prefix, (in_rate, out_rate) in PRICING_PER_TOKEN.items():
            if served.startswith(prefix):
                return usage["tokens_in"] * in_rate + usage["tokens_out"] * out_rate
        return 0.0

    llm = OpenAILLM(model=model, cost_fn=cost_fn)

    tools = ToolRegistry().register(HTTPFetch())
    allowed_tools = ["http_fetch"]

    if _EXECUTOR:
        bridge = ExecutorBridge(ExecutorConfig(allowed_tools=("shell",)))
        tools.register(ExecutorTool("shell", "shell", bridge))
        allowed_tools.append("shell")
        print(f"ah-executor: {_EXECUTOR} (shell tool enabled)")
    else:
        print("ah-executor: not found — shell tool disabled (cargo install --path executor to enable)")

    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="researcher",
            role="fetches URLs and runs shell commands to answer questions",
            system_prompt=(
                "You are a research assistant. "
                "Use `http_fetch` (takes `url`) to retrieve remote content. "
                "Use `shell` (takes `cmd`) to run shell commands on the local machine. "
                "Answer based on actual tool output — never guess. "
                "Always use the ReAct JSON format below — never reply in plain prose."
            ),
            allowed_tools=allowed_tools,
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

    final: dict = {}
    async for event in runtime.run_stream(GOAL):
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
        elif event.type == EventType.REPLAN:
            print(f"[replan]    #{event.payload['replan_count']}")
        elif event.type == EventType.SYNTHESIS:
            print(f"[synthesis] confidence={event.payload.get('confidence')}")
        elif event.type == EventType.DONE:
            final = event.payload
        elif event.type == EventType.ERROR:
            print(f"[error]     {event.agent_id}: {event.error}", file=sys.stderr)

    print("─" * 60)
    print(f"Final answer:\n{final.get('answer', '(no answer)')}")
    print(f"Confidence:  {final.get('confidence')}")
    print(f"Replans:     {final.get('replan_count')}")
    print(f"Elapsed:     {final.get('elapsed_seconds', 0):.2f}s")
    print(f"Cost:        ${final.get('cost_usd', 0):.6f}  "
          f"(computed via local cost_fn; route through a gateway for authoritative cost)")


if __name__ == "__main__":
    asyncio.run(main())
