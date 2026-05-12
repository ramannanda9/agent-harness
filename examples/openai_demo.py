"""
examples/openai_demo.py — end-to-end against OpenAI + a real HTTP tool.

Wires up:
  - OpenAILLM (reads OPENAI_API_KEY from env; override with OPENAI_MODEL)
  - HTTPFetch builtin tool
  - A single-agent registry that uses http_fetch
  - In-memory semantic + episodic stores

Streams the run live to stdout so you can watch plan → action → observation →
synthesis as it happens.

    OPENAI_API_KEY=sk-... python examples/openai_demo.py
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
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.http_fetch import HTTPFetch

GOAL = (
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

    llm = OpenAILLM(model=model)

    tools = ToolRegistry().register(HTTPFetch())

    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="researcher",
            role="fetches a URL and reasons about its contents",
            system_prompt=(
                "You are a research assistant. You have one tool, `http_fetch`, "
                "which takes a `url` and returns the body. Fetch the URL the user "
                "asks about, then answer based on the actual response. "
                "Always use the ReAct JSON format below — never reply in plain prose."
            ),
            allowed_tools=["http_fetch"],
            max_steps=4,
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


if __name__ == "__main__":
    asyncio.run(main())
