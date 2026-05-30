"""
examples/vision_demo.py — vision agent: fetch images, describe them, store findings.

A single agent with fetch_image + http_fetch fetches two public images in
parallel, describes what it sees in each using the LLM's vision capability,
then synthesises a combined report.

What this demonstrates:
  - fetch_image returning image_url content blocks
  - WorkingMemory passing image blocks through to a vision-capable LLM
  - OBSERVATION events showing "[image]" instead of raw base64
  - write_run_end capturing text descriptions — no binary data in memory stores

    OPENAI_API_KEY=sk-... python examples/vision_demo.py

Requires a vision-capable model (gpt-4o by default; override with OPENAI_MODEL).

Install:
    pip install -e ".[openai,http]"

Optional — customise the images via env vars:
    IMAGE_URL_1=https://... IMAGE_URL_2=https://... python examples/vision_demo.py
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
from tools.builtin.fetch_image import FetchImage
from tools.builtin.http_fetch import HTTPFetch

# ── Config ────────────────────────────────────────────────────────────────────

# gpt-5.4-mini supports vision.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

# Public domain test images — stable Wikipedia/httpbin URLs.
# httpbin /image/jpeg → JPEG of a horse; /image/png → PNG badge.
IMAGE_URL_1 = os.environ.get("IMAGE_URL_1", "https://httpbin.org/image/jpeg")
IMAGE_URL_2 = os.environ.get("IMAGE_URL_2", "https://httpbin.org/image/png")

GOAL = (
    f"Fetch these two images in parallel and describe what you see in each:\n"
    f"  1. {IMAGE_URL_1}\n"
    f"  2. {IMAGE_URL_2}\n"
    "Then combine the descriptions into a brief two-paragraph report. "
    "Note any colours, subjects, or artistic style that stand out."
)


_renderer = ConsoleRenderer()


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(2)

    _renderer.sep("═")
    print(f"Model:   {MODEL}")
    print(f"Image 1: {IMAGE_URL_1}")
    print(f"Image 2: {IMAGE_URL_2}")
    _renderer.sep("═")

    llm = OpenAILLM(model=MODEL)

    tools = ToolRegistry().register(FetchImage()).register(HTTPFetch())

    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="vision_agent",
            role="fetches images and describes their visual content",
            system_prompt=(
                "You are a visual analyst. "
                "Use `fetch_image` (takes `url`, optional `detail`: auto/low/high) "
                "to retrieve images — the result is an image you can see directly. "
                "Use parallel actions when fetching multiple independent images. "
                "Describe what you observe in concrete, specific terms. "
                "Use the ReAct JSON format — never reply in plain prose."
            ),
            allowed_tools=["fetch_image", "http_fetch"],
            max_steps=6,
            working_memory_max_tokens=16_000,  # images are ~500 tokens each in budget
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
            max_total_cost_usd=1.0,
            max_wall_time_seconds=60,
            max_replan_count=0,
            confidence_threshold=0.5,
        ),
    )

    print(f"\nGoal: {GOAL}\n")
    _renderer.sep()

    final: dict = {}
    async for event in runtime.dispatch_stream(GOAL):
        if event.type == EventType.TASK_DONE:
            final = event.payload
        else:
            _renderer.render(event)

    _renderer.sep("═")
    print("VISION REPORT")
    _renderer.sep("═")
    print(final.get("answer", "(no answer)"))
    _renderer.sep()
    print(
        f"Confidence: {final.get('confidence', 0):.2f}  |  "
        f"Steps: {final.get('steps', '?')}  |  "
        f"Summarizations: {final.get('metadata', {}).get('summarizations', 0)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
