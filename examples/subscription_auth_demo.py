"""
examples/subscription_auth_demo.py — run the harness through subscription/CLI auth.

This example is intentionally provider-bridge shaped:

  1. openai-codex
     Uses OpenAICodexLLM as a direct Codex backend adapter. It reads Pi-style
     OAuth credentials from an auth file and calls:

         https://chatgpt.com/backend-api/codex/responses

     By default it reads:

         ~/.agent-harness/auth/auth.json

     You can point it at an existing Pi auth file:

         OPENAI_CODEX_AUTH_FILE=~/.pi/agent/auth.json \
           python examples/subscription_auth_demo.py openai-codex

         python examples/subscription_auth_demo.py openai-codex

  2. claude-code
     Uses ClaudeCodeLLM as a direct Anthropic Messages adapter with Claude
     Pro/Max OAuth credentials. Log in first, then:

         agent-harness login claude-code

         python examples/subscription_auth_demo.py claude-code

No browser refresh tokens are scraped by this demo. The openai-codex path reads
an explicit auth file entry for `openai-codex`; the claude-code path reads an
explicit auth file entry for `claude-code`. The normal API-key adapters remain
the stable fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agents.base import AgentConfig
from harness.events import EventType
from harness.llm.claude_code import ClaudeCodeLLM
from harness.llm.openai_codex import OpenAICodexLLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

GOAL = (
    "Reply with one concise sentence confirming that this run used the configured "
    "credential-backed LLM provider."
)


def _truncate(s: str, n: int = 140) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _build_llm(provider: str):
    if provider == "openai-codex":
        return OpenAICodexLLM(
            model=os.environ.get("OPENAI_CODEX_MODEL", "gpt-5.5"),
            auth_file=Path(
                os.environ.get(
                    "OPENAI_CODEX_AUTH_FILE",
                    "~/.agent-harness/auth/auth.json",
                )
            ).expanduser(),
            base_url=os.environ.get(
                "OPENAI_CODEX_BASE_URL",
                "https://chatgpt.com/backend-api",
            ),
            request_timeout_seconds=float(os.environ.get("CODEX_TIMEOUT_SECONDS", "180")),
        )

    if provider == "claude-code":
        return ClaudeCodeLLM(
            model=os.environ.get("CLAUDE_CODE_MODEL", "claude-sonnet-4-6"),
            auth_file=Path(
                os.environ.get(
                    "CLAUDE_CODE_AUTH_FILE",
                    "~/.agent-harness/auth/auth.json",
                )
            ).expanduser(),
            base_url=os.environ.get("CLAUDE_CODE_BASE_URL", "https://api.anthropic.com"),
            request_timeout_seconds=float(os.environ.get("CLAUDE_CODE_TIMEOUT_SECONDS", "120")),
        )

    raise SystemExit(f"unknown provider: {provider}")


async def run(provider: str) -> dict:
    llm = _build_llm(provider)
    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="subscription_agent",
            role="verifies credential-backed provider wiring",
            system_prompt=(
                "You are a provider wiring smoke-test agent. "
                "Return valid ReAct JSON. Do not call tools. Finish immediately."
            ),
            allowed_tools=[],
            max_steps=2,
        )
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    runtime = AgentRuntime(
        agent_registry=agents,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=1.0,
            max_wall_time_seconds=120,
            max_replan_count=0,
        ),
    )

    final: dict = {}
    async for event in runtime.dispatch_stream(GOAL):
        if event.type == EventType.DISPATCH:
            print(f"[dispatch] {event.payload['path']} ({event.payload['complexity']})")
        elif event.type == EventType.ROUTE:
            print(
                f"[route]    {event.payload['agent_id']}: {_truncate(event.payload['rationale'])}"
            )
        elif event.type == EventType.THOUGHT:
            thought = event.payload.get("thought", "")
            if thought:
                print(f"[think]    {_truncate(thought)}")
        elif event.type == EventType.TASK_DONE:
            final = event.payload
        elif event.type == EventType.ERROR:
            print(f"[error]    {event.error}")
            final = {"error": event.error}

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-harness through subscription/CLI auth")
    parser.add_argument("provider", choices=["openai-codex", "claude-code"])
    args = parser.parse_args()

    result = asyncio.run(run(args.provider))
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
