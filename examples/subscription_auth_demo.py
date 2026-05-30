"""
examples/subscription_auth_demo.py — run the harness through subscription/CLI auth.

> WARNING: these adapters bridge subscription OAuth (ChatGPT Plus/Pro,
> Claude Pro/Max) into the harness by talking to undocumented CLI
> endpoints. They may violate provider ToS and can result in account
> suspension. See the "Subscription adapters" section in the README for
> the full caveat — and prefer `OpenAILLM` with `OPENAI_API_KEY` (or the
> Anthropic Messages API with an API key) for anything beyond personal
> research on accounts you own.

Both adapters stream incrementally, so this demo prints tokens as they
arrive (the dots/text streaming under `[stream]` come straight from the
SSE delta events).

  1. openai-codex
     OpenAICodexLLM is a direct Codex backend adapter. Reads OAuth from
     an auth file and calls:

         https://chatgpt.com/backend-api/codex/responses

     Login (writes ~/.agent-harness/auth/auth.json):

         agent-harness login openai-codex

         python examples/subscription_auth_demo.py openai-codex

     Point at an existing Pi auth file:

         OPENAI_CODEX_AUTH_FILE=~/.pi/agent/auth.json \
           python examples/subscription_auth_demo.py openai-codex

  2. claude-code
     ClaudeCodeLLM is a direct Anthropic Messages adapter with Claude
     Pro/Max OAuth credentials. Log in first:

         agent-harness login claude-code

         python examples/subscription_auth_demo.py claude-code
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agents.base import AgentConfig
from harness.console import ConsoleRenderer
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

_renderer = ConsoleRenderer()


def _auth_path(provider: str) -> Path:
    env_key = "OPENAI_CODEX_AUTH_FILE" if provider == "openai-codex" else "CLAUDE_CODE_AUTH_FILE"
    return Path(os.environ.get(env_key, "~/.agent-harness/auth/auth.json")).expanduser()


def _check_auth(provider: str) -> None:
    """Fail fast with a useful message if the user hasn't logged in yet."""
    import json

    path = _auth_path(provider)
    if not path.exists():
        print(
            f"No auth file at {path}.\nRun: agent-harness login {provider}\nThen re-run this demo.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"Could not read {path}: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    if not isinstance(data, dict) or provider not in data:
        print(
            f"{path} has no entry for {provider!r}.\nRun: agent-harness login {provider}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_llm(provider: str):
    if provider == "openai-codex":
        return OpenAICodexLLM(
            model=os.environ.get("OPENAI_CODEX_MODEL", "gpt-5.5"),
            auth_file=_auth_path(provider),
            base_url=os.environ.get(
                "OPENAI_CODEX_BASE_URL",
                "https://chatgpt.com/backend-api",
            ),
            request_timeout_seconds=float(os.environ.get("CODEX_TIMEOUT_SECONDS", "180")),
        )

    if provider == "claude-code":
        return ClaudeCodeLLM(
            model=os.environ.get("CLAUDE_CODE_MODEL", "claude-sonnet-4-6"),
            auth_file=_auth_path(provider),
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
        if event.type == EventType.TASK_DONE:
            final = event.payload
        elif event.type == EventType.ERROR:
            _renderer.render(event)
            final = {"error": event.error}
        else:
            _renderer.render(event)

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-harness through subscription/CLI auth")
    parser.add_argument("provider", choices=["openai-codex", "claude-code"])
    args = parser.parse_args()

    _check_auth(args.provider)
    result = asyncio.run(run(args.provider))
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
