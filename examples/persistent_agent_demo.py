"""Persistent coordinator chat with SQLite-backed session state.

The persistent wrapper does not build sub-agents or MCP tools for you. Wire
agents exactly as usual, including ``SubAgentTool`` and MCP adapters/auth, then
wrap the top-level coordinator with ``PersistentAgent``.

The researcher sub-agent in this demo drives a real Chromium browser via
``@playwright/mcp`` so it can handle JS-rendered pages and bot-walled news
sites (Reuters, NYT, AP, etc.) that plain HTTP fetches can't reach. The
first run downloads ~150 MB of Chromium; subsequent runs reuse it.

Requirements:
    - ``npx`` on PATH (Node 18+)
    - On first run, Playwright auto-installs Chromium into its cache

Run:
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --session-id pr-review
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --new-session
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --show-capabilities
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --headed   # see the browser
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from mcp import StdioServerParameters

from agents.base import AgentConfig, BaseAgent
from harness.console import ConsoleRenderer
from harness.events import EventType
from harness.persistent import PersistentAgent, PersistentAgentConfig, SQLiteSessionStore
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool
from tools.mcp import MCPServerConnection

SYSTEM_PROMPT = (
    "You are a persistent coordinator. Wait for the user's turn, then decide "
    "whether to answer directly or delegate to researcher. Use recent session "
    "context for references like 'above', but avoid treating old tool outputs "
    "as current facts unless you re-check them."
)

RESEARCHER_SYSTEM_PROMPT = (
    "You are a research sub-agent driving a real Chromium browser via MCP. "
    "Typical flow: `browser_navigate(url=...)` then `browser_snapshot()` to "
    "read the page's accessibility tree (cleaner than raw HTML — includes "
    "headings, links, and visible text). For multi-page reading, "
    "`browser_navigate` to each URL in turn. Real-browser headers + JS "
    "execution mean Reuters / AP / NYT / WSJ all work, not just static "
    "sites. Finish with the requested information cited to its URL; if a "
    "site genuinely fails (paywall, CAPTCHA, navigation timeout) say so "
    "rather than guessing."
)


def _build_agent(
    *,
    agent_id: str,
    role: str,
    system_prompt: str,
    tools: dict,
    llm,
    memory: MemoryManager,
    guard: BudgetGuard,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            allowed_tools=list(tools.keys()),
            max_steps=6,
        ),
        tools=tools,
        memory=memory,
        tracer=Tracer(),
        guard=guard,
        llm=llm,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a persistent coordinator chat session.")
    parser.add_argument(
        "--session-id",
        default=os.environ.get("AGENT_SESSION_ID", "default"),
        help="Session/thread id to resume or create. Defaults to AGENT_SESSION_ID or 'default'.",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Create a new random session id and print it.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("AGENT_SESSION_DB", ".agent_harness_sessions.sqlite"),
        help="SQLite session database path. Defaults to AGENT_SESSION_DB or .agent_harness_sessions.sqlite.",
    )
    parser.add_argument(
        "--show-capabilities",
        action="store_true",
        help="Print coordinator, sub-agent, and MCP tool wiring before chat starts.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Run Playwright with a visible browser window. Useful for watching "
            "what the researcher does; defaults to headless."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(2)
    if shutil.which("npx") is None:
        print(
            "ERROR: npx not found on PATH. Install Node 18+ — "
            "`@playwright/mcp` is spawned via npx.",
            file=sys.stderr,
        )
        sys.exit(2)

    from harness.llm.openai import OpenAILLM  # noqa: PLC0415

    llm = OpenAILLM(model=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
        memory_scope="persistent-demo",
        memory_subject="persistent-demo",
    )
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=2.0, max_wall_time_seconds=120))

    # ── Playwright MCP ────────────────────────────────────────────────────
    # ``--isolated`` keeps Chromium state ephemeral so sessions don't bleed
    # cookies/storage into each other. ``--headless`` is the default; pass
    # ``--headed`` on the CLI to watch the researcher drive a visible
    # browser window — useful for debugging.
    playwright_args = ["-y", "@playwright/mcp@latest", "--isolated"]
    if not args.headed:
        playwright_args.append("--headless")
    playwright_params = StdioServerParameters(command="npx", args=playwright_args)

    async with MCPServerConnection(playwright_params, server_name="playwright") as browser:
        # Each MCPToolAdapter has a ``name``; build the dict BaseAgent
        # consumes. Tool names are ``browser_navigate``,
        # ``browser_snapshot``, ``browser_click``, ``browser_evaluate``,
        # etc. — see @playwright/mcp's docs for the full list.
        browser_tools = {tool.name: tool for tool in browser.tools}
        if not browser_tools:
            print(
                "ERROR: Playwright MCP started but advertised no tools. "
                "Try `npx -y @playwright/mcp@latest --help` to verify the "
                "install.",
                file=sys.stderr,
            )
            sys.exit(2)

        researcher = _build_agent(
            agent_id="researcher",
            role="navigates and reads web pages with a real Chromium browser",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            tools=browser_tools,
            llm=llm,
            memory=memory,
            guard=guard,
        )
        delegate_research = SubAgentTool(researcher, name="delegate_research")

        coordinator = _build_agent(
            agent_id="coordinator",
            role="persistent chat coordinator",
            system_prompt=(
                SYSTEM_PROMPT + " Delegate with delegate_research(task=...) when external "
                "evidence or current information is needed — the researcher "
                "drives a real browser so JS-rendered pages and bot-walled "
                "news sites are all reachable."
            ),
            tools={delegate_research.name: delegate_research},
            llm=llm,
            memory=memory,
            guard=guard,
        )

        session_id = f"sess_{uuid4().hex[:12]}" if args.new_session else args.session_id
        session_path = Path(args.db)
        app = PersistentAgent(
            coordinator=coordinator,
            session_store=SQLiteSessionStore(session_path),
            memory=memory,
            llm=llm,
            guard_factory=lambda: BudgetGuard(
                GuardrailConfig(max_total_cost_usd=2.0, max_wall_time_seconds=120)
            ),
            config=PersistentAgentConfig(
                recent_messages=8,
                reconcile_every_turns=6,
                compact_every_turns=12,
                compact_message_threshold=24,
            ),
        )
        renderer = ConsoleRenderer()

        print("Persistent agent ready.")
        print(f"Session: {session_id}")
        print(f"Session DB: {session_path}")
        print(f"Browser tools: {len(browser_tools)} ({'headed' if args.headed else 'headless'})")
        print("\nSystem prompt:")
        print(SYSTEM_PROMPT)
        if args.show_capabilities:
            print("\nCapabilities:")
            print(json.dumps(app.capabilities(), indent=2, default=str))
        print("\nType a message. Use Ctrl-D or an empty line to exit.\n")

        while True:
            try:
                message = input("> ").strip()
            except EOFError:
                print()
                return
            if not message:
                return

            final_answer = ""
            async for event in app.chat(message, session_id=session_id):
                renderer.render(event)
                if event.type == EventType.TASK_DONE and not event.parent_agent_id:
                    final_answer = str(event.payload.get("answer") or "")
            if final_answer:
                renderer.sep("═")
                print(final_answer)
                renderer.sep("═")


if __name__ == "__main__":
    asyncio.run(main())
