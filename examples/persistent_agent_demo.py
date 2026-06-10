"""Persistent coordinator chat with SQLite-backed session state.

The persistent wrapper does not build sub-agents or MCP tools for you. Wire
agents exactly as usual, including ``SubAgentTool`` and MCP adapters/auth, then
wrap the top-level coordinator with ``PersistentAgent``.

The researcher sub-agent in this demo drives a real Chromium browser via
``@playwright/mcp`` so it can handle JS-rendered pages better than plain HTTP
fetch. Some sites still block automation, redirect aggressively, or abort
navigation; the researcher is prompted to report those limits instead of
guessing.

Requirements:
    - ``npx`` on PATH (Node 18+)
    - ``ah-executor`` on PATH for the shell tool
    - On first run, Playwright auto-installs Chromium into its cache

Run:
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --session-id pr-review
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --new-session
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --show-capabilities
    OPENAI_API_KEY=sk-... python examples/persistent_agent_demo.py --headed   # see the browser
    python -m harness.cli login openai-codex
    python examples/persistent_agent_demo.py --provider openai-codex
    python -m harness.cli login claude-code
    python examples/persistent_agent_demo.py --provider claude-code
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
from harness.executor_bridge import ExecutorBridge, ExecutorConfig, ExecutorTool, find_executor
from harness.persistent import PersistentAgent, PersistentAgentConfig, SQLiteSessionStore
from harness.persistent_controls import PersistentCommandHandler
from harness.persistent_prompt import build_chat_prompt_session
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import SQLiteSemanticStore
from tools.builtin.subagent import SubAgentTool
from tools.mcp import MCPServerConnection

SYSTEM_PROMPT = (
    "You are a persistent coordinator. Wait for the user's turn, then decide "
    "whether to answer directly or delegate to researcher. Use recent session "
    "context for references like 'above', but avoid treating old tool outputs "
    "as current facts unless you re-check them. Use shell for local project or "
    "machine inspection and for local date/time context when the user says "
    "'today', 'now', or similar. Keep shell commands non-interactive, focused, "
    "and safe."
)

RESEARCHER_SYSTEM_PROMPT = (
    "You are a research sub-agent driving a real Chromium browser via MCP. "
    "Keep `thought` to one short sentence — long reasoning in the JSON "
    "envelope risks truncation mid-stream. "
    "You also have shell for local date/time context; use it when the task "
    "depends on words like 'today', 'now', or the current date. "
    "Typical flow: `browser_navigate(url=...)` then `browser_snapshot()` to "
    "read the page's accessibility tree (cleaner than raw HTML — includes "
    "headings, links, and visible text). "
    "For pages with structured lists (headlines, prices, repo files), prefer "
    "`browser_evaluate` with a small JS snippet returning ONLY the elements "
    "you need — full snapshots of news homepages can be thousands of tokens. "
    "For multi-page reading, `browser_navigate` to each URL in turn, then "
    "snapshot before deciding. "
    "Browser automation is not magic: paywalls, CAPTCHA, consent flows, "
    "redirect loops, and aborted navigations can still happen. If navigation "
    "or snapshot fails once for a source, try at most one alternate reputable "
    "source, then finish with a clear limitation. Do not keep retrying the "
    "same site. Cite URLs for anything you report."
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
    # Match the framework default. The previous 6 left no headroom for
    # realistic browser flows where consent banners, slow page loads,
    # and one-source retries each cost a ReAct step. Override per-agent
    # when more is needed (the researcher does — see below).
    max_steps: int = 10,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            allowed_tools=list(tools.keys()),
            max_steps=max_steps,
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
        "--provider",
        choices=("openai", "openai-codex", "claude-code"),
        default=os.environ.get("AGENT_LLM_PROVIDER", "openai"),
        help=(
            "LLM provider to use. 'openai' uses OPENAI_API_KEY; "
            "'openai-codex' and 'claude-code' use stored subscription OAuth credentials."
        ),
    )
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
        default=os.environ.get("AGENT_SESSION_DB"),
        help="SQLite session database path. Defaults to ~/.agent-harness/sessions.sqlite.",
    )
    parser.add_argument(
        "--state-dir",
        default=os.environ.get("AGENT_HARNESS_HOME", "~/.agent-harness"),
        help="Directory for local persistent state. Defaults to AGENT_HARNESS_HOME or ~/.agent-harness.",
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
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=int(os.environ.get("AGENT_TIME_BUDGET_SECONDS", "300")),
        help=(
            "Per-turn wall-time budget for coordinator + sub-agents. "
            "Defaults to AGENT_TIME_BUDGET_SECONDS or 300."
        ),
    )
    return parser.parse_args()


def _build_llm(provider: str):
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "ERROR: set OPENAI_API_KEY before running with --provider openai.", file=sys.stderr
            )
            sys.exit(2)
        from harness.llm.openai import OpenAILLM  # noqa: PLC0415

        model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
        return OpenAILLM(model=model), f"openai:{model}"

    if provider == "openai-codex":
        from harness.llm.openai_codex import OpenAICodexLLM  # noqa: PLC0415

        auth_file = Path(
            os.environ.get("OPENAI_CODEX_AUTH_FILE", "~/.agent-harness/auth/auth.json")
        ).expanduser()
        if not auth_file.exists():
            print(
                "ERROR: OpenAI Codex credentials not found. Run: "
                "python -m harness.cli login openai-codex",
                file=sys.stderr,
            )
            sys.exit(2)
        model = os.environ.get("OPENAI_CODEX_MODEL", "gpt-5.5")
        return OpenAICodexLLM(model=model, auth_file=auth_file), f"openai-codex:{model}"

    if provider == "claude-code":
        from harness.llm.claude_code import ClaudeCodeLLM  # noqa: PLC0415

        auth_file = Path(
            os.environ.get("CLAUDE_CODE_AUTH_FILE", "~/.agent-harness/auth/auth.json")
        ).expanduser()
        if not auth_file.exists():
            print(
                "ERROR: Claude Code credentials not found. Run: "
                "python -m harness.cli login claude-code",
                file=sys.stderr,
            )
            sys.exit(2)
        model = os.environ.get("CLAUDE_CODE_MODEL", "claude-sonnet-4-6")
        return ClaudeCodeLLM(model=model, auth_file=auth_file), f"claude-code:{model}"

    raise ValueError(f"unsupported provider: {provider}")


async def main() -> None:
    args = _parse_args()
    if shutil.which("npx") is None:
        print(
            "ERROR: npx not found on PATH. Install Node 18+ — "
            "`@playwright/mcp` is spawned via npx.",
            file=sys.stderr,
        )
        sys.exit(2)
    executor = find_executor()
    if not executor:
        print(
            "ERROR: ah-executor not found on PATH. Install it with: cargo install --path executor",
            file=sys.stderr,
        )
        sys.exit(2)

    state_dir = Path(args.state_dir).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    session_path = Path(args.db).expanduser() if args.db else state_dir / "sessions.sqlite"
    semantic_path = state_dir / "memory" / "semantic.sqlite"
    lance_path = state_dir / "memory" / "lance_episodic"

    try:
        from memory.episodic_lance import LanceDBEpisodicStore, LocalEmbedder, MockEmbedder
    except ImportError as e:
        print(f"ERROR: LanceDB episodic memory is not installed: {e}", file=sys.stderr)
        print('Install with: pip install -e ".[lance]"', file=sys.stderr)
        sys.exit(2)

    try:
        embedder = LocalEmbedder()
        embedder_kind = "LocalEmbedder"
    except Exception:
        embedder = MockEmbedder()
        embedder_kind = "MockEmbedder"
    episodic_store = LanceDBEpisodicStore(uri=str(lance_path), embedder=embedder)
    await episodic_store.initialize()

    llm, llm_label = _build_llm(args.provider)
    memory = MemoryManager(
        semantic_store=SQLiteSemanticStore(semantic_path),
        episodic_store=episodic_store,
        llm=llm,
        memory_scope="persistent-demo",
        memory_subject="persistent-demo",
    )
    guard_config = GuardrailConfig(
        max_total_cost_usd=2.0,
        max_wall_time_seconds=args.time_budget_seconds,
    )
    guard = BudgetGuard(guard_config)
    shell_tool = ExecutorTool(
        "shell",
        "shell",
        ExecutorBridge(
            ExecutorConfig(
                allowed_tools=("shell",),
                binary_path=executor,
                default_timeout_ms=20_000,
                max_output_bytes=200_000,
            )
        ),
    )

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
            tools={**browser_tools, "shell": shell_tool},
            llm=llm,
            memory=memory,
            guard=guard,
            # Browser-driving headroom: a typical "navigate → snapshot →
            # follow link → snapshot → extract → finish" flow already uses
            # 6 steps. Real pages add consent banners, slow loads, and
            # one-source-fails-try-alternate retries; 15 leaves a comfortable
            # margin without inviting infinite loops.
            max_steps=15,
        )
        delegate_research = SubAgentTool(researcher, name="delegate_research")

        coordinator = _build_agent(
            agent_id="coordinator",
            role="persistent chat coordinator",
            system_prompt=(
                SYSTEM_PROMPT + " Delegate with delegate_research(task=...) when external "
                "evidence or current information is needed — the researcher "
                "drives a real browser, but some sites may still block or "
                "abort automation. If research comes back too broad or blocked, "
                "refine the task with concrete date/source constraints and try "
                "once more before asking the user for missing context."
            ),
            tools={delegate_research.name: delegate_research, "shell": shell_tool},
            llm=llm,
            memory=memory,
            guard=guard,
        )

        session_id = f"sess_{uuid4().hex[:12]}" if args.new_session else args.session_id
        persistent_config = PersistentAgentConfig(
            # Defaults are sensible — explicit here for visibility.
            # ``compact_at_context_fraction=0.5`` reads
            # ``coordinator._llm.input_token_budget`` (~123K for
            # gpt-5.4-mini) so compaction fires only under real
            # context pressure, not on arbitrary turn counts.
            # ``retain_context_fraction=0.15`` keeps the newest ~15%
            # of the model's input budget verbatim after compaction
            # and folds older messages into the rolling summary.
            # ``async_reconcile_every_turns=10`` runs a non-blocking
            # background ``write_run_end`` every 10 turns over the
            # last 10 turn-pairs from the durable transcript —
            # memory accumulates without thrashing the prefix cache.
            compact_at_context_fraction=0.5,
            retain_context_fraction=0.15,
            async_reconcile_every_turns=10,
        )
        app = PersistentAgent(
            coordinator=coordinator,
            session_store=SQLiteSessionStore(session_path),
            memory=memory,
            llm=llm,
            guard_factory=lambda: BudgetGuard(guard_config),
            config=persistent_config,
        )
        renderer = ConsoleRenderer()
        controls = PersistentCommandHandler(app)

        # Shift-Tab toggles plan mode for the *current* session id.
        # ``session_id`` is rebound across the loop (by /switch, /new,
        # /delete), so the toggle reads it via a thunk rather than
        # capturing a stale value at construction time.
        async def _toggle_plan_mode() -> bool:
            new_state = not await app.plan_mode_enabled(session_id)
            await app.set_plan_mode(session_id, new_state)
            return new_state

        # Chat-style input: multi-line buffer with Enter to submit,
        # Ctrl+J / Esc-Enter for newline, Shift-Tab to toggle plan mode,
        # slash-command + session-id Tab-completion, command history
        # persisted in ``state_dir``. All configured in one place — see
        # ``build_chat_prompt_session``.
        prompt_session = build_chat_prompt_session(
            app,
            history_path=Path(args.state_dir).expanduser() / "demo_history",
            plan_mode_toggle=_toggle_plan_mode,
        )

        print("Persistent agent ready.")
        print(f"Session: {session_id}")
        print(f"LLM provider: {llm_label}")
        print(f"Session DB: {session_path}")
        print(f"Semantic memory: {semantic_path}")
        print(f"Episodic memory: {lance_path} ({embedder_kind})")
        print(f"Shell executor: {executor}")
        print(f"Turn time budget: {args.time_budget_seconds}s")
        print(f"Browser tools: {len(browser_tools)} ({'headed' if args.headed else 'headless'})")
        print("\nSystem prompt:")
        print(SYSTEM_PROMPT)
        if args.show_capabilities:
            print("\nCapabilities:")
            print(json.dumps(app.capabilities(), indent=2, default=str))
        print(
            "\nType a message. Enter submits; Ctrl+J or Esc-Enter inserts a newline. "
            "Use /help for controls. Shift-Tab (or `/plan on`) toggles plan mode — "
            "the agent will propose a plan and wait for approval before each turn. "
            "Press Esc while the agent is running to cancel "
            "the current turn. Ctrl-D, /end, or an empty line exits.\n"
        )

        while True:
            try:
                message = (await prompt_session.prompt_async("> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not message:
                return
            command_result = await controls.handle(message, session_id=session_id)
            if command_result.session_id is not None:
                session_id = command_result.session_id
            if command_result.text:
                print(command_result.text)
            if command_result.should_exit:
                return
            if command_result.handled:
                continue

            # Esc during a long turn cancels it cleanly.
            # ``PersistentAgent.chat`` only writes to the session store
            # inside ``_finalize_turn`` (TASK_DONE / ERROR / clean stream
            # end), so a mid-stream cancel writes nothing — the
            # transcript looks like the turn never started.
            cancelled, terminal = await renderer.render_stream(
                app.chat(message, session_id=session_id),
                terminal_event_type=EventType.TASK_DONE,
                cancel_message="[cancelled by user — type your next message or /end to exit]",
            )
            if cancelled:
                continue
            final_answer = str(terminal.payload.get("answer") or "") if terminal else ""
            if final_answer:
                renderer.sep("═")
                print(final_answer)
                renderer.sep("═")


if __name__ == "__main__":
    asyncio.run(main())
