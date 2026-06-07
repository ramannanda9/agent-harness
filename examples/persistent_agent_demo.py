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
from harness.persistent import (
    PersistentAgent,
    PersistentAgentConfig,
    SessionState,
    SQLiteSessionStore,
)
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


def _estimate_session_tokens(state: SessionState) -> int:
    total = max(0, len(state.summary) // 4) if state.summary else 0
    for message in state.messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        total += max(1, len(content) // 4)
    return total


def _print_capabilities(app: PersistentAgent) -> None:
    caps = app.capabilities()
    coordinator = caps["coordinator"]
    print(f"Coordinator: {coordinator['agent_id']}")
    for tool in coordinator.get("tools", []):
        sub = next((s for s in caps["subagents"] if s["tool_name"] == tool), None)
        if sub is None:
            print(f"  - {tool}")
            continue
        print(f"  - {tool} -> {sub['agent_id']}")
        for sub_tool in sub.get("tools", []):
            print(f"      - {sub_tool}")


def _print_agents(app: PersistentAgent) -> None:
    caps = app.capabilities()
    coordinator = caps["coordinator"]
    print(f"{coordinator['agent_id']}: {coordinator['role']}")
    for sub in caps["subagents"]:
        print(f"{sub['agent_id']}: {sub['role']} (via {sub['tool_name']})")


def _print_mcp(app: PersistentAgent) -> None:
    tools = app.capabilities()["mcp_tools"]
    if not tools:
        print("No MCP tools are wired.")
        return
    by_owner: dict[str, list[dict]] = {}
    for tool in tools:
        by_owner.setdefault(tool["owner_agent_id"], []).append(tool)
    for owner, owned_tools in sorted(by_owner.items()):
        print(f"{owner}: {len(owned_tools)} MCP tools")
        for tool in sorted(owned_tools, key=lambda t: t["name"]):
            schema = tool.get("input_schema") or {}
            props = schema.get("properties") if isinstance(schema, dict) else None
            args = ", ".join(sorted(props)) if isinstance(props, dict) else ""
            suffix = f"({args})" if args else "()"
            description = tool.get("description") or ""
            print(f"  - {tool['name']}{suffix}" + (f" — {description}" if description else ""))


async def _print_session(
    app: PersistentAgent,
    *,
    session_id: str,
    config: PersistentAgentConfig,
    llm,
) -> None:
    state = await app.session_state(session_id)
    tokens = _estimate_session_tokens(state)
    budget = getattr(llm, "input_token_budget", None)
    if isinstance(budget, int) and budget > 0:
        pct = (tokens / budget) * 100
        token_line = f"{tokens:,} / {budget:,} tokens ({pct:.1f}%)"
    else:
        token_line = f"{tokens:,} estimated tokens"
    interval = config.async_reconcile_every_turns
    if interval > 0:
        remaining = interval - (state.turn_count % interval)
        reconcile = f"turn {state.turn_count + remaining} ({remaining} turn(s))"
    else:
        reconcile = "disabled"
    print(f"session_id: {state.session_id}")
    print(f"turns: {state.turn_count}")
    print(f"transcript: {token_line}")
    print(f"last reconcile turn: {state.last_reconcile_turn or 'never'}")
    print(f"next async reconcile: {reconcile}")
    print(f"last compaction turn: {state.last_compact_turn or 'never'}")
    print(f"summary: {state.summary or '(none)'}")


def _print_memory(app: PersistentAgent, *, session_id: str) -> None:
    context = app.cached_memory_context(session_id)
    if context is None:
        print("memory cache: not loaded yet")
    elif not context:
        print("memory cache: loaded, no relevant facts or episodes")
    else:
        print("memory cache:")
        print(context)


def _session_arg(command: str, message: str) -> str:
    arg = message[len(command) :].strip()
    if not arg:
        return ""
    parts = arg.split()
    if len(parts) != 1:
        raise ValueError("session ids cannot contain whitespace")
    if parts[0].startswith("/"):
        raise ValueError("session ids cannot start with '/'")
    return parts[0]


async def _print_sessions(app: PersistentAgent, *, current_session_id: str) -> None:
    sessions = await app.list_sessions()
    if not sessions:
        print("No sessions found.")
        return
    for state in sessions:
        marker = "*" if state.session_id == current_session_id else " "
        summary = state.summary or "(no summary)"
        if len(summary) > 80:
            summary = summary[:77] + "..."
        print(f"{marker} {state.session_id}  turns={state.turn_count}  updated={state.updated_at}")
        print(f"    {summary}")


async def _handle_slash_command(
    message: str,
    *,
    app: PersistentAgent,
    session_id: str,
    config: PersistentAgentConfig,
    llm,
) -> tuple[bool, str, bool]:
    if not message.startswith("/"):
        return False, session_id, False
    command = message.split(maxsplit=1)[0].lower()

    if command in {"/help", "/?"}:
        print("Inspect:  /capabilities, /agents, /mcp, /session, /memory")
        print("Memory:   /save     reconcile recent turns into long-term memory")
        print("          /compact  structural reorg: summary + trim + reconcile")
        print("          /forget   drop this session's cached memory prior")
        print("Session:  /sessions list known sessions")
        print("          /switch <id> switch to an existing or new logical session")
        print("          /new [id] create a new session; refuses if id exists")
        print("          /clear    clear current transcript; memory retained")
        print("          /delete [id] confirm delete transcript; memory retained")
        print("          /end      exit the demo; transcript stays in SQLite")
    elif command == "/capabilities":
        _print_capabilities(app)
    elif command == "/agents":
        _print_agents(app)
    elif command == "/mcp":
        _print_mcp(app)
    elif command == "/session":
        await _print_session(app, session_id=session_id, config=config, llm=llm)
    elif command == "/memory":
        _print_memory(app, session_id=session_id)
    elif command == "/sessions":
        await _print_sessions(app, current_session_id=session_id)
    elif command == "/save":
        count = await app.save_to_memory(session_id)
        if count:
            print(f"saved {count} pending message(s) from {session_id} to long-term memory")
        else:
            print(f"session {session_id} has no pending messages to save")
    elif command == "/compact":
        state = await app.force_compact(session_id)
        print(f"compacted session {session_id}; last_compact_turn={state.last_compact_turn}")
    elif command == "/forget":
        app.forget_memory_cache(session_id)
        print(f"forgot cached memory context for session {session_id}")
    elif command == "/switch":
        try:
            next_session_id = _session_arg(command, message)
        except ValueError as exc:
            print(f"cannot switch session: {exc}")
            return True, session_id, False
        if not next_session_id:
            print("usage: /switch <session_id>")
            return True, session_id, False
        existed = await app.session_exists(next_session_id)
        # ``session_state`` intentionally creates the row if it is missing,
        # so switching to a new logical workspace makes it visible in
        # /sessions immediately.
        await app.session_state(next_session_id)
        session_id = next_session_id
        print(f"{'resumed' if existed else 'created'} session: {session_id}")
    elif command == "/new":
        try:
            requested_session_id = _session_arg(command, message)
        except ValueError as exc:
            print(f"cannot create session: {exc}")
            return True, session_id, False
        next_session_id = requested_session_id or f"sess_{uuid4().hex[:12]}"
        if await app.session_exists(next_session_id):
            print(f"session already exists: {next_session_id}")
            print(f"use /switch {next_session_id} to resume it")
            return True, session_id, False
        await app.session_state(next_session_id)
        session_id = next_session_id
        print(f"new session: {session_id} (previous transcript preserved in SQLite)")
    elif command == "/clear":
        await app.clear_session(session_id)
        print(f"cleared session {session_id} transcript; long-term memory retained")
    elif command == "/delete":
        raw = message[len(command) :].strip()
        parts = raw.split()
        if parts == ["confirm"]:
            delete_session_id = session_id
        elif len(parts) == 2 and parts[1] == "confirm" and not parts[0].startswith("/"):
            delete_session_id = parts[0]
        else:
            print("usage: /delete confirm OR /delete <session_id> confirm")
            print("deletes transcript only; long-term memory is retained")
            return True, session_id, False
        deleted = await app.delete_session(delete_session_id)
        if not deleted:
            print(f"session not found: {delete_session_id}")
            return True, session_id, False
        print(f"deleted session {delete_session_id}; long-term memory retained")
        if delete_session_id == session_id:
            session_id = "default"
            await app.session_state(session_id)
            print(f"switched to session: {session_id}")
    elif command == "/end":
        print(f"exiting; session {session_id} transcript preserved for next run")
        return True, session_id, True
    else:
        print(f"Unknown command: {command}. Try /help.")
    return True, session_id, False


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
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=2.0, max_wall_time_seconds=120))
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
            guard_factory=lambda: BudgetGuard(
                GuardrailConfig(max_total_cost_usd=2.0, max_wall_time_seconds=120)
            ),
            config=persistent_config,
        )
        renderer = ConsoleRenderer()

        print("Persistent agent ready.")
        print(f"Session: {session_id}")
        print(f"LLM provider: {llm_label}")
        print(f"Session DB: {session_path}")
        print(f"Semantic memory: {semantic_path}")
        print(f"Episodic memory: {lance_path} ({embedder_kind})")
        print(f"Shell executor: {executor}")
        print(f"Browser tools: {len(browser_tools)} ({'headed' if args.headed else 'headless'})")
        print("\nSystem prompt:")
        print(SYSTEM_PROMPT)
        if args.show_capabilities:
            print("\nCapabilities:")
            print(json.dumps(app.capabilities(), indent=2, default=str))
        print("\nType a message. Use /help for controls. Ctrl-D, /end, or an empty line exits.\n")

        while True:
            try:
                message = input("> ").strip()
            except EOFError:
                print()
                return
            if not message:
                return
            handled, session_id, should_exit = await _handle_slash_command(
                message,
                app=app,
                session_id=session_id,
                config=persistent_config,
                llm=llm,
            )
            if should_exit:
                return
            if handled:
                continue

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
