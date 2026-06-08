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

  3. --cache-demo (claude-code only)
     Demonstrates Anthropic prompt caching. Uses a long system prompt
     (>1024 tokens) and a multi-step task so that:
       - Step 1 writes the system prompt to the cache  (cache_new=N)
       - Steps 2+ read from the cache instead           (cache_hit=N)

     Watch the ctx line change from [cache_new=...] to [cache_hit=...]:

         python examples/subscription_auth_demo.py claude-code --cache-demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agents.base import AgentConfig
from harness.cancellation import consume_with_cancel
from harness.console import ConsoleRenderer
from harness.events import BusEvent, EventType
from harness.llm.claude_code import ClaudeCodeLLM
from harness.llm.openai_codex import OpenAICodexLLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

GOAL = (
    "Reply with one concise sentence confirming that this run used the configured "
    "credential-backed LLM provider."
)

# ── Cache demo ────────────────────────────────────────────────────────────────
# System prompt intentionally long (>1024 tokens) so Anthropic actually caches
# it. The first LLM call in the run writes the KV cache (cache_new); every
# subsequent call in the same run reads from it (cache_hit).
_CACHE_DEMO_SYSTEM = """\
You are a meticulous software quality analyst specialising in multi-agent LLM
systems. Your job is to review agent runs step by step, recording observations
and producing a structured final report.

RESPONSIBILITIES
----------------
1. Observe: at each step, call the `record` tool with a single, precise
   observation about what you have done or decided. Observations must be
   written in the third person, past tense, and must be under 120 characters.

2. Investigate: decompose the assigned task into at most four discrete steps.
   Each step must produce exactly one `record` call before proceeding. Do not
   batch observations — one step, one record.

3. Conclude: after all steps are recorded, call `finish` with a structured
   summary. The summary must include: (a) a one-paragraph overall assessment,
   (b) a bullet list of findings keyed to your recorded observations, and
   (c) a confidence score between 0.0 and 1.0 with a one-sentence rationale.

TOOL USAGE
----------
record(observation: str) -> str
    Persists one observation to the run log. Returns "recorded". You MUST call
    this tool at least once per step before calling `finish`. Do not skip it
    even when the observation seems obvious — the audit trail is mandatory.

finish(answer: str) -> never
    Signals end of run. The `answer` field must be your full structured
    summary as described above. Do not call `finish` until you have recorded
    at least three observations.

QUALITY STANDARDS
-----------------
- Precision over brevity: a vague observation is worse than no observation.
- Never fabricate evidence. If you cannot observe something directly, say so.
- Confidence scores above 0.9 require explicit justification.
- If a step is ambiguous, record your interpretation before proceeding.
- Avoid hedging language such as "might", "possibly", "perhaps" in findings.
  State what you observed, not what you imagine could be true.

OUTPUT FORMAT
-------------
Your `finish` answer must follow this template exactly:

    ASSESSMENT
    <one paragraph>

    FINDINGS
    - Observation 1: <finding>
    - Observation 2: <finding>
    - Observation 3: <finding>

    CONFIDENCE: <0.0–1.0> — <one-sentence rationale>

BEHAVIOURAL CONSTRAINTS
-----------------------
- You must not call `finish` before recording at least three observations.
- You must not call `record` more than six times in a single run.
- You must not repeat the same observation text verbatim across steps.
- You must not include tool call syntax in the content of a `record` call.
- Each step in your ReAct loop must include a non-empty "thought" field that
  explains your reasoning before committing to an action.

EXAMPLES
--------
Good observation:
    "Agent verified that the LLM adapter returned a non-empty response on
     the first call without retrying."

Bad observation (too vague):
    "Something happened with the LLM."

Good thought:
    "I have recorded two observations so far. The task requires at least
     three before I can call finish. I will record one more observation
     about the token usage pattern before concluding."

Bad thought (no reasoning):
    "I will call record."

These standards exist because audit logs from this agent feed directly into
human review workflows. Incomplete or imprecise logs cause reviewer confusion
and erode trust in automated quality signals. Treat every observation as if
it will be read by a senior engineer who has no other context about the run.

ERROR HANDLING
--------------
If a tool returns an error string, record the error as an observation and
attempt to continue. Do not silently ignore tool failures. If `record` fails
three times in a row, proceed to `finish` with a note that the audit log is
incomplete, and set confidence to 0.0.

VERSIONING
----------
This prompt conforms to audit-agent spec v2.1. Agents running under this spec
must include the string "audit-agent/2.1" in the ASSESSMENT section of their
final answer so downstream consumers can verify schema compatibility.

COMMON MISTAKES TO AVOID
-------------------------
1. Calling `finish` before three `record` calls — this violates the audit
   contract and will cause the run to be flagged as incomplete by reviewers.
2. Writing observations in first person ("I observed...") — use third person
   ("Agent observed...") to maintain separation between agent narration and
   audit voice.
3. Including line breaks inside the observation string passed to `record` —
   the audit log renderer is single-line-per-entry; newlines corrupt the
   display. Keep each observation to one sentence on one line.
4. Using the word "successfully" without citing evidence — write what the
   agent specifically checked, not just that something "worked".
"""

_CACHE_DEMO_GOAL = (
    "Audit this agent run: verify that (1) the LLM adapter produced a response, "
    "(2) the ReAct loop executed at least two steps, and (3) the context usage "
    "events appeared in the event stream. Record one observation per finding, "
    "then produce a structured summary."
)


class RecordTool:
    """Append one observation to the in-memory run log."""

    name = "record"
    _log: list[str] = []

    def __init__(self) -> None:
        self._log = []

    async def execute(self, observation: str = "") -> str:
        """
        Persist one observation to the audit log.

        Args:
            observation: A precise, third-person past-tense sentence describing
                         what was observed in this step (max 120 characters).

        Returns the string "recorded" on success.
        """
        self._log.append(observation)
        return "recorded"

    @property
    def entries(self) -> list[str]:
        return list(self._log)


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


async def run(provider: str, *, trace_path: str | None = None) -> dict:
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

    stream = runtime.dispatch_stream(GOAL)
    if trace_path:
        from harness.trace import record_trace

        stream = record_trace(stream, trace_path)

    # Press Esc during the run to cancel cleanly. ERROR events get
    # rendered AND recorded as the final payload so the caller can
    # surface "ran but failed" distinctly from "completed".
    final: dict = {}

    def _on_event(event: BusEvent) -> None:
        nonlocal final
        if event.type == EventType.TASK_DONE:
            final = event.payload
        elif event.type == EventType.ERROR:
            _renderer.render(event)
            final = {"error": event.error}
        else:
            _renderer.render(event)

    cancelled = await consume_with_cancel(stream, on_event=_on_event)
    if cancelled:
        _renderer.sep("═")
        print("[cancelled by user]")
        _renderer.sep("═")
        final = {"cancelled": True}

    if trace_path:
        print(f"\n[trace] Wrote {trace_path} — view with: agent-harness trace view {trace_path}")
    return final


async def run_cache_demo(provider: str, *, trace_path: str | None = None) -> dict:
    """Multi-step run with a long system prompt to demonstrate prompt caching.

    The system prompt is >1024 tokens, which is Anthropic's minimum cache block
    size. Step 1 writes the KV cache (cache_new=N in the ctx line); steps 2+
    read from it (cache_hit=N), cutting input costs by ~90% for those tokens.
    """
    if provider != "claude-code":
        print(
            "Cache demo requires claude-code (Anthropic prompt caching). "
            "Pass 'claude-code' as the provider.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    record_tool = RecordTool()
    llm = _build_llm(provider)

    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="audit_agent",
            role="software quality analyst",
            system_prompt=_CACHE_DEMO_SYSTEM,
            allowed_tools=["record"],
            max_steps=8,
        )
    )
    tools = ToolRegistry().register(record_tool)
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
            max_total_cost_usd=2.0,
            max_wall_time_seconds=180,
            max_replan_count=0,
        ),
    )

    print(
        "\n[cache-demo] System prompt length: "
        f"{len(_CACHE_DEMO_SYSTEM):,} chars  "
        f"(~{len(_CACHE_DEMO_SYSTEM) // 4:,} tokens estimated)\n"
        "[cache-demo] Watch the ctx line: first call → cache_new=N, "
        "subsequent calls → cache_hit=N\n"
    )

    stream = runtime.dispatch_stream(_CACHE_DEMO_GOAL)
    if trace_path:
        from harness.trace import record_trace

        stream = record_trace(stream, trace_path)

    # Press Esc during the run to cancel cleanly. ERROR events get
    # rendered AND recorded as the final payload so the caller can
    # surface "ran but failed" distinctly from "completed".
    final: dict = {}

    def _on_event(event: BusEvent) -> None:
        nonlocal final
        if event.type == EventType.TASK_DONE:
            final = event.payload
        elif event.type == EventType.ERROR:
            _renderer.render(event)
            final = {"error": event.error}
        else:
            _renderer.render(event)

    cancelled = await consume_with_cancel(stream, on_event=_on_event)
    if cancelled:
        _renderer.sep("═")
        print("[cancelled by user]")
        _renderer.sep("═")
        final = {"cancelled": True}

    if record_tool.entries:
        print("\n[cache-demo] Recorded observations:")
        for i, entry in enumerate(record_tool.entries, 1):
            print(f"  {i}. {entry}")
    if trace_path:
        print(f"\n[trace] Wrote {trace_path} — view with: agent-harness trace view {trace_path}")

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-harness through subscription/CLI auth")
    parser.add_argument("provider", choices=["openai-codex", "claude-code"])
    parser.add_argument(
        "--cache-demo",
        action="store_true",
        help="Run a multi-step audit task with a long system prompt to show prompt caching "
        "(claude-code only; requires >=1024 token system prompt for cache to activate)",
    )
    parser.add_argument(
        "--trace",
        metavar="PATH",
        help="Record every BusEvent to a JSONL trace file; view with "
        "'agent-harness trace view PATH'",
    )
    args = parser.parse_args()

    _check_auth(args.provider)
    if args.cache_demo:
        result = asyncio.run(run_cache_demo(args.provider, trace_path=args.trace))
    else:
        result = asyncio.run(run(args.provider, trace_path=args.trace))
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
