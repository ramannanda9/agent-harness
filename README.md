# react-agent-harness

Bring-your-own-LLM multi-agent harness: hybrid DAG planning with replan-on-failure,
two-tier memory (semantic KV + episodic vector), a streaming-primary event model,
and cost/token budgets with per-call-site attribution (classifier, router,
planner, synthesizer, agent).

Config-driven — register tools and agents, run any goal. No subclassing.

## Install

```bash
pip install -e ".[dev]"                              # core + tests + lint
pip install -e ".[openai,http,dev]"                  # OpenAI adapter + HTTPFetch tool
pip install -e ".[lance,redis,dev]"                  # LanceDB + Redis durable memory
pip install -e ".[mcp]"                              # MCP tool adapter
pip install -e ".[otel]"                             # OpenTelemetry tracing
```

The core package has no runtime dependencies. The `openai`, `http`, `lance`,
`redis`, `anthropic`, `mcp`, and `otel` extras pull in heavier libs only if
you opt in.

## Quickstart

```bash
python examples/quickstart.py                          # mock LLM, no keys, no network
OPENAI_API_KEY=sk-... python examples/openai_demo.py   # real OpenAI + HTTPFetch
```

`quickstart.py` exercises the full wiring against a deterministic `MockLLM`.
`openai_demo.py` runs the same shape against a real model — streams live
events to stdout and prints elapsed time + cost at the end.

## Architecture

```
harness/runtime.py          AgentRuntime — single entry point; BudgetGuard with cost/token caps + per-call-site breakdown
harness/events.py           BusEvent + EventType — canonical event vocabulary
harness/llm/openai.py       OpenAILLM — OpenAI API-key adapter with usage + cost tracking
harness/llm/anthropic.py    AnthropicLLM — direct Anthropic API-key adapter with prompt-caching support
harness/llm/claude_code.py  ClaudeCodeLLM — Claude subscription OAuth adapter (experimental, ToS caveats)
harness/llm/openai_codex.py OpenAICodexLLM — ChatGPT subscription OAuth adapter (experimental, ToS caveats)
harness/llm/auth.py         Shared OAuth + auth-file primitives for the subscription adapters
harness/llm/fallback.py     FallbackLLM — transparent retry on transient upstream errors
harness/llm/routing.py      RoutingLLM — dispatch calls to different adapters by a selector
harness/trace.py            JSONL trace recorder + replay — durable, per-event flush
harness/trace_viewer.py     Local web timeline viewer for recorded JSONL traces
harness/annotation.py       Annotation store + AnnotationHook — RLHF trajectory capture
harness/hitl.py             HITL approval gate — interactive CLI, session-allow list
harness/tool_policy.py      Persistent tool policy — user-scoped allow rules, CLI management
harness/console.py          ConsoleRenderer — centralised BusEvent formatting + render_budget helper
harness/steering.py         Async steering — agent.steer(text), StdinRouter pub/sub, FileSteer, factory helpers
harness/checkpoint.py       CheckpointStore + _ResumeHint + maybe_resume_key — pluggable run-state persistence (file + Redis); auto-resume built into dispatch_stream / run_stream
harness/otel.py             OTELHook — OpenTelemetry span exporter (opt-in)
harness/executor_bridge.py  ExecutorBridge + ExecutorTool — controlled subprocess launcher with optional Docker sandboxing
harness/oauth_browser.py    Localhost OAuth callback server + open_or_print_url — shared by MCP browser-OAuth and LLM login flows
orchestrator/planner.py     Hybrid DAG orchestrator — plan, replan, synthesize
agents/base.py              Generic BaseAgent — ReAct loop, no subclassing needed
memory/manager.py           MemoryManager — semantic KV + episodic vector
memory/working.py           WorkingMemory — LLM summarization eviction, checkpoint/restore
memory/episodic_lance.py    LanceDB episodic store — IVF_PQ ANN, batch writes
memory/redis_store.py       Redis semantic store — durable KV with TTL
memory/stores.py            InMemory stores — local dev default, no deps
tools/builtin/http_fetch.py HTTPFetch — minimal read-only GET tool
tools/builtin/fetch_image.py FetchImage — fetch URL and return OpenAI image_url block
tools/mcp/adapter.py        MCP tool adapter — stdio, SSE, streamable-HTTP transports
tools/mcp/auth.py           ApiKeyMCPAuth + BrowserOAuthMCPAuth — auth primitives for remote MCP servers
```

Execution is **streaming-primary**: every path yields `BusEvent`s for
dispatch, routing, plan, thoughts, tool calls, observations, task completions,
replans, and synthesis. The blocking variants drain the same stream.

`dispatch_stream(goal)` is the recommended entry point — it classifies
complexity with one cheap LLM call and delegates automatically to the routed
or orchestrated path. Use the lower-level paths directly only when you need
explicit control.

## Examples

| Script | What it shows | Requires |
|---|---|---|
| `examples/quickstart.py` | End-to-end against `MockLLM` + `EchoTool` — reference wiring. | nothing |
| `examples/openai_demo.py` | Real OpenAI + `HTTPFetch` + `shell` (via `ah-executor`), routed single-agent run, live event stream, cost reporting. | `OPENAI_API_KEY`, `[openai,http]` |
| `examples/vision_demo.py` | Multimodal agent: fetches two images in parallel via `FetchImage`, describes each using the LLM's vision capability, synthesises a report. | `OPENAI_API_KEY`, `[openai,http]` |
| `examples/complex_sysaudit_demo.py` | Three heterogeneous agents in parallel: `shell_agent` (ah-executor), `filesystem_agent` (MCP), `web_agent` (HTTPFetch) — orchestrated path, DAG plan, synthesis. | `OPENAI_API_KEY`, `[openai,http,mcp]`, `ah-executor`, `npx` |
| `examples/executor_bridge_demo.py` | `ExecutorBridge` backends side-by-side: allowlist, env scrubbing, Docker network/fs isolation, timeout, positional-arg tools. | `ah-executor` and/or Docker |
| `examples/durable_memory_demo.py` | Redis (semantic) + LanceDB (episodic) memory persistence across two related goals. | `OPENAI_API_KEY`, `[openai,redis,lance]`, Redis reachable |
| `examples/mcp_demo.py` | Connects to an MCP filesystem server and gives the agent its tools. | `OPENAI_API_KEY`, `[openai,mcp]`, `npx` |
| `examples/mcp_auth_demo.py` | Connects to an authenticated remote MCP server using bearer or auth-file credentials. | `OPENAI_API_KEY`, `[openai,mcp]`, `MCP_URL`, `MCP_BEARER_TOKEN` or `MCP_AUTH_PROVIDER` |
| `examples/subscription_auth_demo.py` | Runs an agent through subscription-backed providers: direct `openai-codex` OAuth or direct `claude-code` OAuth. | `agent-harness login openai-codex` or `agent-harness login claude-code` |

## Adding a new domain (3 steps)

```python
# 1. Register tools
tools.register(MyTool())

# 2. Register agent config — no subclassing
agents.register(AgentConfig(
    agent_id="my_agent",
    role="does X using tools Y and Z",
    system_prompt="You are an expert at X...",
    allowed_tools=["my_tool"],
))

# 3. Run
result = await runtime.dispatch("my goal")
```

## LLM clients

The harness is **BYO-LLM.** Any object with
`async def complete(system, messages, **kwargs) -> dict` works. Optional
`async def stream_complete(system, messages) -> AsyncGenerator[str, None]`
enables `TOKEN` events.

The OpenAI adapter is shipped because it's the easiest spin:

```python
from harness.llm.openai import OpenAILLM

llm = OpenAILLM(model="gpt-4o-mini")                # reads OPENAI_API_KEY from env
runtime = AgentRuntime(..., llm=llm)
```

Credential-backed adapters can also plug into the same contract. This is the
shape used for provider-specific subscription or OAuth flows without teaching
agents about auth:

```bash
agent-harness login openai-codex
agent-harness auth status openai-codex
agent-harness login claude-code
agent-harness auth status claude-code
```

> **⚠️ Subscription adapters are experimental — use the metered API in production.**
>
> `OpenAICodexLLM` and `ClaudeCodeLLM` bridge **ChatGPT / Claude
> subscription OAuth credentials** into the harness by talking to
> internal CLI endpoints with CLI-shaped User-Agent and billing headers.
> This route:
>
> - **May violate OpenAI's and Anthropic's Terms of Service.** Both
>   providers prohibit using subscription accounts (ChatGPT Plus/Pro,
>   Claude Pro/Max) for arbitrary programmatic access — subscriptions
>   price for the official CLI's intended use only.
> - **May result in account suspension** if abuse detection classifies
>   harness traffic as misuse.
> - **Depends on undocumented internal endpoints**
>   (`/backend-api/codex/responses`, the Anthropic Messages API with
>   `claude-code-*` beta flags) that providers can change or revoke at
>   any time.
>
> **Use these adapters only for personal research on accounts you own.**
> Do not use them to serve other users. For anything else, prefer the
> metered API path:
>
> - `OpenAILLM` with `OPENAI_API_KEY` (optionally routed through a
>   gateway like LiteLLM/Helicone for cost headers).
> - The standard Anthropic Messages API with an Anthropic API key.

Direct `openai-codex` OAuth follows the Codex/Pi-style ChatGPT
subscription route rather than the stable OpenAI Platform API. The
Codex OAuth client id can be overridden with
`AGENT_HARNESS_OPENAI_CODEX_CLIENT_ID`.

```python
from harness.llm.openai_codex import OpenAICodexLLM

llm = OpenAICodexLLM(
    model="gpt-5.5",
    auth_file="~/.agent-harness/auth/auth.json",  # Pi-shaped openai-codex OAuth entry
)
runtime = AgentRuntime(..., llm=llm)
```

`OpenAICodexLLM` calls the Codex backend directly
(`https://chatgpt.com/backend-api/codex/responses`) with OAuth credentials.
The stable fallback remains `OpenAILLM` with `OPENAI_API_KEY`.

For Claude Code-style setups, use `ClaudeCodeLLM` with Claude Pro/Max OAuth
credentials stored in the same auth file. It calls the Anthropic Messages API
directly with Claude-Code-compatible OAuth headers:

```bash
agent-harness login claude-code
python examples/subscription_auth_demo.py claude-code
```

```python
from harness.llm.claude_code import ClaudeCodeLLM

llm = ClaudeCodeLLM(
    model="claude-sonnet-4-6",
    auth_file="~/.agent-harness/auth/auth.json",
)
```

### Cost shaping + reliability

Two patterns, ordered by how production teams actually solve this:

**1. Per-call-site LLM injection (the recommended pattern)**

`AgentRuntime` exposes one slot per orchestrator call site. Each defaults to
`llm` when unset, so existing code keeps working. The classifier and router
both see only the goal + agent descriptions (~300 tokens) and emit a
one-token decision — natural candidates for a cheaper model. The planner
and synthesiser produce structured DAGs and final answers and usually want
to stay on the main model.

```python
runtime = AgentRuntime(
    agent_registry=agents,
    tool_registry=tools,
    memory=memory,
    llm=premium,                 # default — agent ReAct loops use this
    classifier_llm=cheap,        # simple vs complex dispatch decision
    router_llm=cheap,            # single-agent picker
    # planner_llm=...            # defaults to llm; override only if you want
    # synthesizer_llm=...        # defaults to llm
)
```

No guessing, no keyword matching, no fragility — you read the runtime
construction and you know exactly which model serves which purpose. The
budget guard is wired into every distinct LLM instance automatically
(deduped by object identity, so injecting the same wrapper into multiple
slots costs no extra calls).

**2. `FallbackLLM` for resilience**

Try each adapter in order; transparently switch to the next on rate
limits, timeouts, or 5xx errors:

```python
from harness.llm.fallback import FallbackLLM

llm = FallbackLLM([
    AnthropicLLM(model="claude-sonnet-4-6"),   # primary
    OpenAILLM(model="gpt-4o-mini"),            # backup
])
runtime = AgentRuntime(..., llm=llm)
print(llm.last_route)   # 0 if primary worked, 1 if backup did
```

Permanent errors (auth, bad request) propagate immediately — only transient
upstream errors trigger fallback. Customise with `transient_errors=...`.
Streaming retries only fire before the first token; mid-stream failures
propagate to preserve response integrity.

**3. `RoutingLLM` for bring-your-own-selector cases**

When you need runtime routing — capability gating (`vision` vs
`long_context`), learned classifiers (RouteLLM-style), cascade
routing (cheap-then-escalate-on-low-confidence) — wrap a routes dict
with your own selector callable:

```python
from harness.llm.routing import RoutingLLM

def by_capability(system, messages):
    if _needs_vision(messages):
        return "vision"
    if _estimated_tokens(system, messages) > 100_000:
        return "long_context"
    return "default"

llm = RoutingLLM(
    routes={
        "default":      OpenAILLM(model="gpt-4o-mini"),
        "vision":       OpenAILLM(model="gpt-4o"),
        "long_context": AnthropicLLM(model="claude-sonnet-4-6"),
    },
    selector=by_capability,
    default_route="default",
)
```

The harness intentionally does not ship default selectors. Naive selectors
(keyword matching, fixed token thresholds) misroute in subtle ways and
encourage the wrong mental model — if you're reaching for one, you almost
certainly want per-call-site injection instead.

Compose freely: `FallbackLLM([premium, backup])` injected into the
`llm=` slot gives the agent loops resilience, with `classifier_llm=cheap`
and `router_llm=cheap` shaping the cheap-call cost — all without a custom
selector.

---

`ClaudeCodeLLM` reads a `claude-code` OAuth entry, refreshes it automatically
when expired, and retries once after `401`/`403`. This mirrors Pi's Claude
Pro/Max extension approach rather than shelling out to the Claude CLI. The
default model is the current canonical Sonnet release ID, `claude-sonnet-4-6`;
set `CLAUDE_CODE_MODEL` or pass `model="claude-opus-4-7"` to choose another
model.

Both adapters stream incrementally — `stream_complete()` yields each
SSE delta token as it arrives, and `complete()` consumes the same
stream and returns the concatenated text once finished. Cost / token
usage is captured from the final stream event into `last_usage`.

The Claude billing header's `cc_version` is read from
`CLAUDE_CODE_VERSION` (env) or from `claude --version` if the CLI is
installed; falls back to `unknown` otherwise. Pinning a specific
version with `CLAUDE_CODE_VERSION=2.1.150` is recommended if you want
stable behavior across CLI upgrades.

Do not copy browser/app refresh tokens into repo files. Store OAuth auth files
under `~/.agent-harness/auth` or reuse an existing Pi auth file with private
file permissions (`0600`).

To use Anthropic / Gemini / Ollama / a local SGLang or vLLM server / anything
else — write a 30-line adapter implementing those two methods. See
`harness/llm/openai.py` for the reference shape; the harness never imports a
provider SDK directly.

## Built-in tools

One tool ships out of the box — `HTTPFetch`, intentionally boring. Anything
heavier (auth, retries, connection pooling, scraping) belongs in a tool you
write yourself. Tools just need `.name` and `async def execute(**kwargs)`.

```python
from tools.builtin.http_fetch import HTTPFetch

tools.register(HTTPFetch(max_bytes=64 * 1024))      # body capped at 64 KiB
```

Returns `{status, content_type, body, truncated, url}`. Errors come back as
`{"error": "..."}` so the agent treats them as observations rather than
crashing the loop.

## Memory write timing

- **During run**: `write_working_fact()` — lightweight KV, namespaced, short TTL
- **End of run**: `write_run_end()` — LLM extraction → global semantic + episodic vector

Defaults are in-memory (`InMemorySemanticStore`, `InMemoryEpisodicStore`).
For durable storage:

```python
# Semantic: Redis
import redis.asyncio as redis
from memory.redis_store import RedisSemanticStore

client = redis.Redis(host="localhost", decode_responses=True)
semantic = RedisSemanticStore(client, key_prefix="agent-harness:")

# Episodic: LanceDB
from memory.episodic_lance import LanceDBEpisodicStore
episodic = LanceDBEpisodicStore(db_path="./lance_episodic")

memory = MemoryManager(semantic_store=semantic, episodic_store=episodic, llm=llm)
```

Or write your own backend conforming to the `SemanticStore` / `EpisodicStore`
protocols in `memory/manager.py`.

## Execution paths

### Recommended: `dispatch` / `dispatch_stream`

One call. The harness classifies the goal, picks the right path, and runs it.

```python
from harness.events import EventType

async for event in runtime.dispatch_stream("investigate the GPU spike on worker-07"):
    if event.type == EventType.DISPATCH:
        print(f"complexity={event.payload['complexity']} path={event.payload['path']}")
    elif event.type == EventType.ROUTE:
        print(f"→ {event.payload['agent_id']}: {event.payload['rationale']}")
    elif event.type == EventType.ACTION:
        print(f"  tool: {event.payload['tool']}")
    elif event.type == EventType.DONE:
        print(event.payload["answer"])
    elif event.type == EventType.TASK_DONE:
        print(event.payload["answer"])   # routed path ends here, no DONE

result = await runtime.dispatch("what is the capital of France?")  # blocking
```

**Classification logic:**
- 1 agent registered → always `"simple"`, no LLM call made.
- Multiple agents → one cheap LLM call classifies `"simple"` (one agent handles
  it end-to-end) or `"complex"` (benefits from task decomposition and specialist
  routing). Unknown response defaults to `"simple"`.
- `"simple"` path: LLM router picks the best agent → `run_routed_stream`.
- `"complex"` path: planner decomposes into a DAG → `run_stream`.

---

Four lower-level paths are available when you need explicit control:

### 1. Routed — `run_routed` / `run_routed_stream`

One lightweight LLM call picks the best agent, then that agent runs its
ReAct loop directly. No task decomposition, no synthesis step. Use this
for single-turn goals where one agent handles everything end-to-end.

```python
from harness.events import EventType

async for event in runtime.run_routed_stream("what is the current disk usage?"):
    if event.type == EventType.ROUTE:
        print(f"→ {event.payload['agent_id']}: {event.payload['rationale']}")
    elif event.type == EventType.ACTION:
        print(f"  tool: {event.payload['tool']}")
    elif event.type == EventType.TASK_DONE:
        print(event.payload["answer"])
```

Fast-path: if only one agent is registered, `route()` returns it immediately
without an LLM call.

### 2. Direct — `run_agent` / `run_agent_stream`

You name the agent; it runs its ReAct loop directly. Use when you already
know which agent to call and want to skip routing entirely.

```python
result = await runtime.run_agent("researcher", "summarise this document")
```

### 3. Orchestrated — `run` / `run_stream`

The planner decomposes the goal into a task DAG, assigns tasks to specialist
agents, runs them (in parallel where dependencies allow), and synthesises a
final answer. Use for multi-agent goals where different specialists handle
different parts of the work.

```python
async for event in runtime.run_stream("investigate GPU spike on worker-07"):
    if event.type == EventType.PLAN:
        for t in event.payload["plan"]["tasks"]:
            print(f"  {t['id']}@{t['agent_id']}: {t['instruction']}")
    elif event.type == EventType.REPLAN:
        print(f"[replan #{event.payload['replan_count']}]")
    elif event.type == EventType.DONE:
        print(event.payload["answer"])
```

### 4. Pre-built — `run_with_plan` / `run_with_plan_stream`

Supply a hand-written `Plan` and bypass the LLM planner entirely. Use
this for deterministic, repeatable workflows where the decomposition is
known upfront — CI pipelines, ETL jobs, scheduled tasks. The plan is
validated against registered agents before execution; everything
downstream (parallel batches, replan-on-failure, synthesis, memory
writes, steering) is identical to `run_stream`.

```python
from orchestrator.planner import Plan, Task

plan = Plan([
    Task("t1", "analyst",  "Analyse error logs from the last hour"),
    Task("t2", "reporter", "Write an incident summary", depends_on=["t1"]),
])

# streaming
async for event in runtime.run_with_plan_stream(plan, goal="Incident report"):
    if event.type == EventType.DONE:
        print(event.payload["answer"])

# blocking
result = await runtime.run_with_plan(plan, goal="Incident report")
```

The `goal` string is passed to the synthesiser and used for memory
context injection into agents — even though the plan shape is fixed, the
agents themselves still read from memory.

If a task fails mid-run and `on_failure="replan"`, the replan call does
go to the LLM — the bypass is for the *initial* plan only.

---

Event types by path:

| Event | Dispatch | Routed | Direct | Orchestrated | Pre-built |
|---|---|---|---|---|---|
| `DISPATCH` | ✓ | — | — | — | — |
| `ROUTE` | ✓ (simple) | ✓ | — | — | — |
| `THOUGHT` / `TOKEN` / `ACTION` / `OBSERVATION` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TASK_DONE` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PLAN` / `REPLAN` / `SYNTHESIS` / `DONE` | ✓ (complex) | — | — | ✓ | ✓ |
| `ERROR` | ✓ | ✓ | ✓ | ✓ | ✓ |

`TOKEN` events fire only when your LLM client exposes
`async def stream_complete(system, messages) -> AsyncGenerator[str, None]`.
Non-streaming clients still work — they emit the full response in one
`THOUGHT` event per step.

## Console rendering

`ConsoleRenderer` handles all `BusEvent` types with consistent label
and truncation formatting so event-loop boilerplate stays out of your
scripts.

```python
from harness.console import ConsoleRenderer, trunc

renderer = ConsoleRenderer(
    truncate=140,          # max chars for long text fields
    sep_char="─",          # separator character
    sep_width=72,          # separator width
    agent_label_width=16,  # width of [agent_id] column
    show_tokens=False,     # True to print TOKEN events inline
)

async for event in runtime.dispatch_stream(goal):
    renderer.render(event)   # handles every EventType
```

For events with custom section headers (e.g. a "PROJECT HEALTH REPORT"
block), handle that event yourself and skip `render` for it — the
renderer is additive:

```python
async for event in runtime.run_stream(goal):
    if event.type == EventType.DONE:
        renderer.sep("═")
        print("MY CUSTOM HEADER")
        renderer.sep("═")
        print(event.payload["answer"])
    else:
        renderer.render(event)
```

`trunc(s, n)` is exported for standalone use when you need to truncate
a string to `n` characters with a trailing `…`.

## Working memory budget

`AgentConfig.working_memory_max_tokens` controls per-agent eviction (default
`8000`). Counting defaults to a `chars/4` heuristic (stable for code/JSON/text
within ~10–20% of real BPE counts, zero deps). For exact counts plug your own
counter into `WorkingMemory` directly:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
wm = WorkingMemory(llm=..., token_counter=lambda s: len(enc.encode(s)))
```

## Cost tracking

The harness deliberately **does not maintain a pricing table** — tables go
stale, and per-org rollups don't belong in an SDK. Token counts come from the
provider authoritatively (free). Dollars are optional and follow this
precedence, per call:

**1. Gateway header (recommended).** Route the OpenAI client through LiteLLM
proxy, Helicone, OpenRouter via `base_url=...` and `OpenAILLM` reads
`x-litellm-response-cost` / `x-cost-usd` / `x-helicone-cost-usd` from the
response automatically. Real cost, gateway-maintained pricing.

```python
llm = OpenAILLM(model="gpt-4o-mini", base_url="https://my-litellm/v1")
```

**2. `cost_fn(usage) -> float`** — caller-supplied. For when you hit a
provider directly and want a local pricing function:

```python
def my_pricing(usage):
    rates = {"gpt-4o-mini": (0.15e-6, 0.60e-6)}     # (input, output) USD/token
    served = usage.get("model", "")
    for prefix, (in_rate, out_rate) in rates.items():
        if served.startswith(prefix):
            return usage["tokens_in"] * in_rate + usage["tokens_out"] * out_rate
    return 0.0

llm = OpenAILLM(model="gpt-4o-mini", cost_fn=my_pricing)
```

**3. Neither.** Tokens still flow on every call (visible via `last_usage` and
in OTEL span attributes when OTEL is enabled). `cost_usd` is just omitted. No
crash, no surprise charges from a stale rate table.

`AgentRuntime` wires a fresh `BudgetGuard` per run and calls
`llm.set_budget(guard)` automatically (duck-typed; safe if the adapter
doesn't support it). When `max_total_cost_usd` is exceeded, the next
`BudgetGuard.check()` raises and aborts the run. The final `DONE` event
payload carries `cost_usd` and `elapsed_seconds` at the top level:

```python
async for event in runtime.dispatch_stream(goal):
    if event.type == EventType.DONE:
        print(f"${event.payload['cost_usd']:.4f} in {event.payload['elapsed_seconds']:.1f}s")
```

Cost ceiling fires on the *next* `check()` (start of next ReAct step or
orchestrator batch), not synchronously mid-call — accept this for 0.0.1, the
guard's job is preventing runaway loops, not bounding individual calls.

### Token limits + per-call-site breakdown

`GuardrailConfig.max_input_tokens` / `max_output_tokens` cap raw token
usage independently of dollar cost. This is the only enforcement available
to subscription-auth runs (`ClaudeCodeLLM`, `OpenAICodexLLM`) — those tiers
don't expose pricing, so cost stays 0 and only token caps can fire.

```python
runtime = AgentRuntime(
    ...,
    guardrail_config=GuardrailConfig(
        max_total_cost_usd=2.0,
        max_input_tokens=100_000,
        max_output_tokens=20_000,
    ),
)
```

Per-call-site attribution lives on the terminal event's `budget` payload
— a snapshot of spending bucketed by the LLM slot that ran each call.
The runtime tags classifier / router / planner / synthesizer calls
automatically; ReAct agent calls go into the totals but don't get a
bucket. So `cheap` (used for both `classifier_llm` and `router_llm`) and
`premium` (used for `planner_llm`) report separately even though one is
the same physical LLM instance shared across slots:

```python
async for event in runtime.dispatch_stream(goal):
    # Routed (simple) goals terminate with TASK_DONE; orchestrated goals
    # with DONE. Both carry the same ``budget`` shape.
    if event.type in (EventType.TASK_DONE, EventType.DONE):
        budget = event.payload["budget"]
        print(f"total: in={budget['tokens_in']} out={budget['tokens_out']} "
              f"${budget['cost_usd']:.4f}")
        for slot, stats in budget["breakdown"].items():
            print(f"  {slot}: in={stats['tokens_in']} out={stats['tokens_out']}")
```

The same `budget` dict is attached to `runtime.run(...)` and
`runtime.dispatch(...)` return values under the `budget` key, so blocking
callers don't need to read events.

Anthropic / Claude Code adapters count input tokens as the *total* that
hit the wire (non-cached + cache-creation + cache-read), so token caps
reflect actual consumption regardless of cache hit rate. Cost calculation
via `cost_fn` still respects cache pricing.

### Evals via the trace recorder

There's no shipped evals framework — opinions on scorers, judge models,
and golden-set management belong outside the orchestration core. The
[trace recorder](#trace-recorder--replay--local-viewer) already writes
per-event token/cost/latency to JSONL, so a few lines of glue cover most
in-house eval setups:

```python
import json
from harness.trace import record_trace

# 1. Record traces while running a fixture set.
for fixture in fixtures:
    async for _event in record_trace(
        runtime.dispatch_stream(fixture["input"]),
        path=f"runs/{fixture['id']}.jsonl",
    ):
        pass

# 2. Score offline by replaying.
def score_run(path: str, expected: str) -> dict:
    answer = ""
    budget = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "breakdown": {}}
    for line in open(path):
        event = json.loads(line)
        if event["type"] in ("done", "task_done"):
            answer = event["payload"].get("answer", "")
            budget = event["payload"].get("budget", budget)
    return {
        "success": expected.lower() in answer.lower(),
        **budget,  # tokens_in, tokens_out, cost_usd, breakdown
    }
```

Plug in your own scorer (exact-match, LLM-judge, semantic similarity) on
top. External tools like Braintrust, LangSmith, and Weave are
purpose-built for this and ingest the same JSONL shape directly.

## Tool execution

Tools that shell out (`kubectl`, `curl`, `sh -c …`) should not run inside the
agent process. `ExecutorBridge` provides a controlled subprocess launcher with
two backends.

### What every backend enforces

- **Tool allowlist** — set at startup; the LLM cannot extend it.
- **Wall-clock timeout** per call.
- **Output size cap** (default 1 MiB, configurable).

### `backend="none"` (default) — Rust executor

Routes each call through the compiled Rust binary at `executor/`. Adds
process-level isolation (a tool crash cannot reach the agent) and a scrubbed
environment (only `PATH` is forwarded). Does **not** provide syscall filtering,
filesystem namespacing, or network isolation.

```bash
cargo install --path executor   # installs ah-executor to ~/.cargo/bin
```

```python
from harness.executor_bridge import ExecutorBridge, ExecutorConfig, ExecutorTool

# binary_path auto-discovered from PATH via shutil.which("ah-executor")
bridge = ExecutorBridge(ExecutorConfig(
    allowed_tools=("kubectl", "curl"),   # "shell" is opt-in only
))

kubectl_tool = ExecutorTool("kubectl", "kubectl", bridge, arg_key="args")
```

### `backend="docker"` — real OS-level isolation

Each tool call runs in a fresh Docker container. Provides network isolation,
read-only filesystem, and memory/CPU limits. The Rust binary is not used in
this mode. Requires Docker daemon on the host.

```python
from harness.executor_bridge import ExecutorBridge, ExecutorConfig, ExecutorTool

bridge = ExecutorBridge(ExecutorConfig(
    allowed_tools=("kubectl",),
    backend="docker",
    docker_image="bitnami/kubectl:latest",
    docker_network="none",      # no outbound network
    docker_memory="256m",
    docker_cpus="1.0",
    docker_read_only=True,
))

kubectl_tool = ExecutorTool("kubectl", "kubectl", bridge, arg_key="args")
```

The `shell` tool (dict-style args) works with both backends:

```python
shell_tool = ExecutorTool("shell", "shell", bridge)
# LLM calls: {"action": "shell", "args": {"cmd": "jq '.name' data.json"}}
```

## Tests

```bash
pytest
```

## MCP Tools

Connect any [MCP](https://modelcontextprotocol.io)-compatible server and its
tools become available to agents — no wrapper code needed.

```bash
pip install -e ".[mcp]"
```

```python
from mcp import StdioServerParameters
from tools.mcp import MCPServerConnection

params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

async with MCPServerConnection(params, server_name="filesystem") as conn:
    conn.register_tools(tool_registry)          # bulk-register all discovered tools
    agents.register(AgentConfig(
        agent_id="explorer",
        role="explores the filesystem",
        system_prompt="...",
        allowed_tools=conn.tool_names,           # auto-populated from MCP server
    ))
    result = await runtime.run("list files in /tmp")
```

Supports **stdio**, **SSE**, and **streamable-HTTP** transports. The
`MCPServerConnection` context manager handles the full lifecycle —
connect, discover, and cleanup.

### Auth options

Pick the provider that matches how your MCP server authenticates:

| Provider | When to use |
|---|---|
| `StaticMCPAuth` | Literal header/env values you have in hand |
| `BearerMCPAuth` | A single bearer token string |
| `ApiKeyMCPAuth` | API-key headers backed by environment variables |
| `OAuthMCPAuth` | Bearer token cached in the shared `auth.json` file |
| `BrowserOAuthMCPAuth` | Full OAuth 2.0 + PKCE flow with browser login |

**API keys backed by env vars** — generic, no vendor coupling:

```python
import os
from tools.mcp.auth import ApiKeyMCPAuth, StreamableHttpServerParams
from tools.mcp import MCPServerConnection

auth = ApiKeyMCPAuth({
    "DD-Api-Key": "DD_API_KEY",
    "DD-Application-Key": "DD_APP_KEY",
})
params = StreamableHttpServerParams(url="https://mcp.datadoghq.com/")

async with MCPServerConnection(params, server_name="datadog", auth=auth) as conn:
    conn.register_tools(tool_registry)
```

**Browser-based OAuth (PKCE) for hosted MCP servers**:

```python
from tools.mcp.auth import BrowserOAuthMCPAuth, StreamableHttpServerParams
from tools.mcp import MCPServerConnection

auth = BrowserOAuthMCPAuth(
    server_url="https://mcp.example.com/",
    provider_name="mcp:example",
    client_id="abc123",         # from the provider's developer console
    client_secret="shh",        # optional (PKCE-only flows omit)
    scopes=["read", "write"],
)
params = StreamableHttpServerParams(url="https://mcp.example.com/")

async with MCPServerConnection(params, auth=auth) as conn:
    conn.register_tools(tool_registry)
```

First connect opens the browser, captures the redirect on
`http://127.0.0.1:8765/callback`, persists tokens to
`~/.agent-harness/auth/auth.json` (chmod 0600), and refreshes them
transparently on every subsequent run. Register your OAuth app with that
redirect URI.

Servers that support RFC 7591 dynamic client registration work without
supplying `client_id` — the MCP SDK registers a fresh client on first
connect.

**Cached OAuth from the auth.json file** (for tokens you already minted
elsewhere):

```python
from tools.mcp import OAuthMCPAuth, MCPServerConnection

auth = OAuthMCPAuth.from_auth_file(
    "~/.agent-harness/auth/auth.json",
    provider="my-service",
)
```

See `examples/mcp_demo.py` for local stdio MCP and `examples/mcp_auth_demo.py`
for authenticated remote MCP.

## OpenTelemetry Tracing

Visualize agent runs in Jaeger, Datadog, or any OTEL-compatible backend.

```bash
pip install -e ".[otel]"
```

One flag enables tracing:

```python
runtime = AgentRuntime(
    agent_registry=agents,
    tool_registry=tools,
    memory=memory,
    llm=llm,
    enable_otel=True,           # ← that's it
)
```

Span hierarchy:

```
[run]  goal="Fetch httpbin.org/json..."
  ├── [plan]         task_count=2
  ├── [task]         agent=researcher, task_id=t1
  │     ├── thought  "I need to fetch..."
  │     ├── action   tool=http_fetch
  │     └── thought  "Got the response..."
  ├── [task]         agent=researcher, task_id=t2
  │     └── thought  "Extracting title..."
  └── [synthesis]    confidence=0.95
```

Local Jaeger setup:

```bash
# Start Jaeger (OTLP on :4318, UI on :16686)
docker run -d --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Run your agent
OPENAI_API_KEY=sk-... python examples/openai_demo.py

# View traces at http://localhost:16686
```

The OTEL hook is a side-channel on the existing `Tracer` — the in-memory trace
is always available via `result["trace"]` regardless of whether OTEL is enabled.
Zero overhead and zero imports when `enable_otel=False`.

## Trace recorder + replay + local viewer

For local debug and post-mortem inspection without an OTEL backend, the
harness ships a JSONL trace recorder and a stdlib-only HTML viewer. Wrap
any streaming call:

```python
from harness.trace import record_trace, replay

async for event in record_trace(runtime.dispatch_stream(goal), "run.jsonl"):
    ...  # your normal handling
```

Each `BusEvent` is flushed per-line, so a partial trace survives a crash.
View the trace in your browser:

```bash
agent-harness trace view run.jsonl     # opens http://127.0.0.1:8765/
```

The viewer is a single embedded HTML page — vertical timeline, filter by
agent / event type / text, expandable per-event JSON. No build step, no
external services.

Replay a trace through `ConsoleRenderer` (great for grepping or piping
into another script):

```bash
agent-harness trace replay run.jsonl
agent-harness trace replay run.jsonl --realtime --speed 2.0
```

Programmatic replay yields reconstructed `BusEvent` objects:

```python
async for event in replay("run.jsonl", realtime=False):
    ...  # reuse the same loops you write for live streams
```

This is complementary to OTEL — OTEL is for production observability and
long-term storage in Jaeger/Datadog; the JSONL recorder is for local
debugging, sharing reproductions, and replaying past runs.

## Vision / multimodal agents

`WorkingMemory` accepts `str | list` content so image blocks pass through to
vision-capable LLMs without modification.

```bash
pip install -e ".[openai,http]"
```

```python
from tools.builtin.fetch_image import FetchImage

tools.register(FetchImage())

agents.register(AgentConfig(
    agent_id="vision_agent",
    role="fetches images and describes their visual content",
    system_prompt="Use fetch_image to retrieve images — you can see them directly.",
    allowed_tools=["fetch_image"],
    working_memory_max_tokens=16_000,   # images use ~500 tokens each in budget
))

result = await runtime.run_agent("vision_agent", "describe https://example.com/photo.jpg")
```

`FetchImage` downloads the URL, base64-encodes the body, and returns an OpenAI
`image_url` content block. The agent appends it to `WorkingMemory` as a content
list; the LLM receives the actual image. `OBSERVATION` events and the
summarization LLM see `[image]` as a placeholder so text-only paths are never
handed raw base64.

Image token budget: a fixed `500` token estimate per image block (conservative
mid-point of GPT-4o `auto` detail range). Override with a real counter if you
need exact figures.

## Trajectory capture and RLHF

Every agent run automatically logs its full `WorkingMemory` message history as
a `"trajectory"` tracer event. Wire `InMemoryAnnotationStore` to collect and
rate trajectories:

```python
from harness.annotation import InMemoryAnnotationStore

store = InMemoryAnnotationStore()
runtime = AgentRuntime(..., annotation_store=store)

await runtime.run_agent("my_agent", "task")

# drain unrated trajectories
for annotation in store.list_unrated():
    print(annotation.agent_id, annotation.answer, annotation.confidence)
    store.rate(annotation.annotation_id, rating=0.9)          # 1.0 = ideal

# export to training pipeline
training_data = store.list_all()
```

`Annotation` fields:

| Field | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | Full `WorkingMemory` trajectory — system prompt, every thought, tool call, observation, and final answer |
| `answer` | `str` | Agent's final answer (`""` on failure) |
| `confidence` | `float` | Agent's self-reported confidence `[0, 1]` |
| `steps` | `int` | ReAct steps taken |
| `error` | `str` | `""` on success; failure reason otherwise |
| `summarization_count` | `int` | Number of `WorkingMemory` compression passes |
| `rating` | `float \| None` | Human rating `[0, 1]` — `None` until rated |
| `correction` | `str \| None` | Human-supplied correct answer when `rating < 1` |

Trajectory capture fires in a `finally` block — it records on success, on
max-steps exhaustion, and on crash. Swap `InMemoryAnnotationStore` for any
backend that implements the same interface (`write`, `get`, `list_all`,
`list_run`, `list_unrated`, `rate`, `count`).

## Human-in-the-Loop (HITL)

Gate specific tool calls behind an interactive CLI prompt. Opt-in per agent via
`hitl_tools`; zero overhead when unused. No extra dependencies — checkpoints
are stored as JSON files by default.

```python
agents.register(AgentConfig(
    agent_id="file_agent",
    role="manages files",
    system_prompt="...",
    allowed_tools=["read_file", "write_file", "delete_file"],
    hitl_tools=["write_file", "delete_file"],   # these two require human approval
))

# AgentRuntime auto-creates a FileCheckpointStore when hitl_tools are present.
runtime = AgentRuntime(...)
await runtime.run_agent("file_agent", "clean up the logs directory")
```

Checkpoints are written to `~/.agent-harness/checkpoints/` by default.
Override the directory:

```python
from harness.checkpoint import FileCheckpointStore

runtime = AgentRuntime(..., checkpoint_store=FileCheckpointStore("/var/lib/myapp/ckp"))
```

For Redis-backed storage (shared across processes or machines):

```python
import redis.asyncio as aioredis
from harness.checkpoint import RedisCheckpointStore

client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
runtime = AgentRuntime(..., checkpoint_store=RedisCheckpointStore(client))
```

When the agent calls `write_file` or `delete_file` a prompt appears:

```
────────────────────────────────────────────────────────────
  HITL Approval Required
────────────────────────────────────────────────────────────
  Tool:  delete_file
  Args:  {"path": "/var/log/app.log"}
  Agent: file_agent  step=2
  Run:   3f7a1b2c-...:file_agent
  ID:    a1b2-c3d4
────────────────────────────────────────────────────────────
  y = approve once  |  a = allow 'delete_file' for session  |  A = always allow 'delete_file'  |  n = reject  |  <text> = steer
  Ctrl-C to pause. Resume: python my_script.py --resume 3f7a1b2c-...:file_agent
────────────────────────────────────────────────────────────
  Approve? [y/n/a/A/correction]:
```

**Prompt semantics:**

| Input | Effect |
|---|---|
| `y` / `yes` | Tool runs once |
| `n` / `no` | Tool skipped; agent sees a rejection observation |
| `a` / `allow` | Tool runs **and** added to session allow-list; no further prompts for this tool (or command prefix for shell-like tools) |
| `A` / `always` | Tool runs **and** a user-scoped allow rule is stored in `~/.agent-harness/policies/tool_policy.json` |
| any other text | Correction: tool skipped, text injected into `WorkingMemory` as a user message; LLM self-corrects on the next step |

For shell-like tools (`shell`, `bash`, `run`, `exec`), `a` and `A` allow the
**first word** of the command — e.g. approving `shell git commit ...` allows
all `git` commands in that scope but still prompts for `shell rm ...`.
Persistent rules are user-local, not repo files. Manage them with:

```bash
agent-harness policy list
agent-harness policy revoke <rule-id>
agent-harness policy clear
```

**Wall-time budget** is suspended while waiting for input — human think-time
does not count against `max_wall_time_seconds`.

### Step-level checkpointing

Enable periodic crash-resume independent of HITL:

```python
AgentConfig(
    agent_id="long_runner",
    ...
    checkpoint_every=3,   # checkpoint before every 3rd step (0 = disabled)
)
```

The same `CheckpointStore` is used for both HITL and step checkpoints. Resume
works with `runtime.resume(key)` regardless of how the checkpoint was created.

### Checkpoint namespacing

Each agent writes to its own key so orchestrated runs never overwrite each other:

| Path | Checkpoint key | Stored at |
|---|---|---|
| Single-agent (`run_agent`, `run_routed`) | `<run_id>:<agent_id>` | `~/.agent-harness/checkpoints/<run_id>:<agent_id>.json` |
| Orchestrated (`run`, `run_stream`) | `<run_id>` (orchestrator) + `<run_id>:<agent_id>` (each agent) | one file per agent, one file for the orchestrator |

The orchestrator checkpoint stores the goal, the full plan, completed task
results, and the replan count. It is updated after each parallel batch
completes and deleted on clean `DONE`.

### Crash / Ctrl-C resume

The checkpoint (step number + full `WorkingMemory`) is written before every
HITL prompt and (if `checkpoint_every > 0`) at each periodic step.

**What the banner prints:**

- **Single-agent run**: `--resume <run_id>:<agent_id>` — restores just that agent.
- **Orchestrated run**: `--resume <run_id>` — restores the full orchestration.

```
  Run interrupted — checkpoint saved.
  Resume: python my_script.py --resume 3f7a1b2c-...
```

**Auto-resume — no script changes required.** When `checkpoint_store` is
configured, `dispatch_stream` and `run_stream` detect `--resume <key>` in
`sys.argv` automatically. Your existing script resumes transparently:

```bash
python my_script.py --resume 3f7a1b2c-...
```

The runtime detects the flag, loads the checkpoint, and streams events
identically to a fresh run. Scripts need zero resume-specific code.

For **explicit control** — streaming resume or blocking resume:

```python
# streaming (same event sequence as the original run)
async for event in runtime.resume_stream("3f7a1b2c-..."):
    ...

# blocking
result = await runtime.resume("3f7a1b2c-...:file_agent")  # single-agent
result = await runtime.resume("3f7a1b2c-...")              # orchestrated
```

Both `resume_stream` and `resume` auto-detect the checkpoint type (agent vs
orchestrator) from the stored data and call the right path.

If you need the resume key from `sys.argv` directly:

```python
from harness.checkpoint import maybe_resume_key

key = maybe_resume_key()   # returns None if --resume is absent
```

**Orchestrated resume** skips completed tasks (injects their stored results
directly into the synthesis step) and re-runs only the tasks that had not yet
finished. If an individual agent's HITL checkpoint is still on disk, that agent
is resumed at its saved step rather than re-run from scratch.

### Correction steering and replanning

When the human types a correction instead of y/n:

- **Single-agent run**: correction is injected as a `user` message in
  `WorkingMemory`. The LLM sees it on the next think step and self-corrects
  without replanning. Suitable for redirecting tool choice or adjusting
  parameters.
- **Orchestrated run**: the correction steers only the current agent. Because
  the orchestrator checkpoint records task results as they complete, a full
  `runtime.resume(run_id)` after the agent finishes will continue the remaining
  tasks with correct upstream context.

The `annotation_store` and `checkpoint_store` are independent — both can be
wired simultaneously for RLHF data collection with HITL review.

## Async steering

HITL is synchronous — it only fires when a gated tool is about to run. For
out-of-band course-correction (HTTP handler, supervisor agent, file watcher,
or a human typing in the terminal), each `BaseAgent` exposes a
non-blocking `steer(text)` method. Items are drained at the **top of each
ReAct iteration**, before the per-step checkpoint write and before the
next think, then appended to `WorkingMemory` as a `Human guidance: <text>`
user message. The LLM sees them on the next think and adjusts. One
`HUMAN_GUIDANCE` `BusEvent` fires per drained item.

Why a queue instead of writing straight to `WorkingMemory`: `steer()` is
synchronous and callable from any coroutine; `WorkingMemory.append` is
async (eviction can call the LLM). The queue is the producer/consumer
boundary, enforces step-boundary delivery, and keeps WM single-writer.

### Programmatic API (always available)

```python
agent.steer("skip the legal database, use academic sources only")
```

Fires immediately; the agent picks it up at the next step boundary.
Worst-case latency = remaining tool time + next-think time.

### Sources via factory (so orchestrated agents are reachable)

`BaseAgent` and `AgentRuntime` both accept `steering_source_factory` — a
callable `(agent) -> async ctx mgr`. The agent enters the source on
`run_stream`, exits on completion. No live-agent registry; agents the
runtime constructs internally still get steering.

Two built-in factories:

```python
from harness.steering import file_steering_factory, stdin_steering_factory

# 1. File-based — one file per agent, polled for appends (no shared resource)
runtime = AgentRuntime(
    ...,
    steering_source_factory=file_steering_factory(
        "/tmp/ah-{run_id}-{agent_id}.steer"
    ),
)
# Steer from any other terminal:
#   echo "wrap up and synthesise" >> /tmp/ah-<run_id>-researcher.steer

# 2. Stdin-based — single shared StdinRouter with prefix routing
runtime = AgentRuntime(
    ...,
    steering_source_factory=stdin_steering_factory(),
)
# At the terminal:
#   researcher: skip the legal db, focus on academic
#   writer:     keep the report under 500 words
#   *:          stop after this step
```

Single-agent stdin runs accept lines with no prefix. Multi-agent runs
require `agent_id: text` (or `*: text` for broadcast); unknown or
unprefixed lines print a stderr hint and are discarded.

The stdin factory's underlying `StdinRouter` is started/stopped
automatically — the runtime detects the factory's async-context-manager
shape and wraps `dispatch_stream` / `run_stream` / `run_routed_stream`
around it. Ref-counted so nested calls (`dispatch_stream → run_stream`)
don't double-start the router.

### HITL coordination

When a `StdinRouter` is active, HITL calls `router.claim_next_line()`
**before** printing its approval banner — the next stdin line resolves
HITL's pending Future and bypasses pub/sub. After resolution, subsequent
lines route to steering subscribers normally. When no router is active,
HITL falls back to a standalone `prompt_toolkit` session, ensuring consistent
key-bindings (like Enter-submits and Alt-Enter/Ctrl-J-newline) across both paths.

### Constraints

- Steering arrives **between steps**, never mid-tool, never mid-think.
  Tools that are already running complete; the LLM stream that's
  already producing completes; guidance lands at the next safe boundary.
- Guidance queued **after** the LLM emits `action: "finish"` is lost —
  the agent already decided it's done.
- Crash between drain and next checkpoint write → the queued items are
  in the persisted WM. Crash between checkpoint write and next drain →
  lost; re-steer after `--resume`.

See `examples/complex_sysaudit_demo.py` for stdin steering across three
agents alongside HITL on the shell tool.

## AgentConfig reference

| Field | Default | Description |
|---|---|---|
| `agent_id` | required | Unique identifier for the agent |
| `role` | required | Plain-English description used by the planner for agent selection |
| `system_prompt` | required | Base system prompt for the agent |
| `allowed_tools` | required | Tool names the agent may call |
| `max_steps` | `10` | Maximum ReAct iterations before the run is terminated |
| `max_wall_time_seconds` | (guardrail) | See `GuardrailConfig` |
| `memory_context_enabled` | `True` | Prepend relevant long-term memory to the system prompt |
| `confidence_from_llm` | `True` | Use the `confidence` field from the LLM response; set `False` to always return `1.0` |
| `working_memory_max_tokens` | `8000` | Token budget for in-context working memory before rolling summarisation kicks in |
| `hitl_tools` | `[]` | Tool names that require human approval before execution |
| `checkpoint_every` | `0` | Write a crash-resumable checkpoint every N steps; `0` disables periodic checkpoints |
| `stream_tokens` | `False` | Emit `TOKEN` events as the LLM streams. Disabled by default — enable if you want to render partial output in real time: `AgentConfig(..., stream_tokens=True)` |
