# react-agent-harness

Bring-your-own-LLM multi-agent harness: hybrid DAG planning with replan-on-failure,
two-tier memory (semantic KV + episodic vector), and a streaming-primary event model.

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
harness/runtime.py          AgentRuntime — single entry point, wire once run anything
harness/events.py           BusEvent + EventType — canonical event vocabulary
harness/llm/openai.py       OpenAILLM — OpenAI adapter with usage + cost tracking
harness/annotation.py       Annotation store + AnnotationHook — RLHF trajectory capture
harness/hitl.py             HITL approval gate — interactive CLI, session-allow list
harness/steering.py         Async steering — agent.steer(text), StdinRouter pub/sub, FileSteer, factory helpers
harness/checkpoint.py       CheckpointStore + _ResumeHint + maybe_resume_key — pluggable run-state persistence (file + Redis); auto-resume built into dispatch_stream / run_stream
harness/otel.py             OTELHook — OpenTelemetry span exporter (opt-in)
harness/executor_bridge.py  ExecutorBridge + ExecutorTool — controlled subprocess launcher with optional Docker sandboxing
orchestrator/planner.py     Hybrid DAG orchestrator — plan, replan, synthesize
agents/base.py              Generic BaseAgent — ReAct loop, no subclassing needed
memory/manager.py           MemoryManager — semantic KV + episodic vector
memory/working.py           WorkingMemory — LLM summarization eviction, checkpoint/restore
memory/episodic_lance.py    LanceDB episodic store — IVF_PQ ANN, batch writes
memory/redis_store.py       Redis semantic store — durable KV with TTL
memory/stores.py            InMemory stores — local dev default, no deps
tools/builtin/http_fetch.py HTTPFetch — minimal read-only GET tool
tools/builtin/fetch_image.py FetchImage — fetch URL and return OpenAI image_url block
tools/mcp/adapter.py        MCP tool adapter — connect any MCP server
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

Event types by path:

| Event | Dispatch | Routed | Direct | Orchestrated |
|---|---|---|---|---|
| `DISPATCH` | ✓ | — | — | — |
| `ROUTE` | ✓ (simple) | ✓ | — | — |
| `THOUGHT` / `TOKEN` / `ACTION` / `OBSERVATION` | ✓ | ✓ | ✓ | ✓ |
| `TASK_DONE` | ✓ | ✓ | ✓ | ✓ |
| `PLAN` / `REPLAN` / `SYNTHESIS` / `DONE` | ✓ (complex) | — | — | ✓ |
| `ERROR` | ✓ | ✓ | ✓ | ✓ |

`TOKEN` events fire only when your LLM client exposes
`async def stream_complete(system, messages) -> AsyncGenerator[str, None]`.
Non-streaming clients still work — they emit the full response in one
`THOUGHT` event per step.

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

Supports **stdio** and **SSE** transports. The `MCPServerConnection` context
manager handles the full lifecycle — connect, discover, and cleanup.

See `examples/mcp_demo.py` for a runnable example.

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
  y = approve once  |  a = allow 'delete_file' for session  |  n = reject  |  <text> = steer
  Ctrl-C to pause. Resume: python my_script.py --resume 3f7a1b2c-...:file_agent
────────────────────────────────────────────────────────────
  Approve? [y/n/a/correction]:
```

**Prompt semantics:**

| Input | Effect |
|---|---|
| `y` / `yes` | Tool runs once |
| `n` / `no` | Tool skipped; agent sees a rejection observation |
| `a` / `allow` | Tool runs **and** added to session allow-list; no further prompts for this tool (or command prefix for shell-like tools) |
| any other text | Correction: tool skipped, text injected into `WorkingMemory` as a user message; LLM self-corrects on the next step |

For shell-like tools (`shell`, `bash`, `run`, `exec`), `a` allows the **first
word** of the command — e.g. typing `a` when approving `shell git commit ...`
allows all `git` commands for the session but still prompts for `shell rm ...`.

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
