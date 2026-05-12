# agent-harness

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
harness/otel.py             OTELHook — OpenTelemetry span exporter (opt-in)
harness/executor_bridge.py  ExecutorBridge + ExecutorTool — controlled subprocess launcher with optional Docker sandboxing
orchestrator/planner.py     Hybrid DAG orchestrator — plan, replan, synthesize
agents/base.py              Generic BaseAgent — ReAct loop, no subclassing needed
memory/manager.py           MemoryManager — semantic KV + episodic vector
memory/working.py           WorkingMemory — LLM summarization eviction
memory/episodic_lance.py    LanceDB episodic store — IVF_PQ ANN, batch writes
memory/redis_store.py       Redis semantic store — durable KV with TTL
memory/stores.py            InMemory stores — local dev default, no deps
tools/builtin/http_fetch.py HTTPFetch — minimal read-only GET tool
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
| `examples/complex_sysaudit_demo.py` | Three heterogeneous agents in parallel: `shell_agent` (ah-executor), `filesystem_agent` (MCP), `web_agent` (HTTPFetch) — orchestrated path, DAG plan, synthesis. | `OPENAI_API_KEY`, `[openai,http,mcp]`, `ah-executor`, `npx` |
| `examples/executor_bridge_demo.py` | `ExecutorBridge` backends side-by-side: allowlist, env scrubbing, Docker network/fs isolation, timeout, positional-arg tools. | `ah-executor` and/or Docker |
| `examples/durable_memory_demo.py` | Redis (semantic) + LanceDB (episodic) memory persistence across two related goals. | `OPENAI_API_KEY`, `[openai,redis,lance]`, Redis reachable |
| `examples/mcp_demo.py` | Connects to an MCP filesystem server and gives the agent its tools. | `OPENAI_API_KEY`, `[openai,mcp]`, `npx` |

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
