# agent-harness

Bring-your-own-LLM multi-agent harness: hybrid DAG planning with replan-on-failure,
two-tier memory (semantic KV + episodic vector), and a streaming-primary event model.

Config-driven — register tools and agents, run any goal. No subclassing.

## Install

```bash
pip install -e ".[dev]"                   # core + tests + lint
pip install -e ".[lance,redis,dev]"       # also install LanceDB + Redis stores
```

The core package has no runtime dependencies. The `lance`, `redis`,
`anthropic`, and `openai` extras pull in heavier libs only if you opt in.

## Quickstart

```bash
python examples/quickstart.py
```

Runs end-to-end against a deterministic mock LLM — no API keys needed.
Swap `MockLLM` for an Anthropic or OpenAI client to run against a real model.

## Architecture

```
harness/runtime.py          AgentRuntime — single entry point, wire once run anything
harness/events.py           BusEvent + EventType — canonical event vocabulary
harness/sandbox.py          Sandbox + SandboxedTool — Rust-executor adapter
orchestrator/planner.py     Hybrid DAG orchestrator — plan, replan, synthesize
agents/base.py              Generic BaseAgent — ReAct loop, no subclassing needed
memory/manager.py           MemoryManager — semantic KV + episodic vector
memory/working.py           WorkingMemory — LLM summarization eviction
memory/episodic_lance.py    LanceDB episodic store — IVF_PQ ANN, batch writes
memory/redis_store.py       Redis semantic store — durable KV with TTL
memory/stores.py            InMemory stores — local dev default, no deps
```

Execution is **streaming-primary**: every run flows through `run_stream()`,
which yields `BusEvent`s for plan, thoughts, tool calls, observations, task
completions, replans, and synthesis. The blocking `run()` is a thin drain
over the same stream.

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
result = await runtime.run("my goal")
```

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

## Streaming and blocking — same execution path

Both calls drive the same orchestrator; `run()` is just `.collect()` over
`run_stream()`.

```python
from harness.events import EventType

# Blocking — returns the final result dict
result = await runtime.run("investigate worker-07")

# Streaming — yields events live; display or drain however you want
async for event in runtime.run_stream("investigate worker-07"):
    if event.type == EventType.TOKEN:                # only if LLM streams tokens
        print(event.token, end="", flush=True)
    elif event.type == EventType.ACTION:
        print(f"\n→ {event.payload['tool']}({event.payload['args']})")
    elif event.type == EventType.OBSERVATION:
        print(f"  ← {event.payload['observation']}")
    elif event.type == EventType.REPLAN:
        print(f"\n[replan #{event.payload['replan_count']}]")
    elif event.type == EventType.DONE:
        print(f"\n{event.payload['answer']}")
```

Event types: `PLAN`, `THOUGHT`, `TOKEN`, `ACTION`, `OBSERVATION`, `TASK_DONE`,
`REPLAN`, `SYNTHESIS`, `DONE`, `ERROR` (see `harness/events.py`).

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

## Sandboxed tools

Tools that shell out (`kubectl`, `curl`, `sh -c …`) should not run inside the
agent process. The Rust executor at `executor/` is a one-shot subprocess
sandbox that the Python `Sandbox` client invokes per tool call.

What the sandbox enforces:

- **Tool allowlist** — set at startup; the LLM cannot extend it.
- **Wall-clock timeout** per call.
- **Output size cap** (default 1 MiB, configurable).
- **Subprocess isolation** — a tool crash cannot reach the agent.
- **Scrubbed environment** — only `PATH` is forwarded; everything else dropped.

What it does **not** do — for syscall / fs / network isolation, deploy the
harness inside a container or VM:

- No seccomp / landlock filters.
- No fs or network namespacing.
- No rlimit-based CPU / memory caps.

Build and wire up:

```bash
cd executor && cargo build --release
# binary at executor/target/release/executor
```

```python
from harness.sandbox import Sandbox, SandboxConfig, SandboxedTool

sandbox = Sandbox(SandboxConfig(
    binary_path="executor/target/release/executor",
    allowed_tools=("kubectl", "curl"),   # "shell" is opt-in only
))

tools.register(SandboxedTool(
    name="kubectl", executor_tool="kubectl", sandbox=sandbox, arg_key="args",
))
```

## Tests

```bash
pytest
```
