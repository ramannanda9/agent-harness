# agent-harness

Generic multi-agent orchestration harness with hybrid planning, LanceDB episodic memory, and streaming token bus.

## Architecture

```
harness/runtime.py          AgentRuntime — single entry point, wire once run anything
orchestrator/planner.py     Hybrid DAG orchestrator — static plan + replan on failure
orchestrator/streaming.py   Streaming orchestrator — asyncio.Queue token bus, backpressured
agents/base.py              Generic BaseAgent — ReAct loop, no subclassing needed
memory/manager.py           MemoryManager — semantic (Redis) + episodic (LanceDB)
memory/working.py           WorkingMemory — LLM summarization eviction, non-blocking
memory/episodic_lance.py    LanceDB episodic store — IVF_PQ ANN, batch writes, versioned
memory/stores.py            InMemory stores — local dev, no deps
```

## Quickstart

```bash
pip install -e ".[dev]"
python examples/quickstart.py
```

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

## Rust tool executor

```bash
cd executor
cargo build --release
# binary at executor/target/release/executor
```

## Memory write timing

- **During run**: `write_working_fact()` — lightweight KV, namespaced, short TTL
- **End of run**: `write_run_end()` — LLM extraction → global semantic + episodic vector

## Streaming

```python
async for event in orchestrator.run_stream(goal):
    if event.type == EventType.TOKEN:
        print(event.token, end="", flush=True)
    elif event.type == EventType.DONE:
        print(event.result["answer"])
```
