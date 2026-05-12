"""
examples/durable_memory_demo.py — Redis (semantic) + LanceDB (episodic).

Demonstrates the durable memory backends by running two related goals back-to-back
and showing how the second run sees episodic memory written by the first.

Layout:
  RedisSemanticStore   → semantic KV facts (run-end extraction)
  LanceDBEpisodicStore → episodic summaries with vector similarity search

Requirements:
  pip install -e ".[redis,lance,openai,dev]"
  # Redis server reachable. Default: redis://localhost:6379/0
  #   docker run --rm -p 6379:6379 redis
  # LanceDB is embedded — just needs a writable path (default ./lance_episodic).
  # OPENAI_API_KEY in environment.

Run:
  OPENAI_API_KEY=sk-... python examples/durable_memory_demo.py

Optional env:
  REDIS_URL      default redis://localhost:6379/0
  LANCE_PATH     default ./lance_episodic
  OPENAI_MODEL   default gpt-4o-mini

Note: this demo writes to Redis under the key prefix `agent-harness-demo:` and
writes a Lance table at LANCE_PATH. Both persist after the script ends —
re-running picks up where the previous run left off, which is the point.
"""
from __future__ import annotations

import asyncio
import os
import sys

from agents.base import AgentConfig
from harness.llm.openai import OpenAILLM
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager


def _truncate(s: str, n: int = 80) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


async def _check_or_die_redis(redis_url: str):
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print('ERROR: redis not installed. Run: pip install -e ".[redis]"', file=sys.stderr)
        sys.exit(2)

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception as e:
        print(f"ERROR: Redis not reachable at {redis_url}: {e}", file=sys.stderr)
        print("  Start a local Redis: docker run --rm -p 6379:6379 redis", file=sys.stderr)
        sys.exit(2)
    return client


def _build_episodic(lance_path: str):
    try:
        from memory.episodic_lance import LanceDBEpisodicStore, LocalEmbedder, MockEmbedder
    except ImportError as e:
        print(f'ERROR: lancedb not installed: {e}', file=sys.stderr)
        print('  Run: pip install -e ".[lance]"', file=sys.stderr)
        sys.exit(2)

    # Prefer LocalEmbedder if sentence-transformers is installed — gives real
    # semantic similarity. Fall back to MockEmbedder so the demo still runs.
    try:
        embedder = LocalEmbedder()
        embedder_kind = "LocalEmbedder (sentence-transformers, real similarity)"
    except Exception:
        embedder = MockEmbedder()
        embedder_kind = "MockEmbedder (random vectors — similarity scores meaningless)"
    return LanceDBEpisodicStore(uri=lance_path, embedder=embedder), embedder_kind


async def _dump_memory_state(semantic, episodic, query: str | None = None):
    keys = await semantic.search_prefix("")
    print(f"  semantic keys: {len(keys)}")
    for k, v in list(keys.items())[:5]:
        print(f"    {k}: {_truncate(v, 70)}")
    if query:
        hits = await episodic.search(query, top_k=3)
        print(f"  episodic hits for {query!r}: {len(hits)}")
        for h in hits:
            ts = h.get("metadata", {}).get("timestamp", "?")
            print(f"    [{ts[:19]}] {_truncate(h['text'], 70)}")


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running this demo.", file=sys.stderr)
        sys.exit(2)

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    lance_path = os.environ.get("LANCE_PATH", "./lance_episodic")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    redis_client = await _check_or_die_redis(redis_url)
    episodic, embedder_kind = _build_episodic(lance_path)
    await episodic.initialize()

    from memory.redis_store import RedisSemanticStore
    semantic = RedisSemanticStore(redis_client, key_prefix="agent-harness-demo:")

    print(f"Redis:    {redis_url}")
    print(f"LanceDB:  {lance_path}")
    print(f"Embedder: {embedder_kind}")
    print(f"Model:    {model}")

    llm = OpenAILLM(model=model)
    memory = MemoryManager(semantic_store=semantic, episodic_store=episodic, llm=llm)

    agents = AgentRegistry().register(AgentConfig(
        agent_id="explainer",
        role="briefly explains Python standard-library modules",
        system_prompt=(
            "You are a concise Python expert. Answer the user's question in 2-3 sentences. "
            "Always use the ReAct JSON format. You have no tools — answer from knowledge "
            "and finish on the first step."
        ),
        allowed_tools=[],
        max_steps=2,
        working_memory_max_tokens=4000,
    ))

    runtime = AgentRuntime(
        agent_registry=agents,
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=1.0, max_wall_time_seconds=60,
            confidence_threshold=0.5, max_replan_count=0,
        ),
    )

    # ── Show memory state before any runs ──────────────────────────────────────
    print("\n" + "─" * 64)
    print("Memory state BEFORE any runs")
    print("─" * 64)
    await _dump_memory_state(semantic, episodic, query=None)

    # ── Run 1 — cold memory ────────────────────────────────────────────────────
    goal1 = "What does Python's `json` module do?"
    print("\n" + "─" * 64)
    print(f"RUN 1  (cold memory): {goal1}")
    print("─" * 64)
    result = await runtime.run(goal1)
    print(f"answer:     {_truncate(result['answer'], 200)}")
    print(f"confidence: {result['confidence']}")
    await _dump_memory_state(semantic, episodic, query=goal1)

    # ── Show what run 2 will pick up ───────────────────────────────────────────
    goal2 = "What does Python's `pickle` module do?"
    print("\n" + "─" * 64)
    print(f"Memory context that run 2 will see for goal: {goal2!r}")
    print("─" * 64)
    ctx = await memory.build_context(goal=goal2, agent_id="explainer")
    rendered = ctx.render() if not ctx.is_empty() else "(empty)"
    print(rendered or "(empty)")

    # ── Run 2 — warm memory ────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print(f"RUN 2  (warm — episodic memory from run 1 is in scope): {goal2}")
    print("─" * 64)
    result = await runtime.run(goal2)
    print(f"answer:     {_truncate(result['answer'], 200)}")
    print(f"confidence: {result['confidence']}")
    await _dump_memory_state(semantic, episodic, query=goal2)

    print("\nBoth backends are durable — re-running this script picks up these "
          "episodes / keys from disk. Wipe with:")
    print(f"  redis-cli --no-auth-warning -u {redis_url} KEYS 'agent-harness-demo:*' "
          f"| xargs -r redis-cli --no-auth-warning -u {redis_url} DEL")
    print(f"  rm -rf {lance_path}")

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
