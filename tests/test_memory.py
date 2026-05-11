"""Memory store + manager smoke tests."""
from __future__ import annotations

from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tests.conftest import ScriptedLLM

# ── Semantic store ────────────────────────────────────────────────────────────


async def test_semantic_round_trip():
    store = InMemorySemanticStore()
    await store.write("k", "v")
    assert await store.read("k") == "v"
    await store.delete("k")
    assert await store.read("k") is None


async def test_semantic_ttl_expiry():
    store = InMemorySemanticStore()
    await store.write("k", "v", ttl_seconds=1)
    assert await store.read("k") == "v"  # still alive
    # monkey-patch time forward instead of sleeping
    import time as _time

    real_time = _time.time
    store._store["k"] = (store._store["k"][0], real_time() - 1)  # backdate expiry
    assert await store.read("k") is None
    assert store.size() == 0  # expired entry should be evicted on read


async def test_semantic_search_prefix():
    store = InMemorySemanticStore()
    await store.write("ns:a", 1)
    await store.write("ns:b", 2)
    await store.write("other:c", 3)
    matched = await store.search_prefix("ns:")
    assert matched == {"ns:a": 1, "ns:b": 2}


# ── Episodic store ────────────────────────────────────────────────────────────


async def test_episodic_write_and_get():
    store = InMemoryEpisodicStore()
    eid = await store.write("hello world", metadata={"src": "test"})
    record = await store.get(eid)
    assert record is not None
    assert record["text"] == "hello world"
    assert record["metadata"]["src"] == "test"


async def test_episodic_search_ranks_by_overlap():
    store = InMemoryEpisodicStore()
    await store.write("alpha beta gamma", {})
    await store.write("delta epsilon", {})
    await store.write("alpha beta unrelated", {})

    hits = await store.search("alpha beta", top_k=2)
    assert len(hits) == 2
    # both top hits should contain "alpha beta"
    for h in hits:
        assert "alpha" in h["text"] and "beta" in h["text"]


async def test_episodic_search_empty_store():
    store = InMemoryEpisodicStore()
    assert await store.search("anything") == []


# ── MemoryManager ─────────────────────────────────────────────────────────────


async def test_working_fact_round_trip():
    llm = ScriptedLLM()
    mgr = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    await mgr.write_working_fact("run1", "agent_x", "step_0_echo", {"v": 1})
    facts = await mgr.read_working_facts("run1")
    assert any("step_0_echo" in k for k in facts.keys())


async def test_working_fact_conflict_logged():
    llm = ScriptedLLM()
    mgr = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    await mgr.write_working_fact("run1", "agent_x", "k", "old")
    await mgr.write_working_fact("run1", "agent_x", "k", "new")
    log = mgr.get_conflict_log()
    assert len(log) == 1
    assert log[0]["old"] == "old"
    assert log[0]["new"] == "new"


async def test_run_end_extraction_writes_to_both_stores():
    """Mock the extraction-LLM call; verify semantic + episodic stores get populated."""

    def extract(system, messages, kwargs):
        return {
            "semantic_facts": {"thing:status": "ok"},
            "episodic_summary": "we did the thing successfully",
            "metadata": {},
            "ttl_seconds": None,
        }

    llm = ScriptedLLM(routes={"memory extraction": extract})
    semantic = InMemorySemanticStore()
    episodic = InMemoryEpisodicStore()
    mgr = MemoryManager(semantic_store=semantic, episodic_store=episodic, llm=llm)

    req = await mgr.write_run_end(
        goal="do the thing",
        agent_results=[{"agent_id": "a", "answer": "done", "confidence": 0.9}],
        trace=[],
    )

    assert req.semantic_facts == {"thing:status": "ok"}
    assert await semantic.read("thing:status") == "ok"
    assert episodic.count() == 1


async def test_run_end_extraction_failure_degrades_gracefully():
    """If the LLM blows up, write_run_end must not raise."""

    def boom(system, messages, kwargs):
        raise RuntimeError("LLM offline")

    llm = ScriptedLLM(routes={"memory extraction": boom})
    mgr = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )

    req = await mgr.write_run_end(goal="g", agent_results=[], trace=[])

    assert req.semantic_facts == {}
    assert "Extraction failed" in req.episodic_summary


async def test_build_context_returns_episodes_for_goal():
    llm = ScriptedLLM()
    semantic = InMemorySemanticStore()
    episodic = InMemoryEpisodicStore()
    await episodic.write("worker-07 had high gpu usage", {"timestamp": "now"})
    mgr = MemoryManager(semantic_store=semantic, episodic_store=episodic, llm=llm)

    ctx = await mgr.build_context(goal="check worker-07 gpu", agent_id="diag")
    assert not ctx.is_empty()
    assert any("worker-07" in e["text"] for e in ctx.episodes)
    rendered = ctx.render()
    assert "worker-07" in rendered
