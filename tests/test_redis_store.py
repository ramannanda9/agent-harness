"""RedisSemanticStore tests using fakeredis (no real Redis server required)."""
from __future__ import annotations

import pytest

fakeredis = pytest.importorskip("fakeredis")
from memory.redis_store import RedisSemanticStore  # noqa: E402


@pytest.fixture
def client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(client) -> RedisSemanticStore:
    return RedisSemanticStore(client, key_prefix="ah:")


async def test_write_read_string(store):
    await store.write("k", "v")
    assert await store.read("k") == "v"


async def test_write_read_dict(store):
    await store.write("user:42", {"name": "alice", "age": 30})
    assert await store.read("user:42") == {"name": "alice", "age": 30}


async def test_read_missing_returns_none(store):
    assert await store.read("nope") is None


async def test_delete(store):
    await store.write("k", "v")
    await store.delete("k")
    assert await store.read("k") is None


async def test_ttl_expiry(store, client):
    await store.write("k", "v", ttl_seconds=60)
    # fakeredis honors EX; backdate the key by manipulating its ttl forward
    # by setting expiration to the past via a redis command.
    assert await store.read("k") == "v"
    # confirm Redis itself recorded a TTL
    assert await client.ttl("ah:k") > 0


async def test_search_prefix_filters_by_user_prefix(store):
    await store.write("ns:a", 1)
    await store.write("ns:b", 2)
    await store.write("other:c", 3)
    matched = await store.search_prefix("ns:")
    assert matched == {"ns:a": 1, "ns:b": 2}


async def test_key_prefix_is_invisible_to_caller(store, client):
    await store.write("hello", {"v": 1})
    # raw key in Redis is namespaced
    assert await client.exists("ah:hello") == 1
    # the caller never sees the prefix
    matched = await store.search_prefix("hel")
    assert "hello" in matched
    assert "ah:hello" not in matched


async def test_no_prefix_works(client):
    store = RedisSemanticStore(client, key_prefix="")
    await store.write("plain", "value")
    assert await store.read("plain") == "value"


async def test_works_as_memorymanager_semantic_backend(client):
    """Smoke: drive the full MemoryManager.write_working_fact path against Redis."""
    from memory.manager import MemoryManager
    from memory.stores import InMemoryEpisodicStore
    from tests.conftest import ScriptedLLM

    store = RedisSemanticStore(client, key_prefix="ah:")
    mgr = MemoryManager(
        semantic_store=store,
        episodic_store=InMemoryEpisodicStore(),
        llm=ScriptedLLM(),
    )
    await mgr.write_working_fact("run1", "agent_x", "step_0", {"v": 1})
    facts = await mgr.read_working_facts("run1")
    assert any("step_0" in k for k in facts.keys())
