"""
Default store implementations for local development.

InMemorySemanticStore  — dict-backed, TTL via asyncio, no deps
InMemoryEpisodicStore  — list-backed, keyword similarity (no embeddings needed for dev)

For production:
  Semantic  → RedisSemanticStore  (swap in, same protocol)
  Episodic  → ChromaEpisodicStore or PineconeEpisodicStore
"""
from __future__ import annotations

import time
import uuid
from typing import Any

# ── In-Memory Semantic Store ──────────────────────────────────────────────────

class InMemorySemanticStore:
    """
    Dict-backed semantic store with TTL support.
    Drop-in replacement interface for Redis in local dev.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}
        # (value, expiry_timestamp or None)

    async def write(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expiry = time.time() + ttl_seconds if ttl_seconds else None
        self._store[key] = (value, expiry)

    async def read(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and time.time() > expiry:
            del self._store[key]
            return None
        return value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def search_prefix(self, prefix: str) -> dict[str, Any]:
        now = time.time()
        result = {}
        expired = []
        for key, (value, expiry) in self._store.items():
            if key.startswith(prefix):
                if expiry is not None and now > expiry:
                    expired.append(key)
                else:
                    result[key] = value
        for key in expired:
            del self._store[key]
        return result

    def size(self) -> int:
        return len(self._store)


# ── In-Memory Episodic Store ──────────────────────────────────────────────────

class InMemoryEpisodicStore:
    """
    List-backed episodic store with keyword similarity search.

    Search uses token overlap (Jaccard-like) — no embeddings needed for dev.
    In production, swap for ChromaEpisodicStore which uses real embeddings.

    Schema per episode:
      { "id": str, "text": str, "metadata": dict, "timestamp": float }
    """

    def __init__(self) -> None:
        self._episodes: list[dict] = []

    async def write(self, text: str, metadata: dict) -> str:
        episode_id = str(uuid.uuid4())
        self._episodes.append({
            "id": episode_id,
            "text": text,
            "metadata": metadata,
            "timestamp": time.time(),
        })
        return episode_id

    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self._episodes:
            return []

        query_tokens = set(query.lower().split())

        def score(episode: dict) -> float:
            ep_tokens = set(episode["text"].lower().split())
            if not ep_tokens:
                return 0.0
            intersection = len(query_tokens & ep_tokens)
            union = len(query_tokens | ep_tokens)
            return intersection / union if union > 0 else 0.0

        scored = sorted(self._episodes, key=score, reverse=True)
        return scored[:top_k]

    async def get(self, episode_id: str) -> dict | None:
        return next((e for e in self._episodes if e["id"] == episode_id), None)

    def count(self) -> int:
        return len(self._episodes)
