"""
Default store implementations for local development.

InMemorySemanticStore  — dict-backed, TTL via asyncio, no deps
SQLiteSemanticStore    — sqlite-backed durable KV, no server
InMemoryEpisodicStore  — list-backed, keyword similarity (no embeddings needed for dev)

For production:
  Semantic  → RedisSemanticStore  (swap in, same protocol)
  Episodic  → ChromaEpisodicStore or PineconeEpisodicStore
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
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


# ── SQLite Semantic Store ─────────────────────────────────────────────────────


class SQLiteSemanticStore:
    """Durable semantic key-value store backed by stdlib SQLite."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._ready = False

    async def write(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        self._ensure_schema()
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        payload = json.dumps(value, default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_memory(key, value_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value_json = excluded.value_json,
                  expires_at = excluded.expires_at,
                  updated_at = excluded.updated_at
                """,
                (key, payload, expires_at, time.time()),
            )

    async def read(self, key: str) -> Any | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json, expires_at FROM semantic_memory WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] is not None and time.time() > float(row["expires_at"]):
                conn.execute("DELETE FROM semantic_memory WHERE key = ?", (key,))
                return None
            return _decode_json(row["value_json"])

    async def delete(self, key: str) -> None:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute("DELETE FROM semantic_memory WHERE key = ?", (key,))

    async def search_prefix(self, prefix: str) -> dict[str, Any]:
        self._ensure_schema()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM semantic_memory WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            rows = conn.execute(
                """
                SELECT key, value_json
                FROM semantic_memory
                WHERE key LIKE ? ESCAPE '\\'
                ORDER BY key ASC
                """,
                (f"{_escape_like(prefix)}%",),
            ).fetchall()
        return {row["key"]: _decode_json(row["value_json"]) for row in rows}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    expires_at REAL,
                    updated_at REAL NOT NULL
                )
                """
            )
        self._ready = True


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

    async def write(self, text: str, metadata: dict, agent_id: str = "") -> str:
        episode_id = str(uuid.uuid4())
        # ``latest`` policy: hard-delete prior episodes with the same
        # ``memory_key`` so the store stays bounded — no soft-delete tombstones
        # accumulating per run.
        if metadata.get("memory_policy") == "latest" and metadata.get("memory_key"):
            target = metadata["memory_key"]
            self._episodes = [
                ep
                for ep in self._episodes
                if (ep.get("metadata") or {}).get("memory_key") != target
            ]
        self._episodes.append(
            {
                "id": episode_id,
                "text": text,
                "metadata": metadata,
                "agent_id": agent_id,
                "timestamp": time.time(),
            }
        )
        return episode_id

    async def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        memory_scope: str | None = None,
        agent_id: str | None = None,
        include_shared: bool = True,
        include_legacy: bool = True,
    ) -> list[dict]:
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

        candidates = [
            episode
            for episode in self._episodes
            if _episode_matches(
                episode,
                memory_scope=memory_scope,
                agent_id=agent_id,
                include_shared=include_shared,
                include_legacy=include_legacy,
            )
        ]
        scored = sorted(candidates, key=score, reverse=True)
        return scored[:top_k]

    async def get(self, episode_id: str) -> dict | None:
        return next((e for e in self._episodes if e["id"] == episode_id), None)

    async def invalidate(self, memory_key: str) -> int:
        """Hard-delete every episode with the given ``memory_key``. Returns count.

        Used by the reconciler's DELETE action and called transitively by
        ``latest``-policy writes. Hard-delete so the store doesn't balloon
        with superseded records over many runs.
        """
        before = len(self._episodes)
        self._episodes = [
            ep
            for ep in self._episodes
            if (ep.get("metadata") or {}).get("memory_key") != memory_key
        ]
        return before - len(self._episodes)

    def count(self) -> int:
        return len(self._episodes)


def _episode_matches(
    episode: dict,
    *,
    memory_scope: str | None,
    agent_id: str | None,
    include_shared: bool,
    include_legacy: bool,
) -> bool:
    metadata = episode.get("metadata") or {}
    # Back-compat: any episodes written before the hard-delete switch may
    # still carry active=False from the old soft-delete path. Filter them.
    if metadata.get("active") is False:
        return False
    if memory_scope is not None and metadata.get("memory_scope") != memory_scope:
        return False
    if memory_scope is None and metadata.get("memory_scope") is not None:
        return False
    if agent_id is None:
        return True
    if include_shared and metadata.get("shared") is True:
        return True
    if metadata.get("agent_id") == agent_id or episode.get("agent_id") == agent_id:
        return True
    if agent_id in (metadata.get("agent_ids") or []):
        return True
    if include_legacy and not metadata.get("memory_kind") and not metadata.get("agent_id"):
        return True
    return False


def _decode_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _escape_like(value: str) -> str:
    # Prefix keys are controlled by harness code, but escaping keeps LIKE
    # semantics predictable if callers use '%' or '_' in a key namespace.
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
