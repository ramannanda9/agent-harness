"""
RedisSemanticStore — durable KV semantic store backed by Redis.

Implements the SemanticStore protocol from memory.manager.

Values are JSON-serialised on write and parsed on read. `default=str` is
used as a fallback for non-JSON-native values (timestamps, dataclasses, etc),
so a roundtripped value is not guaranteed to match the original Python type
— it's the best-effort string form of it.

Install:
    pip install -e ".[redis]"

Usage:
    import redis.asyncio as redis
    client = redis.Redis(host="localhost", decode_responses=True)
    store = RedisSemanticStore(client, key_prefix="agent-harness:")
    mgr = MemoryManager(semantic_store=store, episodic_store=..., llm=...)

The client must be created with `decode_responses=True` so SCAN/GET return
strings rather than bytes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisSemanticStore:
    def __init__(self, client: Any, key_prefix: str = "") -> None:
        self._client = client
        self._prefix = key_prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def write(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, default=str)
        if ttl_seconds:
            await self._client.set(self._k(key), payload, ex=ttl_seconds)
        else:
            await self._client.set(self._k(key), payload)

    async def read(self, key: str) -> Any | None:
        raw = await self._client.get(self._k(key))
        if raw is None:
            return None
        return _decode(raw)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._k(key))

    async def search_prefix(self, prefix: str) -> dict[str, Any]:
        match = f"{self._prefix}{prefix}*"
        prefix_len = len(self._prefix)
        result: dict[str, Any] = {}
        async for raw_key in self._client.scan_iter(match=match, count=200):
            user_key = raw_key[prefix_len:] if self._prefix else raw_key
            raw = await self._client.get(raw_key)
            if raw is None:
                continue
            result[user_key] = _decode(raw)
        return result


def _decode(raw: Any) -> Any:
    """Best-effort JSON decode; fall back to the raw string."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw
