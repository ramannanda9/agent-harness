"""
harness/checkpoint.py — Pluggable checkpoint store for run state persistence.

Used by both the HITL approval gate (crash-resume) and periodic step
checkpointing (checkpoint_every on AgentConfig).

A checkpoint is a plain dict written under a run_id key.  The schema is
defined by the caller — agents write {run_id, agent_id, task, step, memory}
with an optional {pending: ...} field added by the HITL gate.

Two backends ship out of the box:

    FileCheckpointStore  — zero deps, one JSON file per run_id
    RedisCheckpointStore — for distributed / multi-process setups

Both share the same three-method interface so callers are backend-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class FileCheckpointStore:
    """
    Zero-dependency checkpoint store backed by JSON files.

    Default directory: ~/.agent-harness/checkpoints/
    Override with the CHECKPOINT_DIR env var or by passing checkpoint_dir.
    """

    def __init__(self, checkpoint_dir: str | Path | None = None) -> None:
        self._dir = Path(
            checkpoint_dir
            or os.environ.get("CHECKPOINT_DIR", Path.home() / ".agent-harness" / "checkpoints")
        )
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    async def write(self, run_id: str, data: dict) -> None:
        self._path(run_id).write_text(json.dumps(data, default=str, indent=2))

    async def read(self, run_id: str) -> dict | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    async def delete(self, run_id: str) -> None:
        path = self._path(run_id)
        if path.exists():
            path.unlink()

    @classmethod
    def purge_old(cls, days: int = 7, checkpoint_dir: str | Path | None = None) -> int:
        """Delete checkpoint files older than `days`. Returns count removed."""
        store = cls(checkpoint_dir)
        cutoff = time.time() - days * 86_400
        removed = 0
        for p in store._dir.glob("*.json"):
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        return removed


class RedisCheckpointStore:
    """
    Checkpoint store backed by Redis.

    Checkpoints expire after ttl_seconds (default 24 h).

    Usage:
        import redis.asyncio as redis
        client = redis.Redis(host="localhost", decode_responses=True)
        store = RedisCheckpointStore(client)
    """

    _KEY = "ckp:{}"

    def __init__(self, client: Any, ttl_seconds: int = 86_400) -> None:
        self._r = client
        self._ttl = ttl_seconds

    async def write(self, run_id: str, data: dict) -> None:
        await self._r.set(self._KEY.format(run_id), json.dumps(data, default=str), ex=self._ttl)

    async def read(self, run_id: str) -> dict | None:
        raw = await self._r.get(self._KEY.format(run_id))
        return json.loads(raw) if raw else None

    async def delete(self, run_id: str) -> None:
        await self._r.delete(self._KEY.format(run_id))
