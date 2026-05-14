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
import sys
import time
from pathlib import Path
from typing import Any


def maybe_resume_key() -> str | None:
    """
    Extract the --resume <key> value from sys.argv, or return None.

    Called automatically by AgentRuntime.dispatch_stream / run_stream so that
    scripts resume transparently without any resume-specific code.  Also
    available for scripts that need the key explicitly.
    """
    args = sys.argv[1:]
    if "--resume" not in args:
        return None
    idx = args.index("--resume")
    if idx + 1 >= len(args):
        print("Usage: --resume <ckp_id>", file=sys.stderr)
        sys.exit(1)
    return args[idx + 1]


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


class _ResumeHint:
    """
    Async context manager that prints a --resume hint to stderr on interruption.

    Set ``hint.done = True`` before leaving the managed block to suppress the
    message (i.e. on clean success). If ``checkpoint_store`` is None the hint
    is never printed since there is no saved state to resume from.
    """

    def __init__(self, resume_key: str, checkpoint_store: Any, label: str = "Run") -> None:
        self._resume_key = resume_key
        self._store = checkpoint_store
        self._label = label
        self.done: bool = False

    async def __aenter__(self) -> _ResumeHint:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.done and self._store is not None and self._resume_key:
            import sys

            script = sys.argv[0] if sys.argv else "your_script.py"
            print(
                f"\n  {self._label} interrupted — checkpoint saved."
                f"\n  Resume: python {script} --resume {self._resume_key}\n",
                file=sys.stderr,
            )
        return False
