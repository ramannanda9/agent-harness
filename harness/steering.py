"""
harness/steering.py — async steering front-ends for BaseAgent.

Three pieces, each composable as an async context manager that pushes into
the agent's `steer()` queue:

  StdinSteer(agents)         — interactive terminal use; reads stdin lines
                               and routes them to the right agent. In
                               multi-agent mode, requires a `<agent_id>:`
                               prefix (or `*:` for broadcast).

  FileSteer(agent, path?)    — deployed / headless use; tails an append-only
                               file and forwards each new line to the
                               agent's queue.

  StdinRouter                — internal coordinator: a single stdin reader
                               task that routes lines to either a pending
                               HITL prompt (when one has claimed the next
                               line) or the registered steering callback.

The agent itself only knows about `agent.steer(text)`. The shims are pure
adapters. Programmatic callers (HTTP handlers, MCP tools, supervisor
agents) skip the shims and call `agent.steer()` directly.

HITL coordination
-----------------
`harness/hitl.py:request_approval` queries `get_active_router()`. If a
router is active, HITL calls `router.claim_next_line()` BEFORE printing
its banner — this hands the router a Future to fulfill with the next
stdin line, instead of forwarding to steering. If no router is active,
HITL falls back to its current `input()` path. Existing non-interactive
tests are unaffected.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent


# ── Active-router registry ────────────────────────────────────────────────────
#
# StdinSteer registers itself here on enter and clears on exit. HITL reads
# this to decide whether to claim the next stdin line via the router or
# fall back to a direct input() call.

_active_router: StdinRouter | None = None


def get_active_router() -> StdinRouter | None:
    """Return the currently-active StdinRouter, or None if no shim is active."""
    return _active_router


def _set_active_router(router: StdinRouter | None) -> None:
    global _active_router
    _active_router = router


# ── Stdin router ──────────────────────────────────────────────────────────────


async def _default_readline() -> str | None:
    """Read one line from real stdin in a worker thread. Returns None on EOF."""
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    return line if line else None


class StdinRouter:
    """Single stdin reader task that routes lines to HITL or steering.

    Routing rules:
      - If HITL has claimed the next line (via `claim_next_line()`), that
        line fulfills HITL's Future and steering is not invoked.
      - Otherwise the line is dispatched to the registered steering
        callback, if any.
      - Empty / whitespace-only lines are skipped.

    The reader is injectable for testability. Production code constructs
    the router with no arguments (uses real stdin); tests pass a fake
    readline that pops from a queue.
    """

    def __init__(
        self,
        readline: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._readline = readline or _default_readline
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._hitl_future: asyncio.Future[str] | None = None
        self._steer_callback: Callable[[str], None] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_steer_callback(self, cb: Callable[[str], None] | None) -> None:
        """Register (or clear) the callback invoked for non-HITL stdin lines."""
        self._steer_callback = cb

    def claim_next_line(self) -> asyncio.Future[str]:
        """Reserve the next stdin line for HITL.

        Returns a Future that resolves with the next line read. HITL must
        call this BEFORE printing its banner so the router knows to route
        the user's response to it rather than to steering.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._hitl_future = fut
        return fut

    async def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        # Drop any pending HITL claim — caller is shutting down.
        if self._hitl_future is not None and not self._hitl_future.done():
            self._hitl_future.cancel()
            self._hitl_future = None

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = await self._readline()
            except asyncio.CancelledError:
                raise
            except Exception:
                # stdin closed or read failure — stop quietly.
                return
            if line is None:
                return  # EOF
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if self._hitl_future is not None and not self._hitl_future.done():
                fut, self._hitl_future = self._hitl_future, None
                fut.set_result(line)
                continue
            if self._steer_callback is not None:
                try:
                    self._steer_callback(line)
                except Exception:
                    # One bad callback shouldn't kill the router.
                    pass


# ── Stdin shim (interactive) ──────────────────────────────────────────────────


class StdinSteer:
    """Async context manager: forwards stdin lines to agent steering queues.

    Single agent: every line goes to that agent's queue (no prefix).
    Multiple agents: each line must begin with `<agent_id>:` to target one
        agent, or `*:` to broadcast. Unprefixed or unknown-prefix lines
        print a hint to stdout and are discarded (never silently misrouted).

    HITL prompts still work — see `StdinRouter` for coordination details.
    Only one StdinSteer should be active at a time per process (stdin is
    a shared resource); entering a second instance overrides the first.
    """

    def __init__(
        self,
        agents: BaseAgent | list[BaseAgent],
        *,
        router: StdinRouter | None = None,
    ) -> None:
        if not isinstance(agents, list):
            agents = [agents]
        if not agents:
            raise ValueError("StdinSteer requires at least one agent")
        self._agents: dict[str, BaseAgent] = {a.config.agent_id: a for a in agents}
        self._router = router or StdinRouter()
        self._owned_router = router is None

    async def __aenter__(self) -> StdinSteer:
        self._router.set_steer_callback(self._route)
        if self._owned_router:
            await self._router.start()
        _set_active_router(self._router)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _set_active_router(None)
        self._router.set_steer_callback(None)
        if self._owned_router:
            await self._router.stop()

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, line: str) -> None:
        # Single-agent: no prefix required, every line goes to the one agent.
        if len(self._agents) == 1:
            next(iter(self._agents.values())).steer(line)
            return

        # Multi-agent: require an explicit "agent_id:" or "*:" prefix.
        if ":" not in line:
            self._hint_unprefixed(line)
            return
        prefix, _, text = line.partition(":")
        prefix = prefix.strip()
        text = text.strip()
        if not text:
            return

        if prefix == "*":
            for agent in self._agents.values():
                agent.steer(text)
            return

        agent = self._agents.get(prefix)
        if agent is None:
            self._hint_unknown(prefix)
            return
        agent.steer(text)

    def _hint_unprefixed(self, line: str) -> None:
        ids = ", ".join(sorted(self._agents))
        print(
            f"[steering] no agent prefix on input; ignored. "
            f"Use 'agent_id: <text>' or '*: <text>'. Active agents: {ids}",
            file=sys.stderr,
        )

    def _hint_unknown(self, prefix: str) -> None:
        ids = ", ".join(sorted(self._agents))
        print(
            f"[steering] unknown agent '{prefix}'; ignored. Active agents: {ids}",
            file=sys.stderr,
        )


# ── File shim (deployed) ──────────────────────────────────────────────────────


class FileSteer:
    """Async context manager: tails an append-only file into an agent's queue.

    Polls `path` every `interval` seconds. New content (since the last
    read offset) is split into lines and each non-empty line is forwarded
    to `agent.steer()`. The file is never mutated — callers can `tail -f`
    it independently for visibility.

    Default path: f"/tmp/ah-{run_id}-{agent.config.agent_id}.steer".
    Either `path` or `run_id` must be provided.

    Semantics:
      - On enter: positions at current EOF so pre-existing content is NOT
        replayed as guidance. Watcher starts fresh.
      - Missing file: no-op until the file appears.
      - File truncated or recreated (size < last offset): offset resets
        to 0 so the new content is read from the beginning.
      - Stop signal: cancels the polling task; subsequent writes ignored.
    """

    def __init__(
        self,
        agent: BaseAgent,
        path: str | None = None,
        *,
        run_id: str | None = None,
        interval: float = 0.25,
    ) -> None:
        if path is None:
            if run_id is None:
                raise ValueError("FileSteer requires either `path` or `run_id`")
            path = f"/tmp/ah-{run_id}-{agent.config.agent_id}.steer"
        self._agent = agent
        self._path = path
        self._interval = interval
        self._offset = 0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def path(self) -> str:
        return self._path

    async def __aenter__(self) -> FileSteer:
        try:
            self._offset = os.path.getsize(self._path)
        except FileNotFoundError:
            self._offset = 0
        self._stop.clear()
        self._task = asyncio.create_task(self._watch())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _watch(self) -> None:
        # Loop: poll → wait for either stop or interval to elapse.
        while not self._stop.is_set():
            self._poll_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue
            else:
                return  # stop signaled

    def _poll_once(self) -> None:
        try:
            size = os.path.getsize(self._path)
        except FileNotFoundError:
            return
        if size < self._offset:
            # Truncated or recreated — restart from the beginning.
            self._offset = 0
        if size == self._offset:
            return
        try:
            with open(self._path) as f:
                f.seek(self._offset)
                new_content = f.read()
                self._offset = f.tell()
        except (FileNotFoundError, OSError):
            return
        for line in new_content.splitlines():
            line = line.strip()
            if line:
                self._agent.steer(line)
