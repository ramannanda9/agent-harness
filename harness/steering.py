"""
harness/steering.py — async steering sources for BaseAgent.

A steering source is any async context manager that, while active, may call
`agent.steer(text)` to inject guidance into the agent's queue. The agent
owns the source's lifecycle: it enters the source at the top of
`run_stream` and exits it when the run finishes. No live-agent registry,
no shared mutable state.

Two transport-level pieces:

  StdinRouter         — process-wide pub/sub over stdin lines. Subscribers
                        register a prefix and a callback; each line is
                        parsed as `[prefix:] text` and routed to matching
                        subscribers. Coordinates with HITL via
                        `claim_next_line()`.

  FileSteer           — per-agent file watcher. Polls an append-only file
                        and forwards new lines to `agent.steer()`. The
                        agent never shares this file with another agent.

Two agent-bound sources for use with `steering_source_factory`:

  FileSteer           — also serves as a per-agent source (constructed by
                        the factory with the right path / agent pair).
  StdinAgentSource    — subscribes to one prefix on a shared
                        `StdinRouter`. Forwards matched lines to the
                        bound agent's `steer()`.

Two convenience helpers that return factories ready for
`AgentRuntime(steering_source_factory=...)`:

  file_steering_factory(path_template)
  stdin_steering_factory(router)

Direct-use shims (for code that holds explicit `BaseAgent` references and
does not go through `AgentRuntime`):

  StdinSteer(agents)   — wraps a `StdinRouter`, subscribes each agent
                         with its `agent_id` prefix (or with the empty
                         prefix when there's only one agent so the user
                         can skip prefixing entirely).

HITL coordination (unchanged in shape, updated for pub/sub):
  `harness/hitl.py:request_approval` calls `get_active_router()`. If a
  router is active, it calls `router.claim_next_line()` BEFORE printing
  the banner. That line bypasses pub/sub and resolves HITL's Future
  directly. Subsequent lines return to pub/sub.
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


# ── Active-router registry (HITL coordination only) ───────────────────────────

_active_router: StdinRouter | None = None


def get_active_router() -> StdinRouter | None:
    """Return the active StdinRouter for HITL coordination, or None."""
    return _active_router


def _set_active_router(router: StdinRouter | None) -> None:
    global _active_router
    _active_router = router


# ── Stdin router (process-wide pub/sub) ───────────────────────────────────────


async def _default_readline() -> str | None:
    """Read one line from real stdin in a worker thread. Returns None on EOF."""
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    return line if line else None


class StdinRouter:
    """Process-wide stdin reader with prefix-keyed pub/sub.

    A single background coroutine reads `readline()` in a loop. Each
    non-empty line is parsed:

      - `prefix: text`    → delivered to subscribers registered with
                            that exact prefix, plus any wildcard
                            subscribers registered with prefix=`*`.
      - `text` (no colon) → delivered to subscribers registered with
                            prefix=`None` (catch-all).
      - leading `*: text` → broadcast to every subscriber.

    HITL claims pre-empt pub/sub: if `claim_next_line()` was called and
    the Future is unresolved, the next line resolves it and is NOT
    routed to subscribers.

    Unknown-prefix lines (a `prefix:` that no one is listening for) are
    surfaced to stderr so the user knows their input was discarded.
    """

    def __init__(
        self,
        readline: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._readline = readline or _default_readline
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._hitl_future: asyncio.Future[str] | None = None
        # subscription_id → (prefix, callback). prefix=None is the catch-all.
        self._subs: dict[int, tuple[str | None, Callable[[str], None]]] = {}
        self._next_sub_id: int = 0

    # ── Pub/sub API ───────────────────────────────────────────────────────────

    def subscribe(
        self,
        prefix: str | None,
        callback: Callable[[str], None],
    ) -> int:
        """Register `callback` to receive text matching `prefix`.

        prefix=None    → catch-all: receives any line that has no `:`.
        prefix=str     → receives lines `prefix: text` (or `*: text`
                         broadcasts). Wildcard `*` may not be used as
                         a subscription prefix (it's reserved for
                         broadcast sends).

        Returns an integer subscription id for `unsubscribe()`.
        """
        if prefix == "*":
            raise ValueError("'*' is reserved for broadcast sends, not subscription")
        sid = self._next_sub_id
        self._next_sub_id += 1
        self._subs[sid] = (prefix, callback)
        return sid

    def unsubscribe(self, sub_id: int) -> None:
        self._subs.pop(sub_id, None)

    def active_prefixes(self) -> list[str]:
        """Return the list of currently-subscribed prefixes (excludes catch-all)."""
        return sorted({p for p, _ in self._subs.values() if p is not None})

    def has_catchall(self) -> bool:
        return any(p is None for p, _ in self._subs.values())

    # ── HITL API ──────────────────────────────────────────────────────────────

    def claim_next_line(self) -> asyncio.Future[str]:
        """Reserve the next stdin line for HITL. Resolves the returned Future."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._hitl_future = fut
        return fut

    # ── Lifecycle ─────────────────────────────────────────────────────────────

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
        if self._hitl_future is not None and not self._hitl_future.done():
            self._hitl_future.cancel()
            self._hitl_future = None

    async def __aenter__(self) -> StdinRouter:
        """Start the reader AND register as active so HITL can coordinate."""
        await self.start()
        _set_active_router(self)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _set_active_router(None)
        await self.stop()

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = await self._readline()
            except asyncio.CancelledError:
                raise
            except Exception:
                return  # stdin closed / read failure
            if line is None:
                return  # EOF
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if self._hitl_future is not None and not self._hitl_future.done():
                fut, self._hitl_future = self._hitl_future, None
                fut.set_result(line)
                continue
            self._dispatch(line)

    def _dispatch(self, line: str) -> None:
        prefix: str | None
        text: str
        if ":" in line:
            prefix, _, text = line.partition(":")
            prefix = prefix.strip()
            text = text.strip()
            if not text:
                return
            if prefix == "*":
                for p, cb in self._subs.values():
                    if p is not None:  # broadcasts target prefixed subscribers
                        self._safe_call(cb, text)
                return
        else:
            prefix = None
            text = line.strip()
            if not text:
                return

        matched = False
        for p, cb in list(self._subs.values()):
            if p == prefix:
                self._safe_call(cb, text)
                matched = True
        if not matched:
            self._warn_unrouted(prefix)

    def _safe_call(self, cb: Callable[[str], None], text: str) -> None:
        try:
            cb(text)
        except Exception:
            pass  # one bad subscriber must not kill the router

    def _warn_unrouted(self, prefix: str | None) -> None:
        active = self.active_prefixes()
        if prefix is None:
            msg = (
                "[steering] no catch-all subscriber; line ignored. "
                f"Active prefixes: {active or '(none)'}"
            )
        else:
            msg = (
                f"[steering] no subscriber for prefix {prefix!r}; line ignored. "
                f"Active prefixes: {active or '(none)'}"
            )
        print(msg, file=sys.stderr)


# ── Per-agent steering sources (used by steering_source_factory) ──────────────


class StdinAgentSource:
    """Async ctx mgr: subscribe one agent to a shared StdinRouter.

    On enter, registers `agent.steer` against the router under `prefix`
    (defaults to `agent.config.agent_id`). On exit, unsubscribes. The
    shared router is started/stopped by the caller — usually the
    `stdin_steering_factory` wrapper or the user's setup code.
    """

    def __init__(
        self,
        agent: BaseAgent,
        router: StdinRouter,
        prefix: str | None = None,
    ) -> None:
        self._agent = agent
        self._router = router
        self._prefix = prefix if prefix is not None else agent.config.agent_id
        self._sub_id: int | None = None

    async def __aenter__(self) -> StdinAgentSource:
        self._sub_id = self._router.subscribe(self._prefix, self._agent.steer)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._sub_id is not None:
            self._router.unsubscribe(self._sub_id)
            self._sub_id = None


class FileSteer:
    """Async ctx mgr: tail an append-only file into agent.steer().

    Polls `path` every `interval` seconds. New content (since the last
    read offset) is split into lines; non-empty lines forward to
    `agent.steer()`. Never mutates the file.

    Default path: f"/tmp/ah-{run_id}-{agent.config.agent_id}.steer".
    Pass `path` or `run_id` (run_id derives the default path).

    Semantics:
      - On enter: positions at current EOF → pre-existing content is NOT
        replayed as guidance.
      - Missing file → no-op until it appears.
      - Truncation (size < offset) → offset resets to 0.
      - Stop signal cancels the polling task.
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

    async def _watch(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue
            else:
                return

    def _poll_once(self) -> None:
        try:
            size = os.path.getsize(self._path)
        except FileNotFoundError:
            return
        if size < self._offset:
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


# ── Factory helpers (for AgentRuntime(steering_source_factory=...)) ───────────


def file_steering_factory(
    path_template: str = "/tmp/ah-{run_id}-{agent_id}.steer",
    *,
    interval: float = 0.25,
) -> Callable[[BaseAgent], FileSteer]:
    """Return a factory that builds a per-agent FileSteer source.

    The template may reference `{run_id}` and `{agent_id}`. The factory
    reads `agent._ckp_id` (set by BaseAgent.run_stream) to extract the
    run_id.
    """

    def factory(agent: BaseAgent) -> FileSteer:
        run_id, _, _ = agent._ckp_id.partition(":")
        path = path_template.format(run_id=run_id, agent_id=agent.config.agent_id)
        return FileSteer(agent, path, interval=interval)

    return factory


class _StdinSteeringFactory:
    """Callable + async context manager.

    As a callable: returns a per-agent `StdinAgentSource` subscribed to
    the agent's `agent_id` prefix on the shared `StdinRouter`.

    As an async context manager: starts the router (and registers it for
    HITL coordination) on enter, stops it on exit. The AgentRuntime
    detects this shape and wraps its `dispatch_stream` automatically, so
    user code doesn't need to manage the router lifecycle.

    Bring-your-own-router: pass `router=...` to skip the lifecycle hooks
    (you become responsible for start/stop).
    """

    def __init__(
        self,
        router: StdinRouter | None = None,
        prefix_template: str = "{agent_id}",
    ) -> None:
        self._router = router or StdinRouter()
        self._owned = router is None
        self._prefix_template = prefix_template
        # Ref-counted lifecycle: nested AgentRuntime wraps (dispatch_stream
        # → run_stream) re-enter the factory; only the outermost
        # enter/exit actually starts/stops the router.
        self._enter_count = 0

    def __call__(self, agent: BaseAgent) -> StdinAgentSource:
        prefix = self._prefix_template.format(agent_id=agent.config.agent_id)
        return StdinAgentSource(agent, self._router, prefix=prefix)

    async def __aenter__(self) -> _StdinSteeringFactory:
        if self._owned and self._enter_count == 0:
            await self._router.__aenter__()
        self._enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._enter_count = max(0, self._enter_count - 1)
        if self._owned and self._enter_count == 0:
            await self._router.__aexit__(exc_type, exc, tb)


def stdin_steering_factory(
    router: StdinRouter | None = None,
    prefix_template: str = "{agent_id}",
) -> _StdinSteeringFactory:
    """Return a steering factory that lifecycles its own StdinRouter.

    The returned object is both a callable (per-agent source factory)
    and an async context manager (shared-router lifecycle). When passed
    to `AgentRuntime(steering_source_factory=...)`, the runtime enters
    the context manager around `dispatch_stream`/`run_stream` so callers
    don't manage the router themselves.

    Pass `router=...` to bring your own (caller manages start/stop).
    `prefix_template` may reference `{agent_id}`; default subscribes
    each agent to its own `agent_id`.
    """
    return _StdinSteeringFactory(router=router, prefix_template=prefix_template)


# ── Direct-use shims (no AgentRuntime / no factory) ───────────────────────────


class StdinSteer:
    """Async ctx mgr for direct (non-runtime) stdin steering.

    Wraps a `StdinRouter` and subscribes each agent under its
    `agent_id` prefix. Single-agent mode also subscribes a catch-all so
    the user can type lines without any prefix at all.

    For orchestrated runs via `AgentRuntime`, prefer
    `stdin_steering_factory(router)` so each agent owns its own
    subscription lifecycle.
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
        self._agents = agents
        self._router = router or StdinRouter()
        self._owned_router = router is None
        self._sub_ids: list[int] = []

    async def __aenter__(self) -> StdinSteer:
        if self._owned_router:
            await self._router.start()
        # Always register one subscription per agent under its agent_id.
        for a in self._agents:
            sid = self._router.subscribe(a.config.agent_id, a.steer)
            self._sub_ids.append(sid)
        # Single-agent ergonomics: also register a catch-all so unprefixed
        # lines work. (Multi-agent: unprefixed lines hit the router's
        # unrouted-warning path because there's no catch-all subscriber.)
        if len(self._agents) == 1:
            sid = self._router.subscribe(None, self._agents[0].steer)
            self._sub_ids.append(sid)
        _set_active_router(self._router)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _set_active_router(None)
        for sid in self._sub_ids:
            self._router.unsubscribe(sid)
        self._sub_ids.clear()
        if self._owned_router:
            await self._router.stop()
