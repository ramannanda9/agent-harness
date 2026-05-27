"""
harness/steering.py — async steering sources for BaseAgent.

A steering source is any async context manager that, while active, may call
`agent.steer(text)` to inject guidance into the agent's queue. The agent
owns the source's lifecycle: it enters the source at the top of
`run_stream` and exits it when the run finishes. No live-agent registry,
no shared mutable state.

Transport-level pieces:

  StdinRouter         — process-wide pub/sub over stdin, built on
                        prompt_toolkit. Persistent prompt anchored at the
                        bottom of the terminal (output scrolls above it
                        without breaking the input line), multi-line
                        composition (Enter submits, Ctrl+J or Esc-Enter
                        insert a newline), in-memory history (up/down),
                        and tab-completion of active agent prefixes.
                        Coordinates with HITL via `claim_next_line()`.

  FileSteer           — per-agent file watcher. Polls an append-only file
                        and forwards new lines to `agent.steer()`. The
                        agent never shares this file with another agent.

Per-agent sources for `steering_source_factory`:

  StdinAgentSource    — subscribes to one prefix on a shared StdinRouter
                        and forwards matched text to the bound agent's
                        `steer()`. Body may be multi-line if the user
                        composed it that way.
  FileSteer           — also serves as a per-agent source (the factory
                        constructs it with the right path / agent).

Factory helpers (drop straight into `AgentRuntime(steering_source_factory=…)`):

  file_steering_factory(path_template)
  stdin_steering_factory(router=None)   — returned object is callable AND
                                          async context manager; runtime
                                          auto-manages router lifecycle.

Direct-use shims (for code that holds explicit BaseAgent references and
bypasses AgentRuntime):

  StdinSteer(agents)   — subscribes each agent on a fresh or supplied
                         router; single-agent mode also subscribes a
                         catch-all so unprefixed lines work.

HITL coordination:
   `harness/hitl.py:request_approval` calls `get_active_router()`. If a
   router is active, it calls `router.claim_next_line(prompt)` BEFORE
   printing the banner. The next text submitted by the user into the
   already-active steering prompt (`> `) is intercepted by
   `_serve_steering` and routed to the HITL Future instead of being
   dispatched to subscribers. If the user submits empty text (or the
   claim is set between prompt cycles), `_serve_hitl` shows a dedicated
   HITL prompt as a fallback. No `app.exit()` is used — that caused a
   race where user input was consumed and lost by the dying prompt.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout

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


class _PrefixCompleter(Completer):
    """Tab-completion for active agent prefixes. Only completes the first
    word (before any `:`) so completion stops once the user is typing the
    body of the message.
    """

    def __init__(self, prefixes_fn: Callable[[], list[str]]) -> None:
        self._prefixes_fn = prefixes_fn

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if ":" in text or "\n" in text:
            return  # past the prefix; don't suggest anything
        for prefix in self._prefixes_fn():
            if prefix.startswith(text):
                yield Completion(f"{prefix}: ", start_position=-len(text))
        if "*".startswith(text):
            yield Completion("*: ", start_position=-len(text))


class StdinRouter:
    """Process-wide stdin reader with prefix-keyed pub/sub.

    Built on prompt_toolkit: persistent prompt anchored at the bottom
    (agent output scrolls above it via `patch_stdout`), multi-line
    composition (Enter submits, Ctrl+J or Esc-Enter insert a newline),
    in-memory history, and tab-completion of currently-active prefixes.

    Routing rules — each submitted block parses its FIRST line for a
    `prefix:` pattern; everything after the colon (plus any further
    lines) is the body:

      - `prefix: body`     → delivered to subscribers with that prefix
                             (or to the `*` broadcast set when
                             prefix=`*`).
      - `body` (no colon)  → delivered to catch-all subscribers
                             (prefix=None).
      - Unknown prefix     → warned to stderr, dropped.

    HITL claims pre-empt pub/sub delivery: when `claim_next_line(prompt)`
    is called, the next submitted text resolves the HITL Future instead
    of routing to subscribers. If the router reaches a pending claim
    between steering prompt cycles, it shows HITL's prompt text directly.
    """

    # Testability: tests construct with custom prompt_toolkit Input/Output
    # via the input_/output kwargs (prompt_toolkit.input.create_pipe_input
    # + DummyOutput) so we don't need a real TTY.

    def __init__(
        self,
        *,
        input_: Any | None = None,
        output: Any | None = None,
        history: Any | None = None,
        patch_stdout_: bool = True,
    ) -> None:
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # HITL claim: (prompt_text, Future) or None.
        self._hitl_claim: tuple[str, asyncio.Future[str]] | None = None
        # subscription_id → (prefix, callback). prefix=None is catch-all.
        self._subs: dict[int, tuple[str | None, Callable[[str], None]]] = {}
        self._next_sub_id: int = 0
        # Tests turn off patch_stdout to avoid interfering with pytest capture.
        self._patch_stdout = patch_stdout_
        self._session: PromptSession = PromptSession(
            history=history or InMemoryHistory(),
            input=input_,
            output=output,
        )
        self._key_bindings = self._build_key_bindings()
        self._completer = _PrefixCompleter(self.active_prefixes)

    @staticmethod
    def _build_key_bindings() -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event):
            event.current_buffer.validate_and_handle()

        @kb.add("c-j")
        def _newline_ctrl_j(event):
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.Escape, "enter")
        def _newline_alt_enter(event):
            event.current_buffer.insert_text("\n")

        return kb

    # ── Pub/sub API ───────────────────────────────────────────────────────────

    def subscribe(
        self,
        prefix: str | None,
        callback: Callable[[str], None],
    ) -> int:
        """Register `callback` to receive text matching `prefix`.

        prefix=None  → catch-all (lines with no `:`).
        prefix=str   → exact prefix match plus `*` broadcasts.
        `*` is reserved for broadcast SENDS — not a valid subscription
        prefix.

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
        return sorted({p for p, _ in self._subs.values() if p is not None})

    def has_catchall(self) -> bool:
        return any(p is None for p, _ in self._subs.values())

    # ── HITL API ──────────────────────────────────────────────────────────────

    def claim_next_line(self, prompt: str | None = None) -> asyncio.Future[str]:
        """Reserve the next stdin read for HITL.

        Returns a Future that resolves with the user's typed answer.
        The claim is satisfied in one of two ways:

        1. **Fast path** — the user types their answer into the
           already-active steering prompt (``> ``).  When the steering
           prompt returns text while a claim is pending,
           ``_serve_steering`` routes it to the HITL future instead of
           dispatching to subscribers.
        2. **Fallback** — the user submits empty text (or the claim is
           set between prompt cycles).  The ``_run`` loop sees the
           still-pending claim and calls ``_serve_hitl``, which shows
           the dedicated HITL prompt text.

        No ``app.exit()`` is used — that caused a race where input
        typed between the exit and the new prompt was consumed and lost
        by the dying steering prompt.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._hitl_claim = (prompt or "> ", fut)
        return fut

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            # Trigger exit so prompt_async returns; then cancel as backup.
            app = self._session.app
            if app is not None and getattr(app, "is_running", False):
                app.exit()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        if self._hitl_claim is not None:
            _, fut = self._hitl_claim
            if not fut.done():
                fut.cancel()
            self._hitl_claim = None

    async def __aenter__(self) -> StdinRouter:
        await self.start()
        _set_active_router(self)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _set_active_router(None)
        await self.stop()

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        # patch_stdout makes prints from other tasks scroll above the prompt
        # instead of corrupting the input line. Tests skip it because it
        # interferes with pytest's stdout capture.
        cm = patch_stdout(raw=True) if self._patch_stdout else contextlib.nullcontext()
        with cm:
            while not self._stop.is_set():
                claim = self._hitl_claim
                if claim is not None:
                    await self._serve_hitl(*claim)
                    self._hitl_claim = None
                else:
                    await self._serve_steering()

    async def _serve_steering(self) -> None:
        try:
            text = await self._session.prompt_async(
                "> ",
                multiline=True,
                key_bindings=self._key_bindings,
                completer=self._completer,
                complete_while_typing=True,
            )
        except (KeyboardInterrupt, EOFError):
            self._stop.set()
            return
        except asyncio.CancelledError:
            # stop() was called.  Just return — loop decides.
            return
        if text is None or not text.strip():
            return
        # HITL fast-path: if a claim arrived while the steering prompt
        # was active, the user's typed answer should go to HITL, not to
        # steering dispatch.  This avoids the app.exit() race where
        # input was consumed and lost by the dying prompt.
        claim = self._hitl_claim
        if claim is not None:
            _, fut = claim
            if not fut.done():
                fut.set_result(text.strip())
            self._hitl_claim = None
            return
        self._dispatch(text)

    async def _serve_hitl(
        self,
        prompt_text: str,
        fut: asyncio.Future[str],
    ) -> None:
        try:
            # Same multiline=True + our Enter-submits / Ctrl+J-newline
            # bindings as steering so the UX is consistent. Single-token
            # answers (y/n/a) still submit on a single Enter.
            text = await self._session.prompt_async(
                prompt_text,
                multiline=True,
                key_bindings=self._key_bindings,
            )
        except (KeyboardInterrupt, EOFError) as e:
            if not fut.done():
                fut.set_exception(e)
            self._stop.set()
            return
        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            return
        if not fut.done():
            fut.set_result(text if text is not None else "")

    def _dispatch(self, text: str) -> None:
        # Find the prefix by inspecting the FIRST line only.
        first_line, sep, rest = text.partition("\n")
        if ":" in first_line:
            prefix, _, head = first_line.partition(":")
            prefix = prefix.strip()
            body = head.lstrip()
            if rest:
                body = f"{body}\n{rest}" if body else rest
            body = body.strip()
            if not body:
                return
            if prefix == "*":
                for p, cb in list(self._subs.values()):
                    if p is not None:
                        self._safe_call(cb, body)
                return
            matched = False
            for p, cb in list(self._subs.values()):
                if p == prefix:
                    self._safe_call(cb, body)
                    matched = True
            if not matched:
                self._warn_unrouted(prefix)
            return

        body = text.strip()
        if not body:
            return
        matched = False
        for p, cb in list(self._subs.values()):
            if p is None:
                self._safe_call(cb, body)
                matched = True
        if not matched:
            self._warn_unrouted(None)

    @staticmethod
    def _safe_call(cb: Callable[[str], None], text: str) -> None:
        try:
            cb(text)
        except Exception:
            pass  # one bad subscriber must not kill the router

    def _warn_unrouted(self, prefix: str | None) -> None:
        active = self.active_prefixes()
        if prefix is None:
            msg = (
                "[steering] no catch-all subscriber; input ignored. "
                f"Active prefixes: {active or '(none)'}"
            )
        else:
            msg = (
                f"[steering] no subscriber for prefix {prefix!r}; input ignored. "
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
