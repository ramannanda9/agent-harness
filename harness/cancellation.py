"""User-initiated cancellation for streaming agent runs.

Two layers:

- ``run_until_cancelled`` runs an awaitable as an ``asyncio.Task`` and
  cancels it when a caller-supplied ``trigger`` event fires. Pure
  asyncio; no UI dependency. Reusable for any "run X but stop early on
  Y" pattern — REPL turns, batch jobs, scheduled probes.

- ``escape_listener`` is a prompt_toolkit-backed async context manager
  that sets a trigger event when the user presses Escape. Composes with
  ``run_until_cancelled`` to give CLIs the standard "press Esc to
  cancel the current turn" affordance.

Agent code itself does not need to know about cancellation —
``asyncio.CancelledError`` propagates naturally through every ``await``
and through ``async for`` over an async generator. Cleanup happens via
ordinary ``try/finally`` blocks where they already exist. The
``PersistentAgent.chat`` flow is shaped so that mid-stream cancellation
writes nothing to the session store (``_finalize_turn`` only runs on
``TASK_DONE``/``ERROR`` or at clean generator end), which matches the
standard chat-UX semantic of "cancelled turn = never happened".
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from harness.events import BusEvent

T = TypeVar("T")


async def run_until_cancelled(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    trigger: asyncio.Event,
) -> tuple[bool, T | None]:
    """Run ``coro_factory()`` until it finishes or ``trigger`` fires.

    Returns ``(cancelled, result)``:

    - ``cancelled`` is ``True`` if the trigger fired before the
      coroutine finished; the task was then cancelled and awaited so any
      ``finally`` cleanup completed before this function returned.
      ``result`` is ``None`` in this case.
    - ``cancelled`` is ``False`` if the coroutine completed first;
      ``result`` is the coroutine's return value. Any waiter on the
      trigger is cancelled before returning.

    The trigger is cleared at entry so the caller can reuse the same
    event across many turns. If the trigger was already set before this
    call, the coroutine still gets a chance to start and run for at
    least one event-loop tick — clearing-then-checking would otherwise
    race with a key press that arrived just before this turn began.
    """
    trigger.clear()
    task: asyncio.Task[T] = asyncio.create_task(coro_factory())
    waiter: asyncio.Task[bool] = asyncio.create_task(trigger.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # Outer cancellation — propagate inward so neither child leaks.
        task.cancel()
        waiter.cancel()
        raise

    if waiter in done and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Cancellation path is best-effort; the caller has already
            # observed "user wanted to stop" — don't shadow that with a
            # late exception from the cancelled task.
            pass
        return True, None

    # Task completed (or errored) first; stop the trigger waiter.
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass
    return False, task.result()


@asynccontextmanager
async def escape_listener(trigger: asyncio.Event) -> AsyncIterator[None]:
    """Set ``trigger`` when the user presses Escape on stdin.

    Active only for the duration of the ``async with`` block. Uses
    prompt_toolkit's input abstraction so terminal mode handling stays
    correct across platforms and so it composes cleanly with
    ``PromptSession`` used elsewhere in the same process.

    No-ops (yields immediately, never sets ``trigger``) when:

    - ``stdin`` is not a TTY (pipe / file / no terminal) — keypress
      capture doesn't apply.
    - prompt_toolkit's input cannot be created on this platform / shell
      configuration — surfacing a startup error here would mask the
      user's actual workload. The trigger simply stays unset, matching
      the pre-Esc-feature behavior.
    """
    if not sys.stdin.isatty():
        yield
        return

    # Local import so consumers that don't need ESC handling (e.g. unit
    # tests that exercise ``run_until_cancelled`` alone) don't pay the
    # prompt_toolkit import cost or its terminal capability probing.
    try:
        from prompt_toolkit.input import create_input
        from prompt_toolkit.keys import Keys
    except Exception:  # noqa: BLE001 — best-effort UI feature
        yield
        return

    try:
        input_obj = create_input()
    except Exception:  # noqa: BLE001 — terminal config refused; degrade silently
        yield
        return

    def _on_input_ready() -> None:
        # Read all currently-available key presses; trigger on the first
        # Escape seen. Other keys are deliberately dropped — this is a
        # cancel listener, not an input layer, and the active
        # ``async for`` over chat events doesn't expect any user input.
        try:
            for key_press in input_obj.read_keys():
                if key_press.key == Keys.Escape:
                    trigger.set()
                    return
        except Exception:  # noqa: BLE001 — listener must not raise into the loop
            return

    # ``raw_mode()`` disables canonical line buffering + echo so we
    # actually see Escape as a keypress rather than after the user hits
    # Enter. ``attach()`` registers ``_on_input_ready`` with the running
    # event loop's reader for ``input_obj``'s file descriptor.
    try:
        with input_obj.raw_mode():
            with input_obj.attach(_on_input_ready):
                yield
    except Exception:  # noqa: BLE001 — terminal manipulation failed mid-flight
        yield


async def consume_with_cancel(
    events: AsyncIterator[BusEvent],
    *,
    on_event: Callable[[BusEvent], None],
) -> bool:
    """Consume a ``BusEvent`` async iterator until completion or user Esc.

    Returns ``True`` if the user pressed Escape (the stream was
    cancelled mid-flight); ``False`` if the stream completed naturally.
    The caller's ``on_event`` callback runs for every event up to the
    cancel point — rendering, final-result extraction, parent-agent
    filtering all live in the callback so this helper stays
    UI-agnostic.

    Most consumers should use ``ConsoleRenderer.render_stream`` instead;
    this lower-level helper is for renderers that aren't
    ``ConsoleRenderer`` (web UI, JSON-stream output, custom orchestrator
    demos that render a bespoke report on DONE) or that need
    per-event logic the renderer method doesn't expose.
    """
    cancel_event = asyncio.Event()

    async def _drain() -> None:
        async for event in events:
            on_event(event)

    async with escape_listener(cancel_event):
        cancelled, _ = await run_until_cancelled(_drain, trigger=cancel_event)
    return cancelled
