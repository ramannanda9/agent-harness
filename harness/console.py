"""Standard console renderer for BusEvent streams."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import NamedTuple, TextIO

from harness.events import BusEvent, EventType


class StreamResult(NamedTuple):
    """Return shape of ``ConsoleRenderer.render_stream``.

    ``cancelled`` is True when the user pressed Esc mid-stream (only
    possible on TTY stdin; non-TTY hosts never set it). ``terminal`` is
    the last event matching the requested ``terminal_event_type`` (and,
    when ``top_level_only`` is True, having no ``parent_agent_id``), or
    None when no such event arrived before cancellation / end-of-stream.

    Tuple shape so callers can write ``cancelled, terminal = await
    renderer.render_stream(...)`` without importing the class.
    """

    cancelled: bool
    terminal: BusEvent | None


def trunc(s: str, n: int) -> str:
    """Truncate *s* to *n* characters, appending '…' when clipped."""
    return s if len(s) <= n else s[:n] + "…"


class _TextTail:
    """Bounded tail buffer for text that may arrive in chunks."""

    def __init__(self, *, max_chars: int, max_lines: int) -> None:
        self._max_chars = max(0, max_chars)
        self._max_lines = max(0, max_lines)
        self._tail = ""
        self.total_chars = 0
        self.total_lines = 0
        self.omitted = False

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.total_chars += len(chunk)
        self.total_lines += len(chunk.splitlines())
        if self._max_chars <= 0 or self._max_lines <= 0:
            self.omitted = True
            self._tail = ""
            return

        self._tail += chunk
        self._trim()

    @property
    def text(self) -> str:
        return self._tail

    def _trim(self) -> None:
        lines = self._tail.splitlines()
        if len(lines) > self._max_lines:
            self.omitted = True
            self._tail = "\n".join(lines[-self._max_lines :])
        if len(self._tail) > self._max_chars:
            self.omitted = True
            self._tail = self._tail[-self._max_chars :]
            if "\n" in self._tail:
                # Avoid starting mid-line when a newline still exists inside the slice.
                self._tail = self._tail[self._tail.find("\n") + 1 :]


class ConsoleRenderer:
    """Renders BusEvent objects to a text stream.

    Centralises all event-type formatting so callers don't duplicate
    THOUGHT/ACTION/OBSERVATION/... blocks and separator/truncation helpers.

    Args:
        truncate:           Max characters for long text fields.
        sep_char:           Character used for separator lines.
        sep_width:          Width of separator lines.
        agent_label_width:  Width of the ``[agent_id]`` label column.
        show_tokens:        If True, TOKEN events are printed inline.
        spinner:            If True, show a TTY-only busy spinner between events.
        spinner_delay:      Seconds to wait before drawing the spinner.
        tail_large_outputs: Render large observations as a bounded tail block
                            instead of first-N-character truncation.
        tail_lines:         Maximum lines to show in an observation tail.
        tail_chars:         Maximum characters to show in an observation tail.
        out:                Output stream (defaults to ``sys.stdout``).
    """

    def __init__(
        self,
        *,
        truncate: int = 140,
        sep_char: str = "─",
        sep_width: int = 72,
        agent_label_width: int = 16,
        show_tokens: bool = False,
        spinner: bool = True,
        spinner_delay: float = 0.4,
        tail_large_outputs: bool = True,
        tail_lines: int = 5,
        tail_chars: int = 4_000,
        out: TextIO | None = None,
    ) -> None:
        self._truncate = truncate
        self._sep_char = sep_char
        self._sep_width = sep_width
        self._label_w = agent_label_width
        self._show_tokens = show_tokens
        self._tail_large_outputs = tail_large_outputs
        self._tail_lines = tail_lines
        self._tail_chars = tail_chars
        self._out = out or sys.stdout
        self._in_token_stream = False
        is_tty = bool(getattr(self._out, "isatty", lambda: False)())
        self._spinner = _Spinner(
            out=self._out,
            enabled=spinner and is_tty,
            delay=spinner_delay,
        )
        self._subagent_panel = _SubagentPanel(
            out=self._out,
            enabled=is_tty,
            label_width=agent_label_width,
        )

    # ── public helpers ────────────────────────────────────────────────────────

    def sep(self, char: str | None = None, w: int | None = None) -> None:
        """Print a separator line."""
        print((char or self._sep_char) * (w or self._sep_width), file=self._out)

    def render(self, event: BusEvent) -> None:
        """Print formatted output for one BusEvent."""
        self._spinner.stop()
        self._subagent_panel.erase()
        if event.type == EventType.TOKEN:
            if self._show_tokens:
                if not self._in_token_stream:
                    self._in_token_stream = True
                self._out.write(event.token)
                self._out.flush()
            return

        # Close any in-progress token stream before the next event line.
        if self._in_token_stream:
            self._out.write("\n")
            self._out.flush()
            self._in_token_stream = False

        t = event.type
        p = event.payload

        if t == EventType.DISPATCH:
            print(
                f"\n[dispatch]   complexity={p.get('complexity')}  path={p.get('path')}",
                file=self._out,
            )

        elif t == EventType.ROUTE:
            print(
                f"[route]      → {p.get('agent_id')}: {trunc(p.get('rationale', ''), 90)}",
                file=self._out,
            )

        elif t == EventType.PLAN:
            tasks = p.get("plan", {}).get("tasks", [])
            print(f"\n[plan]       {len(tasks)} tasks", file=self._out)
            for task in tasks:
                deps = f"  ← {task['depends_on']}" if task.get("depends_on") else ""
                print(
                    f"             {task['id']}@{task['agent_id']}: "
                    f"{trunc(task.get('instruction', ''), 70)}{deps}",
                    file=self._out,
                )

        elif t == EventType.PLAN_PROPOSED:
            # PersistentAgent plan-mode preview, before HITL approval.
            # Renders the full plan so the user can read it before the
            # approval banner prints; HITL's banner repeats the summary
            # in a yes/no/correction context.
            #
            # Steps whose args are deferred to runtime render as
            # ``args: (resolved at runtime)`` rather than fabricating
            # placeholder JSON — keep the renderer convention in lock-
            # step with ``_render_plan_for_banner`` so the user sees the
            # same shape twice.
            plan = p.get("plan", {}) if isinstance(p.get("plan"), dict) else {}
            revision = p.get("revision", 0)
            summary = plan.get("summary") or "(no summary)"
            steps = plan.get("steps") or []
            header = "[plan propose]"
            if revision:
                header = f"[plan rev {revision}]"
            print(f"\n{header}  {summary}", file=self._out)
            for step in steps:
                idx = step.get("step")
                intent = step.get("intent") or "(no intent)"
                tool = step.get("tool")
                why = step.get("why")
                prefix = f"  {idx}." if idx is not None else "  -"
                print(f"{prefix} {intent}", file=self._out)
                if tool:
                    # Convention: ``"args"`` key missing or ``args is None``
                    # → deferred. ``args == {}`` → "tool takes no args".
                    args_present = "args" in step and step["args"] is not None
                    if not args_present:
                        print(f"      tool: {tool}  args: (resolved at runtime)", file=self._out)
                    else:
                        args = step["args"] or {}
                        args_repr = (
                            json.dumps(args, ensure_ascii=False, default=str) if args else "{}"
                        )
                        print(f"      tool: {tool}  args: {args_repr}", file=self._out)
                if why:
                    print(f"      why:  {why}", file=self._out)

        elif t == EventType.THOUGHT:
            thought = p.get("thought", "")
            if thought:
                print(
                    f"{self._label(event)} think   {trunc(thought, 110)}",
                    file=self._out,
                )

        elif t == EventType.ACTION:
            args = json.dumps(p.get("args", {}), default=str)
            print(
                f"{self._label(event)} action  {p.get('tool')}({trunc(args, 90)})",
                file=self._out,
            )

        elif t == EventType.OBSERVATION:
            obs = p.get("observation", "")
            self._render_observation(event, str(obs))

        elif t == EventType.CONTEXT:
            tokens = int(p.get("tokens") or 0)
            max_tokens = int(p.get("max_tokens") or 0)
            pct = float(p.get("percent") or 0.0) * 100
            level = p.get("level") or "normal"
            suffix = "" if level == "normal" else f"  {level}"
            llm_parts: list[str] = []
            if p.get("tokens_in") is not None:
                llm_parts.append(f"in={int(p['tokens_in']):,}")
            if p.get("tokens_out") is not None:
                llm_parts.append(f"out={int(p['tokens_out']):,}")
            if p.get("cache_read_tokens"):
                llm_parts.append(f"cache_hit={int(p['cache_read_tokens']):,}")
            if p.get("cache_creation_tokens"):
                llm_parts.append(f"cache_new={int(p['cache_creation_tokens']):,}")
            llm_suffix = f"  [{' '.join(llm_parts)}]" if llm_parts else ""
            print(
                f"{self._label(event)} ctx     {tokens:,} / {max_tokens:,} tokens  "
                f"{pct:.0f}%{suffix}{llm_suffix}",
                file=self._out,
            )

        elif t == EventType.MEMORY:
            before = p.get("before") if isinstance(p.get("before"), dict) else {}
            after = p.get("after") if isinstance(p.get("after"), dict) else {}
            print(
                f"{self._label(event)} memory  summarized  "
                f"{int(before.get('tokens') or 0):,} -> {int(after.get('tokens') or 0):,} tokens",
                file=self._out,
            )

        elif t == EventType.HUMAN_GUIDANCE:
            print(
                f"\n{self._label(event)} ▶ steered  step={p.get('step')}  text={p.get('text')!r}",
                file=self._out,
            )

        elif t == EventType.SUBAGENT_START:
            indent = "  " if event.parent_agent_id else ""
            print(
                f"{indent}[{event.agent_id:<{self._label_w}}] → start  {trunc(p.get('task', ''), 90)}",
                file=self._out,
            )

        elif t == EventType.SUBAGENT_DONE:
            indent = "  " if event.parent_agent_id else ""
            if p.get("success"):
                status = (
                    f"✓ done   confidence={p.get('confidence', 0):.2f}  steps={p.get('steps', '?')}"
                )
            else:
                err = trunc(p.get("error", "unknown error"), 80)
                status = f"✗ failed  {err}"
            print(
                f"{indent}[{event.agent_id:<{self._label_w}}] {status}",
                file=self._out,
            )

        elif t == EventType.TASK_DONE:
            print(
                f"{self._label(event)} ✓ done  "
                f"confidence={p.get('confidence', 0):.2f}  steps={p.get('steps', '?')}",
                file=self._out,
            )

        elif t == EventType.REPLAN:
            print(
                f"\n[replan]     #{p.get('replan_count')} — trigger={p.get('trigger_task', '?')}",
                file=self._out,
            )

        elif t == EventType.SYNTHESIS:
            print(
                f"\n[synthesis]  confidence={p.get('confidence', 0):.2f}",
                file=self._out,
            )

        elif t == EventType.DONE:
            print(file=self._out)
            self.sep("═")
            print(p.get("answer", "(no answer)"), file=self._out)
            self.sep()
            # ``budget`` snapshot supersedes the flat cost/elapsed fields when
            # present (added with token caps + per-call-site breakdown).
            budget = p.get("budget") or {}
            cost = budget.get("cost_usd", p.get("cost_usd", 0))
            elapsed = budget.get("elapsed_seconds", p.get("elapsed_seconds", 0))
            print(
                f"Confidence: {p.get('confidence', 0):.2f}  |  "
                f"Replans: {p.get('replan_count', 0)}  |  "
                f"Cost: ${cost:.4f}  |  "
                f"Time: {elapsed:.1f}s",
                file=self._out,
            )
            self.render_budget(budget)

        elif t == EventType.ERROR:
            print(f"\n[error]      {event.error}", file=sys.stderr)

        self._subagent_panel.update(event)
        if self._subagent_panel.should_draw:
            self._subagent_panel.draw()
        else:
            self._schedule_spinner(event)

    def close(self) -> None:
        """Stop any background spinner owned by this renderer."""
        self._spinner.stop()
        self._subagent_panel.erase()

    def render_budget(self, budget: dict | None) -> None:
        """Print tokens + per-call-site breakdown from a ``BudgetGuard.snapshot()``
        dict. Safe to call with ``{}`` or ``None`` — prints nothing when
        there's no usage to show.

        Exposed publicly so demos and other consumers that own their own
        DONE / TASK_DONE rendering can still surface the breakdown without
        duplicating the formatting.
        """
        if not budget:
            return
        tokens_in = budget.get("tokens_in")
        tokens_out = budget.get("tokens_out")
        if tokens_in is not None or tokens_out is not None:
            print(
                f"Tokens:     in={int(tokens_in or 0):,}  out={int(tokens_out or 0):,}",
                file=self._out,
            )
        breakdown = budget.get("breakdown") or {}
        if breakdown:
            # Right-pad the slot label so columns line up — matters when
            # the demo prints multiple slots in sequence.
            width = max(len(name) for name in breakdown)
            for slot, stats in breakdown.items():
                print(
                    f"  {slot:<{width}}  "
                    f"in={int(stats.get('tokens_in', 0)):>7,}  "
                    f"out={int(stats.get('tokens_out', 0)):>6,}",
                    file=self._out,
                )

    async def render_stream(
        self,
        events: AsyncIterator[BusEvent],
        *,
        terminal_event_type: EventType | None = None,
        top_level_only: bool = True,
        cancel_message: str = "[cancelled by user]",
        print_cancel_banner: bool = True,
    ) -> StreamResult:
        """Render an event stream until completion or user Esc.

        Composes the rendering loop (``self.render``), the per-event
        terminal capture (so callers don't repeat the "find last
        TASK_DONE / DONE" boilerplate), and the Esc-cancel listener from
        ``harness.cancellation`` into one call. Most consumers only need
        this — the lower-level ``consume_with_cancel`` is the escape
        hatch for bespoke per-event handling (orchestrator demos that
        print a custom report on DONE, etc.).

        Returns ``StreamResult(cancelled, terminal)``. Defaults to
        printing a banner when ``cancelled`` so call sites don't repeat
        the ``sep / print / sep`` pattern; pass
        ``print_cancel_banner=False`` to suppress.
        """
        # Local import keeps ``ConsoleRenderer`` light to import for
        # consumers that never render a stream (snapshot/JSON output etc.).
        from harness.cancellation import consume_with_cancel  # noqa: PLC0415

        terminal_holder: list[BusEvent | None] = [None]

        def _on_event(event: BusEvent) -> None:
            self.render(event)
            if terminal_event_type is None or event.type != terminal_event_type:
                return
            if top_level_only and event.parent_agent_id:
                return
            terminal_holder[0] = event

        try:
            cancelled = await consume_with_cancel(events, on_event=_on_event)
        finally:
            self.close()
        if cancelled and print_cancel_banner:
            self.sep("═")
            print(cancel_message, file=self._out)
            self.sep("═")
        return StreamResult(cancelled=cancelled, terminal=terminal_holder[0])

    # ── private helpers ───────────────────────────────────────────────────────

    def _label(self, event: BusEvent) -> str:
        if event.agent_id:
            return f"[{event.agent_id:<{self._label_w}}]"
        return f"[{event.type.value:<{self._label_w}}]"

    def _render_observation(self, event: BusEvent, obs: str) -> None:
        label = self._label(event)
        should_tail = self._tail_large_outputs and (len(obs) > self._truncate or "\n" in obs)
        if not should_tail:
            print(f"{label} obs     {trunc(obs, self._truncate)}", file=self._out)
            return

        tail = _TextTail(
            max_chars=self._tail_chars,
            max_lines=self._tail_lines,
        )
        tail.append(obs)
        self._print_observation_tail(label, tail)

    def _print_observation_tail(self, label: str, tail: _TextTail) -> None:
        self._write_observation_tail(label, tail)

    def _write_observation_tail(self, label: str, tail: _TextTail) -> int:
        if tail.omitted:
            self._out.write(
                f"{label} obs     [tail: last {self._tail_lines} lines / "
                f"{len(tail.text):,} of {tail.total_chars:,} chars]\n"
            )
        else:
            self._out.write(
                f"{label} obs     [tail: {tail.total_lines} lines / {tail.total_chars:,} chars]\n"
            )
        drawn = 1
        if tail.text:
            for line in tail.text.splitlines():
                self._out.write(f"{'':<{self._label_w + 11}}{line}\n")
                drawn += 1
        return drawn

    def _schedule_spinner(self, event: BusEvent) -> None:
        label = self._spinner_label(event)
        if label:
            self._spinner.start_later(label)

    def _spinner_label(self, event: BusEvent) -> str | None:
        if event.type in {
            EventType.TASK_DONE,
            EventType.DONE,
            EventType.ERROR,
            EventType.PLAN_PROPOSED,
            EventType.HUMAN_GUIDANCE,
            EventType.SYNTHESIS,
        }:
            return None
        agent = event.agent_id or (
            str(event.payload.get("agent_id")) if isinstance(event.payload, dict) else ""
        )
        agent = agent or event.type.value
        if event.type == EventType.ACTION:
            tool = event.payload.get("tool") if isinstance(event.payload, dict) else None
            return f"[{agent}] using {tool or 'tool'}..."
        if event.type in {EventType.OBSERVATION, EventType.CONTEXT, EventType.ROUTE}:
            return f"[{agent}] thinking..."
        if event.type in {EventType.DISPATCH, EventType.PLAN, EventType.REPLAN, EventType.MEMORY}:
            return "[agent-harness] working..."
        return None


class _Spinner:
    _FRAMES = ("|", "/", "-", "\\")

    def __init__(self, *, out: TextIO, enabled: bool, delay: float) -> None:
        self._out = out
        self._enabled = enabled
        self._delay = max(0.0, delay)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start_later(self, text: str) -> None:
        if not self._enabled:
            return
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(text,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=self._delay + 0.2)
        self._thread = None
        self._clear_line()

    def _run(self, text: str) -> None:
        if self._stop.wait(self._delay):
            return
        idx = 0
        while not self._stop.is_set():
            self._out.write(f"\r{text} {self._FRAMES[idx % len(self._FRAMES)]}")
            self._out.flush()
            idx += 1
            if self._stop.wait(0.1):
                break

    def _clear_line(self) -> None:
        if not self._enabled:
            return
        self._out.write("\r\033[K")
        self._out.flush()


@dataclass
class _AgentRow:
    agent_id: str
    parent_agent_id: str
    phase: str = "starting"
    detail: str = ""
    step: int | None = None
    confidence: float | None = None
    steps: int | None = None
    active: bool = True


class _SubagentPanel:
    """TTY-only live table for concurrent sub-agent delegations."""

    def __init__(self, *, out: TextIO, enabled: bool, label_width: int) -> None:
        self._out = out
        self._enabled = enabled
        self._label_width = label_width
        self._rows: dict[str, _AgentRow] = {}
        self._drawn = 0
        self._saw_parallel = False

    @property
    def should_draw(self) -> bool:
        if not self._enabled or not self._rows:
            return False
        active = self._active_count()
        return active >= 2 or (self._saw_parallel and active > 0)

    def update(self, event: BusEvent) -> None:
        if not self._enabled:
            return
        if event.type == EventType.SUBAGENT_START:
            key = self._key(event)
            self._rows[key] = _AgentRow(
                agent_id=event.agent_id,
                parent_agent_id=event.parent_agent_id,
            )
            if self._active_count() >= 2:
                self._saw_parallel = True
            return

        if not event.parent_agent_id:
            return

        row = self._row_for(event)
        if row is None:
            return

        if event.type == EventType.THOUGHT:
            row.phase = "thinking"
            row.detail = trunc(str(event.payload.get("thought", "")), 36)
            step = _payload_int(event.payload.get("step"))
            row.step = step if step is not None else row.step
        elif event.type == EventType.ACTION:
            row.phase = "action"
            row.detail = str(event.payload.get("tool") or "tool")
            step = _payload_int(event.payload.get("step"))
            row.step = step if step is not None else row.step
        elif event.type == EventType.CONTEXT:
            row.phase = "context"
            row.detail = f"{int(event.payload.get('tokens') or 0):,} tokens"
        elif event.type == EventType.SUBAGENT_DONE:
            row.active = False
            row.phase = "done" if event.payload.get("success") else "failed"
            row.detail = str(event.payload.get("error") or "")
            row.confidence = _payload_float(event.payload.get("confidence"))
            row.steps = _payload_int(event.payload.get("steps"))
            if self._active_count() == 0:
                self._rows.clear()
                self._saw_parallel = False

    def draw(self) -> None:
        if not self.should_draw:
            return
        lines = ["Subagents"]
        for row in self._rows.values():
            lines.append(self._format_row(row))
        self._out.write("\n".join(lines) + "\n")
        self._out.flush()
        self._drawn = len(lines)

    def erase(self) -> None:
        if not self._enabled or self._drawn <= 0:
            return
        for _ in range(self._drawn):
            self._out.write("\033[A\r\033[2K")
        self._out.flush()
        self._drawn = 0

    def _active_count(self) -> int:
        return sum(1 for row in self._rows.values() if row.active)

    def _row_for(self, event: BusEvent) -> _AgentRow | None:
        key = self._key(event)
        row = self._rows.get(key)
        if row is not None:
            return row
        candidates = [
            item
            for item in self._rows.values()
            if item.agent_id == event.agent_id
            and item.parent_agent_id == event.parent_agent_id
            and item.active
        ]
        return candidates[-1] if candidates else None

    def _key(self, event: BusEvent) -> str:
        invocation_id = ""
        if isinstance(event.payload, dict):
            invocation_id = str(event.payload.get("invocation_id") or "")
        if invocation_id:
            return invocation_id
        return f"{event.parent_agent_id}:{event.agent_id}"

    def _format_row(self, row: _AgentRow) -> str:
        agent = row.agent_id[: self._label_width]
        status = "running" if row.active else row.phase
        if row.active:
            detail = f"{row.phase} {row.detail}".strip()
            suffix = f"step {row.step}" if row.step is not None else ""
        elif row.phase == "done":
            detail = (
                f"confidence {row.confidence:.2f}" if row.confidence is not None else "complete"
            )
            suffix = f"{row.steps} steps" if row.steps is not None else ""
        else:
            detail = trunc(row.detail or "failed", 36)
            suffix = f"{row.steps} steps" if row.steps is not None else ""
        return f"  {agent:<{self._label_width}}  {status:<8}  {detail:<28}  {suffix}".rstrip()


def _payload_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _payload_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
