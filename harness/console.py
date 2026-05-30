"""Standard console renderer for BusEvent streams."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from harness.events import BusEvent, EventType


def trunc(s: str, n: int) -> str:
    """Truncate *s* to *n* characters, appending '…' when clipped."""
    return s if len(s) <= n else s[:n] + "…"


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
        out: TextIO | None = None,
    ) -> None:
        self._truncate = truncate
        self._sep_char = sep_char
        self._sep_width = sep_width
        self._label_w = agent_label_width
        self._show_tokens = show_tokens
        self._out = out or sys.stdout
        self._in_token_stream = False

    # ── public helpers ────────────────────────────────────────────────────────

    def sep(self, char: str | None = None, w: int | None = None) -> None:
        """Print a separator line."""
        print((char or self._sep_char) * (w or self._sep_width), file=self._out)

    def render(self, event: BusEvent) -> None:
        """Print formatted output for one BusEvent."""
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
            print(
                f"{self._label(event)} obs     {trunc(obs, 110)}",
                file=self._out,
            )

        elif t == EventType.HUMAN_GUIDANCE:
            print(
                f"\n{self._label(event)} ▶ steered  step={p.get('step')}  text={p.get('text')!r}",
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
            print(
                f"Confidence: {p.get('confidence', 0):.2f}  |  "
                f"Replans: {p.get('replan_count', 0)}  |  "
                f"Cost: ${p.get('cost_usd', 0):.4f}  |  "
                f"Time: {p.get('elapsed_seconds', 0):.1f}s",
                file=self._out,
            )

        elif t == EventType.ERROR:
            print(f"\n[error]      {event.error}", file=sys.stderr)

    # ── private helpers ───────────────────────────────────────────────────────

    def _label(self, event: BusEvent) -> str:
        if event.agent_id:
            return f"[{event.agent_id:<{self._label_w}}]"
        return f"[{event.type.value:<{self._label_w}}]"
