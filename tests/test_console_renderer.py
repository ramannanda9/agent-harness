from __future__ import annotations

from io import StringIO

from harness.console import ConsoleRenderer
from harness.events import BusEvent, EventType


def test_console_renderer_context_levels():
    out = StringIO()
    renderer = ConsoleRenderer(out=out)

    renderer.render(
        BusEvent(
            type=EventType.CONTEXT,
            agent_id="agent",
            payload={
                "tokens": 8200,
                "max_tokens": 10000,
                "percent": 0.82,
                "level": "warning",
            },
        )
    )

    text = out.getvalue()
    assert "ctx" in text
    assert "8,200 / 10,000 tokens" in text
    assert "82%" in text
    assert "warning" in text


def test_console_renderer_memory_summary_marker():
    out = StringIO()
    renderer = ConsoleRenderer(out=out)

    renderer.render(
        BusEvent(
            type=EventType.MEMORY,
            agent_id="agent",
            payload={
                "event": "summarized",
                "before": {"tokens": 12000},
                "after": {"tokens": 4200},
            },
        )
    )

    text = out.getvalue()
    assert "memory" in text
    assert "summarized" in text
    assert "12,000 -> 4,200 tokens" in text
