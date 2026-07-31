from __future__ import annotations

import time
from io import StringIO

from harness.console import ConsoleRenderer, excerpt
from harness.events import BusEvent, EventType


class _TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


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


def test_console_renderer_done_event_renders_budget_breakdown():
    """When the DONE event carries a ``budget`` snapshot, the renderer should
    show total tokens + the per-call-site breakdown so demos surface
    classifier/router/planner/synthesizer spending."""
    out = StringIO()
    renderer = ConsoleRenderer(out=out)

    renderer.render(
        BusEvent(
            type=EventType.DONE,
            agent_id="orchestrator",
            payload={
                "answer": "all systems nominal",
                "confidence": 0.91,
                "replan_count": 0,
                "budget": {
                    "cost_usd": 0.0142,
                    "elapsed_seconds": 23.4,
                    "tokens_in": 12_340,
                    "tokens_out": 2_890,
                    "breakdown": {
                        "classifier": {"tokens_in": 156, "tokens_out": 24},
                        "planner": {"tokens_in": 8_432, "tokens_out": 1_200},
                    },
                },
            },
        )
    )

    text = out.getvalue()
    assert "all systems nominal" in text
    assert "$0.0142" in text
    assert "23.4s" in text
    assert "in=12,340" in text
    assert "out=2,890" in text
    assert "classifier" in text and "8,432" in text


def test_render_budget_handles_empty_input():
    """Demos call ``render_budget`` directly with whatever they pulled off
    the payload; ``None`` and ``{}`` must be no-ops, not crashes."""
    out = StringIO()
    renderer = ConsoleRenderer(out=out)
    renderer.render_budget(None)
    renderer.render_budget({})
    assert out.getvalue() == ""


def test_render_budget_helper_emits_tokens_and_breakdown():
    out = StringIO()
    renderer = ConsoleRenderer(out=out)
    renderer.render_budget(
        {
            "tokens_in": 1234,
            "tokens_out": 567,
            "breakdown": {
                "classifier": {"tokens_in": 100, "tokens_out": 10},
                "planner": {"tokens_in": 800, "tokens_out": 400},
            },
        }
    )
    text = out.getvalue()
    assert "in=1,234" in text
    assert "out=567" in text
    assert "classifier" in text
    assert "planner" in text


def test_console_renderer_done_event_back_compat_without_budget():
    """Old-shape DONE events without a ``budget`` key still render cost/time
    from the legacy flat fields — no breakdown printed."""
    out = StringIO()
    renderer = ConsoleRenderer(out=out)

    renderer.render(
        BusEvent(
            type=EventType.DONE,
            agent_id="orchestrator",
            payload={
                "answer": "done",
                "confidence": 0.9,
                "cost_usd": 0.005,
                "elapsed_seconds": 1.2,
            },
        )
    )

    text = out.getvalue()
    assert "$0.0050" in text
    assert "1.2s" in text
    assert "Tokens:" not in text


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


def test_console_renderer_spinner_is_tty_only():
    out = StringIO()
    renderer = ConsoleRenderer(out=out, spinner=True, spinner_delay=0)

    renderer.render(
        BusEvent(
            type=EventType.ACTION,
            agent_id="agent",
            payload={"tool": "browser_snapshot", "args": {}},
        )
    )
    time.sleep(0.05)
    renderer.close()

    assert "using browser_snapshot" not in out.getvalue()


def test_console_renderer_spinner_draws_and_clears_before_next_event():
    out = _TTYStringIO()
    renderer = ConsoleRenderer(out=out, spinner=True, spinner_delay=0)

    renderer.render(
        BusEvent(
            type=EventType.ACTION,
            agent_id="agent",
            payload={"tool": "browser_snapshot", "args": {}},
        )
    )
    time.sleep(0.05)
    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": "done"},
        )
    )
    renderer.close()

    text = out.getvalue()
    assert "[agent] using browser_snapshot..." in text
    assert "\r\033[K" in text
    assert "[agent           ] obs" in text


def test_console_renderer_small_observation_stays_single_line():
    out = StringIO()
    renderer = ConsoleRenderer(out=out)

    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": "done"},
        )
    )

    text = out.getvalue()
    assert "[agent           ] obs     done" in text
    assert "lines /" not in text


def test_excerpt_splits_head_and_tail_with_elided_middle():
    view = excerpt("\n".join(f"line {i}" for i in range(1, 11)), head_lines=2, tail_lines=3)

    assert view.head == ["line 1", "line 2"]
    assert view.tail == ["line 8", "line 9", "line 10"]
    assert view.elided == 5
    assert view.total_lines == 10
    assert view.truncated is True


def test_excerpt_keeps_everything_when_text_fits():
    view = excerpt("a\nb\nc", head_lines=2, tail_lines=3)

    assert view.head == ["a", "b", "c"]
    assert view.tail == []
    assert view.elided == 0
    assert view.truncated is False


def test_console_renderer_large_observation_keeps_head_and_tail():
    out = StringIO()
    renderer = ConsoleRenderer(out=out, truncate=20, width=200)
    observation = "\n".join(f"line {idx}" for idx in range(1, 21))

    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": observation},
        )
    )

    text = out.getvalue()
    assert "[7 of 20 lines /" in text
    assert "line 1" in text
    assert "line 2" in text
    assert "… 13 lines elided …" in text
    assert "line 20" in text
    assert "line 10" not in text


def test_console_renderer_short_multiline_observation_shows_every_line():
    out = StringIO()
    renderer = ConsoleRenderer(out=out, truncate=20, width=200)
    observation = "\n".join(f"line {idx}" for idx in range(1, 6))

    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": observation},
        )
    )

    text = out.getvalue()
    assert "[5 lines /" in text
    assert "elided" not in text
    for idx in range(1, 6):
        assert f"line {idx}" in text


def test_console_renderer_clips_wide_lines_to_display_width():
    out = StringIO()
    renderer = ConsoleRenderer(out=out, truncate=20, width=80)
    observation = "prefix-" + ("x" * 4_000) + "-tail"

    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": observation},
        )
    )

    text = out.getvalue()
    assert "prefix-" in text
    assert "-tail" in text
    assert len(observation) not in {len(line) for line in text.splitlines()}
    assert max(len(line) for line in text.splitlines()) <= 80


def test_console_renderer_tty_observation_renders_once_for_completed_event():
    out = _TTYStringIO()
    renderer = ConsoleRenderer(out=out, truncate=20, width=200)
    observation = "\n".join(f"line {idx}" for idx in range(1, 21))

    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": observation},
        )
    )

    text = out.getvalue()
    assert text.count("lines elided") == 1
    assert "\033[A\r\033[2K" not in text


def test_console_renderer_excerpt_can_be_disabled_for_old_truncation():
    out = StringIO()
    renderer = ConsoleRenderer(out=out, truncate=20, excerpt_large_outputs=False)

    renderer.render(
        BusEvent(
            type=EventType.OBSERVATION,
            agent_id="agent",
            payload={"observation": "prefix-" + ("x" * 80) + "-tail"},
        )
    )

    text = out.getvalue()
    assert "prefix-" in text
    assert "-tail" not in text
    assert "lines /" not in text


def test_console_renderer_terminal_events_do_not_restart_spinner():
    out = _TTYStringIO()
    renderer = ConsoleRenderer(out=out, spinner=True, spinner_delay=0)

    renderer.render(
        BusEvent(
            type=EventType.TASK_DONE,
            agent_id="agent",
            payload={"confidence": 1.0, "steps": 1},
        )
    )
    time.sleep(0.05)
    renderer.close()

    assert "thinking..." not in out.getvalue()


def test_console_renderer_subagent_panel_is_tty_only():
    out = StringIO()
    renderer = ConsoleRenderer(out=out, spinner=False)

    renderer.render(
        BusEvent(
            type=EventType.SUBAGENT_START,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"task": "research", "invocation_id": "run-a"},
        )
    )
    renderer.render(
        BusEvent(
            type=EventType.SUBAGENT_START,
            agent_id="analyst",
            parent_agent_id="coordinator",
            payload={"task": "analyze", "invocation_id": "run-b"},
        )
    )

    text = out.getvalue()
    assert "→ start" in text
    assert "Subagents" not in text
    assert "\033[A" not in text


def test_console_renderer_draws_parallel_subagent_panel():
    out = _TTYStringIO()
    renderer = ConsoleRenderer(out=out, spinner=False)

    renderer.render(
        BusEvent(
            type=EventType.SUBAGENT_START,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"task": "research", "invocation_id": "run-a"},
        )
    )
    renderer.render(
        BusEvent(
            type=EventType.SUBAGENT_START,
            agent_id="analyst",
            parent_agent_id="coordinator",
            payload={"task": "analyze", "invocation_id": "run-b"},
        )
    )
    renderer.render(
        BusEvent(
            type=EventType.ACTION,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"tool": "browser_snapshot", "step": 4, "invocation_id": "run-a"},
        )
    )
    renderer.close()

    text = out.getvalue()
    assert "Subagents" in text
    assert "researcher" in text
    assert "analyst" in text
    assert "action browser_snapshot" in text
    assert "step 4" in text
    assert "\033[A\r\033[2K" in text


def test_console_renderer_subagent_panel_handles_duplicate_agent_ids():
    out = _TTYStringIO()
    renderer = ConsoleRenderer(out=out, spinner=False)

    renderer.render(
        BusEvent(
            type=EventType.SUBAGENT_START,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"task": "first", "invocation_id": "run-a"},
        )
    )
    renderer.render(
        BusEvent(
            type=EventType.SUBAGENT_START,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"task": "second", "invocation_id": "run-b"},
        )
    )
    renderer.render(
        BusEvent(
            type=EventType.ACTION,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"tool": "browser_snapshot", "invocation_id": "run-a"},
        )
    )
    renderer.render(
        BusEvent(
            type=EventType.ACTION,
            agent_id="researcher",
            parent_agent_id="coordinator",
            payload={"tool": "http_fetch", "invocation_id": "run-b"},
        )
    )
    renderer.close()

    text = out.getvalue()
    assert "action browser_snapshot" in text
    assert "action http_fetch" in text
