"""
Tests for harness/otel.py — OTEL hook, span creation, and Tracer integration.

Uses the OTEL InMemorySpanExporter so no collector is needed.
"""
from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from harness.otel import OTELHook
from harness.runtime import Tracer


class InMemorySpanExporter(SpanExporter):
    """Minimal in-memory exporter for tests (SDK no longer ships one)."""

    def __init__(self):
        self._spans = []

    def export(self, spans):
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        return list(self._spans)

    def shutdown(self):
        self._spans.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def otel_env():
    """Set up an in-memory OTEL provider and return (hook, exporter)."""
    exporter = InMemorySpanExporter()
    resource = Resource.create({"service.name": "test-harness"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Build hook that uses this provider directly (bypass global set)
    hook = OTELHook.__new__(OTELHook)
    hook._tracer = provider.get_tracer("agent-harness")
    hook._trace_api = trace
    hook._provider = provider
    hook._root_span = None
    hook._root_ctx = None
    hook._task_spans = {}
    hook._task_counter = {}

    yield hook, exporter

    # Cleanup
    provider.shutdown()


def span_names(exporter) -> list[str]:
    """Get names of all finished spans."""
    return [s.name for s in exporter.get_finished_spans()]


def get_span(exporter, name: str):
    """Get the first span with the given name."""
    for s in exporter.get_finished_spans():
        if s.name == name:
            return s
    raise ValueError(f"No span named {name!r}. Found: {span_names(exporter)}")


# ── OTELHook unit tests ──────────────────────────────────────────────────────

class TestOTELHookLifecycle:
    def test_start_and_end_run_creates_root_span(self, otel_env):
        hook, exporter = otel_env

        hook.on_start_run("run-123", "test goal")
        assert hook._root_span is not None

        hook.on_end_run()
        assert hook._root_span is None

        root = get_span(exporter, "run")
        assert root.attributes["run.id"] == "run-123"
        assert root.attributes["run.goal"] == "test goal"

    def test_plan_creates_child_span(self, otel_env):
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        hook.on_event("plan", "orchestrator", {
            "plan": {
                "rationale": "simple plan",
                "tasks": [
                    {"id": "t1", "agent_id": "researcher", "instruction": "do stuff"},
                ],
            },
        })
        hook.on_end_run()

        plan = get_span(exporter, "plan")
        assert plan.attributes["plan.task_count"] == 1
        assert plan.attributes["plan.rationale"] == "simple plan"

    def test_thought_creates_task_span(self, otel_env):
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        hook.on_event("thought", "agent-a", {"step": 0, "thought": "thinking...", "action": "tool"})
        hook.on_event("task_result", "agent-a", {
            "task_id": "t1", "success": True, "confidence": 0.9,
        })
        hook.on_end_run()

        task = get_span(exporter, "task:agent-a")
        assert task.attributes["task.id"] == "t1"
        assert task.attributes["task.success"] is True
        assert task.attributes["task.confidence"] == 0.9

        # Thought should be recorded as an event on the task span
        events = task.events
        assert len(events) == 1
        assert events[0].name == "thought"

    def test_action_recorded_as_event(self, otel_env):
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        hook.on_event("thought", "agent-a", {"step": 0, "thought": "let me fetch"})
        hook.on_event("action", "agent-a", {
            "step": 0, "tool": "http_fetch", "args": {"url": "http://example.com"},
            "observation": "200 OK",
        })
        hook.on_event("task_result", "agent-a", {
            "task_id": "t1", "success": True, "confidence": 1.0,
        })
        hook.on_end_run()

        task = get_span(exporter, "task:agent-a")
        action_events = [e for e in task.events if e.name == "action"]
        assert len(action_events) == 1
        assert action_events[0].attributes["tool"] == "http_fetch"

    def test_synthesis_creates_span(self, otel_env):
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        hook.on_event("synthesis", "orchestrator", {
            "confidence": 0.95, "conflicts": [], "unknowns": [],
        })
        hook.on_end_run()

        synth = get_span(exporter, "synthesis")
        assert synth.attributes["synthesis.confidence"] == 0.95

    def test_replan_recorded_on_root(self, otel_env):
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        hook.on_event("replan", "orchestrator", {
            "replan_count": 1, "trigger_task": "t1", "new_task_count": 2,
        })
        hook.on_end_run()

        root = get_span(exporter, "run")
        replan_events = [e for e in root.events if e.name == "replan"]
        assert len(replan_events) == 1
        assert replan_events[0].attributes["replan_count"] == 1

    def test_multiple_tasks_same_agent(self, otel_env):
        """Sequential tasks for the same agent create separate spans."""
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        # Task 1
        hook.on_event("thought", "agent-a", {"step": 0, "thought": "task 1"})
        hook.on_event("task_result", "agent-a", {
            "task_id": "t1", "success": True, "confidence": 1.0,
        })
        # Task 2 (same agent)
        hook.on_event("thought", "agent-a", {"step": 0, "thought": "task 2"})
        hook.on_event("task_result", "agent-a", {
            "task_id": "t2", "success": True, "confidence": 0.8,
        })
        hook.on_end_run()

        task_spans = [s for s in exporter.get_finished_spans() if s.name == "task:agent-a"]
        assert len(task_spans) == 2
        # Verify they have different task IDs
        task_ids = {s.attributes["task.id"] for s in task_spans}
        assert task_ids == {"t1", "t2"}

    def test_orphan_task_spans_closed_on_end_run(self, otel_env):
        """If a task span isn't closed by task_result, end_run cleans it up."""
        hook, exporter = otel_env

        hook.on_start_run("run-1", "goal")
        hook.on_event("thought", "agent-a", {"step": 0, "thought": "orphan"})
        # No task_result — simulate crash
        hook.on_end_run()

        task = get_span(exporter, "task:agent-a")
        assert task.attributes.get("orphaned") is True


# ── Tracer + OTELHook integration ─────────────────────────────────────────────

class TestTracerHookIntegration:
    def test_tracer_hook_receives_events(self, otel_env):
        hook, exporter = otel_env

        tracer = Tracer()
        tracer.add_hook(hook)

        tracer.start_run("run-99", "integration test")
        tracer.log("plan", "orchestrator", {
            "plan": {"rationale": "test", "tasks": []},
        })
        tracer.log("thought", "agent-x", {"step": 0, "thought": "hi"})
        tracer.log("task_result", "agent-x", {
            "task_id": "t1", "success": True, "confidence": 1.0,
        })
        tracer.log("synthesis", "orchestrator", {"confidence": 0.9})
        tracer.end_run()

        names = span_names(exporter)
        assert "run" in names
        assert "plan" in names
        assert "task:agent-x" in names
        assert "synthesis" in names

    def test_tracer_in_memory_still_works_with_hook(self, otel_env):
        """Adding a hook doesn't break the in-memory trace."""
        hook, _ = otel_env

        tracer = Tracer()
        tracer.add_hook(hook)

        tracer.start_run("run-1", "test")
        tracer.log("thought", "agent-a", {"step": 0, "thought": "hello"})
        tracer.end_run()

        dump = tracer.dump()
        assert len(dump) == 1
        assert dump[0]["event_type"] == "thought"

    def test_tracer_without_hooks_unchanged(self):
        """Tracer with no hooks behaves exactly as before."""
        tracer = Tracer()
        tracer.start_run("run-1", "test")  # no-op
        tracer.log("thought", "agent-a", {"step": 0})
        tracer.end_run()  # no-op

        assert len(tracer.dump()) == 1
