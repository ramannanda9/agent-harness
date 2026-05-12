"""
harness/otel.py — OpenTelemetry hook for the harness Tracer.

Translates internal trace events into proper OTEL spans so you can
visualise agent runs in Jaeger, Datadog, Honeycomb, etc.

Architecture:
    Tracer.log()  →  OTELHook.on_event()  →  OTEL spans/events
    Tracer keeps its in-memory list unchanged; OTEL is a side-channel.

Span hierarchy:
    [run]  goal="..."
      ├── [plan]
      ├── [task] agent_id=researcher, task_id=t1
      │     ├── thought (event)
      │     ├── action  (event)
      │     └── thought (event)
      ├── [task] agent_id=researcher, task_id=t2
      │     └── thought (event)
      ├── [replan]  (event on root)
      └── [synthesis]

Install:
    pip install -e ".[otel]"

Jaeger (local):
    docker run -d --name jaeger -p 16686:16686 -p 4318:4318 \\
        jaegertracing/all-in-one:latest
    # UI at http://localhost:16686

Configuration (standard OTEL env vars):
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   (default)
    OTEL_SERVICE_NAME=agent-harness                      (default)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OTELHook:
    """
    Tracer hook that emits OpenTelemetry spans.

    Attach to a Tracer via ``tracer.add_hook(OTELHook())``.
    The hook maintains a span tree: root → plan/task/synthesis spans,
    with thoughts and actions as span events.
    """

    def __init__(
        self,
        *,
        service_name: str = "agent-harness",
        endpoint: str | None = None,
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
        except ImportError as e:
            raise ImportError(
                'opentelemetry packages not installed. Run: pip install -e ".[otel]"'
            ) from e

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Try OTLP exporter; fall back to console if not available
        exporter = self._build_exporter(endpoint)
        if exporter is None:
            logger.info("OTLP exporter not available, falling back to console")
            exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        self._tracer = trace.get_tracer("agent-harness")
        self._trace_api = trace
        self._provider = provider

        # Span state
        self._root_span: Any = None
        self._root_ctx: Any = None
        self._task_spans: dict[str, Any] = {}  # agent_id → active span
        self._task_counter: dict[str, int] = {}  # agent_id → count

    @staticmethod
    def _build_exporter(endpoint: str | None) -> Any:
        """Try to create an OTLP HTTP exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            kwargs: dict[str, Any] = {}
            if endpoint:
                kwargs["endpoint"] = endpoint
            return OTLPSpanExporter(**kwargs)
        except ImportError:
            return None

    # ── Lifecycle hooks (called by Tracer) ────────────────────────────────

    def on_start_run(self, run_id: str, goal: str) -> None:
        """Create the root span for the entire run."""
        self._root_span = self._tracer.start_span(
            "run",
            attributes={
                "run.id": run_id,
                "run.goal": goal[:1000],
            },
        )
        self._root_ctx = self._trace_api.set_span_in_context(self._root_span)
        self._task_spans.clear()
        self._task_counter.clear()
        logger.debug("OTEL: started root span for run %s", run_id)

    def on_end_run(self) -> None:
        """End the root span and flush pending exports."""
        # Close any orphan task spans
        for _agent_id, span in list(self._task_spans.items()):
            span.set_attribute("orphaned", True)
            span.end()
        self._task_spans.clear()

        if self._root_span:
            self._root_span.end()
            self._root_span = None
            self._root_ctx = None

        self._provider.force_flush()
        logger.debug("OTEL: ended root span")

    def on_event(self, event_type: str, agent_id: str, payload: Any) -> None:
        """Map a Tracer event to OTEL spans/events."""
        handler = self._handlers.get(event_type)
        if handler:
            handler(self, agent_id, payload)

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_plan(self, agent_id: str, payload: Any) -> None:
        plan_data = payload.get("plan", {})
        tasks = plan_data.get("tasks", [])
        span = self._tracer.start_span(
            "plan",
            context=self._root_ctx,
            attributes={
                "plan.task_count": len(tasks),
                "plan.rationale": plan_data.get("rationale", "")[:500],
            },
        )
        # Add each task as an event on the plan span
        for t in tasks:
            span.add_event(
                f"task:{t.get('id', '?')}",
                attributes={
                    "agent_id": t.get("agent_id", ""),
                    "instruction": t.get("instruction", "")[:500],
                },
            )
        span.end()

    def _on_thought(self, agent_id: str, payload: Any) -> None:
        step = payload.get("step", 0)
        # New task starts at step 0 — create a task span
        if step == 0 or agent_id not in self._task_spans:
            # Close previous task span for this agent if still open
            old_span = self._task_spans.pop(agent_id, None)
            if old_span:
                old_span.end()

            count = self._task_counter.get(agent_id, 0) + 1
            self._task_counter[agent_id] = count

            self._task_spans[agent_id] = self._tracer.start_span(
                f"task:{agent_id}",
                context=self._root_ctx,
                attributes={
                    "agent.id": agent_id,
                    "task.sequence": count,
                },
            )

        span = self._task_spans.get(agent_id)
        if span:
            span.add_event(
                "thought",
                attributes={
                    "step": step,
                    "thought": str(payload.get("thought", ""))[:500],
                    "action": str(payload.get("action", "")),
                },
            )

    def _on_action(self, agent_id: str, payload: Any) -> None:
        span = self._task_spans.get(agent_id)
        if not span:
            return
        span.add_event(
            "action",
            attributes={
                "step": payload.get("step", 0),
                "tool": payload.get("tool", ""),
                "args": _safe_json(payload.get("args", {}), max_len=500),
                "observation": str(payload.get("observation", ""))[:500],
            },
        )

    def _on_task_result(self, agent_id: str, payload: Any) -> None:
        span = self._task_spans.pop(agent_id, None)
        if not span:
            return
        task_id = payload.get("task_id", "")
        success = payload.get("success", False)
        confidence = payload.get("confidence", 0.0)

        span.set_attribute("task.id", task_id)
        span.set_attribute("task.success", success)
        span.set_attribute("task.confidence", confidence)

        if not success:
            from opentelemetry.trace import StatusCode

            span.set_status(StatusCode.ERROR, "task failed")

        span.end()

    def _on_dispatch(self, agent_id: str, payload: Any) -> None:
        if self._root_span:
            self._root_span.add_event(
                "dispatch",
                attributes={
                    "complexity": str(payload.get("complexity", "")),
                    "path": str(payload.get("path", "")),
                },
            )

    def _on_route(self, agent_id: str, payload: Any) -> None:
        if self._root_span:
            self._root_span.add_event(
                "route",
                attributes={
                    "agent_id": str(payload.get("agent_id", "")),
                    "rationale": str(payload.get("rationale", ""))[:500],
                },
            )

    def _on_replan(self, agent_id: str, payload: Any) -> None:
        if self._root_span:
            self._root_span.add_event(
                "replan",
                attributes={
                    "replan_count": payload.get("replan_count", 0),
                    "trigger_task": payload.get("trigger_task", ""),
                    "new_task_count": payload.get("new_task_count", 0),
                },
            )

    def _on_synthesis(self, agent_id: str, payload: Any) -> None:
        span = self._tracer.start_span(
            "synthesis",
            context=self._root_ctx,
            attributes={
                "synthesis.confidence": payload.get("confidence", 0.0) or 0.0,
                "synthesis.conflict_count": len(payload.get("conflicts", [])),
                "synthesis.unknown_count": len(payload.get("unknowns", [])),
            },
        )
        span.end()

    # Handler dispatch table
    _handlers: dict[str, Any] = {
        "dispatch": _on_dispatch,
        "route": _on_route,
        "plan": _on_plan,
        "thought": _on_thought,
        "action": _on_action,
        "task_result": _on_task_result,
        "replan": _on_replan,
        "synthesis": _on_synthesis,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_json(obj: Any, max_len: int = 500) -> str:
    """JSON-encode, truncate, and never raise."""
    try:
        text = json.dumps(obj, default=str)
    except Exception:
        text = str(obj)
    return text[:max_len]
