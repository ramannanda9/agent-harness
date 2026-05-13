"""
harness/annotation.py — trajectory capture and RLHF annotation store.

Every agent run logs its full WorkingMemory message history (system prompt,
thoughts, tool calls, observations, final answer) as a "trajectory" tracer
event. An AnnotationHook assembles goal + trajectory + result into an
Annotation and writes it to an InMemoryAnnotationStore.

RLHF workflow:
  1. Wire up: runtime = AgentRuntime(..., annotation_store=InMemoryAnnotationStore())
  2. Run agents normally — Annotations accumulate automatically.
  3. Drain: store.list_unrated() → present trajectories to human raters.
  4. Attach signal: store.rate(annotation_id, rating=0.9, correction=None)
  5. Export: store.list_all() → training pipeline.

The messages list is the raw RLHF training signal: it contains the exact
token stream the model saw (system prompt, user turns, assistant turns) plus
the final answer. Summarization compresses older turns when the token budget
is exceeded — summarization_count records how many times that happened.

Annotation.rating: float [0.0, 1.0] — 1.0 = ideal answer, 0.0 = wrong/harmful.
Annotation.correction: str | None — human-supplied correct answer when rating < 1.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Annotation:
    annotation_id: str
    run_id: str
    agent_id: str
    goal: str
    messages: list[dict]  # WorkingMemory.get_messages() — the full trajectory
    answer: str  # agent's final answer ("" on failure)
    confidence: float  # agent's self-reported confidence
    steps: int  # ReAct steps taken
    error: str  # "" on success; failure reason otherwise
    summarization_count: int  # number of WorkingMemory compression passes
    timestamp: str  # ISO 8601 UTC
    rating: float | None = None  # human rating added post-hoc
    correction: str | None = None  # human correction added post-hoc


# ── Store ─────────────────────────────────────────────────────────────────────


class InMemoryAnnotationStore:
    """
    In-process annotation store. Swap for a persistent backend (SQLite,
    Postgres, etc.) by implementing the same interface.

    All methods are synchronous — write() is called from the synchronous
    on_end_run() hook. Async wrappers are trivial if needed:
        async def awrite(self, a): return self.write(a)
    """

    def __init__(self) -> None:
        self._store: dict[str, Annotation] = {}

    def write(self, annotation: Annotation) -> str:
        """Persist an annotation. Returns annotation_id."""
        self._store[annotation.annotation_id] = annotation
        return annotation.annotation_id

    def get(self, annotation_id: str) -> Annotation | None:
        return self._store.get(annotation_id)

    def list_all(self) -> list[Annotation]:
        return list(self._store.values())

    def list_run(self, run_id: str) -> list[Annotation]:
        return [a for a in self._store.values() if a.run_id == run_id]

    def list_unrated(self) -> list[Annotation]:
        """Return annotations that have not yet received a human rating."""
        return [a for a in self._store.values() if a.rating is None]

    def rate(
        self,
        annotation_id: str,
        rating: float,
        correction: str | None = None,
    ) -> None:
        """
        Attach human feedback to an annotation.

        rating:     [0.0, 1.0] — 1.0 = ideal, 0.0 = wrong / harmful.
        correction: human-supplied correct answer when rating < 1.0.
        """
        annotation = self._store.get(annotation_id)
        if annotation is None:
            raise KeyError(f"annotation {annotation_id!r} not found")
        annotation.rating = rating
        annotation.correction = correction

    def count(self) -> int:
        return len(self._store)


# ── Hook ──────────────────────────────────────────────────────────────────────


class AnnotationHook:
    """
    Tracer hook that assembles Annotations from run events.

    Attach via ``tracer.add_hook(AnnotationHook(store))``, or pass
    ``annotation_store=`` to AgentRuntime and it wires this automatically.

    Event contract (produced by BaseAgent):
      "trajectory"  — {run_id, messages, summarization_count}
      "task_result" — {answer, confidence, steps, error}

    Both events fire before tracer.end_run(), so on_end_run() can write
    a complete Annotation with no gaps.
    """

    def __init__(self, store: InMemoryAnnotationStore) -> None:
        self._store = store
        self._run_id: str = ""
        self._goal: str = ""
        self._trajectories: dict[str, dict[str, Any]] = {}
        self._results: dict[str, dict[str, Any]] = {}

    def on_start_run(self, run_id: str, goal: str) -> None:
        self._run_id = run_id
        self._goal = goal
        self._trajectories.clear()
        self._results.clear()

    def on_event(self, event_type: str, agent_id: str, payload: Any) -> None:
        if event_type == "trajectory":
            self._trajectories[agent_id] = {
                "messages": payload.get("messages", []),
                "summarization_count": payload.get("summarization_count", 0),
            }
        elif event_type == "task_result":
            self._results[agent_id] = payload

    def on_end_run(self) -> None:
        """Write one Annotation per agent that produced a trajectory."""
        for agent_id, traj in self._trajectories.items():
            result = self._results.get(agent_id, {})
            annotation = Annotation(
                annotation_id=str(uuid.uuid4()),
                run_id=self._run_id,
                agent_id=agent_id,
                goal=self._goal,
                messages=traj.get("messages", []),
                answer=result.get("answer", ""),
                confidence=result.get("confidence", 0.0),
                steps=result.get("steps", 0),
                error=result.get("error", ""),
                summarization_count=traj.get("summarization_count", 0),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._store.write(annotation)
