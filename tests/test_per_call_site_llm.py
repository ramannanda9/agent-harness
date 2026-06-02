"""Per-call-site LLM injection on ``AgentRuntime`` and ``Orchestrator``.

Each LLM slot (classifier / router / planner / synthesizer) defaults to
``llm`` when unset, but when supplied the corresponding call site uses it
instead. This is the production-style pattern: cost-shape by wiring the
right model at each call site, not by guessing at runtime.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import AgentConfig
from harness.events import BusEvent, EventType
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore


class _RecordingLLM:
    """LLM stub that remembers every system-prompt prefix it was asked about.

    The orchestrator's prompt templates start with stable identifying
    phrases ("task complexity classifier", "routing agent", "planning
    agent", "synthesis agent") which lets the test confirm which slot
    each call landed in.

    ``per_call_tokens`` controls how many (in, out) tokens each call
    reports to the budget guard so per-call-site breakdown tests can
    assert non-zero attribution.
    """

    def __init__(
        self,
        *,
        name: str,
        answer: dict,
        per_call_tokens: tuple[int, int] = (0, 0),
    ) -> None:
        self.name = name
        self._answer = answer
        self._per_call_tokens = per_call_tokens
        self.systems: list[str] = []
        self.sources: list[str | None] = []
        self.budget: Any = None

    def set_budget(self, guard: Any) -> None:
        self.budget = guard

    async def complete(self, system, messages, **kwargs) -> dict:
        self.systems.append(system or "")
        source = kwargs.get("source")
        self.sources.append(source)
        tokens_in, tokens_out = self._per_call_tokens
        if self.budget is not None and (tokens_in or tokens_out):
            self.budget.add_tokens(tokens_in, tokens_out, source=source)
        import json

        return {
            "text": json.dumps(self._answer),
            "usage": {"tokens_in": tokens_in, "tokens_out": tokens_out},
        }


def _build_two_agent_runtime(
    *,
    llm: Any,
    classifier_llm: Any | None = None,
    router_llm: Any | None = None,
    planner_llm: Any | None = None,
    synthesizer_llm: Any | None = None,
    checkpoint_store: Any | None = None,
) -> AgentRuntime:
    agents = (
        AgentRegistry()
        .register(
            AgentConfig(
                agent_id="alpha",
                role="does alpha work",
                system_prompt="you are alpha",
                allowed_tools=[],
            )
        )
        .register(
            AgentConfig(
                agent_id="beta",
                role="does beta work",
                system_prompt="you are beta",
                allowed_tools=[],
            )
        )
    )
    return AgentRuntime(
        agent_registry=agents,
        tool_registry=ToolRegistry(),
        memory=MemoryManager(
            semantic_store=InMemorySemanticStore(),
            episodic_store=InMemoryEpisodicStore(),
            llm=llm,
        ),
        llm=llm,
        classifier_llm=classifier_llm,
        router_llm=router_llm,
        planner_llm=planner_llm,
        synthesizer_llm=synthesizer_llm,
        guardrail_config=GuardrailConfig(),
        checkpoint_store=checkpoint_store,
    )


class _MemoryCheckpointStore:
    def __init__(self, data: dict[str, dict]) -> None:
        self._data = data

    async def read(self, key: str) -> dict | None:
        return self._data.get(key)

    async def write(self, key: str, data: dict) -> None:
        self._data[key] = data

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


# ── Defaults: slots fall back to main llm ────────────────────────────────────


def test_unset_slots_fall_back_to_main_llm():
    main = _RecordingLLM(name="main", answer={"complexity": "simple"})
    runtime = _build_two_agent_runtime(llm=main)
    # When the user supplies nothing extra, every slot is the same object.
    assert runtime._classifier_llm is main
    assert runtime._router_llm is main
    assert runtime._planner_llm is main
    assert runtime._synthesizer_llm is main


def test_explicit_slots_override_main_llm():
    main = _RecordingLLM(name="main", answer={})
    cheap = _RecordingLLM(name="cheap", answer={})
    plan = _RecordingLLM(name="plan", answer={})
    synth = _RecordingLLM(name="synth", answer={})
    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=cheap,
        router_llm=cheap,
        planner_llm=plan,
        synthesizer_llm=synth,
    )
    assert runtime._classifier_llm is cheap
    assert runtime._router_llm is cheap
    assert runtime._planner_llm is plan
    assert runtime._synthesizer_llm is synth


# ── Budget guard propagates to every distinct LLM ────────────────────────────


def test_attach_budget_reaches_every_distinct_llm_instance():
    """All four slots — even when distinct — receive the per-run guard.

    De-duped by object identity, so when slots share the same wrapper it
    isn't called multiple times.
    """
    main = _RecordingLLM(name="main", answer={})
    cheap = _RecordingLLM(name="cheap", answer={})
    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=cheap,
        router_llm=cheap,  # intentionally same as classifier_llm
        # planner / synth default to main → also de-duped
    )
    guard = object()
    runtime._attach_budget(guard)
    assert main.budget is guard
    assert cheap.budget is guard


def test_attach_budget_is_idempotent_per_instance():
    """An LLM injected into multiple slots receives set_budget exactly once
    per run (set_budget is allowed to be called more than once in practice,
    but doing so unnecessarily is wasted work)."""
    shared = _RecordingLLM(name="shared", answer={})
    runtime = _build_two_agent_runtime(
        llm=shared,
        classifier_llm=shared,
        router_llm=shared,
        planner_llm=shared,
        synthesizer_llm=shared,
    )
    # _budget_targets must contain the shared instance exactly once.
    assert sum(1 for t in runtime._budget_targets if t is shared) == 1


@pytest.mark.asyncio
async def test_resume_stream_agent_checkpoint_attaches_budget_to_all_slots(monkeypatch):
    """The streaming resume path should use the same budget wiring as normal runs."""

    class FakeBaseAgent:
        def __init__(self, **kwargs) -> None:
            self.config = kwargs["config"]

        async def _resume_stream(self, **_kwargs):
            yield BusEvent(
                type=EventType.TASK_DONE,
                agent_id=self.config.agent_id,
                payload={"answer": "ok", "confidence": 1.0},
            )

    import agents.base as base_module

    monkeypatch.setattr(base_module, "BaseAgent", FakeBaseAgent)

    main = _RecordingLLM(name="main", answer={})
    cheap = _RecordingLLM(name="cheap", answer={})
    checkpoint_store = _MemoryCheckpointStore(
        {
            "run-1:alpha": {
                "run_id": "run-1",
                "agent_id": "alpha",
                "task": "resume me",
                "step": 0,
                "memory": {
                    "messages": [],
                    "summarization_count": 0,
                    "max_tokens": 8000,
                    "summarize_ratio": 0.5,
                    "recency_window": 4,
                },
            }
        }
    )
    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=cheap,
        router_llm=cheap,
        checkpoint_store=checkpoint_store,
    )

    events = [event async for event in runtime.resume_stream("run-1:alpha")]

    assert events[-1].type == EventType.TASK_DONE
    assert main.budget is not None
    assert cheap.budget is main.budget


# ── End-to-end: classifier slot actually answers dispatch ────────────────────


@pytest.mark.asyncio
async def test_dispatch_classifier_call_routes_to_classifier_llm():
    """When classifier_llm is set, the dispatch classifier prompt lands
    on it — not on the main llm."""
    main = _RecordingLLM(name="main", answer={"complexity": "simple"})
    cheap = _RecordingLLM(name="cheap", answer={"complexity": "simple"})

    runtime = _build_two_agent_runtime(llm=main, classifier_llm=cheap)
    # Drain the stream — we only care about who got asked, not the result.
    async for _event in runtime.dispatch_stream("do something"):
        pass

    assert any("task complexity classifier" in s for s in cheap.systems), (
        f"classifier_llm never saw the classifier prompt; got: {cheap.systems!r}"
    )
    assert not any("task complexity classifier" in s for s in main.systems), (
        "main llm should not have seen the classifier prompt"
    )


@pytest.mark.asyncio
async def test_router_call_routes_to_router_llm():
    """When router_llm is set, the single-agent router lands on it."""
    main = _RecordingLLM(
        name="main",
        answer={"thought": "done", "action": {"tool": "finish", "args": {"answer": "ok"}}},
    )
    cheap = _RecordingLLM(name="cheap", answer={"agent_id": "alpha", "rationale": "test"})

    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=cheap,  # also route classifier to cheap so dispatch picks simple
        router_llm=cheap,
    )

    # Force a "simple" classification so the router fires.
    cheap._answer = {"complexity": "simple"}

    # We need separate fixed answers for classifier vs router; the recording
    # stub only holds one answer dict. Wire two stubs instead.
    classifier = _RecordingLLM(name="classifier", answer={"complexity": "simple"})
    router = _RecordingLLM(
        name="router", answer={"agent_id": "alpha", "rationale": "alpha is best"}
    )
    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=classifier,
        router_llm=router,
    )

    async for _event in runtime.dispatch_stream("do something simple"):
        pass

    assert any("routing agent" in s for s in router.systems), (
        f"router_llm never saw the router prompt; got: {router.systems!r}"
    )
    assert not any("routing agent" in s for s in main.systems)
    assert not any("routing agent" in s for s in classifier.systems)


# ── Source kwarg + breakdown attribution ─────────────────────────────────────


@pytest.mark.asyncio
async def test_classifier_call_passes_source_kwarg():
    """The classifier call site tags itself ``source="classifier"`` so the
    budget guard's breakdown shows per-call-site spending."""
    cheap = _RecordingLLM(name="cheap", answer={"complexity": "simple"})
    main = _RecordingLLM(
        name="main",
        answer={"thought": "done", "action": {"tool": "finish", "args": {"answer": "ok"}}},
    )
    runtime = _build_two_agent_runtime(llm=main, classifier_llm=cheap)

    async for _event in runtime.dispatch_stream("hello"):
        pass

    assert "classifier" in cheap.sources, (
        f"classifier_llm did not receive source='classifier'; got: {cheap.sources!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_populates_budget_breakdown_with_classifier_and_router():
    """End-to-end: a dispatch with distinct classifier/router LLMs that report
    tokens should land in the BudgetGuard's per-source breakdown under the
    right keys."""
    classifier = _RecordingLLM(
        name="classifier",
        answer={"complexity": "simple"},
        per_call_tokens=(100, 10),
    )
    router = _RecordingLLM(
        name="router",
        answer={"agent_id": "alpha", "rationale": "alpha is best"},
        per_call_tokens=(200, 20),
    )
    main = _RecordingLLM(
        name="main",
        answer={"thought": "done", "action": {"tool": "finish", "args": {"answer": "ok"}}},
        per_call_tokens=(500, 100),  # ReAct call(s); untagged
    )
    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=classifier,
        router_llm=router,
    )

    async for _event in runtime.dispatch_stream("simple task"):
        pass

    # The guard is created internally per-run; we read it back via the LLM
    # that received set_budget (any of them — they all share the same guard).
    guard = classifier.budget
    assert guard is not None
    breakdown = guard.breakdown
    assert breakdown["classifier"]["tokens_in"] == 100
    assert breakdown["classifier"]["tokens_out"] == 10
    assert breakdown["router"]["tokens_in"] == 200
    assert breakdown["router"]["tokens_out"] == 20
    # BaseAgent tags its ReAct calls with `agent:<agent_id>`, so the
    # main LLM's spending lands under that key.
    assert "agent:alpha" in breakdown, (
        f"ReAct should attribute under 'agent:alpha'; got: {list(breakdown)!r}"
    )
    assert guard.tokens_in >= 100 + 200 + 500  # classifier + router + ≥1 ReAct call


@pytest.mark.asyncio
async def test_terminal_event_payload_exposes_budget_snapshot():
    """``dispatch_stream`` consumers should be able to read the budget
    snapshot off the terminal event payload — no need to hold a reference
    to the guard themselves."""
    classifier = _RecordingLLM(
        name="classifier",
        answer={"complexity": "simple"},
        per_call_tokens=(100, 10),
    )
    router = _RecordingLLM(
        name="router",
        answer={"agent_id": "alpha", "rationale": "alpha is best"},
        per_call_tokens=(200, 20),
    )
    # BaseAgent's finish format: {"thought", "action": "finish", "answer", "confidence"}
    main = _RecordingLLM(
        name="main",
        answer={"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0},
    )
    runtime = _build_two_agent_runtime(
        llm=main,
        classifier_llm=classifier,
        router_llm=router,
    )

    terminal = None
    async for event in runtime.dispatch_stream("simple task"):
        if event.type in (EventType.TASK_DONE, EventType.DONE):
            terminal = event

    assert terminal is not None
    budget = terminal.payload.get("budget")
    assert budget is not None, (
        "terminal event must expose `budget` snapshot for streaming consumers"
    )
    assert budget["breakdown"]["classifier"]["tokens_in"] == 100
    assert budget["breakdown"]["router"]["tokens_in"] == 200
    assert budget["tokens_in"] >= 300  # classifier + router (+ any untagged ReAct)
    assert "cost_usd" in budget
    assert "elapsed_seconds" in budget
