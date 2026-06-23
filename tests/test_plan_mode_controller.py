from __future__ import annotations

from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import BusEvent, EventType
from harness.hitl import PlanApprovalResponse
from harness.plan_mode import _PLAN_REVISION_LIMIT, PlanDecision, PlanModeController
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore


class _PlannerLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise RuntimeError("planner called more times than scripted")
        return self.responses.pop(0)


def _coordinator(llm: Any) -> BaseAgent:
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    return BaseAgent(
        config=AgentConfig(
            agent_id="coordinator",
            role="planner",
            system_prompt="Plan.",
            allowed_tools=["browser_snapshot"],
            max_steps=2,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )


def _controller(llm: _PlannerLLM) -> PlanModeController:
    return PlanModeController(
        coordinator=_coordinator(llm),
        llm=lambda: llm,
        tool_names=lambda: ["browser_snapshot"],
    )


def _stub_approval(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[PlanApprovalResponse],
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    queue = list(responses)

    async def _fake_request_plan_approval(**kwargs: Any) -> PlanApprovalResponse:
        captured.append(kwargs)
        if not queue:
            raise RuntimeError("request_plan_approval called more times than scripted")
        return queue.pop(0)

    monkeypatch.setattr("harness.hitl.request_plan_approval", _fake_request_plan_approval)
    return captured


async def _collect(controller: PlanModeController) -> list[BusEvent | PlanDecision]:
    return [
        item
        async for item in controller.run(
            message="do it",
            session_id="s",
            guard=None,
        )
    ]


@pytest.mark.asyncio
async def test_plan_mode_controller_approval_yields_proposal_then_decision(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = {
        "summary": "Snapshot page",
        "steps": [{"step": 1, "intent": "Snapshot", "tool": "browser_snapshot"}],
    }
    llm = _PlannerLLM([plan])
    controller = _controller(llm)
    captured = _stub_approval(monkeypatch, [PlanApprovalResponse(approved=True)])

    items = await _collect(controller)

    assert isinstance(items[0], BusEvent)
    assert items[0].type == EventType.PLAN_PROPOSED
    assert items[0].payload["plan"] == plan
    assert isinstance(items[-1], PlanDecision)
    assert items[-1].approved_plan == plan
    assert items[-1].rejected is False
    assert captured[0]["dynamic_step_count"] == 1


@pytest.mark.asyncio
async def test_plan_mode_controller_rejection_yields_error_then_rejected_decision(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = _PlannerLLM([{"summary": "x", "steps": []}])
    controller = _controller(llm)
    _stub_approval(monkeypatch, [PlanApprovalResponse(approved=False)])

    items = await _collect(controller)

    assert [item.type for item in items if isinstance(item, BusEvent)] == [
        EventType.PLAN_PROPOSED,
        EventType.ERROR,
    ]
    decision = items[-1]
    assert isinstance(decision, PlanDecision)
    assert decision.rejected is True


@pytest.mark.asyncio
async def test_plan_mode_controller_replans_with_correction(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = _PlannerLLM(
        [
            {"summary": "first", "steps": [{"step": 1, "intent": "a"}]},
            {"summary": "second", "steps": [{"step": 1, "intent": "b"}]},
        ]
    )
    controller = _controller(llm)
    _stub_approval(
        monkeypatch,
        [
            PlanApprovalResponse(approved=False, correction="use b"),
            PlanApprovalResponse(approved=True),
        ],
    )

    items = await _collect(controller)
    proposals = [item for item in items if isinstance(item, BusEvent)]

    assert [event.payload["revision"] for event in proposals] == [0, 1]
    assert isinstance(items[-1], PlanDecision)
    assert items[-1].approved_plan is not None
    assert items[-1].approved_plan["summary"] == "second"
    assert "use b" in llm.calls[1]["system"]


@pytest.mark.asyncio
async def test_plan_mode_controller_revision_limit_yields_rejected_decision(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = _PlannerLLM(
        [
            {"summary": f"take {idx}", "steps": [{"step": 1, "intent": "x"}]}
            for idx in range(_PLAN_REVISION_LIMIT + 1)
        ]
    )
    controller = _controller(llm)
    _stub_approval(
        monkeypatch,
        [
            PlanApprovalResponse(approved=False, correction=f"again {idx}")
            for idx in range(_PLAN_REVISION_LIMIT + 1)
        ],
    )

    items = await _collect(controller)
    proposals = [
        item
        for item in items
        if isinstance(item, BusEvent) and item.type == EventType.PLAN_PROPOSED
    ]
    errors = [item for item in items if isinstance(item, BusEvent) and item.type == EventType.ERROR]

    assert len(proposals) == _PLAN_REVISION_LIMIT + 1
    assert "revision limit" in errors[-1].error
    assert isinstance(items[-1], PlanDecision)
    assert items[-1].rejected is True
