"""Plan-mode contract tests for ``PersistentAgent``.

When ``SessionState.plan_mode_enabled`` is True, ``chat()`` should:

1. Call the LLM for a structured plan.
2. Yield a ``PLAN_PROPOSED`` event so renderers can show it.
3. Route plan approval via ``harness.hitl.request_plan_approval`` (the
   plan-specific sibling of ``request_approval`` — Enter / y / n /
   revision only; no session-allow or persistent-allow because those
   defeat the per-turn-approval intent of plan mode).
4. On approval, run the existing ReAct flow with the plan injected as
   a prior message; on rejection, yield an ERROR and write nothing to
   the session store.
5. Honour up to ``_PLAN_REVISION_LIMIT`` free-text revision cycles
   before giving up cleanly.

The tests stub out the approval prompt by monkey-patching
``request_plan_approval`` so the banner doesn't try to read stdin
during pytest.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.events import EventType
from harness.hitl import PlanApprovalResponse
from harness.persistent import (
    InMemorySessionStore,
    PersistentAgent,
    PersistentAgentConfig,
    SessionState,
)
from harness.plan_mode import (
    _PLAN_REVISION_LIMIT,
    _coerce_plan,
    _dynamic_step_count,
    _render_plan_for_banner,
    _render_plan_for_priors,
    _step_args_deferred,
)
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── Fakes ─────────────────────────────────────────────────────────────────


class _ScriptedLLM:
    """LLM stub that returns the next queued response.

    ``planner_responses`` are returned for ``source="planner"`` calls;
    ``react_responses`` are returned for everything else. This lets a
    test script the planner phase and the ReAct execution phase
    independently.
    """

    input_token_budget = 10_000
    last_usage = None

    def __init__(
        self,
        *,
        planner_responses: list[Any] | None = None,
        react_response: dict | None = None,
    ) -> None:
        self.planner_responses = list(planner_responses or [])
        self.react_response = react_response or {
            "thought": "executing approved plan",
            "action": "finish",
            "answer": "done",
            "confidence": 1.0,
        }
        self.calls: list[dict[str, Any]] = []

    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        if kwargs.get("source") == "planner":
            if not self.planner_responses:
                raise RuntimeError("planner called more times than scripted")
            return self.planner_responses.pop(0)
        return self.react_response


def _make_app(
    *,
    llm: _ScriptedLLM,
    allowed_tools: list[str] | None = None,
) -> PersistentAgent:
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    coordinator = BaseAgent(
        config=AgentConfig(
            agent_id="coordinator",
            role="planner test agent",
            system_prompt="You coordinate.",
            allowed_tools=allowed_tools or [],
            max_steps=2,
            stream_tokens=False,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )
    return PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )


def _stub_plan_approval(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[PlanApprovalResponse],
) -> list[dict[str, Any]]:
    """Replace ``harness.hitl.request_plan_approval`` with a script of
    responses. Returns a list capturing the keyword args each call saw
    so tests can assert against the banner shape (summary, step_count,
    dynamic_step_count, agent_id).
    """
    captured: list[dict[str, Any]] = []
    queue = list(responses)

    async def _fake_request_plan_approval(**kwargs: Any) -> PlanApprovalResponse:
        captured.append(kwargs)
        if not queue:
            raise RuntimeError("request_plan_approval called more times than scripted")
        return queue.pop(0)

    monkeypatch.setattr("harness.hitl.request_plan_approval", _fake_request_plan_approval)
    return captured


# ── Pure unit tests ───────────────────────────────────────────────────────


def test_coerce_plan_accepts_already_structured_dict():
    plan = _coerce_plan({"summary": "x", "steps": [{"step": 1, "intent": "y"}]})
    assert plan == {"summary": "x", "steps": [{"step": 1, "intent": "y"}]}


def test_coerce_plan_parses_text_wrapped_response():
    plan = _coerce_plan({"text": '{"summary": "x", "steps": []}'})
    assert plan == {"summary": "x", "steps": []}


def test_coerce_plan_handles_real_adapter_shape_with_usage_field():
    """Every harness LLM adapter (OpenAI / Anthropic / Claude Code)
    returns ``{"text": <content>, "usage": <dict>}`` — two keys. The
    v0.9.x version checked ``len(raw) == 1`` and wrongly rejected this,
    surfacing as "planner returned a response that could not be parsed
    as a {summary, steps} JSON object" on every real plan call. Pin the
    fix so the next refactor doesn't regress it."""
    raw = {
        "text": '{"summary": "fetch and report", "steps": [{"step": 1, "intent": "x"}]}',
        "usage": {"tokens_in": 412, "tokens_out": 88},
    }
    plan = _coerce_plan(raw)
    assert plan == {
        "summary": "fetch and report",
        "steps": [{"step": 1, "intent": "x"}],
    }


def test_coerce_plan_handles_anthropic_style_fenced_and_prose_wrapped_output():
    """``response_format`` is only honoured by the OpenAI adapter; the
    Anthropic and Claude Code adapters silently drop the kwarg
    (``harness/llm/openai.py:178`` passes it; the other two don't even
    mention it). On those providers the planner LLM is free to wrap
    its JSON in markdown fences or sandwich it in prose — and
    routinely does.

    ``_coerce_plan`` delegates to ``_parse_action_json`` (the same
    helper the ReAct loop uses every turn on Anthropic), which walks
    each ``{`` and lets ``JSONDecoder.raw_decode`` handle bracket
    balancing. Pin every shape we've seen:

    1. Markdown code fence with language tag.
    2. Markdown code fence without language tag.
    3. JSON sandwiched in prose without any fence.
    """
    labelled = '```json\n{"summary": "x", "steps": [{"step": 1, "intent": "y"}]}\n```'
    assert _coerce_plan({"text": labelled, "usage": {}}) == {
        "summary": "x",
        "steps": [{"step": 1, "intent": "y"}],
    }

    bare = '```\n{"summary": "x", "steps": []}\n```'
    assert _coerce_plan({"text": bare, "usage": {}}) == {"summary": "x", "steps": []}

    prose_wrapped = (
        "Sure, here's the plan you asked for:\n\n"
        '{"summary": "x", "steps": [{"step": 1, "intent": "y"}]}\n\n'
        "Let me know if you want changes."
    )
    assert _coerce_plan({"text": prose_wrapped, "usage": {}}) == {
        "summary": "x",
        "steps": [{"step": 1, "intent": "y"}],
    }


def test_coerce_plan_returns_none_on_malformed_input():
    assert _coerce_plan("not json") is None
    assert _coerce_plan({"text": "not json either"}) is None
    assert _coerce_plan({"summary": "no steps"}) is None  # steps missing
    assert _coerce_plan({"summary": "wrong steps", "steps": "should be list"}) is None


def test_coerce_plan_drops_non_dict_steps():
    plan = _coerce_plan(
        {
            "summary": "x",
            "steps": [
                {"step": 1, "intent": "ok"},
                "string is not a step",
                42,
                {"step": 2, "intent": "also ok"},
            ],
        }
    )
    assert plan is not None
    assert len(plan["steps"]) == 2


def test_render_plan_for_banner_includes_summary_and_step_lines():
    plan = {
        "summary": "Search HN and report",
        "steps": [
            {
                "step": 1,
                "intent": "Open HN",
                "tool": "browser_navigate",
                "args": {"url": "https://news.ycombinator.com"},
                "why": "Primary source",
            },
            {"step": 2, "intent": "Summarise", "tool": None, "args": {}, "why": "Wrap up"},
        ],
    }
    rendered = _render_plan_for_banner(plan)
    assert "Search HN and report" in rendered
    assert "1." in rendered and "Open HN" in rendered
    assert "browser_navigate" in rendered
    assert "https://news.ycombinator.com" in rendered
    assert "Primary source" in rendered
    # Step 2 has tool=None — no "tool:" line should appear for it.
    step_2_block = rendered.split("2.")[1]
    assert "tool:" not in step_2_block


def test_render_plan_for_priors_marks_approval_clearly():
    plan = {"summary": "x", "steps": []}
    rendered = _render_plan_for_priors(plan)
    assert rendered.startswith("[Approved plan]")
    # Updated executor wording: must spell out the runtime-resolved
    # convention so the LLM doesn't read placeholders as instructions.
    assert "agreed INTENT" in rendered
    assert "resolved at runtime" in rendered
    assert "observations of the prior step" in rendered


# ── Deferred / runtime-resolved args ──────────────────────────────────────


def test_step_args_deferred_recognises_missing_field_and_null():
    """Convention: missing key OR explicit null = deferred to runtime."""
    assert _step_args_deferred({"step": 1, "intent": "x"}) is True
    assert _step_args_deferred({"step": 1, "intent": "x", "args": None}) is True


def test_step_args_deferred_treats_empty_dict_as_no_args_needed():
    """Empty dict means "tool takes no arguments", NOT deferred — keep
    the two cases distinguishable so we don't lie about the plan."""
    assert _step_args_deferred({"step": 1, "intent": "x", "args": {}}) is False


def test_step_args_deferred_treats_concrete_dict_as_not_deferred():
    assert _step_args_deferred({"step": 1, "intent": "x", "args": {"url": "https://x"}}) is False


def test_dynamic_step_count_only_counts_tool_steps_with_deferred_args():
    plan = {
        "summary": "x",
        "steps": [
            # Concrete: not dynamic.
            {"step": 1, "intent": "fetch", "tool": "http_fetch", "args": {"url": "https://x.com"}},
            # Tool + missing args: dynamic.
            {"step": 2, "intent": "navigate", "tool": "browser_navigate"},
            # Tool + null args: dynamic.
            {"step": 3, "intent": "evaluate", "tool": "browser_evaluate", "args": None},
            # Tool + empty dict: NOT dynamic (tool takes no args).
            {"step": 4, "intent": "snapshot", "tool": "browser_snapshot", "args": {}},
            # No tool: NOT dynamic (conversational step).
            {"step": 5, "intent": "summarise", "tool": None},
        ],
    }
    assert _dynamic_step_count(plan) == 2


def test_render_plan_for_banner_shows_resolved_at_runtime_for_deferred_args():
    plan = {
        "summary": "Find and read the most cited paper on X",
        "steps": [
            # Concrete: user-supplied URL.
            {
                "step": 1,
                "intent": "Search arxiv",
                "tool": "browser_navigate",
                "args": {"url": "https://arxiv.org/search?q=X"},
                "why": "Search the source",
            },
            # Deferred: URL is whatever the search returned. Planner
            # honestly omits args rather than fabricating a URL it can't
            # know.
            {
                "step": 2,
                "intent": "Read the top-cited result",
                "tool": "browser_navigate",
                "args": None,
                "why": "Discover from prior step's observation",
            },
        ],
    }
    rendered = _render_plan_for_banner(plan)

    # Step 1: concrete URL shows verbatim.
    assert "arxiv.org/search?q=X" in rendered
    # Step 2: the deferred sentinel shows instead of a fabricated URL.
    assert "(resolved at runtime)" in rendered
    # And the deferred step does NOT show a fake JSON args blob.
    step_2_block = rendered.split("2.")[1]
    assert "{}" not in step_2_block, (
        "deferred step must render as '(resolved at runtime)', not '{}'"
    )


def test_render_plan_for_banner_shows_empty_dict_when_tool_takes_no_args():
    """A tool that genuinely takes no args (e.g. a snapshot) renders as
    ``args: {}`` — distinguishable from the deferred case."""
    plan = {
        "summary": "x",
        "steps": [
            {"step": 1, "intent": "snap", "tool": "browser_snapshot", "args": {}, "why": "y"},
        ],
    }
    rendered = _render_plan_for_banner(plan)
    assert "args: {}" in rendered
    assert "resolved at runtime" not in rendered


# ── set_plan_mode persistence ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_plan_mode_persists_in_session_state():
    app = _make_app(llm=_ScriptedLLM())

    assert await app.plan_mode_enabled("s") is False
    assert await app.set_plan_mode("s", True) is True
    assert await app.plan_mode_enabled("s") is True
    assert await app.set_plan_mode("s", False) is False
    assert await app.plan_mode_enabled("s") is False


@pytest.mark.asyncio
async def test_sqlite_session_store_round_trips_plan_mode_enabled(tmp_path):
    """The SQLite schema migration must persist ``plan_mode_enabled``
    across process restart (reload from disk into a fresh store instance)."""
    from harness.persistent import SQLiteSessionStore

    path = tmp_path / "plan.sqlite"
    store = SQLiteSessionStore(path)
    await store.set_plan_mode("s", True)

    # New store instance to force re-opening the SQLite file from disk.
    reloaded = SQLiteSessionStore(path)
    state = await reloaded.load("s")
    assert state.plan_mode_enabled is True

    await reloaded.set_plan_mode("s", False)
    state = await reloaded.load("s")
    assert state.plan_mode_enabled is False


@pytest.mark.asyncio
async def test_clear_session_preserves_plan_mode_preference():
    """``clear_session`` is a transcript reset, not a settings reset —
    the user's plan-mode preference must survive."""
    app = _make_app(llm=_ScriptedLLM())
    await app.set_plan_mode("s", True)

    # Drop transcript via session store directly (bypasses chat flow).
    cleared = await app._session_store.clear("s")

    assert isinstance(cleared, SessionState)
    assert cleared.plan_mode_enabled is True, (
        "clear() resets transcript/summary/counters but must preserve the "
        "plan-mode preference; otherwise users would have to /plan on every "
        "time they /clear"
    )


# ── chat() with plan mode OFF: existing path unchanged ────────────────────


@pytest.mark.asyncio
async def test_chat_with_plan_mode_off_does_not_call_planner_or_hitl(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = _ScriptedLLM()  # no planner responses queued
    app = _make_app(llm=llm)
    captured = _stub_plan_approval(monkeypatch, [])  # no HITL calls expected

    async for _ in app.chat("hello", session_id="s"):
        pass

    planner_calls = [c for c in llm.calls if c["kwargs"].get("source") == "planner"]
    assert planner_calls == [], "planner must not run when plan mode is off"
    assert captured == [], "HITL must not be invoked when plan mode is off"


# ── chat() with plan mode ON: full happy path ─────────────────────────────


@pytest.mark.asyncio
async def test_chat_with_plan_mode_on_proposes_then_executes_after_approval(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = {
        "summary": "Greet the user",
        "steps": [
            {"step": 1, "intent": "Say hi", "tool": None, "args": {}, "why": "polite"},
        ],
    }
    llm = _ScriptedLLM(planner_responses=[plan])
    app = _make_app(llm=llm)
    await app.set_plan_mode("s", True)

    captured = _stub_plan_approval(
        monkeypatch,
        [PlanApprovalResponse(approved=True)],
    )

    events = [event async for event in app.chat("say hi", session_id="s")]
    types = [e.type for e in events]

    # PLAN_PROPOSED yielded with the plan in payload, before any THOUGHT.
    plan_idx = types.index(EventType.PLAN_PROPOSED)
    plan_event = events[plan_idx]
    assert plan_event.payload["plan"] == plan
    assert plan_event.payload["revision"] == 0
    assert all(t != EventType.THOUGHT for t in types[:plan_idx]), (
        "plan must be proposed BEFORE the ReAct loop starts thinking"
    )

    # Plan-approval primitive was invoked exactly once with the
    # banner-shaped kwargs.
    assert len(captured) == 1
    call = captured[0]
    assert call["summary"] == "Greet the user"
    assert call["step_count"] == 1
    # No tool on this step, so it's not a "dynamic" step.
    assert call["dynamic_step_count"] == 0
    assert call["agent_id"] == "coordinator"

    # The ReAct loop ran (TASK_DONE present) and the session store was
    # written (turn_count incremented).
    assert EventType.TASK_DONE in types
    state = await app.session_state("s")
    assert state.turn_count == 1
    assert any(m.role == "user" and m.content == "say hi" for m in state.messages)


@pytest.mark.asyncio
async def test_plan_proposed_event_yielded_before_hitl_approval_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """The PLAN_PROPOSED event must reach the renderer BEFORE
    ``request_approval`` blocks on stdin, otherwise the HITL banner
    prints first and the user approves a plan they haven't seen.

    The earlier buffered version of plan mode appended events to a list
    and re-yielded them after ``request_approval`` returned — meaning
    the plan only printed AFTER the user typed y/n. This test pins the
    interleaving so a refactor can't quietly regress it.
    """
    plan = {"summary": "x", "steps": [{"step": 1, "intent": "y"}]}
    llm = _ScriptedLLM(planner_responses=[plan])
    app = _make_app(llm=llm)
    await app.set_plan_mode("s", True)

    # Track the order of events vs. the request_plan_approval call.
    # PLAN_PROPOSED must arrive before plan approval is even invoked —
    # only possible if chat() yields before awaiting.
    timeline: list[str] = []

    async def _spy_approval(**kwargs: Any) -> PlanApprovalResponse:  # noqa: ARG001
        timeline.append("request_plan_approval_called")
        return PlanApprovalResponse(approved=True)

    monkeypatch.setattr("harness.hitl.request_plan_approval", _spy_approval)

    async for event in app.chat("do it", session_id="s"):
        if event.type == EventType.PLAN_PROPOSED:
            timeline.append("plan_proposed_yielded")

    # The renderer (this test loop) saw PLAN_PROPOSED before the
    # approval primitive was even invoked. That ordering is the contract.
    assert timeline.index("plan_proposed_yielded") < timeline.index(
        "request_plan_approval_called"
    ), (
        "PLAN_PROPOSED must yield to the consumer BEFORE "
        "request_plan_approval blocks on stdin — otherwise the user "
        f"approves a plan they haven't seen yet. timeline={timeline!r}"
    )


@pytest.mark.asyncio
async def test_hitl_args_surface_dynamic_step_count_for_partially_concrete_plans(
    monkeypatch: pytest.MonkeyPatch,
):
    """The approval banner ``args`` must include ``dynamic_steps`` so a
    reviewer sees up-front how many step args will be filled in at
    runtime — not all approvals are over fully-concrete plans, and
    pretending they are erodes trust."""
    plan = {
        "summary": "Search and read",
        "steps": [
            {
                "step": 1,
                "intent": "Search",
                "tool": "browser_navigate",
                "args": {"url": "https://example.com/search?q=x"},
            },
            {
                "step": 2,
                "intent": "Open top result",
                "tool": "browser_navigate",
                "args": None,  # deferred
            },
            {
                "step": 3,
                "intent": "Snapshot",
                "tool": "browser_snapshot",
                # no "args" key → also deferred
            },
        ],
    }
    llm = _ScriptedLLM(planner_responses=[plan])
    app = _make_app(llm=llm, allowed_tools=["browser_navigate", "browser_snapshot"])
    await app.set_plan_mode("s", True)
    captured = _stub_plan_approval(
        monkeypatch,
        [PlanApprovalResponse(approved=True)],
    )

    async for _ in app.chat("do it", session_id="s"):
        pass

    assert len(captured) == 1
    call = captured[0]
    assert call["step_count"] == 3
    assert call["dynamic_step_count"] == 2, (
        "two of the three steps have deferred args; the approval banner "
        "must reflect that count so the user knows what they're approving"
    )


# ── chat() with plan mode ON: rejection writes nothing ────────────────────


@pytest.mark.asyncio
async def test_plan_rejection_yields_error_and_does_not_commit_turn(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = {"summary": "x", "steps": []}
    llm = _ScriptedLLM(planner_responses=[plan])
    app = _make_app(llm=llm)
    await app.set_plan_mode("s", True)
    _stub_plan_approval(
        monkeypatch,
        [PlanApprovalResponse(approved=False)],
    )

    events = [event async for event in app.chat("do thing", session_id="s")]
    types = [e.type for e in events]

    assert EventType.PLAN_PROPOSED in types
    assert EventType.ERROR in types
    error = next(e for e in events if e.type == EventType.ERROR)
    assert "rejected" in (error.error or "").lower()
    # ReAct loop did NOT run.
    assert EventType.THOUGHT not in types
    assert EventType.TASK_DONE not in types

    state = await app.session_state("s")
    assert state.messages == [], "rejected plan must not commit the user turn"
    assert state.turn_count == 0


# ── chat() with plan mode ON: correction triggers re-plan ─────────────────


@pytest.mark.asyncio
async def test_plan_correction_triggers_revision_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
):
    plan_v1 = {"summary": "First take", "steps": [{"step": 1, "intent": "do a"}]}
    plan_v2 = {"summary": "Revised take", "steps": [{"step": 1, "intent": "do b"}]}
    llm = _ScriptedLLM(planner_responses=[plan_v1, plan_v2])
    app = _make_app(llm=llm)
    await app.set_plan_mode("s", True)

    _stub_plan_approval(
        monkeypatch,
        [
            # First response: free-text revision request. In the new
            # plan-approval shape this is ``approved=False`` with a
            # non-None correction, distinguishable from a plain
            # rejection (``correction=None``).
            PlanApprovalResponse(approved=False, correction="use b instead"),
            # Second response: approve the revised plan.
            PlanApprovalResponse(approved=True),
        ],
    )

    events = [event async for event in app.chat("do it", session_id="s")]
    proposals = [e for e in events if e.type == EventType.PLAN_PROPOSED]
    assert len(proposals) == 2, "expected revision 0 + revision 1"
    assert proposals[0].payload["revision"] == 0
    assert proposals[1].payload["revision"] == 1

    # The planner was called twice — second call's system prompt
    # contains the user's correction text.
    planner_calls = [c for c in llm.calls if c["kwargs"].get("source") == "planner"]
    assert len(planner_calls) == 2
    assert "use b instead" in planner_calls[1]["system"]

    # The session WAS committed (approved on revision 1).
    state = await app.session_state("s")
    assert state.turn_count == 1


# ── chat() with plan mode ON: revision budget exhausted ───────────────────


@pytest.mark.asyncio
async def test_plan_revision_budget_exhausted_yields_clean_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """If the user keeps correcting, we cap at ``_PLAN_REVISION_LIMIT``
    revisions, then yield ERROR rather than silently approving."""
    plans = [
        {"summary": f"take {i}", "steps": [{"step": 1, "intent": "x"}]}
        for i in range(_PLAN_REVISION_LIMIT + 1)
    ]
    llm = _ScriptedLLM(planner_responses=plans)
    app = _make_app(llm=llm)
    await app.set_plan_mode("s", True)

    _stub_plan_approval(
        monkeypatch,
        [
            PlanApprovalResponse(approved=False, correction=f"try again {i}")
            for i in range(_PLAN_REVISION_LIMIT + 1)
        ],
    )

    events = [event async for event in app.chat("do it", session_id="s")]
    types = [e.type for e in events]

    proposals = [e for e in events if e.type == EventType.PLAN_PROPOSED]
    assert len(proposals) == _PLAN_REVISION_LIMIT + 1
    assert EventType.ERROR in types
    error = next(e for e in events if e.type == EventType.ERROR)
    assert "revision limit" in (error.error or "").lower()

    state = await app.session_state("s")
    assert state.messages == [], "exhausted-revision-budget path must not commit the turn"


# ── chat() with plan mode ON: planner failure surfaces as ERROR ───────────


@pytest.mark.asyncio
async def test_planner_failure_yields_error_without_invoking_hitl(
    monkeypatch: pytest.MonkeyPatch,
):
    """If the planner LLM returns something that can't be coerced to a
    plan, surface ``ERROR`` immediately — don't show the user an empty
    or garbled approval banner."""
    llm = _ScriptedLLM(planner_responses=["garbage non-json"])
    app = _make_app(llm=llm)
    await app.set_plan_mode("s", True)
    captured = _stub_plan_approval(monkeypatch, [])  # HITL must not be called

    events = [event async for event in app.chat("do it", session_id="s")]
    types = [e.type for e in events]

    assert EventType.ERROR in types
    error = next(e for e in events if e.type == EventType.ERROR)
    assert "plan generation failed" in (error.error or "").lower()
    assert captured == [], "HITL must not be invoked when plan generation fails"
