"""Pin the typed BusEvent factory contract.

Each factory must emit the exact payload dict the prior hand-authored
construction produced, and route agent_id / parent_agent_id / token / error
to the right BusEvent fields. These are the regression guard for the
producer-side typed-payload migration.
"""

from __future__ import annotations

from harness.events import BusEvent, EventType


def test_dispatch():
    e = BusEvent.dispatch("orchestrator", complexity="simple", path="routed")
    assert e.type == EventType.DISPATCH
    assert e.agent_id == "orchestrator"
    assert e.payload == {"complexity": "simple", "path": "routed"}


def test_route_agent_id_in_both_field_and_payload():
    e = BusEvent.route("researcher", rationale="best fit")
    assert e.type == EventType.ROUTE
    assert e.agent_id == "researcher"
    assert e.payload == {"agent_id": "researcher", "rationale": "best fit"}


def test_plan_initial_has_no_resumed_or_prebuilt_keys():
    plan = {"tasks": [{"id": "t1"}]}
    e = BusEvent.plan("orchestrator", plan=plan)
    assert e.type == EventType.PLAN
    assert e.payload == {"plan": plan}


def test_plan_resumed_and_prebuilt_flags_only_when_true():
    plan = {"tasks": []}
    assert BusEvent.plan("orchestrator", plan=plan, resumed=True).payload == {
        "plan": plan,
        "resumed": True,
    }
    assert BusEvent.plan("orchestrator", plan=plan, pre_built=True).payload == {
        "plan": plan,
        "pre_built": True,
    }


def test_plan_proposed():
    plan = {"summary": "do it", "steps": []}
    e = BusEvent.plan_proposed("coordinator", plan=plan, revision=2)
    assert e.type == EventType.PLAN_PROPOSED
    assert e.payload == {"plan": plan, "revision": 2}


def test_thought_derives_thought_and_action_from_response():
    resp = {"thought": "thinking", "action": "search", "args": {}}
    e = BusEvent.thought("a1", response=resp)
    assert e.type == EventType.THOUGHT
    assert e.payload == {"response": resp, "thought": "thinking", "action": "search"}


def test_thought_handles_none_response():
    e = BusEvent.thought("a1", response=None)
    assert e.payload == {"response": None, "thought": "", "action": None}


def test_token_event_sets_token_field_not_payload():
    e = BusEvent.token_event("a1", token="hel")
    assert e.type == EventType.TOKEN
    assert e.token == "hel"
    assert e.payload == {}


def test_action():
    e = BusEvent.action("a1", step=3, tool="grep", args={"q": "x"})
    assert e.type == EventType.ACTION
    assert e.payload == {"step": 3, "tool": "grep", "args": {"q": "x"}}


def test_observation():
    e = BusEvent.observation("a1", step=3, tool="grep", observation="hit")
    assert e.type == EventType.OBSERVATION
    assert e.payload == {"step": 3, "tool": "grep", "observation": "hit"}


def test_context_usage_merges_usage_and_always_includes_llm_keys():
    usage = {"tokens": 100, "max_tokens": 1000, "percent": 0.1, "level": "normal"}
    e = BusEvent.context_usage(
        "a1",
        usage=usage,
        tokens_in=50,
        tokens_out=None,
        cache_read_tokens=None,
        cache_creation_tokens=None,
    )
    assert e.type == EventType.CONTEXT
    assert e.payload == {
        **usage,
        "tokens_in": 50,
        "tokens_out": None,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
    }


def test_memory_event_key_is_summarized():
    e = BusEvent.memory("a1", before={"tokens": 9}, after={"tokens": 3}, summarizations=1)
    assert e.type == EventType.MEMORY
    assert e.payload == {
        "event": "summarized",
        "before": {"tokens": 9},
        "after": {"tokens": 3},
        "summarizations": 1,
    }


def test_human_guidance():
    e = BusEvent.human_guidance("a1", step=2, text="focus on X")
    assert e.type == EventType.HUMAN_GUIDANCE
    assert e.payload == {"step": 2, "text": "focus on X"}


def test_subagent_start_carries_parent_agent_id():
    e = BusEvent.subagent_start(
        "researcher", task="find facts", invocation_id="run-1", parent_agent_id="coordinator"
    )
    assert e.type == EventType.SUBAGENT_START
    assert e.agent_id == "researcher"
    assert e.parent_agent_id == "coordinator"
    assert e.payload == {"task": "find facts", "invocation_id": "run-1"}


def test_subagent_done():
    e = BusEvent.subagent_done(
        "researcher",
        success=True,
        steps=4,
        confidence=0.9,
        answer="done",
        error="",
        invocation_id="run-1",
        parent_agent_id="coordinator",
    )
    assert e.type == EventType.SUBAGENT_DONE
    assert e.parent_agent_id == "coordinator"
    assert e.payload == {
        "success": True,
        "steps": 4,
        "confidence": 0.9,
        "answer": "done",
        "error": "",
        "invocation_id": "run-1",
    }


def test_task_done_agent_passes_result_through():
    result = {"agent_id": "a1", "answer": "ok", "confidence": 1.0, "steps": 2, "metadata": {}}
    e = BusEvent.task_done_agent("a1", result=result)
    assert e.type == EventType.TASK_DONE
    assert e.payload is result


def test_task_done_task():
    e = BusEvent.task_done_task(
        "researcher", task_id="t1", success=True, confidence=0.8, answer="a", error=None
    )
    assert e.type == EventType.TASK_DONE
    assert e.payload == {
        "task_id": "t1",
        "success": True,
        "confidence": 0.8,
        "answer": "a",
        "error": None,
    }


def test_replan():
    e = BusEvent.replan("orchestrator", replan_count=1, trigger_task="t2", new_task_count=3)
    assert e.type == EventType.REPLAN
    assert e.payload == {"replan_count": 1, "trigger_task": "t2", "new_task_count": 3}


def test_done_passes_payload_through():
    payload = {"run_id": "r", "goal": "g", "answer": "a", "confidence": 0.9}
    e = BusEvent.done("orchestrator", payload=payload)
    assert e.type == EventType.DONE
    assert e.payload is payload


def test_error_event_no_steps_has_empty_payload():
    e = BusEvent.error_event("a1", error="boom")
    assert e.type == EventType.ERROR
    assert e.error == "boom"
    assert e.payload == {}


def test_error_event_with_steps():
    e = BusEvent.error_event("a1", error="max steps", steps=10)
    assert e.error == "max steps"
    assert e.payload == {"steps": 10}
