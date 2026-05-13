"""Trajectory capture and annotation store tests."""

from __future__ import annotations

import pytest

from agents.base import AgentConfig
from harness.annotation import Annotation, AnnotationHook, InMemoryAnnotationStore
from harness.runtime import AgentRegistry, AgentRuntime, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tests.conftest import EchoTool, FailingTool, ScriptedLLM

# ── InMemoryAnnotationStore ───────────────────────────────────────────────────


def _annotation(**kwargs) -> Annotation:
    defaults = dict(
        annotation_id="a1",
        run_id="r1",
        agent_id="ag",
        goal="test goal",
        messages=[],
        answer="42",
        confidence=0.9,
        steps=2,
        error="",
        summarization_count=0,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    defaults.update(kwargs)
    return Annotation(**defaults)


def test_store_write_and_get():
    store = InMemoryAnnotationStore()
    a = _annotation()
    store.write(a)
    assert store.get("a1") is a


def test_store_list_all():
    store = InMemoryAnnotationStore()
    store.write(_annotation(annotation_id="a1"))
    store.write(_annotation(annotation_id="a2"))
    assert len(store.list_all()) == 2


def test_store_list_run_filters_by_run_id():
    store = InMemoryAnnotationStore()
    store.write(_annotation(annotation_id="a1", run_id="r1"))
    store.write(_annotation(annotation_id="a2", run_id="r2"))
    assert [a.annotation_id for a in store.list_run("r1")] == ["a1"]


def test_store_list_unrated():
    store = InMemoryAnnotationStore()
    store.write(_annotation(annotation_id="a1"))
    store.write(_annotation(annotation_id="a2"))
    store.rate("a1", rating=0.9)
    unrated = store.list_unrated()
    assert len(unrated) == 1
    assert unrated[0].annotation_id == "a2"


def test_store_rate_sets_fields():
    store = InMemoryAnnotationStore()
    store.write(_annotation())
    store.rate("a1", rating=0.3, correction="correct answer")
    a = store.get("a1")
    assert a.rating == 0.3
    assert a.correction == "correct answer"


def test_store_rate_unknown_raises():
    store = InMemoryAnnotationStore()
    with pytest.raises(KeyError):
        store.rate("nonexistent", rating=1.0)


# ── AnnotationHook ────────────────────────────────────────────────────────────


def test_hook_writes_annotation_on_end_run():
    store = InMemoryAnnotationStore()
    hook = AnnotationHook(store)

    hook.on_start_run("run1", "my goal")
    hook.on_event(
        "trajectory",
        "agent_a",
        {
            "run_id": "run1",
            "messages": [{"role": "user", "content": "hi"}],
            "summarization_count": 1,
        },
    )
    hook.on_event(
        "task_result",
        "agent_a",
        {"answer": "42", "confidence": 0.95, "steps": 3, "error": ""},
    )
    hook.on_end_run()

    assert store.count() == 1
    a = store.list_all()[0]
    assert a.run_id == "run1"
    assert a.agent_id == "agent_a"
    assert a.goal == "my goal"
    assert a.answer == "42"
    assert a.confidence == 0.95
    assert a.steps == 3
    assert a.error == ""
    assert a.summarization_count == 1
    assert a.messages == [{"role": "user", "content": "hi"}]
    assert a.rating is None


def test_hook_handles_error_result():
    store = InMemoryAnnotationStore()
    hook = AnnotationHook(store)

    hook.on_start_run("run2", "failing goal")
    hook.on_event("trajectory", "agent_b", {"messages": [], "summarization_count": 0})
    hook.on_event(
        "task_result",
        "agent_b",
        {"answer": "", "confidence": 0.0, "steps": 5, "error": "Max steps (5) reached"},
    )
    hook.on_end_run()

    a = store.list_all()[0]
    assert a.error == "Max steps (5) reached"
    assert a.confidence == 0.0


def test_hook_no_trajectory_writes_nothing():
    """If trajectory event never fires (e.g. agent crashed before run_stream finally),
    on_end_run should write nothing."""
    store = InMemoryAnnotationStore()
    hook = AnnotationHook(store)
    hook.on_start_run("run3", "goal")
    hook.on_end_run()
    assert store.count() == 0


def test_hook_multiple_agents_write_separate_annotations():
    store = InMemoryAnnotationStore()
    hook = AnnotationHook(store)
    hook.on_start_run("run4", "multi-agent goal")

    for agent_id in ("agent_x", "agent_y"):
        hook.on_event("trajectory", agent_id, {"messages": [], "summarization_count": 0})
        hook.on_event(
            "task_result",
            agent_id,
            {"answer": agent_id, "confidence": 0.8, "steps": 1, "error": ""},
        )

    hook.on_end_run()
    assert store.count() == 2
    agent_ids = {a.agent_id for a in store.list_all()}
    assert agent_ids == {"agent_x", "agent_y"}


# ── End-to-end via AgentRuntime ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_captures_trajectory_on_success():
    store = InMemoryAnnotationStore()
    llm = ScriptedLLM()
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    tools = ToolRegistry().register(EchoTool())
    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="echo_agent",
            role="echoes",
            system_prompt="You echo.",
            allowed_tools=["echo"],
        )
    )
    runtime = AgentRuntime(
        agent_registry=agents,
        tool_registry=tools,
        memory=memory,
        llm=llm,
        annotation_store=store,
    )

    result = await runtime.run_agent("echo_agent", "say hello")
    assert result.get("answer")

    assert store.count() == 1
    a = store.list_all()[0]
    assert a.agent_id == "echo_agent"
    assert a.goal == "say hello"
    assert a.answer == result["answer"]
    assert a.confidence > 0
    assert a.error == ""
    assert len(a.messages) >= 2  # at minimum: system + user


@pytest.mark.asyncio
async def test_runtime_captures_trajectory_on_failure():
    """Even when the agent fails (tool error → max steps), annotation is written."""
    store = InMemoryAnnotationStore()
    llm = ScriptedLLM(
        routes={
            # Always call the failing tool — agent will hit max_steps
            "you echo": lambda s, m, kw: {
                "thought": "try",
                "action": "fail",
                "args": {},
            }
        }
    )
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    tools = ToolRegistry().register(FailingTool())
    agents = AgentRegistry().register(
        AgentConfig(
            agent_id="fail_agent",
            role="always fails",
            system_prompt="You echo.",
            allowed_tools=["fail"],
            max_steps=2,
        )
    )
    runtime = AgentRuntime(
        agent_registry=agents,
        tool_registry=tools,
        memory=memory,
        llm=llm,
        annotation_store=store,
    )

    await runtime.run_agent("fail_agent", "this will fail")

    assert store.count() == 1
    a = store.list_all()[0]
    assert a.error != ""
    assert a.confidence == 0.0


@pytest.mark.asyncio
async def test_annotation_messages_contain_full_trajectory():
    """messages must include system prompt, user task, and at least one assistant turn."""
    store = InMemoryAnnotationStore()
    llm = ScriptedLLM()
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )
    runtime = AgentRuntime(
        agent_registry=AgentRegistry().register(
            AgentConfig(
                agent_id="ag",
                role="r",
                system_prompt="sys",
                allowed_tools=[],
            )
        ),
        tool_registry=ToolRegistry(),
        memory=memory,
        llm=llm,
        annotation_store=store,
    )

    await runtime.run_agent("ag", "task text")

    a = store.list_all()[0]
    roles = [m["role"] for m in a.messages]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
