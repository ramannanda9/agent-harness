from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.base import AgentConfig, BaseAgent
from examples.persistent_agent_demo import _parse_args
from harness.events import EventType
from harness.persistent import (
    InMemorySessionStore,
    PersistentAgent,
    PersistentAgentConfig,
    SessionMessage,
    SQLiteSessionStore,
)
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from harness.skills import Skill
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool


class _ChatLLM:
    # PersistentAgent reads ``input_token_budget`` to drive the
    # context-fraction compaction trigger. Stubs that omit it would
    # default to "never auto-compact" — tests that want to exercise the
    # compaction path adjust this per-fixture.
    input_token_budget: int = 1_000_000

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_usage: dict | None = None

    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        if kwargs.get("source") == "persistent_session":
            return {"text": "The user likes the proposed approach."}
        if kwargs.get("source") == "reconciler":
            return {
                "text": (
                    '{"semantic_actions": [], '
                    '"episodic_action": {"action": "noop", "memory_key": "k"}}'
                )
            }
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return {
            "thought": "answer",
            "action": "finish",
            "answer": f"answer: {last_user}",
            "confidence": 0.9,
        }


class _NamedLLM(_ChatLLM):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        return {
            "thought": f"using {self.name}",
            "action": "finish",
            "answer": f"model={self.name}",
            "confidence": 0.9,
        }


class _DeepLLM(_NamedLLM):
    created: dict[str, int] = {}

    def __init__(self) -> None:
        self.created["deep"] = self.created.get("deep", 0) + 1
        super().__init__("deep")


class _DelegatingLLM(_ChatLLM):
    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            return {
                "thought": "delegate",
                "action": "delegate_sub",
                "args": {"task": "answer with your model"},
            }
        return {
            "thought": "done",
            "action": "finish",
            "answer": "delegated",
            "confidence": 0.9,
        }


class _BackgroundDelegatingLLM(_ChatLLM):
    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            assert "background_delegate_researcher" in system
            return {
                "thought": "start background",
                "action": "background_delegate_researcher",
                "args": {"task": "find a fact"},
            }
        return {
            "thought": "task started",
            "action": "finish",
            "answer": "Background research is running.",
            "confidence": 0.8,
        }


class _SpyMemory(MemoryManager):
    def __init__(self, llm: Any) -> None:
        super().__init__(
            semantic_store=InMemorySemanticStore(),
            episodic_store=InMemoryEpisodicStore(),
            llm=llm,
        )
        self.run_writes: list[dict[str, Any]] = []

    async def write_run_end(self, goal: str, agent_results: list[dict], trace: list[dict]):
        self.run_writes.append(
            {
                "goal": goal,
                "agent_results": agent_results,
                "trace": trace,
            }
        )
        return await super().write_run_end(goal, agent_results, trace)


class _PlainTool:
    name = "plain"

    async def execute(self) -> str:
        return "plain result"


class _CheckpointStore:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    async def write(self, key: str, value: dict) -> None:
        self.data[key] = value

    async def read(self, key: str) -> dict | None:
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)


class MCPToolAdapter:
    name = "mcp_query"
    description = "query a fake MCP server"
    input_schema = {"type": "object"}

    async def execute(self, **_kwargs):
        return "mcp result"


MCPToolAdapter.__module__ = "tools.mcp.adapter"


class _ToolLLM(_ChatLLM):
    async def complete(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        if kwargs.get("source") == "reconciler":
            return {
                "text": (
                    '{"semantic_actions": [], '
                    '"episodic_action": {"action": "noop", "memory_key": "k"}}'
                )
            }
        if len([c for c in self.calls if c["kwargs"].get("source") != "reconciler"]) == 1:
            return {"thought": "use tool", "action": "plain", "args": {}}
        return {"thought": "done", "action": "finish", "answer": "used tool", "confidence": 0.9}


class _UsageLLM(_ChatLLM):
    def __init__(self) -> None:
        super().__init__()
        self._budget = None

    def set_budget(self, guard: Any) -> None:
        self._budget = guard

    async def complete(self, system, messages, **kwargs):
        if self._budget is not None:
            self._budget.add_tokens(100, 25, source=kwargs.get("source"))
        self.last_usage = {"tokens_in": 100, "tokens_out": 25}
        return await super().complete(system, messages, **kwargs)


def _agent(*, llm: Any, memory: MemoryManager, tools: dict[str, Any] | None = None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="coordinator",
            role="persistent coordinator",
            system_prompt="You are a persistent coordinator.",
            allowed_tools=list((tools or {}).keys()),
            max_steps=4,
        ),
        tools=tools or {},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60)),
        llm=llm,
    )


async def _wait_background_done(
    app: PersistentAgent,
    session_id: str,
    task_id: str,
) -> None:
    for _ in range(20):
        tasks = await app.list_background_tasks(session_id)
        task = next(task for task in tasks if task.task_id == task_id)
        if task.status != "running":
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"background task still running: {task_id}")


@pytest.mark.asyncio
async def test_sqlite_session_store_persists_messages(tmp_path):
    path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(path)
    await store.append_messages(
        "s1",
        [
            SessionMessage(role="user", content="hello"),
            SessionMessage(role="assistant", content="hi"),
        ],
    )

    reloaded = SQLiteSessionStore(path)
    state = await reloaded.load("s1")

    assert state.turn_count == 1
    assert [m.content for m in state.messages] == ["hello", "hi"]


@pytest.mark.asyncio
async def test_sqlite_session_store_lists_clears_and_deletes_sessions(tmp_path):
    path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(path)
    await store.append_messages(
        "alpha",
        [
            SessionMessage(role="user", content="hello"),
            SessionMessage(role="assistant", content="hi"),
        ],
    )
    await store.append_messages("beta", [SessionMessage(role="user", content="question")])

    listed = await store.list_sessions()
    assert {state.session_id for state in listed} == {"alpha", "beta"}
    assert await store.exists("alpha") is True
    assert await store.exists("missing") is False
    assert [state.session_id for state in await store.list_sessions(query="alp")] == ["alpha"]
    usage = await store.record_usage(
        "alpha",
        tokens_in=12,
        tokens_out=5,
        usage={"tokens_in": 12, "tokens_out": 5, "breakdown": {"agent:a": {"tokens_in": 12}}},
    )
    assert usage.tokens_in_total == 12
    assert usage.tokens_out_total == 5
    assert usage.last_run_tokens_in == 12
    assert usage.last_run_tokens_out == 5
    assert usage.last_usage["breakdown"]["agent:a"]["tokens_in"] == 12

    cleared = await store.clear("alpha")
    assert cleared.messages == []
    assert cleared.summary == ""
    assert cleared.turn_count == 0
    assert cleared.last_reconcile_turn == 0
    assert cleared.last_compact_turn == 0
    assert cleared.tokens_in_total == 0
    assert cleared.tokens_out_total == 0
    assert cleared.last_usage == {}
    assert await store.delete("alpha") is True
    assert await store.delete("missing") is False
    assert {state.session_id for state in await store.list_sessions()} == {"beta"}


@pytest.mark.asyncio
async def test_sqlite_session_store_persists_model_overrides(tmp_path):
    path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(path)

    state = await store.set_model_override("s", "coordinator", "fast")

    assert state.model_overrides == {"coordinator": "fast"}
    reloaded = await SQLiteSessionStore(path).load("s")
    assert reloaded.model_overrides == {"coordinator": "fast"}

    cleared = await store.set_model_override("s", "coordinator", None)
    assert cleared.model_overrides == {}


def test_persistent_demo_parser_accepts_session_controls(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "persistent_agent_demo.py",
            "--session-id",
            "pr-review",
            "--db",
            "sessions.sqlite",
            "--time-budget-seconds",
            "420",
        ],
    )

    args = _parse_args()

    assert args.session_id == "pr-review"
    assert args.db == "sessions.sqlite"
    assert args.new_session is False
    assert args.provider == "openai"
    assert args.time_budget_seconds == 420


def test_persistent_demo_parser_accepts_codex_provider(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "persistent_agent_demo.py",
            "--provider",
            "openai-codex",
        ],
    )

    args = _parse_args()

    assert args.provider == "openai-codex"


def test_persistent_demo_parser_accepts_claude_code_provider(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "persistent_agent_demo.py",
            "--provider",
            "claude-code",
        ],
    )

    args = _parse_args()

    assert args.provider == "claude-code"


def test_persistent_agent_capabilities_lists_subagents_and_mcp_tools():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    researcher = _agent(
        llm=llm,
        memory=memory,
        tools={"mcp_query": MCPToolAdapter()},
    )
    researcher.config.agent_id = "researcher"
    researcher.role = "research role"
    researcher.config.skills = [
        Skill(
            name="source-checking",
            description="Check retrieved material against primary sources.",
            instructions="Prefer primary sources.",
            tool_hints=["mcp_query"],
        )
    ]
    delegate = SubAgentTool(researcher, name="delegate_research")
    coordinator = _agent(
        llm=llm,
        memory=memory,
        tools={"delegate_research": delegate, "mcp_query": MCPToolAdapter()},
    )
    coordinator.config.skills = [
        Skill(
            name="coordination",
            description="Coordinate specialist agents.",
            instructions="Delegate precisely.",
        )
    ]
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
    )

    caps = app.capabilities()

    assert caps["coordinator"]["agent_id"] == "coordinator"
    assert caps["coordinator"]["tools"] == [
        "background_delegate_researcher",
        "check_background_task",
        "collect_background_task",
        "delegate_research",
        "mcp_query",
    ]
    assert caps["coordinator"]["skills"] == [
        {
            "name": "coordination",
            "description": "Coordinate specialist agents.",
            "tool_hints": [],
        }
    ]
    assert caps["subagents"][0]["agent_id"] == "researcher"
    assert caps["subagents"][0]["skills"] == [
        {
            "name": "source-checking",
            "description": "Check retrieved material against primary sources.",
            "tool_hints": ["mcp_query"],
        }
    ]
    assert caps["subagents"][0]["tool_name"] == "delegate_research"
    assert caps["subagents"][0]["parent_agent_id"] == "coordinator"
    assert caps["subagents"][0]["mcp_tools"][0]["name"] == "mcp_query"
    assert {tool["owner_agent_id"] for tool in caps["mcp_tools"]} == {
        "coordinator",
        "researcher",
    }


def test_persistent_agent_exposes_config_and_llm():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    config = PersistentAgentConfig(retain_context_fraction=0.2)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=config,
    )

    assert app.config is config
    assert app.llm is llm


@pytest.mark.asyncio
async def test_persistent_agent_session_state_returns_store_state():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    await store.append_messages("s", [SessionMessage(role="user", content="hello")])
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
    )

    state = await app.session_state("s")

    assert state.session_id == "s"
    assert state.turn_count == 1
    assert state.messages[0].content == "hello"


@pytest.mark.asyncio
async def test_persistent_agent_runs_background_subagent_and_collects_result():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    sub = BaseAgent(
        config=AgentConfig(
            agent_id="researcher",
            role="researches",
            system_prompt="You research.",
            allowed_tools=[],
            max_steps=2,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )
    coordinator = _agent(
        llm=llm,
        memory=memory,
        tools={"delegate_researcher": SubAgentTool(sub, name="delegate_researcher")},
    )
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )

    started = await app.start_background_subagent("s", "researcher", "find a fact")
    assert started.status == "running"

    await _wait_background_done(app, "s", started.task_id)
    tasks = await app.list_background_tasks("s")
    done = next(task for task in tasks if task.task_id == started.task_id)
    assert done.status == "done"
    assert "answer: find a fact" in done.answer

    collected = await app.collect_background_task("s", started.task_id)
    assert collected.collected is True
    state = await app.session_state("s")
    assert len(state.messages) == 1
    assert state.messages[0].role == "assistant"
    assert started.task_id in state.messages[0].content
    assert "answer: find a fact" in state.messages[0].content


@pytest.mark.asyncio
async def test_collect_background_task_tool_returns_answer_to_llm():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    sub = BaseAgent(
        config=AgentConfig(
            agent_id="researcher",
            role="researches",
            system_prompt="You research.",
            allowed_tools=[],
            max_steps=2,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )
    coordinator = _agent(
        llm=llm,
        memory=memory,
        tools={"delegate_researcher": SubAgentTool(sub, name="delegate_researcher")},
    )
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )
    started = await app.start_background_subagent("s", "researcher", "find a fact")
    await _wait_background_done(app, "s", started.task_id)

    token = app._active_session_id.set("s")
    try:
        result = await coordinator._tools["collect_background_task"].execute(
            task_id=started.task_id
        )
    finally:
        app._active_session_id.reset(token)

    assert result["status"] == "done"
    assert "answer: find a fact" in result["answer"]
    assert result["collected"] is True
    state = await app.session_state("s")
    assert started.task_id in state.messages[-1].content


@pytest.mark.asyncio
async def test_persistent_agent_installs_llm_visible_background_delegate_tool():
    llm = _BackgroundDelegatingLLM()
    memory = _SpyMemory(llm)
    sub = BaseAgent(
        config=AgentConfig(
            agent_id="researcher",
            role="researches",
            system_prompt="You research.",
            allowed_tools=[],
            max_steps=2,
        ),
        tools={},
        memory=memory,
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )
    coordinator = _agent(
        llm=llm,
        memory=memory,
        tools={"delegate_researcher": SubAgentTool(sub, name="delegate_researcher")},
    )
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )

    events = [event async for event in app.chat("research in the background", session_id="s")]

    observations = [event for event in events if event.type == EventType.OBSERVATION]
    assert observations
    assert "background_delegate_researcher" in coordinator._tools
    assert "check_background_task" in coordinator._tools
    assert "collect_background_task" in coordinator._tools
    tasks = await app.list_background_tasks("s")
    assert len(tasks) == 1
    assert tasks[0].agent_id == "researcher"
    assert tasks[0].instruction == "find a fact"
    await _wait_background_done(app, "s", tasks[0].task_id)


@pytest.mark.asyncio
async def test_persistent_agent_rejects_background_for_unknown_or_coordinator_agent():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    coordinator = _agent(llm=llm, memory=memory)
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(),
    )

    with pytest.raises(ValueError, match="unknown sub-agent"):
        await app.start_background_subagent("s", "coordinator", "work")
    with pytest.raises(ValueError, match="unknown sub-agent"):
        await app.start_background_subagent("s", "missing", "work")


def test_persistent_agent_forget_memory_cache_evicts_session_context():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
    )
    app._mem._cache["s"] = "cached"

    assert app.cached_memory_context("s") == "cached"
    app.forget_memory_cache("s")

    assert app.cached_memory_context("s") is None


@pytest.mark.asyncio
async def test_persistent_agent_lists_clears_and_deletes_sessions():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    await store.append_messages("s", [SessionMessage(role="user", content="hello")])
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
    )
    app._mem._cache["s"] = "cached"

    assert await app.session_exists("s") is True
    assert await app.session_exists("missing") is False
    assert [state.session_id for state in await app.list_sessions()] == ["s"]
    assert [state.session_id for state in await app.list_sessions(query="S")] == ["s"]

    cleared = await app.clear_session("s")
    assert cleared.messages == []
    assert cleared.turn_count == 0
    assert app.cached_memory_context("s") is None

    app._mem._cache["s"] = "cached"
    assert await app.delete_session("s") is True
    assert await app.delete_session("s") is False
    assert app.cached_memory_context("s") is None
    assert await app.list_sessions() == []


@pytest.mark.asyncio
async def test_persistent_agent_force_compact_summarizes_and_trims():
    llm = _ChatLLM()
    llm.input_token_budget = 40
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    await store.append_messages(
        "s",
        [
            SessionMessage(role="user", content="old user"),
            SessionMessage(role="assistant", content="old assistant"),
            SessionMessage(role="user", content="recent user"),
            SessionMessage(role="assistant", content="recent assistant"),
        ],
    )
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(retain_context_fraction=0.15),
    )
    app._mem._cache["s"] = "cached"

    state = await app.force_compact("s")

    assert state.summary == "The user likes the proposed approach."
    assert [m.content for m in state.messages] == ["recent user", "recent assistant"]
    assert state.last_compact_turn == 2
    assert state.last_reconcile_turn == 2
    assert memory.run_writes
    assert [step["content"] for step in memory.run_writes[-1]["trace"]] == [
        "old user",
        "old assistant",
    ]
    assert app.cached_memory_context("s") is None


@pytest.mark.asyncio
async def test_persistent_agent_save_to_memory_reconciles_without_evicting_cache():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    await store.append_messages(
        "s",
        [
            SessionMessage(role="user", content="remember final preference"),
            SessionMessage(role="assistant", content="noted"),
        ],
    )
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(async_reconcile_every_turns=5),
    )
    app._mem._cache["s"] = "cached"

    count = await app.save_to_memory("s")

    assert count == 2
    assert memory.run_writes
    write = memory.run_writes[-1]
    assert write["goal"] == "remember final preference"
    assert write["agent_results"][-1]["answer"] == "noted"
    assert [step["content"] for step in write["trace"]] == ["remember final preference", "noted"]
    # Cache is NOT evicted: the active session keeps its warm prefix.
    assert app.cached_memory_context("s") == "cached"
    state = await store.load("s")
    assert state.last_reconcile_turn == 1


@pytest.mark.asyncio
async def test_persistent_agent_save_to_memory_noops_on_empty_session():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
    )

    count = await app.save_to_memory("s")

    assert count == 0
    assert not memory.run_writes


@pytest.mark.asyncio
async def test_persistent_agent_save_to_memory_uses_reconcile_checkpoint():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    # 8 messages = 4 turn-pairs. Mark the first 2 turns reconciled, then
    # explicit save should include only turns 3-4.
    msgs = []
    for i in range(4):
        msgs.append(SessionMessage(role="user", content=f"u{i}"))
        msgs.append(SessionMessage(role="assistant", content=f"a{i}"))
    await store.append_messages("s", msgs)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(async_reconcile_every_turns=2),
    )
    await store.mark_reconciled("s", 2)

    count = await app.save_to_memory("s")

    assert count == 4
    write = memory.run_writes[-1]
    assert [step["content"] for step in write["trace"]] == ["u2", "a2", "u3", "a3"]


@pytest.mark.asyncio
async def test_persistent_agent_save_to_memory_noops_when_checkpoint_is_current():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    await store.append_messages(
        "s",
        [
            SessionMessage(role="user", content="already saved"),
            SessionMessage(role="assistant", content="ok"),
        ],
    )
    await store.mark_reconciled("s", 1)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
    )

    count = await app.save_to_memory("s")

    assert count == 0
    assert not memory.run_writes


@pytest.mark.asyncio
async def test_persistent_agent_refreshes_guard_per_turn_for_coordinator_and_subagents():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    sub = _agent(llm=llm, memory=memory)
    sub.config.agent_id = "sub"
    delegate = SubAgentTool(sub, name="delegate_sub")
    coordinator = _agent(llm=llm, memory=memory, tools={"delegate_sub": delegate})
    original_coordinator_guard = coordinator._guard
    original_sub_guard = sub._guard
    created_guards: list[BudgetGuard] = []

    def guard_factory() -> BudgetGuard:
        guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60))
        created_guards.append(guard)
        return guard

    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        guard_factory=guard_factory,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    [event async for event in app.chat("hello")]

    assert created_guards
    assert coordinator._guard is created_guards[0]
    assert sub._guard is created_guards[0]
    assert coordinator._guard is not original_coordinator_guard
    assert sub._guard is not original_sub_guard


@pytest.mark.asyncio
async def test_persistent_agent_switches_coordinator_model_per_session():
    default_llm = _NamedLLM("fast")
    memory = _SpyMemory(default_llm)
    coordinator = _agent(llm=default_llm, memory=memory)
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=default_llm,
        llm_registry={
            "fast": lambda: _NamedLLM("fast"),
            "deep": lambda: _NamedLLM("deep"),
        },
        default_model="fast",
    )

    await app.switch_model("s", "coordinator", "fast")
    events = [event async for event in app.chat("hello", session_id="s")]

    done = next(event for event in events if event.type == EventType.TASK_DONE)
    assert done.payload["answer"] == "model=fast"
    assert await app.model_overrides("s") == {"coordinator": "fast"}

    events = [event async for event in app.chat("other session", session_id="other")]
    done = next(event for event in events if event.type == EventType.TASK_DONE)
    assert done.payload["answer"] == "model=fast"

    await app.clear_model_override("s", "coordinator")
    events = [event async for event in app.chat("hello again", session_id="s")]
    done = next(event for event in events if event.type == EventType.TASK_DONE)
    assert done.payload["answer"] == "model=fast"


@pytest.mark.asyncio
async def test_persistent_agent_preserves_explicit_control_llm_after_model_apply():
    coordinator_llm = _NamedLLM("coordinator")
    control_llm = _NamedLLM("control")
    memory = _SpyMemory(control_llm)
    coordinator = _agent(llm=coordinator_llm, memory=memory)
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=control_llm,
    )

    [event async for event in app.chat("hello", session_id="s")]

    assert app.llm is control_llm


@pytest.mark.asyncio
async def test_persistent_agent_registry_uses_class_factories_once_per_model():
    _DeepLLM.created.clear()
    control_llm = _NamedLLM("control")
    memory = _SpyMemory(control_llm)
    coordinator = _agent(llm=_NamedLLM("fast"), memory=memory)
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=control_llm,
        llm_registry={"deep": _DeepLLM},
    )

    await app.switch_model("s", "coordinator", "deep")
    [event async for event in app.chat("first", session_id="s")]
    [event async for event in app.chat("second", session_id="s")]

    assert _DeepLLM.created["deep"] == 1


def test_persistent_agent_rejects_ambiguous_model_registry_names():
    llm = _NamedLLM("fast")
    memory = _SpyMemory(llm)
    coordinator = _agent(llm=llm, memory=memory)

    with pytest.raises(ValueError, match="reserved"):
        PersistentAgent(
            coordinator=coordinator,
            session_store=InMemorySessionStore(),
            memory=memory,
            llm=llm,
            llm_registry={"default": lambda: _NamedLLM("default")},
        )

    with pytest.raises(ValueError, match="not present"):
        PersistentAgent(
            coordinator=coordinator,
            session_store=InMemorySessionStore(),
            memory=memory,
            llm=llm,
            llm_registry={"fast": lambda: _NamedLLM("fast")},
            default_model="deep",
        )


@pytest.mark.asyncio
async def test_persistent_agent_can_clear_stale_model_override():
    store = InMemorySessionStore()
    await store.set_model_override("s", "old_agent", "fast")
    llm = _NamedLLM("fast")
    memory = _SpyMemory(llm)
    coordinator = _agent(llm=llm, memory=memory)
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=store,
        memory=memory,
        llm=llm,
        llm_registry={"fast": lambda: _NamedLLM("fast")},
    )

    state = await app.clear_model_override("s", "old_agent")

    assert state.model_overrides == {}


@pytest.mark.asyncio
async def test_persistent_agent_switches_subagent_model_per_session():
    coordinator_llm = _DelegatingLLM()
    default_sub_llm = _NamedLLM("sub-default")
    memory = _SpyMemory(coordinator_llm)
    sub = _agent(llm=default_sub_llm, memory=memory)
    sub.config.agent_id = "sub"
    delegate = SubAgentTool(sub, name="delegate_sub")
    coordinator = _agent(llm=coordinator_llm, memory=memory, tools={"delegate_sub": delegate})
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=coordinator_llm,
        llm_registry={"sub-fast": lambda: _NamedLLM("sub-fast")},
    )

    await app.switch_model("s", "sub", "sub-fast")
    events = [event async for event in app.chat("delegate", session_id="s")]

    sub_done = next(
        event for event in events if event.type == EventType.TASK_DONE and event.agent_id == "sub"
    )
    assert sub_done.payload["answer"] == "model=sub-fast"


@pytest.mark.asyncio
async def test_persistent_agent_records_turn_usage_from_guard():
    llm = _UsageLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        guard_factory=lambda: BudgetGuard(
            GuardrailConfig(max_total_cost_usd=10.0, max_wall_time_seconds=60)
        ),
        config=PersistentAgentConfig(
            compact_at_context_fraction=1.0,
            async_reconcile_every_turns=0,
        ),
    )

    [event async for event in app.chat("hello", session_id="usage")]
    [event async for event in app.chat("again", session_id="usage")]
    state = await store.load("usage")

    assert state.tokens_in_total == 200
    assert state.tokens_out_total == 50
    assert state.last_run_tokens_in == 100
    assert state.last_run_tokens_out == 25
    assert state.last_usage["breakdown"]["agent:coordinator"]["tokens_in"] == 100


@pytest.mark.asyncio
async def test_persistent_agent_injects_recent_session_into_fresh_turn():
    """Turn N+1's LLM call must include turn N's user/assistant pair as
    real role messages (not inline-rendered text). Cache-friendly shape:
    the prefix between turns is byte-identical except for the new pair
    and the new current task."""
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    [event async for event in app.chat("I like the above", session_id="s")]
    [event async for event in app.chat("Can you do x?", session_id="s")]

    agent_calls = [c for c in llm.calls if c["kwargs"].get("source") != "reconciler"]
    second_call_messages = agent_calls[-1]["messages"]
    user_contents = [m["content"] for m in second_call_messages if m["role"] == "user"]
    assistant_contents = [m["content"] for m in second_call_messages if m["role"] == "assistant"]
    # Turn 1's user message survives verbatim as its OWN user message —
    # not folded into a "Recent conversation:" blob inside the turn-2 user
    # message. That's exactly the difference that lets prefix caching
    # hit.
    assert "I like the above" in user_contents
    # Turn 1's assistant response is also a separate role message.
    assert assistant_contents, "expected turn 1's assistant reply to appear as its own message"
    # The current turn's message is the LAST user entry, not a wrapped
    # "Current user turn:" label.
    assert user_contents[-1] == "Can you do x?", (
        f"expected last user content to be the current turn; got {user_contents[-1]!r}"
    )


@pytest.mark.asyncio
async def test_message_prefix_is_stable_across_turns_for_caching():
    """The message list sent to the LLM at turn N+1 must extend turn N's
    list by exactly the new pair + new task — no rotation, no inline
    rewriting. That stability is what unlocks OpenAI's automatic prefix
    cache (matches longest identical prefix) and Anthropic's
    ``cache_control`` markers."""
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    [event async for event in app.chat("first turn", session_id="cache-test")]
    [event async for event in app.chat("second turn", session_id="cache-test")]
    [event async for event in app.chat("third turn", session_id="cache-test")]

    agent_calls = [c for c in llm.calls if c["kwargs"].get("source") != "reconciler"]
    # First call from turn 2 and first call from turn 3.
    turn2_messages = agent_calls[1]["messages"]
    turn3_messages = agent_calls[2]["messages"]

    # System + first user (turn 1 message) + first assistant (turn 1 reply)
    # is the cacheable prefix shared by turns 2 and 3. Compare role/content
    # of the first 3 messages — they must be byte-identical.
    assert len(turn3_messages) > len(turn2_messages), (
        "turn 3 should include MORE history than turn 2, not slide a window"
    )
    for idx in range(3):
        assert turn3_messages[idx]["role"] == turn2_messages[idx]["role"]
        assert turn3_messages[idx]["content"] == turn2_messages[idx]["content"], (
            f"message {idx} content differs between turns — prefix cache would break here. "
            f"turn2: {turn2_messages[idx]['content']!r}, "
            f"turn3: {turn3_messages[idx]['content']!r}"
        )


@pytest.mark.asyncio
async def test_system_prompt_stays_byte_identical_within_compaction_window():
    """The system prompt must be byte-identical across plain chat turns
    inside a single compaction window. Memory context used to be embedded
    in the system prompt via ``MemoryManager.build_context`` on every
    turn — which made the system block content-dependent on the goal and
    invalidated prefix caching from position 0. The fix moved memory
    context to a pinned user-message prior, cached per-session and only
    refreshed at compaction or high-signal reconcile."""
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    [event async for event in app.chat("hello there", session_id="sysstable")]
    [event async for event in app.chat("how's the weather", session_id="sysstable")]
    [event async for event in app.chat("what about Paris", session_id="sysstable")]

    agent_calls = [c for c in llm.calls if c["kwargs"].get("source") != "reconciler"]
    # System prompt now arrives via the top-level ``system=`` parameter
    # rather than as an inline ``role="system"`` message — that change
    # was required to fix the Anthropic adapter dropping inline system
    # entries silently. The cache-stability contract is the same: every
    # plain chat turn must see byte-identical system text.
    system_prompts = [c["system"] for c in agent_calls]
    assert len(system_prompts) >= 3
    assert all(s is not None and s != "" for s in system_prompts), (
        "system prompt must be passed via the ``system=`` parameter to "
        "every LLM call, not stuffed inline as a role=system message — "
        "Anthropic-style adapters silently drop the inline form"
    )
    assert system_prompts[0] == system_prompts[1] == system_prompts[2], (
        "system prompt must not vary between plain chat turns — varying "
        "system prompt invalidates the prefix cache from position 0"
    )


@pytest.mark.asyncio
async def test_persistent_agent_does_not_reconcile_on_tool_runs_alone():
    """The previous contract reconciled on every tool run — but tool
    outputs are usually situational ("3pm Reuters headlines", "this
    moment's kubectl get pods") and don't generalise across sessions.
    Eagerly writing them invalidated the next turn's cache for marginal
    cross-session benefit. Tools now stay in the session transcript;
    long-term memory picks them up at the next compaction boundary."""
    llm = _ToolLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory, tools={"plain": _PlainTool()}),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    events = [event async for event in app.chat("use a tool")]

    # Tool still ran — observation event reached the consumer.
    assert any(e.type == EventType.OBSERVATION for e in events)
    # But no immediate write_run_end (no compaction crossed; no durable
    # signal terms in the message). Tool fact lives in the transcript
    # only until the next compaction event picks it up.
    assert not memory.run_writes, (
        f"tool-run-only turn should NOT trigger reconciliation; "
        f"got {len(memory.run_writes)} write(s)"
    )


@pytest.mark.asyncio
async def test_persistent_agent_gated_tool_does_not_write_resume_checkpoint(monkeypatch):
    from harness.hitl import ApprovalResponse

    llm = _ToolLLM()
    memory = _SpyMemory(llm)
    checkpoint_store = _CheckpointStore()
    coordinator = _agent(llm=llm, memory=memory, tools={"plain": _PlainTool()})
    coordinator.config.hitl_tools = ["plain"]
    coordinator._checkpoint_store = checkpoint_store
    captured_hints: list[str | None] = []

    async def _approve(req, guard):
        captured_hints.append(req.resume_hint)
        return ApprovalResponse(approval_id=req.approval_id, approved=True)

    monkeypatch.setattr("harness.hitl.request_approval", _approve)
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    events = [event async for event in app.chat("use a tool")]

    assert any(e.type == EventType.OBSERVATION for e in events)
    assert checkpoint_store.data == {}
    assert captured_hints == ["Esc cancels this turn; completed session history is preserved."]
    assert coordinator._checkpoint_resume_enabled is True
    assert coordinator._hitl_resume_hint is None


@pytest.mark.asyncio
async def test_persistent_agent_reconciles_on_explicit_remember_term():
    """User-explicit durable signals ("remember", "always", "prefer", etc.)
    still fire reconciliation immediately — cross-session by intent."""
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(
            compact_at_context_fraction=1.0,
            async_reconcile_every_turns=0,  # disable for focused assertion
        ),
    )

    [event async for event in app.chat("please remember I prefer concise answers")]

    assert memory.run_writes, (
        "explicit 'remember' / 'prefer' signal should trigger immediate reconcile"
    )


# ── Async background reconcile ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_reconcile_fires_at_interval_without_evicting_cache():
    """Every ``async_reconcile_every_turns`` turns, a background reconcile
    runs against the durable session transcript window. The session's
    memory-context cache must NOT be evicted — that's the whole point of
    the async path. Memory accumulates without thrashing prefix cache."""
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(
            compact_at_context_fraction=1.0,  # disable compaction trigger
            async_reconcile_every_turns=3,
        ),
    )

    # Prime the per-session memory context cache by running one turn.
    [event async for event in app.chat("first turn", session_id="async")]
    cached_before = app.cached_memory_context("async")

    # Turn 2: not an interval boundary → no async fire.
    [event async for event in app.chat("second turn", session_id="async")]
    assert not memory.run_writes, "turn 2 should not trigger async reconcile"

    # Turn 3: hits the interval → async reconcile fires. Give the
    # fire-and-forget task a tick to land.
    [event async for event in app.chat("third turn", session_id="async")]
    await asyncio.sleep(0)

    assert memory.run_writes, "turn 3 (interval) should fire async reconcile"
    state = await store.load("async")
    assert state.last_reconcile_turn == 3
    # Crucially: the memory context cache was NOT evicted. Identity matters
    # — if it had been refetched, ``cached_before`` would no longer be
    # the value held by the SessionMemoryController for "async".
    cached_after = app.cached_memory_context("async")
    assert cached_after == cached_before, (
        "async reconcile must NOT evict the per-session memory context cache"
    )


@pytest.mark.asyncio
async def test_async_reconcile_samples_transcript_window():
    """The async reconcile pulls evidence from the durable transcript —
    not just the trigger turn. The reconciler sees the last N turn-pairs
    so it can make MERGE / NOOP decisions across the window instead of
    treating every fact as a fresh ADD."""
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(
            compact_at_context_fraction=1.0,
            async_reconcile_every_turns=2,  # smaller for the test
        ),
    )

    [event async for event in app.chat("turn-A msg", session_id="winfocus")]
    [event async for event in app.chat("turn-B msg", session_id="winfocus")]
    await asyncio.sleep(0)

    assert memory.run_writes, "turn 2 (interval=2) should fire async reconcile"
    trace = memory.run_writes[-1]["trace"]
    trace_contents = [t.get("content") for t in trace]
    # Both turn-A and turn-B's user messages should appear in the trace
    # — confirming the reconciler sees the WINDOW, not just the trigger
    # turn's evidence.
    assert "turn-A msg" in trace_contents
    assert "turn-B msg" in trace_contents


@pytest.mark.asyncio
async def test_async_reconcile_disabled_with_zero_interval():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(
            compact_at_context_fraction=1.0,
            async_reconcile_every_turns=0,
        ),
    )

    for i in range(15):
        [event async for event in app.chat(f"turn {i}", session_id="off")]
    await asyncio.sleep(0)

    assert not memory.run_writes, (
        "async_reconcile_every_turns=0 must fully disable the background path"
    )


@pytest.mark.asyncio
async def test_persistent_agent_persists_before_terminal_event_is_yielded():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(compact_at_context_fraction=1.0),
    )

    async for event in app.chat("finish quickly", session_id="s"):
        if event.type == EventType.TASK_DONE:
            break

    state = await store.load("s")
    assert [m.role for m in state.messages] == ["user", "assistant"]
    assert state.messages[0].content == "finish quickly"


@pytest.mark.asyncio
async def test_persistent_agent_compacts_session_at_context_pressure():
    """Compaction now triggers on transcript-token pressure relative to
    the coordinator LLM's ``input_token_budget`` — not on turn or message
    count. Fixture sets a very small budget so two short turns of content
    push past the 0.5 default and force a compaction."""
    llm = _ChatLLM()
    # 6-token threshold (12 * 0.5). Each turn here adds ~6 transcript
    # tokens (8-char user msg + 16-char assistant reply, chars/4 counter),
    # so the second turn leaves an older turn to fold into the summary.
    llm.input_token_budget = 12
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(retain_context_fraction=0.5),
    )

    [event async for event in app.chat("turn one", session_id="s")]
    [event async for event in app.chat("turn two", session_id="s")]
    state = await store.load("s")

    assert state.summary == "The user likes the proposed approach."
    assert "turn one" not in [m.content for m in state.messages if m.role == "user"]
    assert state.messages[-1].role == "assistant"
    assert state.last_compact_turn == 2
