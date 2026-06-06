from __future__ import annotations

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
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool


class _ChatLLM:
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


def test_persistent_demo_parser_accepts_session_controls(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "persistent_agent_demo.py",
            "--session-id",
            "pr-review",
            "--db",
            "sessions.sqlite",
        ],
    )

    args = _parse_args()

    assert args.session_id == "pr-review"
    assert args.db == "sessions.sqlite"
    assert args.new_session is False
    assert args.provider == "openai"


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
    delegate = SubAgentTool(researcher, name="delegate_research")
    coordinator = _agent(
        llm=llm,
        memory=memory,
        tools={"delegate_research": delegate, "mcp_query": MCPToolAdapter()},
    )
    app = PersistentAgent(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
    )

    caps = app.capabilities()

    assert caps["coordinator"]["agent_id"] == "coordinator"
    assert caps["coordinator"]["tools"] == ["delegate_research", "mcp_query"]
    assert caps["subagents"][0]["agent_id"] == "researcher"
    assert caps["subagents"][0]["tool_name"] == "delegate_research"
    assert caps["subagents"][0]["parent_agent_id"] == "coordinator"
    assert caps["subagents"][0]["mcp_tools"][0]["name"] == "mcp_query"
    assert {tool["owner_agent_id"] for tool in caps["mcp_tools"]} == {
        "coordinator",
        "researcher",
    }


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
        config=PersistentAgentConfig(reconcile_every_turns=99, compact_every_turns=99),
    )

    [event async for event in app.chat("hello")]

    assert created_guards
    assert coordinator._guard is created_guards[0]
    assert sub._guard is created_guards[0]
    assert coordinator._guard is not original_coordinator_guard
    assert sub._guard is not original_sub_guard


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
        config=PersistentAgentConfig(reconcile_every_turns=99, compact_every_turns=99),
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
        config=PersistentAgentConfig(reconcile_every_turns=99, compact_every_turns=99),
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
async def test_persistent_agent_reconciles_when_tools_run():
    llm = _ToolLLM()
    memory = _SpyMemory(llm)
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory, tools={"plain": _PlainTool()}),
        session_store=InMemorySessionStore(),
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(reconcile_every_turns=99, compact_every_turns=99),
    )

    events = [event async for event in app.chat("use a tool")]

    assert any(e.type == EventType.OBSERVATION for e in events)
    assert memory.run_writes
    assert memory.run_writes[0]["goal"] == "use a tool"
    assert any(t.get("type") == "action" for t in memory.run_writes[0]["trace"])


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
        config=PersistentAgentConfig(reconcile_every_turns=99, compact_every_turns=99),
    )

    async for event in app.chat("finish quickly", session_id="s"):
        if event.type == EventType.TASK_DONE:
            break

    state = await store.load("s")
    assert [m.role for m in state.messages] == ["user", "assistant"]
    assert state.messages[0].content == "finish quickly"


@pytest.mark.asyncio
async def test_persistent_agent_compacts_session_at_threshold():
    llm = _ChatLLM()
    memory = _SpyMemory(llm)
    store = InMemorySessionStore()
    app = PersistentAgent(
        coordinator=_agent(llm=llm, memory=memory),
        session_store=store,
        memory=memory,
        llm=llm,
        config=PersistentAgentConfig(
            reconcile_every_turns=99,
            compact_every_turns=99,
            compact_message_threshold=3,
            recent_messages=2,
        ),
    )

    [event async for event in app.chat("turn one", session_id="s")]
    [event async for event in app.chat("turn two", session_id="s")]
    state = await store.load("s")

    assert state.summary == "The user likes the proposed approach."
    assert len(state.messages) == 2
    assert state.last_compact_turn == 2
