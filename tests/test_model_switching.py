from __future__ import annotations

import pytest

from agents.base import AgentConfig, BaseAgent
from harness.model_switching import ModelSwitcher
from harness.persistent import InMemorySessionStore
from harness.runtime import BudgetGuard, GuardrailConfig, Tracer
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.subagent import SubAgentTool


class _LLM:
    input_token_budget = 1000

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.budget = None

    def set_budget(self, guard) -> None:
        self.budget = guard

    async def complete(self, system, messages, **kwargs):
        return {"thought": self.name, "action": "finish", "answer": self.name}


class _FactoryLLM(_LLM):
    created = 0

    def __init__(self) -> None:
        type(self).created += 1
        super().__init__("factory")


def _memory(llm: _LLM) -> MemoryManager:
    return MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )


def _agent(agent_id: str, llm: _LLM, tools: dict | None = None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            role=f"{agent_id} role",
            system_prompt="ReAct.",
            allowed_tools=list((tools or {}).keys()),
            max_steps=2,
        ),
        tools=tools or {},
        memory=_memory(llm),
        tracer=Tracer(),
        guard=BudgetGuard(GuardrailConfig(max_total_cost_usd=10.0)),
        llm=llm,
    )


def test_model_switcher_validates_registry_and_default():
    coordinator = _agent("coordinator", _LLM())
    store = InMemorySessionStore()

    with pytest.raises(ValueError, match="reserved"):
        ModelSwitcher(
            coordinator=coordinator,
            session_store=store,
            llm_registry={"default": lambda: _LLM()},
            default_model=None,
        )
    with pytest.raises(TypeError, match="zero-argument factories"):
        ModelSwitcher(
            coordinator=coordinator,
            session_store=store,
            llm_registry={"fast": _LLM()},
            default_model=None,
        )
    with pytest.raises(ValueError, match="not present"):
        ModelSwitcher(
            coordinator=coordinator,
            session_store=store,
            llm_registry={"fast": lambda: _LLM()},
            default_model="deep",
        )


@pytest.mark.asyncio
async def test_model_switcher_applies_and_resets_session_overrides():
    default_llm = _LLM("default")
    coordinator = _agent("coordinator", default_llm)
    store = InMemorySessionStore()
    switcher = ModelSwitcher(
        coordinator=coordinator,
        session_store=store,
        llm_registry={"fast": lambda: _LLM("fast")},
        default_model="fast",
    )

    state = await switcher.switch("s", "coordinator", "fast")
    assert coordinator._llm.name == "fast"

    empty_state = await store.load("other")
    switcher.apply(empty_state)
    assert coordinator._llm is default_llm
    assert state.model_overrides == {"coordinator": "fast"}


@pytest.mark.asyncio
async def test_model_switcher_memoizes_factories_and_binds_guard():
    _FactoryLLM.created = 0
    coordinator = _agent("coordinator", _LLM())
    store = InMemorySessionStore()
    switcher = ModelSwitcher(
        coordinator=coordinator,
        session_store=store,
        llm_registry={"deep": _FactoryLLM},
        default_model=None,
    )

    await switcher.switch("s", "coordinator", "deep")
    first = coordinator._llm
    await switcher.switch("s", "coordinator", "deep")

    assert coordinator._llm is first
    assert _FactoryLLM.created == 1
    assert coordinator._llm.budget is coordinator._guard


@pytest.mark.asyncio
async def test_model_switcher_supports_subagents():
    sub = _agent("sub", _LLM("sub-default"))
    coordinator = _agent(
        "coordinator",
        _LLM("coordinator-default"),
        tools={"delegate_sub": SubAgentTool(sub, name="delegate_sub")},
    )
    switcher = ModelSwitcher(
        coordinator=coordinator,
        session_store=InMemorySessionStore(),
        llm_registry={"sub-fast": lambda: _LLM("sub-fast")},
        default_model=None,
    )

    await switcher.switch("s", "sub", "sub-fast")

    assert sub._llm.name == "sub-fast"
