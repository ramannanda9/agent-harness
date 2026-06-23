"""Session-scoped model switching for ``PersistentAgent``.

A caller may supply an ``llm_registry`` (label -> zero-arg factory) so the
coordinator and its sub-agents can have their LLM swapped per session via the
``/model`` control. ``ModelSwitcher`` owns the registry, lazily-built instances,
the snapshot of each agent's default LLM, and applying/resetting overrides
across the agent tree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from harness.agent_tree import find_agent, iter_agents

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from harness.persistent import SessionState, SessionStore

_RESERVED_MODEL_NAMES = {"default", "reset", "clear"}


class ModelSwitcher:
    def __init__(
        self,
        *,
        coordinator: BaseAgent,
        session_store: SessionStore,
        llm_registry: dict[str, Callable[[], Any]] | None,
        default_model: str | None,
    ) -> None:
        self._coordinator = coordinator
        self._session_store = session_store
        self._llm_registry = dict(llm_registry or {})

        collisions = sorted(
            model for model in self._llm_registry if model.lower() in _RESERVED_MODEL_NAMES
        )
        if collisions:
            joined = ", ".join(repr(model) for model in collisions)
            raise ValueError(f"model names are reserved for clearing overrides: {joined}")
        non_factories = sorted(
            model for model, factory in self._llm_registry.items() if not callable(factory)
        )
        if non_factories:
            joined = ", ".join(repr(model) for model in non_factories)
            raise TypeError(f"llm_registry entries must be zero-argument factories: {joined}")
        if (
            default_model is not None
            and self._llm_registry
            and default_model not in self._llm_registry
        ):
            raise ValueError(f"default_model {default_model!r} is not present in llm_registry")

        self._default_model = default_model
        self._llm_instances: dict[str, Any] = {}
        self._default_agent_llms: dict[str, Any] = {
            agent.config.agent_id: getattr(agent, "_llm", None)
            for agent in iter_agents(coordinator)
        }

    # ── Introspection ──────────────────────────────────────────────────────────

    def available_models(self) -> list[str]:
        """Return model names available for session-scoped switching."""
        return sorted(self._llm_registry)

    def default_model(self) -> str | None:
        """Return the label for the construction-time/default model, if supplied."""
        return self._default_model

    def enabled(self) -> bool:
        """True when an LLM registry was supplied at construction time."""
        return bool(self._llm_registry)

    async def overrides(self, session_id: str) -> dict[str, str]:
        """Return session-scoped agent_id -> model_name overrides."""
        state = await self._session_store.load(session_id)
        return dict(state.model_overrides)

    # ── Mutation ─────────────────────────────────────────────────────────────────

    async def switch(self, session_id: str, agent_id: str, model_name: str) -> SessionState:
        """Persist and apply a session-scoped model override for one agent."""
        if model_name not in self._llm_registry:
            raise ValueError(f"unknown model {model_name!r}")
        if find_agent(self._coordinator, agent_id) is None:
            raise ValueError(f"unknown agent {agent_id!r}")
        state = await self._session_store.set_model_override(session_id, agent_id, model_name)
        self.apply(state)
        return state

    async def clear(self, session_id: str, agent_id: str) -> SessionState:
        """Remove and apply a session-scoped model override for one agent."""
        state = await self._session_store.set_model_override(session_id, agent_id, None)
        self.apply(state)
        return state

    def apply(self, state: SessionState) -> None:
        """Point each agent's LLM at its session override, or its default.

        Reset first so switching sessions cannot leak a previous session's
        model choice into this one.
        """
        for agent in iter_agents(self._coordinator):
            self._default_agent_llms.setdefault(agent.config.agent_id, getattr(agent, "_llm", None))
            agent._llm = self._default_agent_llms[agent.config.agent_id]

        for agent_id, model_name in state.model_overrides.items():
            agent = find_agent(self._coordinator, agent_id)
            if agent is None or model_name not in self._llm_registry:
                continue
            agent._llm = self._make_registered_llm(model_name)
            guard = getattr(agent, "_guard", None)
            if guard is not None and hasattr(agent._llm, "set_budget"):
                agent._llm.set_budget(guard)

    def _make_registered_llm(self, model_name: str) -> Any:
        llm = self._llm_instances.get(model_name)
        if llm is None:
            llm = self._llm_registry[model_name]()
            self._llm_instances[model_name] = llm
        return llm
