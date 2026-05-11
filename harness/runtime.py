"""
harness/runtime.py — AgentRuntime: single entry point for all agent runs.

harness/tracer.py   — Tracer: records every event in the run.
harness/guardrails.py — BudgetGuard: cost, depth, time limits.
harness/registry.py — AgentRegistry + ToolRegistry.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# Tracer
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceEvent:
    event_type: str      # thought | action | task_result | plan | replan | synthesis
    agent_id: str
    payload: Any
    timestamp: float = field(default_factory=time.time)


class Tracer:
    """
    Records every event during a run.
    In production: emit to OTEL collector, LangSmith, or Arize.
    """
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def log(self, event_type: str, agent_id: str, payload: Any) -> None:
        self._events.append(TraceEvent(event_type, agent_id, payload))

    def dump(self) -> list[dict]:
        result = []
        for e in self._events:
            result.append({
                "event_type": e.event_type,
                "agent_id": e.agent_id,
                "payload": e.payload,
                "timestamp": e.timestamp,
            })
        return result

    def get_agent_trace(self, agent_id: str) -> list[dict]:
        return [
            {"event_type": e.event_type, "payload": e.payload}
            for e in self._events
            if e.agent_id == agent_id
        ]

    def print_trace(self, truncate: int = 300) -> None:
        import json
        for e in self._events:
            payload_str = json.dumps(e.payload, default=str)
            if len(payload_str) > truncate:
                payload_str = payload_str[:truncate] + "…"
            print(f"[{e.event_type.upper():15}] {e.agent_id:20} {payload_str}")


# ══════════════════════════════════════════════════════════════════════════════
# Guardrails
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailConfig:
    max_total_cost_usd: float = 2.0
    max_wall_time_seconds: int = 180
    max_replan_count: int = 2          # forwarded to EvalConfig
    confidence_threshold: float = 0.6  # forwarded to EvalConfig


class BudgetGuard:
    """
    Hard budget limits enforced on every check() call.
    Call check() at the start of each ReAct step and each orchestration loop.
    """
    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config
        self._cost: float = 0.0
        self._start: float = time.time()

    def add_cost(self, usd: float) -> None:
        self._cost += usd

    def check(self) -> None:
        elapsed = time.time() - self._start
        if self._cost > self.config.max_total_cost_usd:
            raise RuntimeError(
                f"Cost budget exceeded: ${self._cost:.4f} > ${self.config.max_total_cost_usd}"
            )
        if elapsed > self.config.max_wall_time_seconds:
            raise RuntimeError(
                f"Time budget exceeded: {elapsed:.1f}s > {self.config.max_wall_time_seconds}s"
            )

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    @property
    def cost(self) -> float:
        return self._cost


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> ToolRegistry:
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def get_subset(self, names: list[str]) -> dict[str, Any]:
        return {n: self.get(n) for n in names}

    def all_names(self) -> list[str]:
        return list(self._tools.keys())


class AgentRegistry:
    def __init__(self) -> None:
        from agents.base import AgentConfig
        self._configs: dict[str, AgentConfig] = {}

    def register(self, config: Any) -> AgentRegistry:
        self._configs[config.agent_id] = config
        return self

    def get(self, agent_id: str) -> Any:
        if agent_id not in self._configs:
            raise KeyError(f"Agent '{agent_id}' not registered")
        return self._configs[agent_id]

    def all_ids(self) -> list[str]:
        return list(self._configs.keys())


# ══════════════════════════════════════════════════════════════════════════════
# AgentRuntime — single entry point
# ══════════════════════════════════════════════════════════════════════════════

class AgentRuntime:
    """
    Wire up once. Run any goal.

    Usage:
        runtime = AgentRuntime(
            agent_registry=agents,
            tool_registry=tools,
            memory=memory_manager,
            llm=llm_client,
        )
        result = await runtime.run("investigate GPU latency spike on worker-07")

    To add a new domain:
        1. Register new tools: tool_registry.register(MyTool())
        2. Register new agents: agent_registry.register(AgentConfig(...))
        3. runtime.run("new domain goal") — orchestrator handles the rest
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        memory: Any,                      # MemoryManager
        llm: Any,
        guardrail_config: GuardrailConfig | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._memory = memory
        self._llm = llm
        self._guardrail_config = guardrail_config or GuardrailConfig()

    async def run(self, goal: str) -> dict:
        from agents.base import BaseAgent
        from orchestrator.planner import EvalConfig, Orchestrator

        tracer = Tracer()
        guard = BudgetGuard(self._guardrail_config)

        # instantiate agents fresh per run — state lives in memory, not agents
        agents = {
            agent_id: BaseAgent(
                config=self._agent_registry.get(agent_id),
                tools=self._tool_registry.get_subset(
                    self._agent_registry.get(agent_id).allowed_tools
                ),
                memory=self._memory,
                tracer=tracer,
                guard=guard,
                llm=self._llm,
            )
            for agent_id in self._agent_registry.all_ids()
        }

        orchestrator = Orchestrator(
            agents=agents,
            memory=self._memory,
            tracer=tracer,
            guard=guard,
            llm=self._llm,
            eval_config=EvalConfig(
                confidence_threshold=self._guardrail_config.confidence_threshold,
                max_replan_count=self._guardrail_config.max_replan_count,
            ),
        )

        result = await orchestrator.run(goal)
        result["trace"] = tracer.dump()
        result["budget"] = {
            "elapsed_seconds": guard.elapsed,
            "cost_usd": guard.cost,
        }
        return result
