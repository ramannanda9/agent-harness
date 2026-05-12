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

    Hooks: attach side-channel exporters (e.g. OTEL) via add_hook().
    Each hook must implement on_event(event_type, agent_id, payload)
    and optionally on_start_run(run_id, goal) / on_end_run().
    """
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._hooks: list = []

    def add_hook(self, hook) -> None:
        """Attach an exporter hook (e.g. OTELHook)."""
        self._hooks.append(hook)

    def log(self, event_type: str, agent_id: str, payload: Any) -> None:
        self._events.append(TraceEvent(event_type, agent_id, payload))
        for hook in self._hooks:
            hook.on_event(event_type, agent_id, payload)

    def start_run(self, run_id: str, goal: str) -> None:
        """Signal run start to hooks (e.g. for OTEL root span)."""
        for hook in self._hooks:
            if hasattr(hook, "on_start_run"):
                hook.on_start_run(run_id, goal)

    def end_run(self) -> None:
        """Signal run end to hooks (e.g. to close OTEL root span)."""
        for hook in self._hooks:
            if hasattr(hook, "on_end_run"):
                hook.on_end_run()

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
    Wire up once. Run any goal — streaming or blocking.

    Usage (blocking):
        runtime = AgentRuntime(agent_registry=..., tool_registry=...,
                               memory=..., llm=...)
        result = await runtime.run("investigate worker-07")

    Usage (streaming):
        async for event in runtime.run_stream("investigate worker-07"):
            if event.type == EventType.TOKEN:
                print(event.token, end="", flush=True)
            elif event.type == EventType.DONE:
                print(event.payload["answer"])

    Both paths execute the same code; `run()` is just a drain over `run_stream()`.

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
        enable_otel: bool = False,
    ) -> None:
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._memory = memory
        self._llm = llm
        self._guardrail_config = guardrail_config or GuardrailConfig()
        self._enable_otel = enable_otel

    def _build_orchestrator(self):
        """Construct fresh tracer, guard, agents, and orchestrator for one run."""
        from agents.base import BaseAgent
        from orchestrator.planner import EvalConfig, Orchestrator

        tracer = Tracer()

        if self._enable_otel:
            from harness.otel import OTELHook
            tracer.add_hook(OTELHook())
        guard = BudgetGuard(self._guardrail_config)

        # state lives in memory, not agents — instantiate fresh per run
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
        return orchestrator, tracer, guard

    async def run(self, goal: str) -> dict:
        from harness.events import EventType

        orchestrator, tracer, guard = self._build_orchestrator()
        result: dict = {}
        async for event in orchestrator.run_stream(goal):
            if event.type == EventType.DONE:
                result = event.payload
        result["trace"] = tracer.dump()
        result["budget"] = {
            "elapsed_seconds": guard.elapsed,
            "cost_usd": guard.cost,
        }
        return result

    async def run_stream(self, goal: str):
        """
        Yield BusEvents as the orchestrator runs. Caller consumes events live
        for UI / partial-result display. The final DONE event's payload is the
        same dict returned by `run()` (minus trace/budget, which are attached
        only in the blocking path).
        """
        orchestrator, _tracer, _guard = self._build_orchestrator()
        async for event in orchestrator.run_stream(goal):
            yield event
