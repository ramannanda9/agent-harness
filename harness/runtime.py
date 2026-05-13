"""
harness/runtime.py — AgentRuntime: single entry point for all agent runs.

harness/tracer.py   — Tracer: records every event in the run.
harness/guardrails.py — BudgetGuard: cost, depth, time limits.
harness/registry.py — AgentRegistry + ToolRegistry.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from harness.utils import parse_llm_json

logger = logging.getLogger(__name__)

# ── Router prompt ─────────────────────────────────────────────────────────────
# Deliberately minimal — one cheap LLM call, no decomposition.

_ROUTER_SYSTEM = """
You are a routing agent. Select the single best agent to handle the goal.

Available agents:
{agent_descriptions}

Return JSON only:
{{
  "agent_id": "<agent_id from the list above>",
  "rationale": "<one sentence explaining why>"
}}
"""

# ── Complexity classifier prompt ───────────────────────────────────────────────
# One cheap LLM call. Decides whether the goal needs sub-task decomposition.

_CLASSIFIER_SYSTEM = """
You are a task complexity classifier.

Available agents:
{agent_descriptions}

Classify the goal as "simple" or "complex":

  simple  — one agent can handle the goal end-to-end in a single focused run.
            It may call multiple tools, but the steps are naturally sequential
            and don't benefit from splitting into separate subtasks.

  complex — the goal benefits from decomposition: either it needs multiple
            specialist agents working in parallel, or it has distinct phases
            where later steps depend on the structured output of earlier ones
            (e.g. discover → analyse → report).

Return JSON only:
{{
  "complexity": "simple" | "complex",
  "rationale": "<one sentence>"
}}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Tracer
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceEvent:
    event_type: str  # thought | action | task_result | plan | replan | synthesis
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
            result.append(
                {
                    "event_type": e.event_type,
                    "agent_id": e.agent_id,
                    "payload": e.payload,
                    "timestamp": e.timestamp,
                }
            )
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
    max_replan_count: int = 2  # forwarded to EvalConfig
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
        memory: Any,  # MemoryManager
        llm: Any,
        guardrail_config: GuardrailConfig | None = None,
        enable_otel: bool = False,
        annotation_store: Any | None = None,  # InMemoryAnnotationStore or compatible
    ) -> None:
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._memory = memory
        self._llm = llm
        self._guardrail_config = guardrail_config or GuardrailConfig()
        self._enable_otel = enable_otel
        self._annotation_store = annotation_store

    def _make_tracer(self) -> Tracer:
        """Create a fresh Tracer, attaching configured hooks."""
        tracer = Tracer()
        if self._enable_otel:
            from harness.otel import OTELHook

            tracer.add_hook(OTELHook())
        if self._annotation_store is not None:
            from harness.annotation import AnnotationHook

            tracer.add_hook(AnnotationHook(self._annotation_store))
        return tracer

    async def _run_agent_with_tracer(self, agent_id: str, task: str, tracer: Tracer, run_id: str):
        """
        Internal: run a single agent using a pre-built tracer.
        The caller is responsible for tracer.start_run() / tracer.end_run().
        """
        from agents.base import BaseAgent

        guard = BudgetGuard(self._guardrail_config)
        if hasattr(self._llm, "set_budget"):
            self._llm.set_budget(guard)

        config = self._agent_registry.get(agent_id)
        agent = BaseAgent(
            config=config,
            tools=self._tool_registry.get_subset(config.allowed_tools),
            memory=self._memory,
            tracer=tracer,
            guard=guard,
            llm=self._llm,
        )
        async for event in agent.run_stream(task, run_id=run_id):
            yield event

    def _build_orchestrator(self):
        """Construct fresh tracer, guard, agents, and orchestrator for one run."""
        from agents.base import BaseAgent
        from orchestrator.planner import EvalConfig, Orchestrator

        tracer = self._make_tracer()
        guard = BudgetGuard(self._guardrail_config)

        # Adapters that implement set_budget(guard) (e.g. OpenAILLM) get the
        # fresh per-run guard so they can call add_cost() on every completion.
        # Duck-typed so users can plug in any LLM client that doesn't.
        if hasattr(self._llm, "set_budget"):
            self._llm.set_budget(guard)

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

    async def _classify(self, goal: str) -> str:
        """
        Classify goal complexity with one cheap LLM call.
        Returns "simple" or "complex".
        Fast-path: single agent registered → always "simple" (nothing to decompose across).
        """
        if len(self._agent_registry.all_ids()) == 1:
            return "simple"

        agent_descriptions = "\n".join(
            f"  {aid}: {self._agent_registry.get(aid).role}"
            for aid in self._agent_registry.all_ids()
        )
        response = await self._llm.complete(
            system=_CLASSIFIER_SYSTEM.format(agent_descriptions=agent_descriptions),
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
            response_format={"type": "json_object"},
        )
        data = _parse_json_response(response)
        complexity = data.get("complexity", "simple")
        rationale = data.get("rationale", "")
        logger.info("Classifier → %s (%s)", complexity, rationale)
        return complexity if complexity in ("simple", "complex") else "simple"

    async def dispatch_stream(self, goal: str):
        """
        Single entry point for any goal. Classifies complexity then delegates:
          simple  → router picks agent, one ReAct loop (OTEL-traced)
          complex → run_stream (planner decomposes into sub-tasks, has its own trace)

        Emits a DISPATCH event first so callers can see which path was chosen.
        """
        from harness.events import BusEvent, EventType

        complexity = await self._classify(goal)
        path = "routed" if complexity == "simple" else "orchestrated"
        yield BusEvent(
            type=EventType.DISPATCH,
            agent_id="orchestrator",
            payload={"complexity": complexity, "path": path},
        )

        if complexity == "simple":
            # Own the full OTEL lifecycle for the simple path so dispatch + route
            # events appear in the same trace as the agent work.
            tracer = self._make_tracer()
            run_id = str(uuid.uuid4())
            tracer.start_run(run_id, goal)
            try:
                tracer.log("dispatch", "orchestrator", {"complexity": complexity, "path": path})
                agent_id, rationale = await self.route(goal)
                logger.info("Router → %s (%s)", agent_id, rationale)
                tracer.log("route", agent_id, {"agent_id": agent_id, "rationale": rationale})
                yield BusEvent(
                    type=EventType.ROUTE,
                    agent_id=agent_id,
                    payload={"agent_id": agent_id, "rationale": rationale},
                )
                async for event in self._run_agent_with_tracer(agent_id, goal, tracer, run_id):
                    yield event
            finally:
                tracer.end_run()
        else:
            # Orchestrated path owns its own trace via _build_orchestrator.
            async for event in self.run_stream(goal):
                yield event

    async def dispatch(self, goal: str) -> dict:
        """Blocking dispatch. Returns TASK_DONE payload for simple goals,
        DONE payload for complex (orchestrated) goals."""
        from harness.events import EventType

        result: dict = {}
        async for event in self.dispatch_stream(goal):
            if event.type in (EventType.TASK_DONE, EventType.DONE):
                result = event.payload
            elif event.type == EventType.ERROR:
                result = {"answer": "", "confidence": 0.0, "error": event.error}
        return result

    async def route(self, goal: str) -> tuple[str, str]:
        """
        Pick the best agent for a goal without decomposing into subtasks.

        Fast-path: if only one agent is registered, return it immediately
        without an LLM call.

        Returns (agent_id, rationale).
        """
        all_ids = self._agent_registry.all_ids()

        if len(all_ids) == 1:
            return all_ids[0], "only one agent registered"

        agent_descriptions = "\n".join(
            f"  {aid}: {self._agent_registry.get(aid).role}" for aid in all_ids
        )
        response = await self._llm.complete(
            system=_ROUTER_SYSTEM.format(agent_descriptions=agent_descriptions),
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
            response_format={"type": "json_object"},
        )

        data = _parse_json_response(response)
        agent_id = data.get("agent_id", "")
        rationale = data.get("rationale", "")

        if agent_id not in all_ids:
            logger.warning(
                "Router returned unknown agent_id %r — falling back to %r",
                agent_id,
                all_ids[0],
            )
            agent_id = all_ids[0]

        return agent_id, rationale

    async def run_routed_stream(self, goal: str):
        """
        Route to the best agent then run its ReAct loop directly.
        Yields a ROUTE event first, then all agent events.
        Use this instead of run_stream when you have a single-turn goal
        that one agent can handle end-to-end — no task decomposition.
        """
        from harness.events import BusEvent, EventType

        tracer = self._make_tracer()
        run_id = str(uuid.uuid4())
        tracer.start_run(run_id, goal)
        try:
            agent_id, rationale = await self.route(goal)
            logger.info("Router → %s (%s)", agent_id, rationale)
            tracer.log("route", agent_id, {"agent_id": agent_id, "rationale": rationale})
            yield BusEvent(
                type=EventType.ROUTE,
                agent_id=agent_id,
                payload={"agent_id": agent_id, "rationale": rationale},
            )
            async for event in self._run_agent_with_tracer(agent_id, goal, tracer, run_id):
                yield event
        finally:
            tracer.end_run()

    async def run_routed(self, goal: str) -> dict:
        """Blocking routed run. Returns the TASK_DONE payload dict."""
        from harness.events import EventType

        result: dict = {}
        async for event in self.run_routed_stream(goal):
            if event.type == EventType.TASK_DONE:
                result = event.payload
            elif event.type == EventType.ERROR:
                result = {"answer": "", "confidence": 0.0, "error": event.error}
        return result

    async def run_agent_stream(self, agent_id: str, task: str):
        """
        Run a single named agent directly, bypassing orchestrator planning.

        Use this when you know exactly which agent should handle the task and
        don't need multi-agent decomposition. The agent runs its ReAct loop and
        yields BusEvents (THOUGHT, TOKEN, ACTION, OBSERVATION, TASK_DONE, ERROR).

        async for event in runtime.run_agent_stream("researcher", "what is 2+2?"):
            ...
        """
        tracer = self._make_tracer()
        run_id = str(uuid.uuid4())
        tracer.start_run(run_id, task)
        try:
            async for event in self._run_agent_with_tracer(agent_id, task, tracer, run_id):
                yield event
        finally:
            tracer.end_run()

    async def run_agent(self, agent_id: str, task: str) -> dict:
        """Blocking single-agent run. Returns the TASK_DONE payload dict."""
        from harness.events import EventType

        result: dict = {}
        async for event in self.run_agent_stream(agent_id, task):
            if event.type == EventType.TASK_DONE:
                result = event.payload
            elif event.type == EventType.ERROR:
                result = {"answer": "", "confidence": 0.0, "error": event.error}
        return result

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


# ── Helpers ───────────────────────────────────────────────────────────────────


_parse_json_response = parse_llm_json
