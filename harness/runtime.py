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
from orchestrator.planner import Plan

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
    # Token caps default to None (unlimited) so existing callers see no
    # behaviour change. Subscription-auth adapters (claude-code, codex)
    # finally have an enforceable dimension here — no pricing required.
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_replan_count: int = 2  # forwarded to EvalConfig
    confidence_threshold: float = 0.6  # forwarded to EvalConfig


class BudgetGuard:
    """
    Hard budget limits enforced on every check() call.
    Call check() at the start of each ReAct step and each orchestration loop.

    suspend() / resume() pause the wall-time clock (e.g. during HITL waits)
    so human think-time is not counted against the agent's time budget.

    Per-call-site attribution
    -------------------------
    `add_cost` / `add_tokens` accept an optional ``source`` tag (e.g.
    ``"classifier"``, ``"planner"``). When supplied, the value also lands
    in ``breakdown[source]`` alongside the running totals. Untagged spending
    contributes to totals only — useful for the BaseAgent ReAct loop, which
    isn't a single call site.
    """

    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config
        self._cost: float = 0.0
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._breakdown: dict[str, dict[str, float]] = {}
        self._start: float = time.time()
        self._suspended_seconds: float = 0.0
        self._suspend_at: float | None = None

    def add_cost(self, usd: float, *, source: str | None = None) -> None:
        self._cost += usd
        if source:
            self._attribute(source)["cost_usd"] += usd

    def add_tokens(
        self,
        tokens_in: int,
        tokens_out: int,
        *,
        source: str | None = None,
    ) -> None:
        self._tokens_in += int(tokens_in)
        self._tokens_out += int(tokens_out)
        if source:
            slot = self._attribute(source)
            slot["tokens_in"] += int(tokens_in)
            slot["tokens_out"] += int(tokens_out)

    def _attribute(self, source: str) -> dict[str, float]:
        return self._breakdown.setdefault(
            source, {"cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}
        )

    def suspend(self) -> None:
        """Pause the wall-time clock. Safe to call multiple times; only the first has effect."""
        if self._suspend_at is None:
            self._suspend_at = time.time()

    def resume(self) -> None:
        """Resume the wall-time clock after a suspend()."""
        if self._suspend_at is not None:
            self._suspended_seconds += time.time() - self._suspend_at
            self._suspend_at = None

    def check(self) -> None:
        elapsed = time.time() - self._start - self._suspended_seconds
        if self._cost > self.config.max_total_cost_usd:
            raise RuntimeError(
                f"Cost budget exceeded: ${self._cost:.4f} > ${self.config.max_total_cost_usd}"
            )
        if elapsed > self.config.max_wall_time_seconds:
            raise RuntimeError(
                f"Time budget exceeded: {elapsed:.1f}s > {self.config.max_wall_time_seconds}s"
            )
        if (
            self.config.max_input_tokens is not None
            and self._tokens_in > self.config.max_input_tokens
        ):
            raise RuntimeError(
                f"Input token budget exceeded: {self._tokens_in} > {self.config.max_input_tokens}"
            )
        if (
            self.config.max_output_tokens is not None
            and self._tokens_out > self.config.max_output_tokens
        ):
            raise RuntimeError(
                f"Output token budget exceeded: {self._tokens_out} > {self.config.max_output_tokens}"
            )

    @property
    def elapsed(self) -> float:
        return time.time() - self._start - self._suspended_seconds

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def tokens_in(self) -> int:
        return self._tokens_in

    @property
    def tokens_out(self) -> int:
        return self._tokens_out

    @property
    def breakdown(self) -> dict[str, dict[str, float]]:
        """Read-only snapshot of per-source spending.

        Returned as a fresh dict-of-dicts so callers can mutate it freely
        without poisoning the guard's internal state.
        """
        return {source: dict(values) for source, values in self._breakdown.items()}

    def snapshot(self) -> dict[str, Any]:
        """Serialisable view of the guard's state — attached to terminal
        events so streaming consumers can read totals + breakdown without
        ever holding a reference to the guard itself.

        Shape::

            {
                "cost_usd": float,
                "elapsed_seconds": float,
                "tokens_in": int,
                "tokens_out": int,
                "breakdown": {slot: {cost_usd, tokens_in, tokens_out}, ...},
            }
        """
        return {
            "cost_usd": self._cost,
            "elapsed_seconds": self.elapsed,
            "tokens_in": self._tokens_in,
            "tokens_out": self._tokens_out,
            "breakdown": self.breakdown,
        }


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
        checkpoint_store: Any | None = None,  # FileCheckpointStore / RedisCheckpointStore
        steering_source_factory: Any | None = None,  # passed to each spawned BaseAgent
        # ── Optional per-call-site LLM overrides ──────────────────────────────
        # Each defaults to ``llm`` when unset. The dispatch classifier and the
        # single-agent router both see only the goal + agent descriptions
        # (~300 tokens) and emit a one-token decision — they're the natural
        # candidates for a cheaper model. The planner and synthesiser produce
        # structured DAGs and final answers and should usually stay on the
        # main model. See README "Smart routing + fallback" for the pattern.
        classifier_llm: Any | None = None,
        router_llm: Any | None = None,
        planner_llm: Any | None = None,
        synthesizer_llm: Any | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._memory = memory
        self._llm = llm
        self._classifier_llm = classifier_llm or llm
        self._router_llm = router_llm or llm
        self._planner_llm = planner_llm or llm
        self._synthesizer_llm = synthesizer_llm or llm
        # ``set_budget`` should reach every distinct LLM instance — if the
        # user injected the same wrapper into multiple slots, dedupe by
        # object identity so we don't call it N times.
        self._budget_targets: list[Any] = []
        for candidate in (llm, classifier_llm, router_llm, planner_llm, synthesizer_llm):
            if candidate is None:
                continue
            if any(candidate is existing for existing in self._budget_targets):
                continue
            self._budget_targets.append(candidate)
        self._guardrail_config = guardrail_config or GuardrailConfig()
        self._enable_otel = enable_otel
        self._annotation_store = annotation_store
        self._steering_source_factory = steering_source_factory
        # Auto-create a FileCheckpointStore if any agent uses hitl_tools or
        # checkpoint_every — zero-dep default, no configuration required.
        if checkpoint_store is None and any(
            getattr(agent_registry.get(aid), "hitl_tools", [])
            or getattr(agent_registry.get(aid), "checkpoint_every", 0) > 0
            for aid in agent_registry.all_ids()
        ):
            from harness.checkpoint import FileCheckpointStore

            checkpoint_store = FileCheckpointStore()
        self._checkpoint_store = checkpoint_store

    def _attach_budget(self, guard: Any) -> None:
        """Wire the per-run budget guard into every distinct LLM instance.

        Duck-typed: adapters that don't implement ``set_budget`` (e.g. a
        bare custom client) are skipped silently.
        """
        for target in self._budget_targets:
            if hasattr(target, "set_budget"):
                target.set_budget(guard)

    def _steering_lifecycle(self):
        """Wrap the dispatch in the steering factory's lifecycle if it has one.

        Factories with shared resources (e.g. a StdinRouter) expose
        `__aenter__/__aexit__`. File-based factories don't. We detect at
        runtime and use nullcontext for the latter so the wrapping is
        always safe.
        """
        import contextlib

        f = self._steering_source_factory
        if f is not None and hasattr(f, "__aenter__") and hasattr(f, "__aexit__"):
            return f
        return contextlib.nullcontext()

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

    async def _run_agent_with_tracer(
        self,
        agent_id: str,
        task: str,
        tracer: Tracer,
        run_id: str,
        *,
        guard: BudgetGuard | None = None,
    ):
        """
        Internal: run a single agent using a pre-built tracer.
        The caller is responsible for tracer.start_run() / tracer.end_run().

        ``guard`` lets dispatch / run_routed reuse the guard that already
        captured classifier / router spending, so per-call-site breakdown
        is intact end-to-end. When None, a fresh guard is created (used by
        direct ``run_agent_stream`` callers where there's no preceding
        classifier or router call).
        """
        from agents.base import BaseAgent

        if guard is None:
            guard = BudgetGuard(self._guardrail_config)
            self._attach_budget(guard)

        config = self._agent_registry.get(agent_id)
        agent = BaseAgent(
            config=config,
            tools=self._tool_registry.get_subset(config.allowed_tools),
            memory=self._memory,
            tracer=tracer,
            guard=guard,
            llm=self._llm,
            checkpoint_store=self._checkpoint_store,
            steering_source_factory=self._steering_source_factory,
        )
        async for event in agent.run_stream(task, run_id=run_id):
            yield event

    async def resume_agent(self, ckp_id: str) -> dict:
        """
        Restore and continue an agent run from a checkpoint.

        Call this after a process crash or Ctrl-C to re-present any pending
        HITL approval prompt and continue the ReAct loop from the saved step.

        ckp_id is the value printed by the HITL banner:  "<run_id>:<agent_id>".
        The banner's --resume flag carries this value verbatim so it can be
        passed directly to maybe_resume() / resume_agent().

        Requires checkpoint_store to have been passed (or auto-created) in AgentRuntime.
        """
        from agents.base import BaseAgent
        from harness.events import EventType

        if self._checkpoint_store is None:
            raise RuntimeError("resume_agent requires checkpoint_store")

        checkpoint = await self._checkpoint_store.read(ckp_id)
        if checkpoint is None:
            raise KeyError(f"No checkpoint found for ckp_id={ckp_id!r}")

        from memory.working import WorkingMemory

        wm = WorkingMemory.from_dict(checkpoint["memory"], llm=self._llm)

        # checkpoint["run_id"] is the outer run_id; _resume_stream recomputes
        # _ckp_id as f"{run_id}:{agent_id}" so the correct checkpoint key is used.
        outer_run_id = checkpoint["run_id"]

        config = self._agent_registry.get(checkpoint["agent_id"])
        guard = BudgetGuard(self._guardrail_config)
        self._attach_budget(guard)
        tracer = self._make_tracer()

        agent = BaseAgent(
            config=config,
            tools=self._tool_registry.get_subset(config.allowed_tools),
            memory=self._memory,
            tracer=tracer,
            guard=guard,
            llm=self._llm,
            checkpoint_store=self._checkpoint_store,
            steering_source_factory=self._steering_source_factory,
        )
        agent._working_memory = wm
        agent._task = checkpoint["task"]

        tracer.start_run(outer_run_id, checkpoint["task"])
        result: dict = {}
        try:
            async for event in agent._resume_stream(
                run_id=outer_run_id,
                start_step=checkpoint["step"],
                pending=checkpoint.get("pending"),
            ):
                if event.type == EventType.TASK_DONE:
                    result = event.payload
                elif event.type == EventType.ERROR:
                    result = {"answer": "", "confidence": 0.0, "error": event.error}
        finally:
            tracer.end_run()
        return result

    async def resume_orchestration(self, run_id: str) -> dict:
        """
        Restore and continue an orchestrated run from an orchestrator checkpoint.

        Skips already-completed tasks (results are injected directly) and
        re-runs or resumes any incomplete tasks from where they left off.

        Requires checkpoint_store to have been passed (or auto-created) in AgentRuntime.
        """
        from harness.events import EventType
        from orchestrator.planner import _plan_from_dict, _task_result_from_dict

        if self._checkpoint_store is None:
            raise RuntimeError("resume_orchestration requires checkpoint_store")

        checkpoint = await self._checkpoint_store.read(run_id)
        if checkpoint is None:
            raise KeyError(f"No orchestrator checkpoint found for run_id={run_id!r}")

        goal = checkpoint["goal"]
        plan = _plan_from_dict(checkpoint["plan"])
        completed = {tid: _task_result_from_dict(r) for tid, r in checkpoint["completed"].items()}
        replan_count = checkpoint["replan_count"]

        orchestrator, tracer, _ = self._build_orchestrator(run_id=run_id)
        result: dict = {}
        try:
            async for event in orchestrator.resume_stream(goal, plan, completed, replan_count):
                if event.type == EventType.DONE:
                    result = event.payload
                elif event.type == EventType.ERROR:
                    result = {"answer": "", "confidence": 0.0, "error": event.error}
        finally:
            tracer.end_run()
        return result

    async def resume_stream(self, key: str):
        """
        Resume a checkpoint and stream BusEvents — auto-detects orchestrator vs agent.

        Orchestrator checkpoint (has 'plan')  → streams up to DONE.
        Agent checkpoint       (has 'agent_id') → streams up to TASK_DONE / ERROR.

        Callers iterate this exactly like dispatch_stream / run_stream:
            async for event in runtime.resume_stream(key):
                ...
        """
        if self._checkpoint_store is None:
            raise RuntimeError("resume_stream requires checkpoint_store")

        checkpoint = await self._checkpoint_store.read(key)
        if checkpoint is None:
            raise KeyError(f"No checkpoint found for key={key!r}")

        if "plan" in checkpoint:
            from orchestrator.planner import _plan_from_dict, _task_result_from_dict

            goal = checkpoint["goal"]
            plan = _plan_from_dict(checkpoint["plan"])
            completed = {
                tid: _task_result_from_dict(r) for tid, r in checkpoint["completed"].items()
            }
            replan_count = checkpoint["replan_count"]
            # Orchestrator calls self._tracer.end_run() inside _execute_plan_stream.
            orchestrator, _tracer, _ = self._build_orchestrator(run_id=key)
            async for event in orchestrator.resume_stream(goal, plan, completed, replan_count):
                yield event
        else:
            from agents.base import BaseAgent
            from memory.working import WorkingMemory

            wm = WorkingMemory.from_dict(checkpoint["memory"], llm=self._llm)
            outer_run_id = checkpoint["run_id"]
            config = self._agent_registry.get(checkpoint["agent_id"])
            guard = BudgetGuard(self._guardrail_config)
            self._attach_budget(guard)
            tracer = self._make_tracer()
            agent = BaseAgent(
                config=config,
                tools=self._tool_registry.get_subset(config.allowed_tools),
                memory=self._memory,
                tracer=tracer,
                guard=guard,
                llm=self._llm,
                checkpoint_store=self._checkpoint_store,
                steering_source_factory=self._steering_source_factory,
            )
            agent._working_memory = wm
            agent._task = checkpoint["task"]
            tracer.start_run(outer_run_id, checkpoint["task"])
            try:
                async for event in agent._resume_stream(
                    run_id=outer_run_id,
                    start_step=checkpoint["step"],
                    pending=checkpoint.get("pending"),
                ):
                    yield event
            finally:
                tracer.end_run()

    async def resume(self, key: str) -> dict:
        """
        Unified resume entry point — auto-detects checkpoint type:
          - orchestrator checkpoint (has 'plan' field) → resume_orchestration
          - agent checkpoint (has 'agent_id' field)   → resume_agent

        Pass the value from --resume directly; no need to know the type upfront.
        """
        if self._checkpoint_store is None:
            raise RuntimeError("resume requires checkpoint_store")

        checkpoint = await self._checkpoint_store.read(key)
        if checkpoint is None:
            raise KeyError(f"No checkpoint found for key={key!r}")

        if "plan" in checkpoint:
            return await self.resume_orchestration(key)
        return await self.resume_agent(key)

    def _build_orchestrator(
        self,
        run_id: str | None = None,
        *,
        guard: BudgetGuard | None = None,
    ):
        """Construct fresh tracer, guard, agents, and orchestrator for one run.

        ``guard`` lets dispatch hand down the guard that already captured
        classifier spending. When None, a fresh guard is created — the
        normal path for direct ``run_stream`` callers.
        """
        from agents.base import BaseAgent
        from orchestrator.planner import EvalConfig, Orchestrator

        tracer = self._make_tracer()
        if guard is None:
            guard = BudgetGuard(self._guardrail_config)
            # Adapters that implement set_budget(guard) (e.g. OpenAILLM) get
            # the fresh per-run guard so they can call add_cost() on every
            # completion. Duck-typed so users can plug in any LLM client
            # that doesn't.
            self._attach_budget(guard)

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
                checkpoint_store=self._checkpoint_store,
                steering_source_factory=self._steering_source_factory,
            )
            for agent_id in self._agent_registry.all_ids()
        }

        orchestrator = Orchestrator(
            agents=agents,
            memory=self._memory,
            tracer=tracer,
            guard=guard,
            llm=self._llm,
            planner_llm=self._planner_llm,
            synthesizer_llm=self._synthesizer_llm,
            eval_config=EvalConfig(
                confidence_threshold=self._guardrail_config.confidence_threshold,
                max_replan_count=self._guardrail_config.max_replan_count,
            ),
            checkpoint_store=self._checkpoint_store,
            run_id=run_id,
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
        response = await self._classifier_llm.complete(
            system=_CLASSIFIER_SYSTEM.format(agent_descriptions=agent_descriptions),
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
            response_format={"type": "json_object"},
            source="classifier",
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

        Auto-resume: when --resume <key> is in sys.argv and a checkpoint store is
        configured, the saved run is transparently restored and streamed — callers
        need no resume-specific handling.

        Steering: if `steering_source_factory` exposes async context manager
        methods (e.g. `stdin_steering_factory()` which owns a shared
        StdinRouter), this method wraps the entire dispatch in that
        lifecycle so callers don't manage the shared resource themselves.
        """
        from harness.events import BusEvent, EventType

        async with self._steering_lifecycle():
            if self._checkpoint_store is not None:
                from harness.checkpoint import maybe_resume_key

                resume_key = maybe_resume_key()
                if resume_key:
                    async for event in self.resume_stream(resume_key):
                        yield event
                    return

            # Create the budget guard up-front so the classifier and router
            # LLM calls (which fire before any agent runs) land in the
            # per-call-site breakdown alongside planner / synthesizer / agent
            # spending. Downstream stream methods reuse this same guard.
            guard = BudgetGuard(self._guardrail_config)
            self._attach_budget(guard)

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
                    async for event in self._run_agent_with_tracer(
                        agent_id, goal, tracer, run_id, guard=guard
                    ):
                        yield event
                finally:
                    tracer.end_run()
            else:
                # Orchestrated path owns its own trace via _build_orchestrator.
                # run_stream re-enters _steering_lifecycle as nullcontext when
                # the factory is already active (idempotent), so no double-start.
                async for event in self.run_stream(goal, guard=guard):
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
        response = await self._router_llm.complete(
            system=_ROUTER_SYSTEM.format(agent_descriptions=agent_descriptions),
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
            response_format={"type": "json_object"},
            source="router",
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

        async with self._steering_lifecycle():
            tracer = self._make_tracer()
            run_id = str(uuid.uuid4())
            tracer.start_run(run_id, goal)
            # Same hoisting as dispatch_stream — the router call below
            # otherwise fires before the budget guard is attached, dropping
            # source="router" from the breakdown.
            guard = BudgetGuard(self._guardrail_config)
            self._attach_budget(guard)
            try:
                agent_id, rationale = await self.route(goal)
                logger.info("Router → %s (%s)", agent_id, rationale)
                tracer.log("route", agent_id, {"agent_id": agent_id, "rationale": rationale})
                yield BusEvent(
                    type=EventType.ROUTE,
                    agent_id=agent_id,
                    payload={"agent_id": agent_id, "rationale": rationale},
                )
                async for event in self._run_agent_with_tracer(
                    agent_id, goal, tracer, run_id, guard=guard
                ):
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
            async with self._steering_lifecycle():
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
        result["budget"] = guard.snapshot()
        return result

    async def run_stream(self, goal: str, *, guard: BudgetGuard | None = None):
        """
        Yield BusEvents as the orchestrator runs. Caller consumes events live
        for UI / partial-result display. The final DONE event's payload is the
        same dict returned by `run()` (minus trace/budget, which are attached
        only in the blocking path).

        Auto-resume: when --resume <key> is in sys.argv and a checkpoint store is
        configured, the saved run is transparently restored and streamed.

        ``guard`` lets dispatch reuse the budget guard that already captured
        classifier-slot spending, so the breakdown remains coherent across
        classify → plan → synth. When None, ``_build_orchestrator`` creates
        a fresh guard — the normal path for direct ``run_stream`` callers.
        """
        async with self._steering_lifecycle():
            if self._checkpoint_store is not None:
                from harness.checkpoint import maybe_resume_key

                resume_key = maybe_resume_key()
                if resume_key:
                    async for event in self.resume_stream(resume_key):
                        yield event
                    return
            orchestrator, _tracer, _guard = self._build_orchestrator(guard=guard)
            async for event in orchestrator.run_stream(goal):
                yield event

    async def run_with_plan_stream(self, plan: Plan, goal: str):
        """Stream a pre-built plan, bypassing the LLM planner entirely.

        Use this for deterministic, repeatable workflows where the task
        decomposition is known upfront (CI pipelines, ETL, scheduled jobs).
        The plan is validated against the registered agents before execution;
        everything downstream — parallel execution, replan-on-failure,
        synthesis, memory writes — is identical to ``run_stream``.

        Args:
            plan: A ``Plan`` instance, e.g.
                  ``Plan([Task("t1", "analyst", "Analyse X"),
                          Task("t2", "reporter", "Report Y", depends_on=["t1"])])``
            goal: Goal text for memory context injection and synthesis prompt.
        """
        async with self._steering_lifecycle():
            orchestrator, _tracer, _guard = self._build_orchestrator()
            async for event in orchestrator.run_with_plan_stream(plan, goal):
                yield event

    async def run_with_plan(self, plan: Plan, goal: str) -> dict:
        """Run a pre-built plan and return the DONE payload.

        Blocking version of ``run_with_plan_stream()``.
        """
        from harness.events import EventType

        orchestrator, tracer, guard = self._build_orchestrator()
        result: dict = {}
        async for event in orchestrator.run_with_plan_stream(plan, goal):
            if event.type == EventType.DONE:
                result = event.payload
        result["trace"] = tracer.dump()
        result["budget"] = guard.snapshot()
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────


_parse_json_response = parse_llm_json
