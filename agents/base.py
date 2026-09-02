"""
BaseAgent — generic ReAct loop agent. Streaming-primary.

Every agent is an instance of BaseAgent configured via AgentConfig.
No subclassing needed for new domains — just register a new AgentConfig
with different role, system_prompt, and allowed_tools.

Execution model:
  - run_stream(task) is the canonical method — yields BusEvents for each
    THOUGHT, TOKEN (when the LLM client streams), ACTION, OBSERVATION,
    and finally TASK_DONE with the result payload.
  - run(task) is a thin drain: collects the stream and returns the final dict.
    Use it when you don't need real-time events.

Memory integration:
  - build_context() injected into system prompt at run start
  - write_working_fact() called after each tool observation
  - run-end write handled by Orchestrator, not BaseAgent

Token management:
  - WorkingMemory handles eviction via LLM summarization
  - max budget is configured per-agent via AgentConfig.working_memory_max_tokens
  - count_tokens defaults to chars/4; pass a custom counter to WorkingMemory
    if you need exact (e.g. tiktoken) counts.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from harness.checkpoint import _ResumeHint
from harness.events import BusEvent, EventType
from harness.llm.reasoning import ReasoningEffort, validate_reasoning_effort
from harness.runstate import ActionState, ActionStatus, ErrorKind, Phase, RunState
from harness.skills import Skill
from harness.utils import fire
from memory.manager import MemoryManager
from memory.working import WorkingMemory

logger = logging.getLogger(__name__)


def _assistant_message(response: dict) -> str:
    """Serialize a model response for working memory.

    One function so the string written to WorkingMemory and the string
    persisted in a checkpoint are produced identically. ``default=str``
    matches what the checkpoint stores use, so a response containing a
    non-JSON-native value cannot serialize on one path and raise on the
    other.
    """
    return json.dumps(response, default=str)


def _freeze_factory(tool: Any, args: dict) -> Any:
    """Bind ``tool`` and ``args`` into a zero-arg factory the fan-in helper
    can call to spawn one driver per parallel streaming tool.

    Defined at module scope (not inline) so the late-binding closure trap —
    every lambda capturing the same final loop variable — is avoided in
    the parallel-actions fan-out.
    """
    return lambda: tool.execute_stream(**args)


# ── Agent Config ──────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    agent_id: str
    role: str  # plain English — used by planner for agent selection
    system_prompt: str
    allowed_tools: list[str]  # tool names from ToolRegistry
    max_steps: int = 10
    memory_context_enabled: bool = True
    confidence_from_llm: bool = True  # if False, confidence=1.0 on success
    stream_tokens: bool = False  # if True, TOKEN events are emitted as the LLM streams
    # Optional per-agent reasoning depth. Overrides the LLM adapter's default
    # for ReAct calls, including retries. Provider adapters validate which
    # levels their backend supports.
    reasoning_effort: ReasoningEffort | None = None
    # ``None`` → derive from ``llm.input_token_budget * 0.8`` at runtime
    # (each adapter reports a per-model context window; OpenAILLM /
    # AnthropicLLM / etc. expose ``input_token_budget``). Pass an explicit
    # int to hard-cap the WorkingMemory eviction threshold — useful for
    # cost-sensitive workloads or when feeding very small models.
    working_memory_max_tokens: int | None = None
    # Cap model-facing text appended after tool observations. Event payloads
    # and traces still carry the original tool result; this only protects the
    # next LLM prompt from oversized browser/MCP/shell observations. Set to
    # 0 or a negative value to disable the cap.
    max_observation_chars: int = 20_000
    hitl_tools: list[str] = None  # tools requiring human approval; None = no HITL
    checkpoint_every: int = 0  # write a resumable checkpoint every N steps; 0 = disabled
    # Cache tool results within a single run, keyed by (tool_name, args).
    # Opt-in because not every tool is idempotent — a tool may also veto
    # caching for itself by exposing ``cacheable = False`` on its instance.
    # Designed for read-mostly multi-agent runs where agents redo each
    # other's lookups (HTTPFetch on stable URLs, ``kubectl get …`` style
    # discovery, MCP filesystem reads).
    cache_tool_results: bool = False
    # Prompt/context bundles attached explicitly to this agent. Skills do
    # not grant tools; they only add reusable instructions and hints to the
    # system prompt.
    skills: list[Skill] = None
    # Hard cap on how deep a SubAgentTool chain may recurse. Depth 0 = the
    # top-level agent invoked by AgentRuntime; depth 1 = a sub-agent
    # delegated to from the top; depth 2 = a sub-agent that itself
    # delegated. The default is conservative — most production setups want
    # one or two levels and a hard stop against an LLM hallucinating an
    # infinite delegation chain.
    max_subagent_depth: int = 3

    def __post_init__(self):
        self.reasoning_effort = validate_reasoning_effort(self.reasoning_effort)
        if self.hitl_tools is None:
            self.hitl_tools = []
        if self.skills is None:
            self.skills = []


# ── ReAct Response Schema ─────────────────────────────────────────────────────

# Injected into every agent's system prompt so LLM knows the expected format.
REACT_FORMAT = """
At each step, respond with a JSON object in one of three forms:

To use a single tool:
{
  "thought": "<brief reason, max 25 words>",
  "action": "<tool_name>",
  "args": { "<arg>": "<value>", ... }
}

To use multiple independent tools at once (they run in parallel — use this when \
the calls don't depend on each other):
{
  "thought": "<brief reason, max 25 words>",
  "actions": [
    {"tool": "<tool_name>", "args": { "<arg>": "<value>", ... }},
    {"tool": "<tool_name_2>", "args": { "<arg>": "<value>", ... }}
  ]
}

To finish:
{
  "thought": "<brief reason, max 25 words>",
  "action": "finish",
  "answer": "<comprehensive answer to the task>",
  "confidence": <0.0-1.0>
}

Available tools: __TOOL_LIST__
Return JSON only — no markdown, no preamble. Keep `thought` short; put details in
tool arguments or final `answer`, not in `thought`.
"""


# ── Base Agent ────────────────────────────────────────────────────────────────


class BaseAgent:
    """
    Generic ReAct agent. Configured entirely via AgentConfig + ToolRegistry.

    To create a new specialist agent:
        config = AgentConfig(
            agent_id="my_agent",
            role="does X using tools Y and Z",
            system_prompt="You are an expert at X...",
            allowed_tools=["tool_y", "tool_z"],
        )
        registry.register(config)
    No subclassing needed.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: dict[str, Any],  # name → Tool instance
        memory: MemoryManager,
        tracer,
        guard,
        llm,
        checkpoint_store: Any | None = None,  # FileCheckpointStore / RedisCheckpointStore
        steering_source_factory: Any | None = None,  # (BaseAgent) -> async ctx mgr
    ) -> None:
        self.config = config
        self.role = config.role  # exposed for orchestrator planner prompt
        self._tools = tools
        self._memory = memory
        self._tracer = tracer
        self._guard = guard
        self._llm = llm
        self._checkpoint_store = checkpoint_store
        self._checkpoint_resume_enabled: bool = True
        self._hitl_resume_hint: str | None = None
        self._working_memory: WorkingMemory | None = None
        self._task: str = ""
        self._last_think_error: str | None = None
        self._ckp_id: str = ""  # f"{run_id}:{agent_id}" — unique per agent per run
        # Async steering queue — items drained at the top of each ReAct
        # step (before checkpoint, before think). Created eagerly so
        # callers can steer() before run_stream starts.
        self._steering: asyncio.Queue[str] = asyncio.Queue()
        # Optional factory: called once at run_stream entry. Must return an
        # async context manager that, while active, may call agent.steer().
        # The agent owns the source's lifecycle — no live-instance registry.
        self._steering_source_factory = steering_source_factory
        self._resume_key: str = (
            ""  # key printed in --resume banner; set by orchestrator to outer run_id
        )
        # Per-run tool-result cache. ``None`` when caching is off so the
        # hot path on ``_execute_tool`` skips the lookup entirely; a fresh
        # dict per BaseAgent instance bounds the lifetime to one run.
        self._tool_cache: dict[tuple[str, str], Any] | None = (
            {} if config.cache_tool_results else None
        )
        # SubAgentTool nesting depth. The top-level agent stays at 0; each
        # delegation hop bumps the sub-agent's depth by one. A
        # ``SubAgentTool.execute_stream`` invocation refuses if bumping
        # would exceed ``config.max_subagent_depth``, so an LLM that
        # hallucinates a recursive delegation chain gets stopped at a
        # bounded level rather than hanging the framework.
        self._subagent_depth: int = 0

    # ── Async steering ────────────────────────────────────────────────────────

    def steer(self, text: str) -> None:
        """Inject human guidance to be consumed at the next ReAct step boundary.

        Non-blocking and safe to call concurrently from any coroutine in the
        same event loop. Drained at the top of the next iteration (before
        the per-step checkpoint write and before the next think call), then
        appended to WorkingMemory as a user message and emitted as a
        HUMAN_GUIDANCE BusEvent.

        Worst-case latency = time remaining in the current tool +
        next-think duration. Guidance arriving after the LLM has already
        emitted action="finish" is lost — the agent has decided it's done.
        """
        if not text or not text.strip():
            return
        self._steering.put_nowait(text.strip())

    async def _drain_steering(self, step: int) -> AsyncGenerator[BusEvent, None]:
        """Drain any queued guidance into WorkingMemory; yield one event each.

        Called at the top of each ReAct iteration. Items are FIFO. Empty
        queue is a no-op (zero overhead when no one is steering).
        """
        while not self._steering.empty():
            try:
                text = self._steering.get_nowait()
            except asyncio.QueueEmpty:
                break  # defensive — single consumer, should never fire
            await self._working_memory.append("user", f"Human guidance: {text}")
            self._tracer.log(
                "human_guidance",
                self.config.agent_id,
                {"step": step, "text": text},
            )
            yield BusEvent.human_guidance(self.config.agent_id, step=step, text=text)

    # ── Streaming entry point (canonical) ─────────────────────────────────────

    async def run_stream(
        self,
        task: str,
        run_id: str | None = None,
        *,
        prior_messages: list[tuple[str, str | list]] | None = None,
        pinned_priors: int = 0,
        precomputed_memory_context: Any = None,
    ) -> AsyncGenerator[BusEvent, None]:
        """Run the ReAct loop on ``task``, optionally seeded with prior
        conversation history.

        ``prior_messages``
            List of ``(role, content)`` pairs to append to WorkingMemory
            after the system prompt and before the current task. Used by
            ``PersistentAgent`` to feed cross-turn conversation history as
            real role messages instead of inline-rendered text — which
            makes the prompt prefix byte-identical between turns until
            the next compaction, unlocking OpenAI's automatic prefix
            cache and Anthropic's ``cache_control``.

        ``pinned_priors``
            How many of the *first* ``prior_messages`` to pin against
            WorkingMemory eviction. Designed for ``PersistentAgent`` to
            pin the rolling-summary priming pair so that even if a busy
            turn's tool observations push WM into summarisation, the
            session-level summary survives.
        """
        run_id = run_id or str(uuid.uuid4())
        self._ckp_id = f"{run_id}:{self.config.agent_id}"
        if not self._resume_key:
            self._resume_key = self._ckp_id
        self._task = task
        self._working_memory = WorkingMemory(
            llm=self._llm,
            max_tokens=self.config.working_memory_max_tokens,
        )

        system = await self._build_system_prompt(
            task, precomputed_memory_context=precomputed_memory_context
        )
        await self._working_memory.append("system", system, pinned=True)
        if prior_messages:
            for idx, (role, content) in enumerate(prior_messages):
                await self._working_memory.append(role, content, pinned=idx < pinned_priors)
        await self._working_memory.append("user", task)

        # Steering source is owned by the agent for the duration of the run.
        # nullcontext when no factory is configured — zero overhead.
        source_cm = (
            self._steering_source_factory(self)
            if self._steering_source_factory is not None
            else contextlib.nullcontext()
        )
        state = RunState(
            run_id=run_id,
            agent_id=self.config.agent_id,
            task=task,
            memory=self._working_memory.to_dict(),
        )

        async with source_cm:
            async with _ResumeHint(
                self._resume_key,
                self._checkpoint_store,
                f"Agent {self.config.agent_id}",
                check_key=self._ckp_id,
            ) as hint:
                async for event in self._run_stream_internal(state):
                    # ``parent_agent_id`` filtering: sub-agent events bubble up
                    # through this loop tagged with their invoker's id. A sub's
                    # TASK_DONE / ERROR is NOT terminal for the outer agent —
                    # the outer keeps running. Without this guard the FIRST
                    # delegated sub-agent that completes would wrongly clear
                    # the outer's checkpoint and suppress its resume hint.
                    if not event.parent_agent_id:
                        if event.type == EventType.TASK_DONE:
                            await self._clear_checkpoint(run_id)
                            hint.done = True
                        elif event.type == EventType.ERROR:
                            # Terminal ERROR (max_steps, budget exceeded, mid-run
                            # crash translated to ERROR by ``_run_stream_internal``)
                            # is "the agent ran to completion but failed", NOT a
                            # user interrupt. Suppress the misleading "interrupted
                            # — Resume:" banner; leave the checkpoint intact so the
                            # user can deliberately resume with new config (higher
                            # max_steps, larger budget) if they want.
                            hint.done = True
                    yield event

    async def _resume_stream(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        """
        Re-enter the ReAct loop from a stored :class:`RunState`.

        The state carries everything needed to continue, so this rebuilds
        working memory and the task itself rather than requiring the caller
        to graft them on first — two callers used to do that identically and
        could drift.

        Actions still ``PENDING`` mean the run died at an approval prompt.
        Each is re-prompted; once the human responds the tool runs (or the
        correction is injected) before the loop continues.
        """
        run_id = state.run_id
        self._ckp_id = f"{run_id}:{self.config.agent_id}"
        if not self._resume_key:
            self._resume_key = self._ckp_id
        self._task = state.task
        if state.memory:
            self._working_memory = WorkingMemory.from_dict(state.memory, llm=self._llm)
        else:
            # A state with no transcript still has a task, so rebuild the
            # opening messages rather than driving with no working memory at
            # all — that used to surface as an AttributeError deep in the
            # think path, which says nothing about what was wrong.
            self._working_memory = WorkingMemory(
                llm=self._llm, max_tokens=self.config.working_memory_max_tokens
            )
            await self._working_memory.append(
                "system", await self._build_system_prompt(state.task), pinned=True
            )
            await self._working_memory.append("user", state.task)

        # A run that reached FAILED still has somewhere to go: max_steps and an
        # unparseable response both leave working memory intact and are resumed
        # by thinking again, typically against a raised limit.
        if state.phase is Phase.FAILED and state.error_kind in (
            ErrorKind.MAX_STEPS,
            ErrorKind.UNPARSEABLE_THINK,
        ):
            state.phase = Phase.THINK
            state.error = None
            state.error_kind = None

        # Spending already recorded counts against this run, so a resumed run
        # cannot quietly start the budget over. The exception is a run that
        # failed *because* of the budget: restoring it there would re-raise at
        # the same point and make resume a no-op loop, so the operator's raised
        # GuardrailConfig is allowed to take effect instead.
        if state.budget and state.error_kind is not ErrorKind.BUDGET:
            if hasattr(self._guard, "restore"):
                self._guard.restore(state.budget)

        yield BusEvent.resumed(
            self.config.agent_id,
            step=state.step,
            phase=state.phase.value,
            actions=[a.tool for a in state.actions],
        )

        source_cm = (
            self._steering_source_factory(self)
            if self._steering_source_factory is not None
            else contextlib.nullcontext()
        )
        async with source_cm:
            async with _ResumeHint(
                self._resume_key,
                self._checkpoint_store,
                f"Agent {self.config.agent_id}",
                check_key=self._ckp_id,
            ) as hint:
                async for event in self._run_stream_internal(state):
                    # See ``run_stream`` for why both branches gate on
                    # ``not event.parent_agent_id`` (sub-agent terminals are
                    # not terminal for the outer) and why a top-level ERROR
                    # marks ``done`` without clearing the checkpoint
                    # (it's "ran-but-failed", not an interrupt).
                    if not event.parent_agent_id:
                        if event.type == EventType.TASK_DONE:
                            await self._clear_checkpoint(run_id)
                            hint.done = True
                        elif event.type == EventType.ERROR:
                            hint.done = True
                    yield event

    async def _run_stream_internal(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        try:
            async for event in self._drive(state):
                yield event
        except Exception as e:
            logger.exception("Agent %s stream crashed", self.config.agent_id)
            state.phase = Phase.FAILED
            state.error = str(e)
            state.error_kind = ErrorKind.CRASH
            yield BusEvent.error_event(self.config.agent_id, error=str(e))
        finally:
            if self._working_memory is not None:
                self._tracer.log(
                    "trajectory",
                    self.config.agent_id,
                    {
                        "run_id": state.run_id,
                        "messages": self._working_memory.get_messages(),
                        "summarization_count": self._working_memory.summarization_count,
                    },
                )

    # ── Blocking entry point (thin drain) ─────────────────────────────────────

    async def run(self, task: str, run_id: str | None = None) -> dict:
        result: dict = {}
        last_step = 0  # tracked from ACTION events so ERROR can report meaningful steps
        async for event in self.run_stream(task=task, run_id=run_id):
            if event.type == EventType.TASK_DONE:
                result = event.payload
            elif event.type == EventType.ACTION:
                last_step = event.payload.get("step", last_step) + 1
            elif event.type == EventType.ERROR:
                steps = event.payload.get("steps", last_step) if event.payload else last_step
                result = self._error_result(event.error, steps=steps)
        return result

    # ── System Prompt ─────────────────────────────────────────────────────────

    async def _build_system_prompt(
        self,
        task: str,
        *,
        precomputed_memory_context: Any = None,
    ) -> str:
        """Build the system prompt.

        When ``precomputed_memory_context`` is the sentinel ``"_skip_"``
        (passed by ``PersistentAgent``), the live ``build_context`` lookup
        is skipped entirely — memory context is placed in user-message
        priors instead so the system prompt stays byte-stable across
        turns. Otherwise (the default for one-shot dispatch via
        AgentRuntime / Orchestrator), memory is fetched + rendered inline
        as before.
        """
        parts = [self.config.system_prompt]

        if self.config.skills:
            rendered_skills = "\n\n".join(skill.render() for skill in self.config.skills)
            if rendered_skills:
                parts.append("## Skills\n" + rendered_skills)

        memory_in_system_prompt = (
            self.config.memory_context_enabled and precomputed_memory_context != "_skip_"
        )
        if memory_in_system_prompt:
            mem_context = await self._memory.build_context(
                goal=task,
                agent_id=self.config.agent_id,
            )
            if not mem_context.is_empty():
                rendered = mem_context.render()
                if os.environ.get("DEBUG_MEMORY_CONTEXT") == "1":
                    print(f"\n[debug:memory] context injected for {self.config.agent_id}")
                    print("─" * 64)
                    print(rendered)
                    print("─" * 64)
                parts.append(rendered)
            elif os.environ.get("DEBUG_MEMORY_CONTEXT") == "1":
                print(f"\n[debug:memory] context injected for {self.config.agent_id}: (empty)")

        tool_list = ", ".join(self._tools.keys()) or "none"
        parts.append(REACT_FORMAT.replace("__TOOL_LIST__", tool_list))
        return "\n\n".join(parts)

    # ── ReAct Loop (stream) ───────────────────────────────────────────────────

    def _state(
        self,
        run_id: str,
        step: int,
        *,
        phase: Phase = Phase.THINK,
        assistant_message: str | None = None,
        actions: list[ActionState] | None = None,
    ) -> RunState:
        """Snapshot the agent's position as a :class:`RunState`.

        One builder for every checkpoint writer, so the stored shape cannot
        drift between the step write, the pre-approval write and the
        post-tool write — which is how the previous format grew a ``pending``
        key that only one of the three knew about.
        """
        return RunState(
            run_id=run_id,
            agent_id=self.config.agent_id,
            task=self._task,
            memory=self._working_memory.to_dict(),
            step=step,
            phase=phase,
            assistant_message=assistant_message,
            actions=actions or [],
            budget=self._guard.snapshot() if hasattr(self._guard, "snapshot") else None,
        )

    async def _write_step_checkpoint(self, run_id: str, step: int) -> None:
        if self._checkpoint_store is None or not self._checkpoint_resume_enabled:
            return
        await self._checkpoint_store.write(self._ckp_id, self._state(run_id, step).to_dict())

    # ── ReAct state machine ───────────────────────────────────────────────────
    #
    # One ReAct step walks THINK → APPROVE → ACT → OBSERVE and back to THINK.
    # Each phase performs its side effects, records them on the state, sets the
    # next phase, persists, and only then yields the events describing what it
    # did. That order matters: an async generator abandoned by its consumer has
    # GeneratorExit raised at the yield, so anything awaited afterwards never
    # runs. Persisting after yielding would leave a checkpoint claiming work is
    # still pending when it has already happened, and the next resume would
    # repeat its side effects.

    async def _drive(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        """Advance a run until it reaches a terminal phase.

        Starting a run and resuming one are the same operation — build a fresh
        state or load a stored one, then drive it. There is no separate resume
        implementation to fall out of step with this one.
        """
        while not state.is_terminal:
            async for event in self._advance(state):
                yield event

    async def _advance(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        """Run exactly one phase."""
        handlers = {
            Phase.THINK: self._phase_think,
            Phase.APPROVE: self._phase_approve,
            Phase.ACT: self._phase_act,
            Phase.OBSERVE: self._phase_observe,
        }
        async for event in handlers[state.phase](state):
            yield event

    def _fail(self, state: RunState, error: str, kind: ErrorKind, *, steps: int | None = None):
        """Move to FAILED and build the ERROR event describing why."""
        state.phase = Phase.FAILED
        state.error = error
        state.error_kind = kind
        self._tracer.log(
            "task_result",
            self.config.agent_id,
            {
                "answer": "",
                "confidence": 0.0,
                "steps": steps if steps is not None else state.step,
                "error": error,
            },
        )
        if steps is not None:
            return BusEvent.error_event(self.config.agent_id, error=error, steps=steps)
        return BusEvent.error_event(self.config.agent_id, error=error)

    async def _persist(self, state: RunState) -> None:
        if self._checkpoint_store is None or not self._checkpoint_resume_enabled:
            return
        state.memory = self._working_memory.to_dict()
        state.budget = self._guard.snapshot() if hasattr(self._guard, "snapshot") else None
        await self._checkpoint_store.write(self._ckp_id, state.to_dict())

    # ── THINK ─────────────────────────────────────────────────────────────────

    async def _phase_think(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        if state.step >= self.config.max_steps:
            yield self._fail(
                state,
                f"Max steps ({self.config.max_steps}) reached",
                ErrorKind.MAX_STEPS,
                steps=self.config.max_steps,
            )
            return

        try:
            self._guard.check()
        except RuntimeError as e:
            yield self._fail(state, str(e), ErrorKind.BUDGET)
            return

        # Drain steering before the checkpoint write so any queued guidance is
        # captured by the persisted working memory.
        async for guidance_event in self._drain_steering(state.step):
            yield guidance_event

        if self.config.checkpoint_every > 0 and state.step % self.config.checkpoint_every == 0:
            await self._persist(state)

        # Think — yields TOKEN events when the LLM client supports streaming.
        response = None
        async for thought_event in self._think_stream():
            if thought_event.type == EventType.THOUGHT:
                response = thought_event.payload.get("response")
            yield thought_event

        if response is None:
            yield self._fail(
                state,
                self._last_think_error or "LLM returned unparseable response",
                ErrorKind.UNPARSEABLE_THINK,
            )
            return

        self._tracer.log(
            "thought",
            self.config.agent_id,
            {
                "step": state.step,
                "thought": response.get("thought", ""),
                "action": response.get("action"),
            },
        )

        if response.get("action") == "finish":
            async for event in self._finish(state, response):
                yield event
            return

        state.assistant_message = _assistant_message(response)
        state.actions, state.parallel = self._actions_for(response)
        state.phase = Phase.APPROVE

    async def _finish(self, state: RunState, response: dict) -> AsyncGenerator[BusEvent, None]:
        await self._working_memory.append("assistant", _assistant_message(response))
        result = {
            "agent_id": self.config.agent_id,
            "answer": response.get("answer", ""),
            "confidence": (
                response.get("confidence", 1.0) if self.config.confidence_from_llm else 1.0
            ),
            "steps": state.step + 1,
            "metadata": {"summarizations": self._working_memory.summarization_count},
        }
        # Attach the current budget snapshot so dispatch_stream consumers can
        # read totals + per-call-site breakdown off the routed path's terminal
        # event, same shape as the orchestrator's DONE event.
        if self._guard is not None and hasattr(self._guard, "snapshot"):
            result["budget"] = self._guard.snapshot()
        logger.info(
            "Agent %s completed: steps=%d confidence=%.2f summarizations=%d",
            self.config.agent_id,
            result["steps"],
            result["confidence"],
            self._working_memory.summarization_count,
        )
        self._tracer.log(
            "task_result",
            self.config.agent_id,
            {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "steps": result["steps"],
                "error": "",
            },
        )
        state.result = result
        state.phase = Phase.DONE
        yield BusEvent.task_done_agent(self.config.agent_id, result=result)

    def _actions_for(self, response: dict) -> tuple[list[ActionState], bool]:
        """Turn one model response into the actions this step will run.

        Single and parallel calls become the same list, so every later phase
        has one shape to handle. ``args`` is deep-copied here: it used to alias
        into the response dict that was also being persisted, so a tool that
        mutated its own kwargs retroactively changed what the record said the
        human had approved.

        Delegations that would exceed the sub-agent depth limit are refused
        *here*, before anyone is asked to approve them — there is no point
        prompting a human for a call that cannot run.
        """
        raw = response.get("actions")
        parallel = bool(raw) and isinstance(raw, list)
        specs = (
            [(a.get("tool", ""), a.get("args", {})) for a in raw]
            if parallel
            else [(response.get("action", ""), response.get("args", {}))]
        )

        actions: list[ActionState] = []
        for tool_name, args in specs:
            action = ActionState(tool=tool_name, args=copy.deepcopy(args or {}))
            refusal = self._delegation_refusal(tool_name)
            if refusal is not None:
                action.status = ActionStatus.EXECUTED
                action.observation = refusal
            actions.append(action)
        return actions, parallel

    def _delegation_refusal(self, tool_name: str) -> str | None:
        """Refusal text if delegating to this tool would exceed the depth cap."""
        from tools.builtin.subagent import SubAgentTool

        tool = self._tools.get(tool_name)
        if not isinstance(tool, SubAgentTool):
            return None
        if self._subagent_depth + 1 <= self.config.max_subagent_depth:
            return None
        return (
            f"Refused to delegate to {tool.name!r}: "
            f"max sub-agent depth {self.config.max_subagent_depth} "
            f"would be exceeded (current depth {self._subagent_depth})."
        )

    # ── APPROVE ───────────────────────────────────────────────────────────────

    async def _phase_approve(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        """Resolve the human gate for each action that still needs one.

        Actions are resolved one at a time and the decision is recorded on the
        state before the next prompt, so a run that dies partway through a
        batch resumes with the earlier answers intact instead of asking again
        or, worse, dropping them.
        """
        for action in state.actions:
            if action.status is not ActionStatus.PENDING:
                continue

            yield BusEvent.action(
                self.config.agent_id, step=state.step, tool=action.tool, args=action.args
            )

            approval = await self._gate_action(state, action)
            if approval is not None and approval.correction:
                # The human redirected instead of answering. Drop the batch,
                # feed the correction back, and think again.
                await self._inject_human_guidance(
                    state.assistant_message or "{}",
                    approval.correction,
                    state.run_id,
                    state.step,
                )
                state.actions = []
                state.assistant_message = None
                state.parallel = False
                state.step += 1
                state.phase = Phase.THINK
                return

            action.status = (
                ActionStatus.APPROVED
                if approval is None or approval.approved
                else ActionStatus.REJECTED
            )

        state.phase = Phase.ACT

    # ── ACT ───────────────────────────────────────────────────────────────────

    async def _phase_act(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        """Run every approved action.

        A step is executed at most once per attempt but not exactly once: no
        tool here carries an idempotency key, so an action approved and started
        before a crash is re-run on resume. ``ActionState.attempts`` records
        that so a crash loop is visible rather than silent.
        """
        for action in state.actions:
            if action.status is ActionStatus.REJECTED:
                action.observation = "Tool rejected by human: no reason given"
                action.status = ActionStatus.EXECUTED

        runnable = [a for a in state.actions if a.status is ActionStatus.APPROVED]
        if not runnable:
            state.phase = Phase.OBSERVE
            return

        gated = any(a.approval_id for a in state.actions)
        if gated:
            await self._persist(state)

        async for event in self._execute_actions(runnable):
            yield event

        if gated:
            await self._persist(state)
        state.phase = Phase.OBSERVE

    async def _execute_actions(self, actions: list[ActionState]) -> AsyncGenerator[BusEvent, None]:
        """Execute actions concurrently, bubbling any streaming tool's events.

        Mixed batches are supported: streaming tools push their events through
        the fan-in helper in arrival order while plain awaitables resolve under
        a single gather.
        """
        streaming: list[tuple[ActionState, Any]] = []
        plain: list[ActionState] = []
        plain_tasks: list[Any] = []

        for action in actions:
            action.attempts += 1
            tool = self._tools.get(action.tool)
            if tool is not None and hasattr(tool, "execute_stream"):
                self._prepare_delegation(tool)
                streaming.append((action, _freeze_factory(tool, action.args)))
            else:
                plain.append(action)
                plain_tasks.append(self._execute_tool(action.tool, action.args))

        if streaming:
            from harness.streaming import fan_in

            # ``asyncio.gather`` schedules its argument coroutines immediately,
            # so the plain tasks start running on this line — no extra
            # ``create_task`` wrapper needed (and modern ``create_task``
            # rejects gather's Future return value with TypeError).
            plain_future = asyncio.gather(*plain_tasks) if plain_tasks else None
            try:
                async for idx, item in fan_in([factory for _, factory in streaming]):
                    if isinstance(item, BusEvent):
                        yield item
                    else:
                        # Streaming tools yield exactly one non-BusEvent
                        # terminal value — the observation the parent records.
                        streaming[idx][0].observation = item
                plain_results = await plain_future if plain_future is not None else []
            except Exception:
                if plain_future is not None:
                    plain_future.cancel()
                raise
        else:
            plain_results = await asyncio.gather(*plain_tasks) if plain_tasks else []

        for action, value in zip(plain, plain_results, strict=False):
            action.observation = value
        for action in actions:
            action.status = ActionStatus.EXECUTED

    def _prepare_delegation(self, tool: Any) -> None:
        """Wire a sub-agent tool to this agent before it runs."""
        from tools.builtin.subagent import SubAgentTool

        if not isinstance(tool, SubAgentTool):
            return
        tool._agent._subagent_depth = self._subagent_depth + 1
        # Share the parent's guard so the sub-agent's check() enforces the
        # run-level budget and its bubbled TASK_DONE snapshot reflects real
        # token usage. Without this, sub-agents track an empty local guard
        # while the LLM reports tokens to the runtime's guard.
        tool._agent._guard = self._guard
        # Tell the tool who's invoking so its bubbled events carry the actual
        # parent's id in ``parent_agent_id``. Without this the tool defaults to
        # its own sub-agent id, which makes ``agent_id == parent_agent_id`` for
        # the immediate sub — useless to renderers that want indentation.
        tool._invoking_agent_id = self.config.agent_id

    # ── OBSERVE ───────────────────────────────────────────────────────────────

    async def _phase_observe(self, state: RunState) -> AsyncGenerator[BusEvent, None]:
        """Record what the tools returned, then advance to the next step."""
        combined: list[dict] = []
        image_blocks: list[tuple[str, Any]] = []
        observation_events: list[BusEvent] = []

        for i, action in enumerate(state.actions):
            obs = action.observation
            is_img = _is_image_block(obs)
            obs_raw = "[image]" if is_img else str(obs)
            obs_display = obs_raw if is_img else obs_raw[:500]
            self._tracer.log(
                "action",
                self.config.agent_id,
                {
                    "step": state.step,
                    "tool": action.tool,
                    "args": action.args,
                    "observation": obs_display,
                },
            )
            observation_events.append(
                BusEvent.observation(
                    self.config.agent_id,
                    step=state.step,
                    tool=action.tool,
                    observation=obs_raw if state.parallel else obs_display,
                )
            )
            combined.append({"tool": action.tool, "result": obs_raw})
            if is_img:
                image_blocks.append((action.tool, obs))
            if obs and not isinstance(obs, str) and not is_img:
                key = (
                    f"step_{state.step}_{i}_{action.tool}"
                    if state.parallel
                    else f"step_{state.step}_{action.tool}"
                )
                fire(
                    self._memory.write_working_fact(
                        run_id=state.run_id,
                        agent_id=self.config.agent_id,
                        key=key,
                        value=obs,
                    )
                )

        message = state.assistant_message or "{}"
        if state.parallel:
            await self._record_parallel_observations(message, combined, image_blocks)
        elif state.actions:
            action = state.actions[0]
            await self._record_tool_observation(message, action.tool, action.observation)

        gated = any(a.approval_id for a in state.actions)
        state.step += 1
        state.actions = []
        state.assistant_message = None
        state.parallel = False
        state.phase = Phase.THINK
        if gated:
            await self._persist(state)

        for event in observation_events:
            yield event

    # ── Think ─────────────────────────────────────────────────────────────────

    async def _think_stream(self) -> AsyncGenerator[BusEvent, None]:
        """
        Streaming think: if the LLM client has `stream_complete`, forwards
        TOKEN events as text arrives, then parses the accumulated response
        into the action JSON and yields it as a THOUGHT event. Otherwise
        falls back to one `complete` call.
        """
        # Working memory stores the system prompt as a role="system"
        # entry for uniform summarisation + token accounting; split it
        # back out here so Anthropic's top-level ``system=`` contract is
        # honoured. OpenAI's adapter re-injects it inline when needed.
        # See ``_split_system`` docstring for the full rationale.
        system_text, messages = _split_system(self._working_memory.get_messages())

        # The ReAct loop should call the LLM only after a user task or
        # user observation. If working memory ends with an assistant
        # message, log the invalid shape, but do not fabricate a user
        # turn. Synthetic cues such as "Continue." hide the missing
        # observation and can make the model continue from the wrong
        # state.
        if messages and messages[-1].get("role") == "assistant":
            logger.warning(
                "Agent %s: messages end with assistant before LLM call; "
                "leaving messages unchanged. role_sequence=%r",
                self.config.agent_id,
                [m.get("role") for m in messages],
            )

        accumulated = ""
        before_usage = self._working_memory.context_usage()
        before_summarizations = self._working_memory.summarization_count

        yield BusEvent(
            type=EventType.CONTEXT,
            agent_id=self.config.agent_id,
            payload=before_usage,
        )

        # Tag ReAct spending so it shows up in BudgetGuard.breakdown alongside
        # classifier/router/planner/synthesizer. Per-agent attribution makes
        # multi-agent demos surface which specialist agent actually drove the
        # bulk of token usage.
        react_source = f"agent:{self.config.agent_id}"
        try:
            if hasattr(self._llm, "stream_complete"):
                # Pass response_format on the streaming path too — without it,
                # OpenAI's JSON mode is off and the model can drift into
                # prose, which then fails _parse_action_json. Adapters that
                # don't take the kwarg (older custom stubs) get it via
                # ``**kwargs`` and ignore it.
                async for token in self._llm.stream_complete(
                    system=system_text,
                    messages=messages,
                    source=react_source,
                    response_format={"type": "json_object"},
                    **self._reasoning_options(),
                ):
                    accumulated += token
                    if self.config.stream_tokens:
                        yield BusEvent.token_event(self.config.agent_id, token=token)
                response = _normalize_response(accumulated)
                if response is None:
                    response = await self._retry_complete_after_bad_stream(
                        system_text=system_text,
                        messages=messages,
                        react_source=react_source,
                        accumulated=accumulated,
                    )
            else:
                raw = await self._llm.complete(
                    system=system_text,
                    messages=messages,
                    response_format={"type": "json_object"},
                    source=react_source,
                    **self._reasoning_options(),
                )
                response = _normalize_response(raw)
                if response is None:
                    logger.warning(
                        "Agent %s got unparseable response: %r",
                        self.config.agent_id,
                        raw,
                    )
                    self._last_think_error = f"Unparseable response: {str(raw)[:300]}"
        except Exception as e:
            logger.error("Agent %s think failed: %s", self.config.agent_id, e)
            response = None
            self._last_think_error = str(e)
        else:
            if response is not None:
                self._last_think_error = None

        after_usage = self._working_memory.context_usage()
        if self._working_memory.summarization_count > before_summarizations:
            yield BusEvent.memory(
                self.config.agent_id,
                before=before_usage,
                after=after_usage,
                summarizations=self._working_memory.summarization_count,
            )
        llm_usage = getattr(self._llm, "last_usage", None) or {}
        if llm_usage or after_usage != before_usage:
            yield BusEvent.context_usage(
                self.config.agent_id,
                usage=after_usage,
                tokens_in=llm_usage.get("tokens_in"),
                tokens_out=llm_usage.get("tokens_out"),
                cache_read_tokens=llm_usage.get("cache_read_tokens"),
                cache_creation_tokens=llm_usage.get("cache_creation_tokens"),
            )

        yield BusEvent.thought(self.config.agent_id, response=response)

    async def _retry_complete_after_bad_stream(
        self,
        *,
        system_text: str | None,
        messages: list[dict],
        react_source: str,
        accumulated: str,
    ) -> dict | None:
        """Retry once non-streaming when streamed JSON is truncated/malformed."""
        logger.warning(
            "Agent %s stream got unparseable response, retrying non-streaming: %r",
            self.config.agent_id,
            accumulated[:300],
        )
        try:
            raw = await self._llm.complete(
                system=system_text,
                messages=[
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous streamed JSON was incomplete or malformed. "
                            "Return one complete valid ReAct JSON object now. Keep "
                            "`thought` under 25 words."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                source=react_source,
                **self._reasoning_options(),
            )
            response = _normalize_response(raw)
        except Exception as e:
            logger.error("Agent %s retry after bad stream failed: %s", self.config.agent_id, e)
            response = None
            self._last_think_error = str(e)
        else:
            if response is None:
                self._last_think_error = f"Unparseable stream response: {accumulated[:300]}"
        return response

    def _reasoning_options(self) -> dict[str, ReasoningEffort]:
        if self.config.reasoning_effort is None:
            return {}
        return {"reasoning_effort": self.config.reasoning_effort}

    async def _record_tool_observation(
        self,
        assistant_message: str,
        tool_name: str,
        observation: Any,
    ) -> None:
        """Record one ReAct action and its observation in working memory.

        Takes the already-serialized assistant message rather than the parsed
        response dict, so the bytes written here are the same bytes a
        checkpoint holds. Re-serializing a dict that has been through a JSON
        round-trip can reorder keys, which changes the prompt prefix and
        silently costs a provider-side cache hit on a resumed run.
        """
        await self._working_memory.append("assistant", assistant_message)
        if _is_image_block(observation):
            await self._working_memory.append(
                "user",
                [
                    {"type": "text", "text": f"Observation ({tool_name}):"},
                    observation,
                ],
            )
            return
        obs_text = (
            json.dumps(observation, default=str)
            if not isinstance(observation, str)
            else observation
        )
        obs_text = _format_model_observation(
            obs_text,
            max_chars=self.config.max_observation_chars,
        )
        await self._working_memory.append("user", f"Observation: {obs_text}")

    async def _record_parallel_observations(
        self,
        assistant_message: str,
        combined: list[dict],
        image_blocks: list[tuple[str, Any]],
    ) -> None:
        """Record one parallel ReAct action batch and its observations.

        See :meth:`_record_tool_observation` on why this takes the serialized
        message rather than the response dict.
        """
        await self._working_memory.append("assistant", assistant_message)
        if image_blocks:
            content: list = [
                {
                    "type": "text",
                    "text": "Observations:\n"
                    + _format_model_observation(
                        json.dumps(combined, default=str),
                        max_chars=self.config.max_observation_chars,
                    ),
                }
            ]
            for tool_name_img, img_block in image_blocks:
                content.append({"type": "text", "text": f"\nImage from {tool_name_img}:"})
                content.append(img_block)
            await self._working_memory.append("user", content)
            return
        await self._working_memory.append(
            "user",
            "Observations:\n"
            + _format_model_observation(
                json.dumps(combined, default=str),
                max_chars=self.config.max_observation_chars,
            ),
        )

    # ── Tool Execution ────────────────────────────────────────────────────────

    async def _execute_tool(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            return (
                f"Error: tool '{name}' not available. Available tools: {list(self._tools.keys())}"
            )
        tool = self._tools[name]

        # Per-run memoization, gated by both agent opt-in AND tool consent.
        # Tools that have side effects or time-dependent output can veto
        # caching by setting ``cacheable = False`` on the instance. Errors
        # are NOT cached — a transient failure should not poison the rest
        # of the run.
        cache_key: tuple[str, str] | None = None
        if self._tool_cache is not None and getattr(tool, "cacheable", True) is True:
            try:
                cache_key = (name, json.dumps(args, sort_keys=True, default=str))
            except (TypeError, ValueError):
                cache_key = None  # un-serialisable args — silently skip
            if cache_key is not None and cache_key in self._tool_cache:
                return self._tool_cache[cache_key]

        try:
            result = await tool.execute(**args)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return f"Tool error ({name}): {e}"

        if cache_key is not None and self._tool_cache is not None:
            self._tool_cache[cache_key] = result
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _error_result(self, reason: str, steps: int) -> dict:
        return {
            "agent_id": self.config.agent_id,
            "answer": "",
            "confidence": 0.0,
            "steps": steps,
            "error": reason,
            "metadata": {},
        }

    async def _gate_action(self, state: RunState, action: ActionState):
        """
        Run the HITL approval gate for one action.

        Returns ApprovalResponse if the action is gated, None if not.

        Durability is deliberately *not* a precondition for gating. This used
        to skip the prompt entirely when no checkpoint store was configured,
        which meant a tool the operator had explicitly listed in
        ``hitl_tools`` ran unsupervised. An approval gate must fail closed:
        with no store we still prompt, we just cannot offer crash-resume.
        """
        if action.tool not in self.config.hitl_tools:
            return None

        from harness.hitl import ApprovalRequest, is_allowed, request_approval

        if is_allowed(action.tool, action.args):
            return None  # fast-path: human already allowed this tool/prefix

        action.approval_id = action.approval_id or str(uuid.uuid4())
        # Persist the whole state — every action, with the decisions already
        # made — *before* blocking on the human. A crash at this prompt used
        # to resume knowing about only the last action gated, silently
        # dropping the rest of the batch.
        await self._persist(state)

        return await request_approval(
            ApprovalRequest(
                approval_id=action.approval_id,
                run_id=self._resume_key,  # standalone: ckp_id; orchestrated: outer run_id
                agent_id=self.config.agent_id,
                tool=action.tool,
                args=action.args,
                step=state.step,
                timestamp=datetime.now(timezone.utc).isoformat(),
                resume_hint=self._hitl_resume_hint if not self._checkpoint_resume_enabled else None,
            ),
            self._guard,
        )

    async def _inject_human_guidance(
        self, assistant_message: str, correction: str, run_id: str, step: int
    ) -> None:
        """Append human correction to WorkingMemory and commit a clean checkpoint."""
        await self._working_memory.append("assistant", assistant_message)
        await self._working_memory.append("user", f"Human guidance: {correction}")
        await self._commit_checkpoint(run_id, step)

    async def _commit_checkpoint(self, run_id: str, step: int) -> None:
        """Overwrite checkpoint with current state (no pending field).

        Called after HITL resolves or a tool completes so the stored state
        always reflects reality — no stale 'pending' approval marker, and
        the step position is preserved for crash-resume.

        Honours ``_checkpoint_resume_enabled`` like the other two writers.
        Without that check this wrote during PersistentAgent turns, which
        opt out of crash-resume; the write only stayed invisible because
        ``run_stream`` deletes the checkpoint on TASK_DONE, so a crash
        mid-turn left an orphan behind.
        """
        if self._checkpoint_store is None or not self._checkpoint_resume_enabled:
            return
        await self._checkpoint_store.write(self._ckp_id, self._state(run_id, step).to_dict())

    async def _clear_checkpoint(self, run_id: str) -> None:
        if self._checkpoint_store:
            await self._checkpoint_store.delete(self._ckp_id)


# ── LLM call shaping (module-level for testability) ──────────────────────────


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull system-role entries out of a messages list and join them.

    Returns ``(system_text, non_system_messages)``. ``system_text`` is the
    concatenation of every ``role == "system"`` entry's ``content``
    (joined with a blank line) or ``None`` when no system entries are
    present.

    Why this exists
    ---------------
    ``BaseAgent`` keeps the system prompt inside ``WorkingMemory`` (as a
    pinned ``role="system"`` entry) so summarisation, token accounting,
    and checkpoint serialisation treat it uniformly with every other
    message. But the two LLM adapter contracts diverge at the wire:

    - **OpenAI** accepts ``role="system"`` entries *inside* the messages
      array — passing ``system=None`` + an inline system entry works.
    - **Anthropic** requires the system prompt as a *top-level* ``system=``
      parameter and ``_build_messages`` silently drops any
      ``role="system"`` entries in the messages list. Passing
      ``system=None`` + an inline system entry causes the entire system
      prompt to be discarded — the model sees only the user turn.

    Splitting at the call boundary picks up any system entries (including
    those that arrived via ``prior_messages`` priors, not just the one
    BaseAgent appended itself), produces one joined system string, and
    leaves the rest as a clean user/assistant transcript. Both adapter
    contracts are then satisfied identically.
    """
    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        rest.append(m)
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, rest


# ── Tool observation shaping ─────────────────────────────────────────────────


def _format_model_observation(text: str, *, max_chars: int) -> str:
    """Return the observation text appended to WorkingMemory.

    The raw tool result is still emitted on the event stream before this
    method is involved. This cap only protects the *next* LLM call from
    oversized observations.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return json.dumps(
        {
            "truncated": True,
            "original_chars": len(text),
            "shown_chars": max_chars,
            "omitted_chars": omitted,
            "content": text[:max_chars],
            "note": "Tool observation was capped before the next LLM call.",
        },
        ensure_ascii=False,
    )


# ── Response normalization (module-level for testability) ────────────────────


def _normalize_response(response: Any) -> dict | None:
    if isinstance(response, dict) and "text" not in response:
        return response if _is_valid_react_response(response) else None
    if isinstance(response, dict) and "text" in response:
        text = response["text"].strip()
    elif isinstance(response, str):
        text = response.strip()
    else:
        text = str(response).strip()
    parsed = _parse_action_json(text)
    return parsed if _is_valid_react_response(parsed) else None


def _is_valid_react_response(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    action = response.get("action")
    if isinstance(action, str) and action.strip():
        return True
    actions = response.get("actions")
    if isinstance(actions, list) and actions:
        return all(
            isinstance(item, dict)
            and isinstance(item.get("tool"), str)
            and bool(item.get("tool", "").strip())
            for item in actions
        )
    return False


def _is_image_block(obs: Any) -> bool:
    """True when a tool observation is an OpenAI-style image content block."""
    return isinstance(obs, dict) and obs.get("type") in ("image_url", "image")


def _parse_action_json(text: str) -> dict | None:
    """Extract and parse the first parseable JSON object in text.

    Scans forward through every '{' so that a malformed preamble (e.g. a
    thought with an unescaped newline) doesn't block the valid action object
    that follows it.
    """
    text = text.strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    pos = 0
    while (start := text.find("{", pos)) >= 0:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        pos = start + 1

    return None
