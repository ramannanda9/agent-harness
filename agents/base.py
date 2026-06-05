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
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

from harness.checkpoint import _ResumeHint
from harness.events import BusEvent, EventType
from harness.utils import fire
from memory.manager import MemoryManager
from memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# Sentinel returned by _run_tool_gated when human injects a correction.
# Caller must `continue` the ReAct loop — WM is already updated.
_HITL_CORRECTION: Final = object()


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
    working_memory_max_tokens: int = 8000  # WorkingMemory eviction threshold; tune per agent
    hitl_tools: list[str] = None  # tools requiring human approval; None = no HITL
    checkpoint_every: int = 0  # write a resumable checkpoint every N steps; 0 = disabled
    # Cache tool results within a single run, keyed by (tool_name, args).
    # Opt-in because not every tool is idempotent — a tool may also veto
    # caching for itself by exposing ``cacheable = False`` on its instance.
    # Designed for read-mostly multi-agent runs where agents redo each
    # other's lookups (HTTPFetch on stable URLs, ``kubectl get …`` style
    # discovery, MCP filesystem reads).
    cache_tool_results: bool = False
    # Hard cap on how deep a SubAgentTool chain may recurse. Depth 0 = the
    # top-level agent invoked by AgentRuntime; depth 1 = a sub-agent
    # delegated to from the top; depth 2 = a sub-agent that itself
    # delegated. The default is conservative — most production setups want
    # one or two levels and a hard stop against an LLM hallucinating an
    # infinite delegation chain.
    max_subagent_depth: int = 3

    def __post_init__(self):
        if self.hitl_tools is None:
            self.hitl_tools = []


# ── ReAct Response Schema ─────────────────────────────────────────────────────

# Injected into every agent's system prompt so LLM knows the expected format.
REACT_FORMAT = """
At each step, respond with a JSON object in one of three forms:

To use a single tool:
{
  "thought": "<reasoning about what to do next>",
  "action": "<tool_name>",
  "args": { "<arg>": "<value>", ... }
}

To use multiple independent tools at once (they run in parallel — use this when \
the calls don't depend on each other):
{
  "thought": "<reasoning>",
  "actions": [
    {"tool": "<tool_name>", "args": { "<arg>": "<value>", ... }},
    {"tool": "<tool_name_2>", "args": { "<arg>": "<value>", ... }}
  ]
}

To finish:
{
  "thought": "<final reasoning>",
  "action": "finish",
  "answer": "<comprehensive answer to the task>",
  "confidence": <0.0-1.0>
}

Available tools: __TOOL_LIST__
Return JSON only — no markdown, no preamble.
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
            yield BusEvent(
                type=EventType.HUMAN_GUIDANCE,
                agent_id=self.config.agent_id,
                payload={"step": step, "text": text},
            )

    # ── Streaming entry point (canonical) ─────────────────────────────────────

    async def run_stream(
        self,
        task: str,
        run_id: str | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        run_id = run_id or str(uuid.uuid4())
        self._ckp_id = f"{run_id}:{self.config.agent_id}"
        if not self._resume_key:
            self._resume_key = self._ckp_id
        self._task = task
        self._working_memory = WorkingMemory(
            llm=self._llm,
            max_tokens=self.config.working_memory_max_tokens,
        )

        system = await self._build_system_prompt(task)
        await self._working_memory.append("system", system, pinned=True)
        await self._working_memory.append("user", task)

        # Steering source is owned by the agent for the duration of the run.
        # nullcontext when no factory is configured — zero overhead.
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
                async for event in self._run_stream_internal(run_id):
                    if event.type == EventType.TASK_DONE:
                        await self._clear_checkpoint(run_id)
                        hint.done = True
                    yield event

    async def _resume_stream(
        self,
        run_id: str,
        start_step: int,
        pending: dict | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        """
        Re-enter the ReAct loop from a checkpoint.

        If pending is set, the last step was interrupted mid-approval.
        The approval prompt is shown again; once the human responds the
        tool runs (or the correction is injected) before the loop continues.
        """
        self._ckp_id = f"{run_id}:{self.config.agent_id}"
        if not self._resume_key:
            self._resume_key = self._ckp_id
        if pending:
            async for event in self._replay_pending_step(run_id, pending):
                yield event
            start_step = pending["step"] + 1

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
                async for event in self._run_stream_internal(run_id, start_step=start_step):
                    if event.type == EventType.TASK_DONE:
                        await self._clear_checkpoint(run_id)
                        hint.done = True
                    yield event

    async def _run_stream_internal(
        self,
        run_id: str,
        start_step: int = 0,
    ) -> AsyncGenerator[BusEvent, None]:
        try:
            async for event in self._react_stream(run_id, start_step=start_step):
                yield event
        except Exception as e:
            logger.exception("Agent %s stream crashed", self.config.agent_id)
            yield BusEvent(
                type=EventType.ERROR,
                agent_id=self.config.agent_id,
                error=str(e),
            )
        finally:
            if self._working_memory is not None:
                self._tracer.log(
                    "trajectory",
                    self.config.agent_id,
                    {
                        "run_id": run_id,
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

    async def _build_system_prompt(self, task: str) -> str:
        parts = [self.config.system_prompt]

        if self.config.memory_context_enabled:
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

    async def _write_step_checkpoint(self, run_id: str, step: int) -> None:
        if self._checkpoint_store is None:
            return
        await self._checkpoint_store.write(
            self._ckp_id,
            {
                "run_id": run_id,
                "agent_id": self.config.agent_id,
                "task": self._task,
                "step": step,
                "memory": self._working_memory.to_dict(),
            },
        )

    async def _react_stream(
        self, run_id: str, start_step: int = 0
    ) -> AsyncGenerator[BusEvent, None]:
        for step in range(start_step, self.config.max_steps):
            self._guard.check()
            # Drain steering queue BEFORE the checkpoint write so any
            # queued guidance is captured by the persisted WM.
            async for guidance_event in self._drain_steering(step):
                yield guidance_event
            if (
                self._checkpoint_store is not None
                and self.config.checkpoint_every > 0
                and step % self.config.checkpoint_every == 0
            ):
                await self._write_step_checkpoint(run_id, step)

            # Think — yields TOKEN events when the LLM client supports streaming.
            response = None
            async for thought_event in self._think_stream():
                if thought_event.type == EventType.TOKEN:
                    yield thought_event
                elif thought_event.type == EventType.THOUGHT:
                    response = thought_event.payload.get("response")
                    yield thought_event
                else:
                    yield thought_event

            if response is None:
                reason = self._last_think_error or "LLM returned unparseable response"
                self._tracer.log(
                    "task_result",
                    self.config.agent_id,
                    {"answer": "", "confidence": 0.0, "steps": step, "error": reason},
                )
                yield BusEvent(
                    type=EventType.ERROR,
                    agent_id=self.config.agent_id,
                    error=reason,
                )
                return

            self._tracer.log(
                "thought",
                self.config.agent_id,
                {
                    "step": step,
                    "thought": response.get("thought", ""),
                    "action": response.get("action"),
                },
            )

            # Finish?
            if response.get("action") == "finish":
                await self._working_memory.append("assistant", json.dumps(response))
                result = {
                    "agent_id": self.config.agent_id,
                    "answer": response.get("answer", ""),
                    "confidence": (
                        response.get("confidence", 1.0) if self.config.confidence_from_llm else 1.0
                    ),
                    "steps": step + 1,
                    "metadata": {
                        "summarizations": self._working_memory.summarization_count,
                    },
                }
                # Attach the current budget snapshot so dispatch_stream
                # consumers can read totals + per-call-site breakdown off
                # the routed path's terminal event, same shape as the
                # orchestrator's DONE event.
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
                yield BusEvent(
                    type=EventType.TASK_DONE,
                    agent_id=self.config.agent_id,
                    payload=result,
                )
                return

            # Act — parallel or single
            parallel_actions = response.get("actions")
            if parallel_actions and isinstance(parallel_actions, list):
                # Gate each gated tool sequentially before fanning out.
                # Correction from any one tool aborts the whole batch.
                approved: list[dict] = []
                correction_injected = False
                for act in parallel_actions:
                    approval = await self._gate_tool(
                        run_id, step, act.get("tool", ""), act.get("args", {}), response
                    )
                    if approval is None or approval.approved:
                        approved.append(act)
                    elif approval.correction:
                        await self._inject_human_guidance(
                            response, approval.correction, run_id, step
                        )
                        correction_injected = True
                        break
                    # else: rejected — drop from batch silently

                if correction_injected:
                    continue

                parallel_actions = approved

                # Emit ACTION events first so callers see what's being launched.
                for act in parallel_actions:
                    yield BusEvent(
                        type=EventType.ACTION,
                        agent_id=self.config.agent_id,
                        payload={
                            "step": step,
                            "tool": act.get("tool", ""),
                            "args": act.get("args", {}),
                        },
                    )

                # Fan out all approved tool calls concurrently. Mixed
                # batches (some streaming sub-agent tools, some plain) are
                # supported — the fan-in helper bubbles streaming events
                # in arrival order while plain awaitables resolve in
                # parallel under a single gather.
                observations: list[Any] = [None] * len(parallel_actions)
                streaming_indices: list[int] = []
                streaming_factories: list[Any] = []
                plain_indices: list[int] = []
                plain_tasks: list[Any] = []

                for i, act in enumerate(parallel_actions):
                    t_name = act.get("tool", "")
                    t_args = act.get("args", {})
                    t_obj = self._tools.get(t_name)
                    if t_obj is not None and hasattr(t_obj, "execute_stream"):
                        # Recursion guard fires here for the parallel path.
                        # Refused delegations land as their own observation
                        # so the rest of the batch still runs.
                        from tools.builtin.subagent import SubAgentTool

                        if isinstance(t_obj, SubAgentTool):
                            if self._subagent_depth + 1 > self.config.max_subagent_depth:
                                observations[i] = (
                                    f"Refused to delegate to {t_obj.name!r}: "
                                    f"max sub-agent depth "
                                    f"{self.config.max_subagent_depth} would be exceeded."
                                )
                                continue
                            t_obj._agent._subagent_depth = self._subagent_depth + 1
                            # Share the run-level guard — see the matching
                            # comment in ``_run_streaming_tool_gated``.
                            t_obj._agent._guard = self._guard
                        streaming_indices.append(i)
                        streaming_factories.append(_freeze_factory(t_obj, t_args))
                    else:
                        plain_indices.append(i)
                        plain_tasks.append(self._execute_tool(t_name, t_args))

                if streaming_factories:
                    from harness.streaming import fan_in

                    plain_future = (
                        asyncio.create_task(asyncio.gather(*plain_tasks)) if plain_tasks else None
                    )
                    try:
                        async for fan_idx, item in fan_in(streaming_factories):
                            real_idx = streaming_indices[fan_idx]
                            if isinstance(item, BusEvent):
                                yield item
                            else:
                                observations[real_idx] = item
                        plain_results = await plain_future if plain_future is not None else []
                    except Exception:
                        if plain_future is not None:
                            plain_future.cancel()
                        raise
                    for slot, val in zip(plain_indices, plain_results, strict=False):
                        observations[slot] = val
                else:
                    plain_results = await asyncio.gather(*plain_tasks)
                    for slot, val in zip(plain_indices, plain_results, strict=False):
                        observations[slot] = val
                await self._commit_checkpoint(run_id, step)

                combined: list[dict] = []
                for i, (act, obs) in enumerate(zip(parallel_actions, observations, strict=False)):
                    tool_name = act.get("tool", "")
                    tool_args = act.get("args", {})
                    obs_display = "[image]" if _is_image_block(obs) else str(obs)[:500]
                    self._tracer.log(
                        "action",
                        self.config.agent_id,
                        {
                            "step": step,
                            "tool": tool_name,
                            "args": tool_args,
                            "observation": obs_display,
                        },
                    )
                    yield BusEvent(
                        type=EventType.OBSERVATION,
                        agent_id=self.config.agent_id,
                        payload={"step": step, "tool": tool_name, "observation": obs_display},
                    )
                    combined.append({"tool": tool_name, "result": obs_display})
                    if obs and not isinstance(obs, str) and not _is_image_block(obs):
                        fire(
                            self._memory.write_working_fact(
                                run_id=run_id,
                                agent_id=self.config.agent_id,
                                key=f"step_{step}_{i}_{tool_name}",
                                value=obs,
                            )
                        )

                await self._working_memory.append("assistant", json.dumps(response))
                # Inject image observations as content blocks; text observations as a string.
                image_blocks = [
                    (act.get("tool", ""), obs)
                    for act, obs in zip(parallel_actions, observations, strict=False)
                    if _is_image_block(obs)
                ]
                if image_blocks:
                    content: list = [
                        {
                            "type": "text",
                            "text": f"Observations:\n{json.dumps(combined, default=str)}",
                        }
                    ]
                    for tool_name_img, img_block in image_blocks:
                        content.append({"type": "text", "text": f"\nImage from {tool_name_img}:"})
                        content.append(img_block)
                    await self._working_memory.append("user", content)
                else:
                    await self._working_memory.append(
                        "user",
                        f"Observations:\n{json.dumps(combined, default=str)}",
                    )
            else:
                # Single action path.
                tool_name = response.get("action", "")
                tool_args = response.get("args", {})
                yield BusEvent(
                    type=EventType.ACTION,
                    agent_id=self.config.agent_id,
                    payload={"step": step, "tool": tool_name, "args": tool_args},
                )

                tool_obj = self._tools.get(tool_name)
                if tool_obj is not None and hasattr(tool_obj, "execute_stream"):
                    # Streaming tool — bubble its events into the parent stream,
                    # collect the terminal observation. HITL gate + recursion
                    # guard are inside _run_streaming_tool_gated.
                    correction_fired = False
                    observation = None
                    async for kind, value in self._run_streaming_tool_gated(
                        run_id, step, tool_name, tool_args, response, tool_obj
                    ):
                        if kind == "event":
                            yield value
                        elif kind == "correction":
                            correction_fired = True
                        elif kind == "result":
                            observation = value
                    if correction_fired:
                        continue
                else:
                    observation = await self._run_tool_gated(
                        run_id, step, tool_name, tool_args, response
                    )
                    if observation is _HITL_CORRECTION:
                        continue

                obs_display = "[image]" if _is_image_block(observation) else str(observation)[:500]
                self._tracer.log(
                    "action",
                    self.config.agent_id,
                    {
                        "step": step,
                        "tool": tool_name,
                        "args": tool_args,
                        "observation": obs_display,
                    },
                )
                yield BusEvent(
                    type=EventType.OBSERVATION,
                    agent_id=self.config.agent_id,
                    payload={
                        "step": step,
                        "tool": tool_name,
                        "observation": obs_display,
                    },
                )

                if (
                    observation
                    and not isinstance(observation, str)
                    and not _is_image_block(observation)
                ):
                    fire(
                        self._memory.write_working_fact(
                            run_id=run_id,
                            agent_id=self.config.agent_id,
                            key=f"step_{step}_{tool_name}",
                            value=observation,
                        )
                    )

                await self._working_memory.append("assistant", json.dumps(response))
                if _is_image_block(observation):
                    await self._working_memory.append(
                        "user",
                        [
                            {"type": "text", "text": f"Observation ({tool_name}):"},
                            observation,
                        ],
                    )
                else:
                    obs_text = (
                        json.dumps(observation, default=str)
                        if not isinstance(observation, str)
                        else observation
                    )
                    await self._working_memory.append("user", f"Observation: {obs_text}")

        # Max steps exhausted.
        self._tracer.log(
            "task_result",
            self.config.agent_id,
            {
                "answer": "",
                "confidence": 0.0,
                "steps": self.config.max_steps,
                "error": f"Max steps ({self.config.max_steps}) reached",
            },
        )
        yield BusEvent(
            type=EventType.ERROR,
            agent_id=self.config.agent_id,
            error=f"Max steps ({self.config.max_steps}) reached",
            payload={"steps": self.config.max_steps},
        )

    # ── Think ─────────────────────────────────────────────────────────────────

    async def _think_stream(self) -> AsyncGenerator[BusEvent, None]:
        """
        Streaming think: if the LLM client has `stream_complete`, forwards
        TOKEN events as text arrives, then parses the accumulated response
        into the action JSON and yields it as a THOUGHT event. Otherwise
        falls back to one `complete` call.
        """
        messages = self._working_memory.get_messages()
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
                    system=None,
                    messages=messages,
                    source=react_source,
                    response_format={"type": "json_object"},
                ):
                    accumulated += token
                    if self.config.stream_tokens:
                        yield BusEvent(
                            type=EventType.TOKEN,
                            agent_id=self.config.agent_id,
                            token=token,
                        )
                response = _parse_action_json(accumulated)
                if response is None:
                    logger.warning(
                        "Agent %s stream got unparseable response: %r",
                        self.config.agent_id,
                        accumulated[:300],
                    )
                    self._last_think_error = f"Unparseable stream response: {accumulated[:300]}"
            else:
                raw = await self._llm.complete(
                    system=None,
                    messages=messages,
                    response_format={"type": "json_object"},
                    source=react_source,
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
            yield BusEvent(
                type=EventType.MEMORY,
                agent_id=self.config.agent_id,
                payload={
                    "event": "summarized",
                    "before": before_usage,
                    "after": after_usage,
                    "summarizations": self._working_memory.summarization_count,
                },
            )
        llm_usage = getattr(self._llm, "last_usage", None) or {}
        if llm_usage or after_usage != before_usage:
            yield BusEvent(
                type=EventType.CONTEXT,
                agent_id=self.config.agent_id,
                payload={
                    **after_usage,
                    "tokens_in": llm_usage.get("tokens_in"),
                    "tokens_out": llm_usage.get("tokens_out"),
                    "cache_read_tokens": llm_usage.get("cache_read_tokens"),
                    "cache_creation_tokens": llm_usage.get("cache_creation_tokens"),
                },
            )

        yield BusEvent(
            type=EventType.THOUGHT,
            agent_id=self.config.agent_id,
            payload={
                "response": response,
                "thought": response.get("thought", "") if response else "",
                "action": response.get("action") if response else None,
            },
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

    async def _gate_tool(
        self,
        run_id: str,
        step: int,
        tool_name: str,
        tool_args: dict,
        llm_response: dict,
    ):
        """
        Run the HITL approval gate for one tool.

        Returns ApprovalResponse if the tool is gated, None if not.
        Writes a crash-resumable checkpoint to the store before blocking on stdin.
        """
        if not (self._checkpoint_store and tool_name in self.config.hitl_tools):
            return None

        from harness.hitl import ApprovalRequest, is_allowed, request_approval

        if is_allowed(tool_name, tool_args):
            return None  # fast-path: human already allowed this tool/prefix

        approval_id = str(uuid.uuid4())
        await self._checkpoint_store.write(
            self._ckp_id,
            {
                "run_id": run_id,
                "agent_id": self.config.agent_id,
                "task": self._task,
                "step": step,
                "memory": self._working_memory.to_dict(),
                "pending": {
                    "approval_id": approval_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "step": step,
                    "llm_response": llm_response,
                },
            },
        )
        return await request_approval(
            ApprovalRequest(
                approval_id=approval_id,
                run_id=self._resume_key,  # standalone: ckp_id; orchestrated: outer run_id
                agent_id=self.config.agent_id,
                tool=tool_name,
                args=tool_args,
                step=step,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            self._guard,
        )

    async def _run_tool_gated(
        self,
        run_id: str,
        step: int,
        tool_name: str,
        tool_args: dict,
        response: dict,
    ) -> Any:
        """
        Gate + execute a single tool.

        Returns _HITL_CORRECTION sentinel if the human typed a correction
        (WorkingMemory already updated; caller must `continue` the ReAct loop).
        Otherwise returns the observation (str or image block).
        """
        approval = await self._gate_tool(run_id, step, tool_name, tool_args, response)
        if approval is not None:
            if approval.correction:
                await self._inject_human_guidance(response, approval.correction, run_id, step)
                return _HITL_CORRECTION
            if not approval.approved:
                await self._commit_checkpoint(run_id, step)
                return f"Tool rejected by human: {approval.correction or 'no reason given'}"
        obs = await self._execute_tool(tool_name, tool_args)
        if approval is not None:
            # HITL was involved — overwrite pending checkpoint with clean state.
            # Non-HITL tools leave the step checkpoint intact for the next iteration.
            await self._commit_checkpoint(run_id, step)
        return obs

    async def _run_streaming_tool_gated(
        self,
        run_id: str,
        step: int,
        tool_name: str,
        tool_args: dict,
        response: dict,
        tool: Any,
    ):
        """Async-iterator variant of ``_run_tool_gated`` for tools that
        implement ``execute_stream``.

        Yields a sequence of ``("event", BusEvent)`` items as the underlying
        streaming tool emits events, and ends with exactly one of:

          ``("result", obs)``     — the observation the ReAct loop records
          ``("correction", None)`` — HITL injected a correction; caller
                                     must ``continue`` the loop. WM is
                                     already updated.

        Recursion guard fires here: if invoking the tool would push a
        sub-agent past ``config.max_subagent_depth``, the tool is refused
        and an error string surfaces as the observation instead.
        """
        from tools.builtin.subagent import SubAgentTool

        # Recursion guard for sub-agent delegation specifically.
        if isinstance(tool, SubAgentTool):
            next_depth = self._subagent_depth + 1
            if next_depth > self.config.max_subagent_depth:
                yield (
                    "result",
                    (
                        f"Refused to delegate to {tool.name!r}: "
                        f"max sub-agent depth {self.config.max_subagent_depth} "
                        f"would be exceeded (current depth {self._subagent_depth})."
                    ),
                )
                return
            tool._agent._subagent_depth = next_depth
            # Share the parent's guard so the sub-agent's check() enforces
            # the run-level budget and its bubbled TASK_DONE snapshot
            # reflects real token usage. Without this, sub-agents track an
            # empty local guard while the LLM reports tokens to the
            # runtime's guard (the one ``_attach_budget`` actually wired).
            tool._agent._guard = self._guard

        approval = await self._gate_tool(run_id, step, tool_name, tool_args, response)
        if approval is not None:
            if approval.correction:
                await self._inject_human_guidance(response, approval.correction, run_id, step)
                yield ("correction", None)
                return
            if not approval.approved:
                await self._commit_checkpoint(run_id, step)
                yield (
                    "result",
                    f"Tool rejected by human: {approval.correction or 'no reason given'}",
                )
                return

        observation: Any = None
        try:
            async for item in tool.execute_stream(**tool_args):
                if isinstance(item, BusEvent):
                    yield ("event", item)
                else:
                    # Streaming tools yield exactly one non-BusEvent terminal
                    # value — the dict observation the parent records.
                    observation = item
        except Exception as e:  # noqa: BLE001 — surface to the loop, not crash
            logger.error("Streaming tool %s failed: %s", tool_name, e)
            observation = f"Tool error ({tool_name}): {e}"

        if approval is not None:
            await self._commit_checkpoint(run_id, step)
        yield ("result", observation)

    async def _inject_human_guidance(
        self, response: dict, correction: str, run_id: str, step: int
    ) -> None:
        """Append human correction to WorkingMemory and commit a clean checkpoint."""
        await self._working_memory.append("assistant", json.dumps(response))
        await self._working_memory.append("user", f"Human guidance: {correction}")
        await self._commit_checkpoint(run_id, step)

    async def _commit_checkpoint(self, run_id: str, step: int) -> None:
        """Overwrite checkpoint with current state (no pending field).

        Called after HITL resolves or a tool completes so the stored state
        always reflects reality — no stale 'pending' approval marker, and
        the step position is preserved for crash-resume.
        """
        if self._checkpoint_store is None:
            return
        await self._checkpoint_store.write(
            self._ckp_id,
            {
                "run_id": run_id,
                "agent_id": self.config.agent_id,
                "task": self._task,
                "step": step,
                "memory": self._working_memory.to_dict(),
            },
        )

    async def _clear_checkpoint(self, run_id: str) -> None:
        if self._checkpoint_store:
            await self._checkpoint_store.delete(self._ckp_id)

    async def _replay_pending_step(
        self,
        run_id: str,
        pending: dict,
    ) -> AsyncGenerator[BusEvent, None]:
        """Re-prompt approval for a step interrupted by a crash, then complete it."""
        from harness.hitl import ApprovalRequest, is_allowed, request_approval

        tool_name = pending["tool"]
        tool_args = pending["args"]
        step = pending["step"]
        llm_response = pending["llm_response"]

        approval = None
        if not is_allowed(tool_name, tool_args):
            approval = await request_approval(
                ApprovalRequest(
                    approval_id=pending["approval_id"],
                    run_id=self._resume_key,  # standalone: ckp_id; orchestrated: outer run_id
                    agent_id=self.config.agent_id,
                    tool=tool_name,
                    args=tool_args,
                    step=step,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                self._guard,
            )

        if approval is not None and approval.correction:
            await self._inject_human_guidance(llm_response, approval.correction, run_id, step)
            return

        observation = (
            await self._execute_tool(tool_name, tool_args)
            if approval is None or approval.approved
            else f"Tool rejected by human: {approval.correction or 'no reason given'}"
        )
        obs_display = "[image]" if _is_image_block(observation) else str(observation)[:500]
        yield BusEvent(
            type=EventType.OBSERVATION,
            agent_id=self.config.agent_id,
            payload={"step": step, "tool": tool_name, "observation": obs_display},
        )
        await self._working_memory.append("assistant", json.dumps(llm_response))
        if _is_image_block(observation):
            await self._working_memory.append(
                "user",
                [{"type": "text", "text": f"Observation ({tool_name}):"}, observation],
            )
        else:
            obs_text = (
                json.dumps(observation, default=str)
                if not isinstance(observation, str)
                else observation
            )
            await self._working_memory.append("user", f"Observation: {obs_text}")
        await self._commit_checkpoint(run_id, step)


# ── Response normalization (module-level for testability) ────────────────────


def _normalize_response(response: Any) -> dict | None:
    if isinstance(response, dict) and ("action" in response or "actions" in response):
        return response
    if isinstance(response, dict) and "text" in response:
        text = response["text"].strip()
    elif isinstance(response, str):
        text = response.strip()
    else:
        text = str(response).strip()
    return _parse_action_json(text)


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
