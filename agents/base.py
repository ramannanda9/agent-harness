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
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from harness.events import BusEvent, EventType
from harness.utils import fire
from memory.manager import MemoryManager
from memory.working import WorkingMemory

logger = logging.getLogger(__name__)


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
    working_memory_max_tokens: int = 8000  # WorkingMemory eviction threshold; tune per agent


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
    ) -> None:
        self.config = config
        self.role = config.role  # exposed for orchestrator planner prompt
        self._tools = tools
        self._memory = memory
        self._tracer = tracer
        self._guard = guard
        self._llm = llm
        self._working_memory: WorkingMemory | None = None
        self._last_think_error: str | None = None

    # ── Streaming entry point (canonical) ─────────────────────────────────────

    async def run_stream(
        self,
        task: str,
        run_id: str | None = None,
    ) -> AsyncGenerator[BusEvent, None]:
        run_id = run_id or str(uuid.uuid4())
        self._working_memory = WorkingMemory(
            llm=self._llm,
            max_tokens=self.config.working_memory_max_tokens,
        )

        system = await self._build_system_prompt(task)
        await self._working_memory.append("system", system, pinned=True)
        await self._working_memory.append("user", task)

        try:
            async for event in self._react_stream(run_id):
                yield event
        except Exception as e:
            logger.exception("Agent %s stream crashed", self.config.agent_id)
            yield BusEvent(
                type=EventType.ERROR,
                agent_id=self.config.agent_id,
                error=str(e),
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
                parts.append(mem_context.render())

        tool_list = ", ".join(self._tools.keys()) or "none"
        parts.append(REACT_FORMAT.replace("__TOOL_LIST__", tool_list))
        return "\n\n".join(parts)

    # ── ReAct Loop (stream) ───────────────────────────────────────────────────

    async def _react_stream(self, run_id: str) -> AsyncGenerator[BusEvent, None]:
        for step in range(self.config.max_steps):
            self._guard.check()

            # Think — yields TOKEN events when the LLM client supports streaming.
            response = None
            async for thought_event in self._think_stream():
                if thought_event.type == EventType.TOKEN:
                    yield thought_event
                elif thought_event.type == EventType.THOUGHT:
                    response = thought_event.payload.get("response")
                    yield thought_event

            if response is None:
                reason = self._last_think_error or "LLM returned unparseable response"
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
                logger.info(
                    "Agent %s completed: steps=%d confidence=%.2f summarizations=%d",
                    self.config.agent_id,
                    result["steps"],
                    result["confidence"],
                    self._working_memory.summarization_count,
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

                # Fan out all tool calls concurrently.
                observations = await asyncio.gather(
                    *[
                        self._execute_tool(act.get("tool", ""), act.get("args", {}))
                        for act in parallel_actions
                    ]
                )

                combined: list[dict] = []
                for i, (act, obs) in enumerate(zip(parallel_actions, observations, strict=False)):
                    tool_name = act.get("tool", "")
                    tool_args = act.get("args", {})
                    self._tracer.log(
                        "action",
                        self.config.agent_id,
                        {
                            "step": step,
                            "tool": tool_name,
                            "args": tool_args,
                            "observation": str(obs)[:500],
                        },
                    )
                    yield BusEvent(
                        type=EventType.OBSERVATION,
                        agent_id=self.config.agent_id,
                        payload={"step": step, "tool": tool_name, "observation": str(obs)[:500]},
                    )
                    combined.append({"tool": tool_name, "result": obs})
                    if obs and not isinstance(obs, str):
                        fire(
                            self._memory.write_working_fact(
                                run_id=run_id,
                                agent_id=self.config.agent_id,
                                key=f"step_{step}_{i}_{tool_name}",
                                value=obs,
                            )
                        )

                await self._working_memory.append("assistant", json.dumps(response))
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

                observation = await self._execute_tool(tool_name, tool_args)

                self._tracer.log(
                    "action",
                    self.config.agent_id,
                    {
                        "step": step,
                        "tool": tool_name,
                        "args": tool_args,
                        "observation": str(observation)[:500],
                    },
                )
                yield BusEvent(
                    type=EventType.OBSERVATION,
                    agent_id=self.config.agent_id,
                    payload={
                        "step": step,
                        "tool": tool_name,
                        "observation": str(observation)[:500],
                    },
                )

                if observation and not isinstance(observation, str):
                    fire(
                        self._memory.write_working_fact(
                            run_id=run_id,
                            agent_id=self.config.agent_id,
                            key=f"step_{step}_{tool_name}",
                            value=observation,
                        )
                    )

                obs_text = (
                    json.dumps(observation, default=str)
                    if not isinstance(observation, str)
                    else observation
                )
                await self._working_memory.append("assistant", json.dumps(response))
                await self._working_memory.append("user", f"Observation: {obs_text}")

        # Max steps exhausted.
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

        try:
            if hasattr(self._llm, "stream_complete"):
                async for token in self._llm.stream_complete(
                    system=None,
                    messages=messages,
                ):
                    accumulated += token
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
        try:
            return await self._tools[name].execute(**args)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return f"Tool error ({name}): {e}"

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
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        pos = start + 1

    return None
