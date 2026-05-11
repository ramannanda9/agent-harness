"""
BaseAgent — generic ReAct loop agent.

Every agent is an instance of BaseAgent configured via AgentConfig.
No subclassing needed for new domains — just register a new AgentConfig
with different role, system_prompt, and allowed_tools.

Memory integration:
  - build_context() injected into system prompt at run start
  - write_working_fact() called after each tool observation
  - run-end write handled by Orchestrator, not BaseAgent

Token management:
  - WorkingMemory handles eviction via LLM summarization
  - Tool schemas compressed: only name + description injected,
    full schema only sent to LLM on tool call
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from memory.working import WorkingMemory
from memory.manager import MemoryManager

logger = logging.getLogger(__name__)


# ── Agent Config ──────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    agent_id: str
    role: str                          # plain English — used by planner for agent selection
    system_prompt: str
    allowed_tools: list[str]           # tool names from ToolRegistry
    max_steps: int = 10
    memory_context_enabled: bool = True
    confidence_from_llm: bool = True   # if False, confidence=1.0 on success


# ── ReAct Response Schema ─────────────────────────────────────────────────────

# Injected into every agent's system prompt so LLM knows the expected format
REACT_FORMAT = """
At each step, respond with a JSON object in one of two forms:

To use a tool:
{
  "thought": "<reasoning about what to do next>",
  "action": "<tool_name>",
  "args": { "<arg>": "<value>", ... }
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
        tools: dict[str, Any],         # name → Tool instance
        memory: MemoryManager,
        tracer,
        guard,
        llm,
    ) -> None:
        self.config = config
        self.role = config.role        # exposed for orchestrator planner prompt
        self._tools = tools
        self._memory = memory
        self._tracer = tracer
        self._guard = guard
        self._llm = llm
        self._working_memory: WorkingMemory | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, task: str, run_id: str | None = None) -> dict:
        run_id = run_id or str(uuid.uuid4())
        self._working_memory = WorkingMemory(
            llm=self._llm,
            max_tokens=2000,
        )

        # build system prompt: base + memory context + react format
        system = await self._build_system_prompt(task)
        await self._working_memory.append("system", system, pinned=True)
        await self._working_memory.append("user", task)

        result = await self._react_loop(run_id=run_id)

        logger.info(
            "Agent %s completed: steps=%d confidence=%.2f summarizations=%d",
            self.config.agent_id,
            result["steps"],
            result["confidence"],
            self._working_memory.summarization_count,
        )
        return result

    # ── System Prompt ─────────────────────────────────────────────────────────

    async def _build_system_prompt(self, task: str) -> str:
        parts = [self.config.system_prompt]

        # inject memory context if enabled
        if self.config.memory_context_enabled:
            mem_context = await self._memory.build_context(
                goal=task,
                agent_id=self.config.agent_id,
            )
            if not mem_context.is_empty():
                parts.append(mem_context.render())

        # inject ReAct format with compressed tool list
        tool_list = ", ".join(self._tools.keys()) or "none"
        parts.append(REACT_FORMAT.replace("__TOOL_LIST__", tool_list))

        return "\n\n".join(parts)

    # ── ReAct Loop ────────────────────────────────────────────────────────────

    async def _react_loop(self, run_id: str) -> dict:
        for step in range(self.config.max_steps):
            self._guard.check()

            # think
            response = await self._think()
            if response is None:
                return self._error_result("LLM returned unparseable response", step)

            self._tracer.log(
                "thought", self.config.agent_id,
                {"step": step, "thought": response.get("thought", ""), "action": response.get("action")},
            )

            # finish?
            if response.get("action") == "finish":
                return {
                    "agent_id": self.config.agent_id,
                    "answer": response.get("answer", ""),
                    "confidence": response.get("confidence", 1.0) if self.config.confidence_from_llm else 1.0,
                    "steps": step + 1,
                    "metadata": {"summarizations": self._working_memory.summarization_count},
                }

            # act
            tool_name = response.get("action", "")
            tool_args = response.get("args", {})
            observation = await self._execute_tool(tool_name, tool_args)

            self._tracer.log(
                "action", self.config.agent_id,
                {"step": step, "tool": tool_name, "args": tool_args, "observation": str(observation)[:500]},
            )

            # write working fact — lightweight, no LLM, namespaced
            # only write if observation is a non-trivial result
            if observation and not isinstance(observation, str):
                await self._memory.write_working_fact(
                    run_id=run_id,
                    agent_id=self.config.agent_id,
                    key=f"step_{step}_{tool_name}",
                    value=observation,
                )

            # update working memory — eviction fires here if over budget
            obs_text = json.dumps(observation, default=str) if not isinstance(observation, str) else observation
            await self._working_memory.append("assistant", json.dumps(response))
            await self._working_memory.append("user", f"Observation: {obs_text}")

        return self._error_result(f"Max steps ({self.config.max_steps}) reached", self.config.max_steps)

    # ── Think ─────────────────────────────────────────────────────────────────

    async def _think(self) -> dict | None:
        try:
            response = await self._llm.complete(
                system=None,
                messages=self._working_memory.get_messages(),
                response_format={"type": "json_object"},
            )
            # normalize: extract dict regardless of response shape
            if isinstance(response, dict) and "action" in response:
                return response
            if isinstance(response, dict) and "text" in response:
                text = response["text"].strip()
            elif isinstance(response, str):
                text = response.strip()
            else:
                text = str(response).strip()
            # extract first JSON object from text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Agent %s think failed: %s", self.config.agent_id, e)
            return None

    # ── Tool Execution ────────────────────────────────────────────────────────

    async def _execute_tool(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            return f"Error: tool '{name}' not available. Available tools: {list(self._tools.keys())}"
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
