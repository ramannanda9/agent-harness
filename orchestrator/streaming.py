"""
Streaming orchestration layer.

Performance model:
  - Each agent is an AsyncGenerator — yields tokens as they arrive from LLM
  - Orchestrator merges streams via asyncio.Queue (token bus)
  - Synthesizer reads from token bus — can start before all agents finish
  - Backpressure: bounded queue (maxsize=512) — fast agents block if synthesizer slow

Token bus event types:
  TokenEvent   — a single token from an agent
  DoneEvent    — agent completed, carries final AgentResult
  ErrorEvent   — agent failed mid-stream

Agent output streaming requires LLM client to support stream_complete():
  async def stream_complete(system, messages) -> AsyncGenerator[str, None]

For Anthropic:
  async with client.messages.stream(...) as stream:
      async for text in stream.text_stream:
          yield text

Working memory summarization:
  - summarize_async() fires in background task — non-blocking
  - ReAct loop continues while summarization runs
  - Summary is injected at next step via _pending_summary flag

Inter-agent result sharing:
  - Completed agent results written to Blackboard immediately on DoneEvent
  - Dependent agents can read partial context from Blackboard
    even before orchestrator synthesizes final answer
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Token Bus Events ──────────────────────────────────────────────────────────

class EventType(str, Enum):
    TOKEN  = "token"
    DONE   = "done"
    ERROR  = "error"


@dataclass
class BusEvent:
    type: EventType
    agent_id: str
    # token event
    token: str = ""
    # done event
    result: dict = field(default_factory=dict)
    # error event
    error: str = ""
    timestamp: float = field(default_factory=time.time)


# ── Streaming Working Memory ──────────────────────────────────────────────────

class StreamingWorkingMemory:
    """
    WorkingMemory variant where summarization is non-blocking.

    When token budget is hit:
      1. Snapshot messages to summarize
      2. Fire summarization as background asyncio.Task
      3. Continue ReAct loop immediately
      4. On next step: if summary ready, inject it and drop originals
         If not ready: continue with full context (accept budget overage)

    This trades occasional budget overage for zero-blocking on hot path.
    """

    def __init__(
        self,
        llm,
        max_tokens: int = 8000,
        summarize_ratio: float = 0.5,
        token_counter=None,
    ) -> None:
        from memory.working import count_tokens

        self._llm = llm
        self.max_tokens = max_tokens
        self._summarize_ratio = summarize_ratio
        self._count_fn = token_counter or count_tokens
        self._messages: list[dict] = []
        self._token_total: int = 0
        self._pending_summary: asyncio.Task | None = None
        self._pending_summary_range: tuple[int, int] | None = None  # (start_idx, end_idx)
        self._summarization_count: int = 0

    def _count(self, text: str) -> int:
        return self._count_fn(text)

    async def append(self, role: str, content: str, pinned: bool = False) -> None:
        self._messages.append({"role": role, "content": content, "pinned": pinned})
        self._token_total += self._count(content)

        # inject pending summary if ready
        await self._maybe_inject_summary()

        # fire background summarization if over budget
        if self._token_total > self.max_tokens and self._pending_summary is None:
            self._fire_summarization()

    async def _maybe_inject_summary(self) -> None:
        if self._pending_summary is None:
            return
        if not self._pending_summary.done():
            return   # not ready yet — continue with full context

        try:
            summary_text = await self._pending_summary
            start, end = self._pending_summary_range

            # replace summarized range with single summary message
            summary_msg = {
                "role": "user",
                "content": f"[Memory summary]: {summary_text}",
                "pinned": False,
            }
            self._messages = (
                self._messages[:start]
                + [summary_msg]
                + self._messages[end:]
            )
            self._token_total = sum(self._count(m["content"]) for m in self._messages)
            self._summarization_count += 1
            logger.debug("Injected async summary — new token count: %d", self._token_total)
        except Exception as e:
            logger.error("Async summarization failed: %s — continuing without summary", e)
        finally:
            self._pending_summary = None
            self._pending_summary_range = None

    def _fire_summarization(self) -> None:
        """Fire background summarization task — non-blocking."""
        evictable = [
            (i, m) for i, m in enumerate(self._messages)
            if not m.get("pinned")
        ]
        if not evictable:
            return

        cutoff = max(1, int(len(evictable) * self._summarize_ratio))
        to_summarize = evictable[:cutoff]
        indices = [i for i, _ in to_summarize]
        messages = [m for _, m in to_summarize]

        start_idx = indices[0]
        end_idx = indices[-1] + 1
        self._pending_summary_range = (start_idx, end_idx)

        formatted = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in messages
        )

        SUMMARIZE_SYSTEM = (
            "You are a memory compressor. Summarize the conversation into one dense paragraph. "
            "Preserve key facts, tool results, decisions. Discard pleasantries and repetition. "
            "Return only the summary paragraph."
        )

        self._pending_summary = asyncio.create_task(
            self._llm.complete(
                system=SUMMARIZE_SYSTEM,
                messages=[{"role": "user", "content": formatted}],
            )
        )
        logger.debug(
            "Fired async summarization for messages[%d:%d]", start_idx, end_idx
        )

    def get_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def token_count(self) -> int:
        return self._token_total

    @property
    def summarization_count(self) -> int:
        return self._summarization_count


# ── Streaming Base Agent ──────────────────────────────────────────────────────

class StreamingAgent:
    """
    Agent variant that yields tokens via AsyncGenerator.

    Enables orchestrator to:
      - Display partial results to end user in real time
      - Start dependent processing before agent fully completes
      - Implement token-level backpressure via bounded queue

    The run_stream() method is the primary interface.
    run() (blocking) is kept for backward compat and testing.
    """

    def __init__(
        self,
        config,           # AgentConfig
        tools: dict,
        memory,           # MemoryManager
        tracer,
        guard,
        llm,
    ) -> None:
        self.config = config
        self.role = config.role
        self._tools = tools
        self._memory = memory
        self._tracer = tracer
        self._guard = guard
        self._llm = llm

    async def run_stream(
        self,
        task: str,
        run_id: str,
    ) -> AsyncGenerator[BusEvent, None]:
        """
        Stream tokens as they arrive from LLM.
        Yields BusEvent(TOKEN) for each token, BusEvent(DONE) at completion.
        """
        wm = StreamingWorkingMemory(
            llm=self._llm,
            max_tokens=self.config.working_memory_max_tokens,
        )

        # build system prompt with memory context
        system = await self._build_system(task, wm)
        await wm.append("system", system, pinned=True)
        await wm.append("user", task)

        try:
            async for event in self._react_stream(wm, run_id, task):
                yield event
        except Exception as e:
            logger.error("Agent %s stream failed: %s", self.config.agent_id, e)
            yield BusEvent(type=EventType.ERROR, agent_id=self.config.agent_id, error=str(e))

    async def _react_stream(
        self,
        wm: StreamingWorkingMemory,
        run_id: str,
        task: str,
    ) -> AsyncGenerator[BusEvent, None]:
        accumulated = ""

        for step in range(self.config.max_steps):
            self._guard.check()
            accumulated = ""

            # stream tokens from LLM
            if hasattr(self._llm, "stream_complete"):
                async for token in self._llm.stream_complete(
                    system=None,
                    messages=wm.get_messages(),
                ):
                    accumulated += token
                    yield BusEvent(
                        type=EventType.TOKEN,
                        agent_id=self.config.agent_id,
                        token=token,
                    )
            else:
                # fallback: non-streaming LLM — emit full response as single token
                response = await self._llm.complete(
                    system=None,
                    messages=wm.get_messages(),
                    response_format={"type": "json_object"},
                )
                accumulated = _normalize_response(response)
                yield BusEvent(
                    type=EventType.TOKEN,
                    agent_id=self.config.agent_id,
                    token=accumulated,
                )

            # parse accumulated response
            parsed = _parse_json_safe(accumulated)
            if parsed is None:
                logger.error("Agent %s unparseable response at step %d", self.config.agent_id, step)
                continue

            self._tracer.log("thought", self.config.agent_id, {
                "step": step,
                "action": parsed.get("action"),
            })

            # finish?
            if parsed.get("action") == "finish":
                result = {
                    "agent_id": self.config.agent_id,
                    "answer": parsed.get("answer", ""),
                    "confidence": parsed.get("confidence", 1.0),
                    "steps": step + 1,
                    "metadata": {"summarizations": wm.summarization_count},
                }
                yield BusEvent(
                    type=EventType.DONE,
                    agent_id=self.config.agent_id,
                    result=result,
                )
                return

            # execute tool
            tool_name = parsed.get("action", "")
            tool_args = parsed.get("args", {})
            observation = await self._execute_tool(tool_name, tool_args)

            obs_str = json.dumps(observation, default=str) if not isinstance(observation, str) else observation

            # write working fact (non-blocking)
            if observation and not isinstance(observation, str):
                await self._memory.write_working_fact(
                    run_id=run_id,
                    agent_id=self.config.agent_id,
                    key=f"step_{step}_{tool_name}",
                    value=observation,
                )

            await wm.append("assistant", accumulated)
            await wm.append("user", f"Observation: {obs_str}")

            self._tracer.log("action", self.config.agent_id, {
                "step": step, "tool": tool_name, "observation": obs_str[:200],
            })

        # max steps reached
        yield BusEvent(
            type=EventType.DONE,
            agent_id=self.config.agent_id,
            result={
                "agent_id": self.config.agent_id,
                "answer": "max steps reached",
                "confidence": 0.0,
                "steps": self.config.max_steps,
            },
        )

    async def _build_system(self, task: str, wm: StreamingWorkingMemory) -> str:
        parts = [self.config.system_prompt]
        if self.config.memory_context_enabled:
            ctx = await self._memory.build_context(goal=task, agent_id=self.config.agent_id)
            if not ctx.is_empty():
                parts.append(ctx.render())
        tool_list = ", ".join(self._tools.keys()) or "none"
        react = (
            'At each step respond with JSON.\n'
            'To use tool: {"thought":"...","action":"<tool>","args":{...}}\n'
            'To finish: {"thought":"...","action":"finish","answer":"...","confidence":0.9}\n'
            f'Available tools: {tool_list}'
        )
        parts.append(react)
        return "\n\n".join(parts)

    async def _execute_tool(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            return f"Error: tool '{name}' not available. Available: {list(self._tools)}"
        try:
            return await self._tools[name].execute(**args)
        except Exception as e:
            return f"Tool error ({name}): {e}"

    # backward-compat blocking run
    async def run(self, task: str, run_id: str | None = None) -> dict:
        run_id = run_id or str(uuid.uuid4())
        result = {}
        async for event in self.run_stream(task=task, run_id=run_id):
            if event.type == EventType.DONE:
                result = event.result
            elif event.type == EventType.ERROR:
                result = {"agent_id": self.config.agent_id, "answer": "", "confidence": 0.0, "error": event.error}
        return result


# ── Streaming Orchestrator ────────────────────────────────────────────────────

class StreamingOrchestrator:
    """
    Orchestrator that uses a bounded asyncio.Queue as the token bus.

    Parallel agent streams are fan-in'd to a single queue.
    Consumers (synthesizer, UI layer) read from the queue.

    Backpressure: if queue is full, fast agents yield — slow consumer
    is not overwhelmed. Queue size = 512 tokens ~= 400ms of buffer at
    typical LLM throughput.

    Dependency handling: tasks with depends_on wait for their dependencies
    to emit DONE events before starting — same hybrid DAG as non-streaming.
    """

    TOKEN_BUS_SIZE = 512

    def __init__(
        self,
        agents: dict[str, StreamingAgent],
        memory,
        tracer,
        guard,
        llm,
        eval_config=None,
    ) -> None:
        self._agents = agents
        self._memory = memory
        self._tracer = tracer
        self._guard = guard
        self._llm = llm
        self._eval_config = eval_config
        self._run_id = str(uuid.uuid4())

    async def run(self, goal: str) -> dict:
        """Non-streaming entry point — drains token bus internally."""
        all_tokens: dict[str, list[str]] = {}
        all_results: list[dict] = []

        async for event in self.run_stream(goal):
            if event.type == EventType.TOKEN:
                all_tokens.setdefault(event.agent_id, []).append(event.token)
            elif event.type == EventType.DONE:
                all_results.append(event.result)

        return {
            "run_id": self._run_id,
            "goal": goal,
            "agent_results": all_results,
            "answer": all_results[-1].get("answer", "") if all_results else "",
        }

    async def run_stream(self, goal: str) -> AsyncGenerator[BusEvent, None]:
        """
        Streaming entry point — yields BusEvents for every token and completion.
        Caller can display tokens in real-time or drain to dict via run().
        """
        from orchestrator.planner import EvalConfig, should_replan

        eval_cfg = self._eval_config or EvalConfig()

        # plan
        plan = await self._plan(goal)
        self._tracer.log("plan", "orchestrator", {"tasks": len(plan.tasks)})

        completed: dict[str, dict] = {}   # task_id → result
        pending = list(plan.tasks)
        replan_count = 0
        bus: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=self.TOKEN_BUS_SIZE)

        while pending:
            self._guard.check()

            ready = [
                t for t in pending
                if all(dep in completed for dep in t.depends_on)
            ]
            if not ready:
                break

            # launch all ready tasks concurrently — each pushes to shared bus
            producers = [
                asyncio.create_task(
                    self._agent_to_bus(t, bus, self._run_id)
                )
                for t in ready
            ]
            expected_dones = {t.id for t in ready}
            received_dones: set[str] = set()

            # drain bus until all ready tasks are done
            while received_dones != expected_dones:
                event = await bus.get()
                yield event   # pass to caller (UI, synthesizer, etc.)

                if event.type == EventType.DONE:
                    task = next(t for t in ready if t.agent_id == event.agent_id
                                and t.id not in received_dones)
                    received_dones.add(task.id)
                    completed[task.id] = event.result
                    pending.remove(task)

                    # replan check
                    result_dict = event.result
                    mock_result = type("R", (), {
                        "success": result_dict.get("confidence", 0) > 0,
                        "confidence": result_dict.get("confidence", 0),
                    })()

                    if should_replan(mock_result, eval_cfg) and replan_count < eval_cfg.max_replan_count:
                        replan_count += 1
                        self._tracer.log("replan", "orchestrator", {
                            "trigger": task.id, "count": replan_count,
                        })
                        # cancel remaining producers
                        for p in producers:
                            if not p.done():
                                p.cancel()
                        # replan remaining tasks
                        new_plan = await self._replan(goal, list(completed.values()), result_dict, pending)
                        pending = list(new_plan.tasks)
                        break

                elif event.type == EventType.ERROR:
                    logger.error("Agent error on bus: %s", event.error)

            # wait for all producers to clean up
            await asyncio.gather(*producers, return_exceptions=True)

        # final synthesis event
        synthesis = await self._synthesize(goal, list(completed.values()))
        yield BusEvent(
            type=EventType.DONE,
            agent_id="orchestrator",
            result=synthesis,
        )

        # run-end memory write
        await self._memory.write_run_end(
            goal=goal,
            agent_results=list(completed.values()),
            trace=self._tracer.dump(),
        )

    async def _agent_to_bus(
        self,
        task,
        bus: asyncio.Queue,
        run_id: str,
    ) -> None:
        """Pump agent stream events onto shared token bus."""
        agent = self._agents.get(task.agent_id)
        if agent is None:
            await bus.put(BusEvent(
                type=EventType.ERROR,
                agent_id=task.agent_id,
                error=f"Agent '{task.agent_id}' not found",
            ))
            return

        async for event in agent.run_stream(task=task.instruction, run_id=run_id):
            await bus.put(event)   # backpressure: blocks if bus full

    async def _plan(self, goal: str):
        from orchestrator.planner import PLAN_SYSTEM, _parse_plan
        agent_descriptions = "\n".join(
            f"  {aid}: {getattr(agent, 'role', '')}"
            for aid, agent in self._agents.items()
        )
        response = await self._llm.complete(
            system=PLAN_SYSTEM.format(agent_descriptions=agent_descriptions),
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
            response_format={"type": "json_object"},
        )
        return _parse_plan(response)

    async def _replan(self, goal, completed, failed, remaining):
        from orchestrator.planner import REPLAN_SYSTEM, _parse_plan
        agent_descriptions = "\n".join(
            f"  {aid}: {getattr(agent, 'role', '')}"
            for aid, agent in self._agents.items()
        )
        response = await self._llm.complete(
            system=REPLAN_SYSTEM.format(
                agent_descriptions=agent_descriptions,
                completed=json.dumps(completed, default=str),
                failed_task=json.dumps(failed, default=str),
                remaining=json.dumps([{"id": t.id, "agent_id": t.agent_id} for t in remaining]),
            ),
            messages=[{"role": "user", "content": f"Replan for: {goal}"}],
            response_format={"type": "json_object"},
        )
        return _parse_plan(response)

    async def _synthesize(self, goal: str, results: list[dict]) -> dict:
        from orchestrator.planner import SYNTHESIZE_SYSTEM
        response = await self._llm.complete(
            system=SYNTHESIZE_SYSTEM,
            messages=[{"role": "user", "content": f"Goal: {goal}\nResults: {json.dumps(results, default=str)}"}],
            response_format={"type": "json_object"},
        )
        if isinstance(response, str):
            return json.loads(response)
        return response if isinstance(response, dict) else {"answer": str(response), "confidence": 0.5}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_response(response: Any) -> str:
    if isinstance(response, dict) and "action" in response:
        return json.dumps(response)
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    if isinstance(response, str):
        return response.strip()
    return json.dumps(response)


def _parse_json_safe(text: str) -> dict | None:
    text = text.strip()
    try:
        # direct parse
        if text.startswith("{"):
            return json.loads(text)
        # extract first JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None
