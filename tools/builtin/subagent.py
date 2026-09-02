"""``SubAgentTool`` — expose a ``BaseAgent`` as a tool callable from another
agent's ReAct loop.

The parent agent invokes ``delegate_<id>(task="…")``; the wrapped sub-agent
runs its own ReAct loop with a fresh ``WorkingMemory``, and its final answer
becomes the parent's observation. Sub-agent events bubble up through the
parent's ``BusEvent`` stream tagged with ``parent_agent_id`` so renderers
can indent / group them.

Memory model (matches the Orchestrator's per-agent shape):
  - Each delegation creates a fresh WM via ``agent.run_stream(task, run_id)``.
  - The sub-agent does NOT see the parent's WM — only the task string the
    parent's LLM emitted.
  - Cross-delegation continuity for the same sub-agent goes through the
    long-term memory layer (``MemoryManager.build_context(agent_id=…)``),
    not through WM carry-over.

This is the dynamic alternative to ``Orchestrator``'s static DAG: instead
of a planner LLM deciding upfront which agents to fan out to, the parent
agent's ReAct loop decides per-step which sub-agent to delegate to. Both
approaches stay in the framework; users pick per use case.
"""

from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from harness.events import BusEvent, EventType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agents.base import BaseAgent


class SubAgentTool:
    """Streaming tool that wraps one ``BaseAgent``.

    Implements ``execute_stream`` (not ``execute``) so the parent agent's
    streaming-tool dispatch picks it up — events bubble through the
    parent's stream as the sub-agent runs.

    ``cacheable=False`` by default: sub-agent runs aren't pure, and the
    per-run tool cache key includes only ``(tool_name, args)``, which
    misses memory / world state changes between calls.

    Parameters
    ----------
    agent:
        The ``BaseAgent`` instance to invoke. Constructed by the caller
        with whatever tools, memory, guard, LLM the sub-agent should use.
    name:
        Tool name visible to the parent agent's LLM. Defaults to
        ``delegate_<agent_id>``; override when you want a more
        action-oriented name (``research``, ``analyse``).
    task_arg:
        Name of the kwarg the parent agent's LLM uses to pass the
        instruction. Defaults to ``task``; some users prefer
        ``instruction`` or ``query`` to nudge the LLM toward clearer
        delegations.
    """

    cacheable: bool = False

    def __init__(
        self,
        agent: BaseAgent,
        *,
        name: str | None = None,
        task_arg: str = "task",
    ) -> None:
        self._agent = agent
        self.name = name or f"delegate_{agent.config.agent_id}"
        self._task_arg = task_arg
        # The invoking agent's id. ``BaseAgent`` sets this before calling
        # ``execute_stream`` so bubbled events carry the *parent*'s id
        # in ``parent_agent_id``, not the sub-agent's own id. Default
        # empty for direct callers (e.g. tests) — "no known parent".
        self._invoking_agent_id: str = ""

    def clone_for_run(self) -> SubAgentTool:
        """A copy safe to use from a separate, concurrent run.

        Both this tool and the agent behind it hold per-invocation state
        (``_invoking_agent_id`` here; working memory and the current task on
        the agent), so two parents delegating through one shared instance
        would overwrite each other exactly as two tasks sharing one agent do.
        """
        clone = copy.copy(self)
        clone._agent = self._agent.clone_for_run()
        clone._invoking_agent_id = ""
        return clone

    @property
    def agent_id(self) -> str:
        return self._agent.config.agent_id

    @property
    def description(self) -> str:
        """Used by the parent agent's tool listing; defaults to the role."""
        return f"Delegate to {self._agent.config.agent_id}: {self._agent.role}"

    async def load_resume_state(self, invocation_id: str) -> Any | None:
        """Stored state for a delegation that started and never finished.

        A delegation's ``run_id`` is minted inside :meth:`execute_stream`, so
        the sub-agent's checkpoint lands under a key the parent could not
        otherwise name. The parent learns it from the ``invocation_id`` on the
        ``subagent_start`` event and hands it back here.

        Returns ``None`` whenever resuming is not possible — no store, no
        checkpoint, an unreadable or legacy one, or a run that already
        reached a terminal phase. Restarting is always safe, so this never
        raises: a delegation that cannot be resumed is simply re-run, which
        is what happened before this existed.
        """
        store = getattr(self._agent, "_checkpoint_store", None)
        if store is None or not invocation_id:
            return None

        from harness.runstate import RunState, load_state

        try:
            stored = await store.read(self._agent._checkpoint_key(invocation_id))
        except Exception as exc:  # noqa: BLE001 — restarting is always safe
            logger.warning("Ignoring unreadable delegation checkpoint: %s", exc)
            return None
        if not stored:
            return None
        try:
            state = load_state(stored)
        except Exception as exc:  # noqa: BLE001 — e.g. a pre-0.13 checkpoint
            logger.warning("Ignoring undecodable delegation checkpoint: %s", exc)
            return None
        if not isinstance(state, RunState) or state.is_terminal:
            return None
        return state

    async def resume_stream(self, state: Any) -> AsyncIterator[BusEvent | dict]:
        """Continue a delegation from stored state instead of restarting it.

        Emits the same ``subagent_start`` / events / terminal-dict shape as
        :meth:`execute_stream`, reusing the original ``invocation_id`` so a
        consumer watching across the crash sees one continuous delegation
        rather than two.
        """
        invoking_parent = self._invoking_agent_id
        yield BusEvent.subagent_start(
            self.agent_id,
            task=state.task[:300],
            invocation_id=state.run_id,
            parent_agent_id=invoking_parent,
        )
        async for item in self._drive(
            self._agent._resume_stream(state),
            run_id=state.run_id,
            invoking_parent=invoking_parent,
        ):
            yield item

    async def execute_stream(
        self,
        **args: Any,
    ) -> AsyncIterator[BusEvent | dict]:
        """Run the sub-agent on the delegated task; yield its events; end
        with one terminal dict the parent records as its OBSERVATION.

        Yields
        ------
        BusEvent
            Every event from the sub-agent's ``run_stream``, with
            ``parent_agent_id`` set to the sub-agent's own ``agent_id``
            so callers can tell the event was emitted *under* a
            delegation context. Top-level events stay unchanged.
        dict
            A single terminal dict ``{"agent_id", "answer", "confidence",
            "steps", "success", "error"}`` distinguishable from
            ``BusEvent`` by type. ``BaseAgent``'s streaming dispatch
            treats this as the tool's return value.
        """
        task = str(args.get(self._task_arg) or "").strip()
        if not task:
            yield {
                "agent_id": self.agent_id,
                "answer": "",
                "confidence": 0.0,
                "steps": 0,
                "success": False,
                "error": f"missing required arg {self._task_arg!r}",
            }
            return

        # Per-delegation run_id keeps checkpoints / OTel spans / memory
        # writes correctly scoped under this specific invocation, even when
        # the same sub-agent is delegated to multiple times in one parent
        # run. It reaches the parent on the ``subagent_start`` event below,
        # which is how a delegation becomes addressable for resume.
        run_id = str(uuid.uuid4())

        # The invoking agent's id (set by BaseAgent before this call). For
        # nested delegations, the inner SubAgentTool already populated
        # ``parent_agent_id`` on its own bubbled events to point at the
        # immediate parent — we preserve that, so a top-level consumer can
        # walk one level up by reading ``parent_agent_id``.
        invoking_parent = self._invoking_agent_id

        yield BusEvent.subagent_start(
            self.agent_id,
            task=task[:300],
            invocation_id=run_id,
            parent_agent_id=invoking_parent,
        )

        async for item in self._drive(
            self._agent.run_stream(task=task, run_id=run_id),
            run_id=run_id,
            invoking_parent=invoking_parent,
        ):
            yield item

    async def _drive(
        self,
        stream: AsyncIterator[BusEvent],
        *,
        run_id: str,
        invoking_parent: str,
    ) -> AsyncIterator[BusEvent | dict]:
        """Tag and forward a sub-agent's events, then yield the terminal dict.

        Shared by :meth:`execute_stream` and :meth:`resume_stream` so a
        resumed delegation is indistinguishable downstream from a fresh one.
        """
        last_done: dict | None = None
        last_error: str | None = None
        try:
            async for event in stream:
                tagged = BusEvent(
                    type=event.type,
                    agent_id=event.agent_id,
                    payload={**dict(event.payload), "invocation_id": run_id},
                    token=event.token,
                    error=event.error,
                    timestamp=event.timestamp,
                    parent_agent_id=event.parent_agent_id or invoking_parent,
                )
                yield tagged
                if event.type == EventType.TASK_DONE:
                    last_done = event.payload
                elif event.type == EventType.ERROR:
                    last_error = event.error
        except Exception as exc:  # noqa: BLE001 — surface to parent
            last_error = f"{type(exc).__name__}: {exc}"

        yield BusEvent.subagent_done(
            self.agent_id,
            success=last_done is not None,
            steps=(last_done or {}).get("steps", 0),
            confidence=(last_done or {}).get("confidence", 0.0),
            answer=(last_done or {}).get("answer", "")[:300],
            error=last_error or "",
            invocation_id=run_id,
            parent_agent_id=invoking_parent,
        )

        if last_done is not None:
            yield {
                "agent_id": self.agent_id,
                "answer": last_done.get("answer", ""),
                "confidence": last_done.get("confidence", 1.0),
                "steps": last_done.get("steps", 0),
                "success": True,
                "metadata": last_done.get("metadata", {}),
            }
        else:
            yield {
                "agent_id": self.agent_id,
                "answer": "",
                "confidence": 0.0,
                "steps": 0,
                "success": False,
                "error": last_error or "sub-agent stream ended without TASK_DONE",
            }
