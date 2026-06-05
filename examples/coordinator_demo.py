"""Coordinator pattern: one main agent delegates to sub-agents via SubAgentTool.

Contrasts with ``complex_sysaudit_demo`` which uses the static Orchestrator:
here, the main agent's ReAct loop decides *per step* whether to delegate
and to whom. Useful when the task path is exploratory — you don't know
which specialist you'll need until you've seen partial results.

What this shows:
  - Three roles wired as sub-agents: ``researcher`` (HTTPFetch), ``analyst``
    (LLM-only reasoning), ``reporter`` (LLM-only synthesis).
  - A ``coordinator`` agent whose only tools are ``delegate_<id>`` —
    everything happens through delegation.
  - The coordinator can delegate **in parallel** by emitting
    ``actions: [delegate_research, delegate_analyse]`` — both sub-agent
    streams interleave in the parent's event output via fan-in.

Run:
    OPENAI_API_KEY=sk-... python examples/coordinator_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents.base import AgentConfig, BaseAgent
from harness.console import ConsoleRenderer
from harness.events import EventType
from harness.runtime import (
    AgentRegistry,
    AgentRuntime,
    BudgetGuard,
    GuardrailConfig,
    ToolRegistry,
    Tracer,
)
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore
from tools.builtin.http_fetch import HTTPFetch
from tools.builtin.subagent import SubAgentTool

GOAL = (
    "Find the current Python release on python.org's downloads page, "
    "evaluate whether agent-harness's >=3.10 requirement is reasonable, "
    "and produce a one-paragraph reporter-style summary."
)


def _build_sub_agent(
    *,
    agent_id: str,
    role: str,
    system_prompt: str,
    tools: dict,
    llm,
    memory: MemoryManager,
    guard: BudgetGuard,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            allowed_tools=list(tools.keys()),
            max_steps=5,
            # Sub-agents won't delegate further in this demo; depth=0 would
            # also do, but leaving the default makes the recursion guard
            # demonstrable if you add another layer.
        ),
        tools=tools,
        memory=memory,
        tracer=Tracer(),
        guard=guard,
        llm=llm,
    )


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(2)

    from harness.llm.openai import OpenAILLM  # noqa: PLC0415 — optional dep

    llm = OpenAILLM(model="gpt-4o-mini")
    semantic = InMemorySemanticStore()
    episodic = InMemoryEpisodicStore()
    memory = MemoryManager(
        semantic_store=semantic,
        episodic_store=episodic,
        llm=llm,
        reconcile_on_write=False,  # keep the demo focused on delegation, not memory
    )
    guard = BudgetGuard(GuardrailConfig(max_total_cost_usd=2.0, max_wall_time_seconds=120))

    # ── Build the sub-agents ──────────────────────────────────────────────
    researcher = _build_sub_agent(
        agent_id="researcher",
        role="fetches and quotes from public URLs",
        system_prompt=(
            "You are a research agent. Use http_fetch to retrieve evidence "
            "from public URLs. Return concrete quotes with the source URL."
        ),
        tools={"http_fetch": HTTPFetch()},
        llm=llm,
        memory=memory,
        guard=guard,
    )
    analyst = _build_sub_agent(
        agent_id="analyst",
        role="evaluates findings against requirements",
        system_prompt=(
            "You are an analyst. Given factual findings and a requirement, "
            "produce a single judgment paragraph: reasonable or not, and why."
        ),
        tools={},
        llm=llm,
        memory=memory,
        guard=guard,
    )
    reporter = _build_sub_agent(
        agent_id="reporter",
        role="writes the final summary paragraph",
        system_prompt=(
            "You are a reporter. Given research findings and analysis, "
            "write exactly one paragraph in clear reporter style. No bullets."
        ),
        tools={},
        llm=llm,
        memory=memory,
        guard=guard,
    )

    # ── Coordinator's only tools: the three delegate_X ────────────────────
    delegate_research = SubAgentTool(researcher, name="delegate_research")
    delegate_analyse = SubAgentTool(analyst, name="delegate_analyse")
    delegate_report = SubAgentTool(reporter, name="delegate_report")

    tool_registry = ToolRegistry()
    tool_registry.register(delegate_research)
    tool_registry.register(delegate_analyse)
    tool_registry.register(delegate_report)

    coordinator_config = AgentConfig(
        agent_id="coordinator",
        role=("decides which sub-agent to invoke and in what order: research → analyse → report"),
        system_prompt=(
            "You coordinate a research / analyse / report pipeline by delegating "
            "to sub-agents. Delegate `delegate_research` first (with a concrete "
            "URL or query in `task`), then `delegate_analyse` (passing the "
            "research findings as `task`), then `delegate_report` (passing the "
            "combined findings + analysis). You may delegate in parallel by "
            "emitting `actions: [...]` when two delegations don't depend on "
            "each other. Finish with the reporter's paragraph as your answer."
        ),
        allowed_tools=[
            delegate_research.name,
            delegate_analyse.name,
            delegate_report.name,
        ],
        max_steps=8,
    )

    agent_registry = AgentRegistry()
    agent_registry.register(coordinator_config)

    runtime = AgentRuntime(
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        memory=memory,
        llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=2.0,
            max_wall_time_seconds=120,
        ),
    )

    renderer = ConsoleRenderer()

    print(f"\nGOAL: {GOAL}\n")
    renderer.sep("═")

    async for event in runtime.run_agent_stream("coordinator", GOAL):
        # ConsoleRenderer handles parent_agent_id-tagged events by labeling
        # the sub-agent's id; the human can see indentation by reading
        # `event.parent_agent_id` themselves if they want deeper formatting.
        renderer.render(event)
        if event.type == EventType.TASK_DONE:
            renderer.sep("═")
            print("\nFINAL ANSWER\n")
            print(event.payload.get("answer", "(no answer)"))
            renderer.sep("═")
            renderer.render_budget(event.payload.get("budget"))


if __name__ == "__main__":
    asyncio.run(main())
