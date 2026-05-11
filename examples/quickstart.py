"""
examples/quickstart.py

Demonstrates the generic harness with a mock LLM client.
No real API keys needed — swap MockLLM for AnthropicLLM or OpenAILLM.

Shows:
  1. Register tools (pluggable — add any tool)
  2. Register agents (config-driven — no subclassing)
  3. Wire memory (semantic + episodic)
  4. Run any goal — orchestrator handles planning, execution, memory
"""
from __future__ import annotations

import asyncio
import json

from agents.base import AgentConfig
from harness.runtime import AgentRegistry, AgentRuntime, GuardrailConfig, ToolRegistry
from memory.manager import MemoryManager
from memory.stores import InMemoryEpisodicStore, InMemorySemanticStore

# ── Mock LLM ─────────────────────────────────────────────────────────────────
# Replace with real LLM client:
#
#   class AnthropicLLM:
#       async def complete(self, system, messages, **kwargs):
#           import anthropic
#           client = anthropic.AsyncAnthropic()
#           resp = await client.messages.create(
#               model="claude-sonnet-4-6",
#               max_tokens=1024,
#               system=system or "",
#               messages=messages,
#           )
#           return {"text": resp.content[0].text}

class MockLLM:
    """Deterministic mock for local testing without API keys."""

    def __init__(self) -> None:
        self._call_count = 0

    async def complete(self, system: str | None, messages: list[dict], **kwargs) -> dict:
        self._call_count += 1
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # planner response
        if "decompose" in (system or "").lower() or "available agents" in (system or "").lower():
            return {
                "tasks": [
                    {
                        "id": "t1",
                        "agent_id": "analyst",
                        "instruction": f"Analyse the following: {last_user[:100]}",
                        "depends_on": [],
                        "on_failure": "replan",
                    },
                    {
                        "id": "t2",
                        "agent_id": "reporter",
                        "instruction": "Summarise findings from t1",
                        "depends_on": ["t1"],
                        "on_failure": "skip",
                    },
                ],
                "rationale": "Analyse first, then report.",
            }

        # memory extraction response
        if "memory extraction" in (system or "").lower():
            return {
                "semantic_facts": {"last_run:goal": last_user[:80]},
                "episodic_summary": f"Completed analysis for: {last_user[:80]}",
                "metadata": {},
                "ttl_seconds": None,
            }

        # summarization response
        if "memory compressor" in (system or "").lower():
            return {"text": f"[Compressed {self._call_count}]: Key facts from prior context."}

        # synthesis response
        if "synthesis agent" in (system or "").lower():
            return {
                "answer": "Analysis complete. No critical issues found.",
                "confidence": 0.85,
                "conflicts": [],
                "unknowns": [],
            }

        # agent ReAct response — finish after one step
        return {
            "thought": "I have enough information to answer.",
            "action": "finish",
            "answer": f"Completed task: {last_user[:100]}",
            "confidence": 0.9,
        }


# ── Mock Tool ─────────────────────────────────────────────────────────────────

class EchoTool:
    """Minimal tool for testing — echoes its input."""
    name = "echo"

    async def execute(self, message: str = "") -> dict:
        return {"echo": message}

    def schema(self):
        return {
            "name": self.name,
            "description": "Echoes input back",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        }


# ── Wiring ────────────────────────────────────────────────────────────────────

async def main() -> None:
    llm = MockLLM()

    # 1 — register tools
    tools = (
        ToolRegistry()
        .register(EchoTool())
        # add real tools here:
        # .register(KubectlTool())
        # .register(DatadogQueryTool())
        # .register(SlackSearchTool())
    )

    # 2 — register agents (config-driven, no subclassing)
    agents = (
        AgentRegistry()
        .register(AgentConfig(
            agent_id="analyst",
            role="analyses problems and extracts key facts",
            system_prompt="You are an expert analyst. Investigate thoroughly.",
            allowed_tools=["echo"],
            max_steps=5,
        ))
        .register(AgentConfig(
            agent_id="reporter",
            role="synthesises findings into clear reports",
            system_prompt="You are a technical writer. Produce clear, concise reports.",
            allowed_tools=["echo"],
            max_steps=3,
        ))
        # add new domain agents here:
        # .register(AgentConfig(
        #     agent_id="diagnosis",
        #     role="diagnoses GPU and Kubernetes failures using metrics and logs",
        #     system_prompt=DIAGNOSIS_PROMPT,
        #     allowed_tools=["kubectl", "datadog_query"],
        # ))
    )

    # 3 — wire memory (swap InMemory* for Redis/Chroma in production)
    memory = MemoryManager(
        semantic_store=InMemorySemanticStore(),
        episodic_store=InMemoryEpisodicStore(),
        llm=llm,
    )

    # 4 — build runtime (wire once, run anything)
    runtime = AgentRuntime(
        agent_registry=agents,
        tool_registry=tools,
        memory=memory,
        llm=llm,
        guardrail_config=GuardrailConfig(
            max_total_cost_usd=5.0,
            max_wall_time_seconds=120,
            max_replan_count=2,
            confidence_threshold=0.6,
        ),
    )

    # 5 — run any goal
    print("=" * 60)
    print("Run 1: First goal (cold memory)")
    print("=" * 60)
    result = await runtime.run(
        "Investigate why GPU utilization dropped to 12% on worker-07"
    )
    print(f"Answer:     {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Replans:    {result['replan_count']}")
    print(f"Elapsed:    {result['budget']['elapsed_seconds']:.2f}s")

    print()
    print("=" * 60)
    print("Run 2: Different goal (episodic memory now populated)")
    print("=" * 60)
    result2 = await runtime.run(
        "Check data pipeline health for the feature store"
    )
    print(f"Answer:     {result2['answer']}")
    print(f"Confidence: {result2['confidence']}")

    print()
    print("=" * 60)
    print("Memory state after 2 runs")
    print("=" * 60)
    print(f"Episodic episodes stored: {memory._episodic.count()}")
    print(f"Semantic keys:            {memory._semantic.size()}")
    print(f"Memory conflicts:         {len(memory.get_conflict_log())}")

    print()
    print("Trace (Run 1):")
    for event in result["trace"]:
        payload_str = json.dumps(event["payload"], default=str)
        if len(payload_str) > 120:
            payload_str = payload_str[:120] + "…"
        print(f"  [{event['event_type']:15}] {event['agent_id']:15} {payload_str}")


if __name__ == "__main__":
    asyncio.run(main())
