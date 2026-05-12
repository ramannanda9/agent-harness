"""ReAct loop smoke tests for BaseAgent."""

from __future__ import annotations

from agents.base import AgentConfig
from harness.events import EventType
from tests.conftest import EchoTool, FailingTool, ScriptedLLM, SlowTool


async def test_finish_on_first_step_returns_answer(agent_factory, llm: ScriptedLLM):
    config = AgentConfig(
        agent_id="a1",
        role="finishes immediately",
        system_prompt="finish.",
        allowed_tools=[],
    )
    agent = agent_factory(config)

    result = await agent.run("what is 2+2?")

    assert result["agent_id"] == "a1"
    assert result["confidence"] == 0.9
    assert result["steps"] == 1
    assert "error" not in result
    assert "done:" in result["answer"]


async def test_tool_call_then_finish(agent_factory, llm: ScriptedLLM):
    """Two-step: call echo tool, then finish using the observation."""
    step = {"n": 0}

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {
                "thought": "call echo",
                "action": "echo",
                "args": {"message": "hello"},
            }
        return {
            "thought": "I have the observation",
            "action": "finish",
            "answer": "echoed hello",
            "confidence": 0.8,
        }

    # ScriptedLLM default routes don't catch agent ReAct (no system match), so we
    # override the default by routing on the BaseAgent system prompt's "ReAct" marker.
    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id="a2",
        role="uses echo",
        system_prompt="You may use the echo tool. Follow the ReAct format.",
        allowed_tools=["echo"],
        max_steps=5,
    )
    agent = agent_factory(config, tools={"echo": EchoTool()})

    result = await agent.run("say hello")

    assert result["steps"] == 2
    assert result["answer"] == "echoed hello"
    assert step["n"] == 2


async def test_unknown_tool_returns_error_observation(agent_factory, llm: ScriptedLLM):
    """Unknown tool name should not crash — returns error string as observation."""
    step = {"n": 0}

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {"thought": "try", "action": "nope", "args": {}}
        # second step: should see the error in the observation
        last = messages[-1]["content"]
        assert "Error: tool 'nope' not available" in last
        return {
            "thought": "give up",
            "action": "finish",
            "answer": "no such tool",
            "confidence": 0.5,
        }

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id="a3",
        role="tries unknown tool",
        system_prompt="ReAct format please.",
        allowed_tools=[],
        max_steps=5,
    )
    agent = agent_factory(config)

    result = await agent.run("do something")

    assert result["steps"] == 2
    assert result["answer"] == "no such tool"


async def test_failing_tool_does_not_crash_loop(agent_factory, llm: ScriptedLLM):
    """A tool that raises should produce an error observation, not propagate."""
    step = {"n": 0}

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {"thought": "try", "action": "fail", "args": {}}
        last = messages[-1]["content"]
        assert "Tool error (fail)" in last
        return {"action": "finish", "answer": "handled", "confidence": 0.7, "thought": ""}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id="a4",
        role="uses failing tool",
        system_prompt="ReAct.",
        allowed_tools=["fail"],
    )
    agent = agent_factory(config, tools={"fail": FailingTool()})

    result = await agent.run("trigger failure")

    assert result["answer"] == "handled"


async def test_max_steps_returns_error_result(agent_factory, llm: ScriptedLLM):
    """Agent that never finishes should hit max_steps and return error."""

    def react(system, messages, kwargs):
        # always call echo, never finish
        return {"thought": "loop", "action": "echo", "args": {"message": "x"}}

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id="a5",
        role="loops",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        max_steps=3,
    )
    agent = agent_factory(config, tools={"echo": EchoTool()})

    result = await agent.run("loop forever")

    assert result["steps"] == 3
    assert result["confidence"] == 0.0
    assert "Max steps" in result["error"]


async def test_unparseable_llm_response_returns_error(agent_factory, llm: ScriptedLLM):
    """If the LLM returns garbage, the agent should error gracefully."""

    def react(system, messages, kwargs):
        return "this is not json at all"

    llm.routes = {"react": react}

    config = AgentConfig(
        agent_id="a6",
        role="parser test",
        system_prompt="ReAct.",
        allowed_tools=[],
    )
    agent = agent_factory(config)

    result = await agent.run("anything")

    assert result["confidence"] == 0.0
    assert "unparseable" in result["error"]


# ── Parallel tool calls ───────────────────────────────────────────────────────


async def test_parallel_actions_emit_two_action_and_two_observation_events(
    agent_factory, llm: ScriptedLLM
):
    """actions=[...] form: both ACTION and OBSERVATION events fire once per tool."""
    step = {"n": 0}

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {
                "thought": "call both at once",
                "actions": [
                    {"tool": "echo", "args": {"message": "a"}},
                    {"tool": "echo", "args": {"message": "b"}},
                ],
            }
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 0.9}

    llm.routes = {"react": react}
    config = AgentConfig(
        agent_id="par",
        role="parallel",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        max_steps=3,
    )
    agent = agent_factory(config, tools={"echo": EchoTool()})

    events = []
    async for ev in agent.run_stream("run both"):
        events.append(ev)

    action_events = [e for e in events if e.type == EventType.ACTION]
    obs_events = [e for e in events if e.type == EventType.OBSERVATION]
    assert len(action_events) == 2
    assert len(obs_events) == 2
    assert {e.payload["tool"] for e in action_events} == {"echo"}
    # Both actions happen before any observation (batch emit order)
    action_indices = [events.index(e) for e in action_events]
    obs_indices = [events.index(e) for e in obs_events]
    assert max(action_indices) < min(obs_indices)


async def test_parallel_actions_run_concurrently(agent_factory, llm: ScriptedLLM):
    """Two slow tools in actions=[...] finish faster than sequential would."""
    step = {"n": 0}
    slow1, slow2 = SlowTool(delay=0.05), SlowTool(delay=0.05)
    slow2.name = "slow2"

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {
                "thought": "run both slow tools",
                "actions": [
                    {"tool": "slow", "args": {"label": "x"}},
                    {"tool": "slow2", "args": {"label": "y"}},
                ],
            }
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0}

    llm.routes = {"react": react}
    config = AgentConfig(
        agent_id="par2",
        role="parallel",
        system_prompt="ReAct.",
        allowed_tools=["slow", "slow2"],
        max_steps=3,
    )
    agent = agent_factory(config, tools={"slow": slow1, "slow2": slow2})

    import time

    t0 = time.monotonic()
    await agent.run("run slow tools")
    elapsed = time.monotonic() - t0

    # Sequential would take ≥0.1s; concurrent should finish in ~0.05s (+overhead)
    assert elapsed < 0.09, f"tools did not run concurrently (elapsed={elapsed:.3f}s)"
    assert len(slow1.starts) == 1
    assert len(slow2.starts) == 1


async def test_parallel_actions_combined_observations_in_working_memory(
    agent_factory, llm: ScriptedLLM
):
    """Both tool results appear in a single Observations: message in working memory."""
    step = {"n": 0}
    seen_obs: list[str] = []

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {
                "thought": "parallel",
                "actions": [
                    {"tool": "echo", "args": {"message": "hello"}},
                    {"tool": "echo", "args": {"message": "world"}},
                ],
            }
        # Capture the last user message to inspect combined observations
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        seen_obs.append(last_user)
        return {"thought": "done", "action": "finish", "answer": "ok", "confidence": 1.0}

    llm.routes = {"react": react}
    config = AgentConfig(
        agent_id="par3",
        role="parallel",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        max_steps=3,
    )
    agent = agent_factory(config, tools={"echo": EchoTool()})
    await agent.run("run both")

    assert seen_obs, "second LLM call never happened"
    obs_msg = seen_obs[0]
    assert "Observations:" in obs_msg
    assert "hello" in obs_msg
    assert "world" in obs_msg
    # Both results present even though both tools share the same name
    assert obs_msg.count('"echo"') >= 2


async def test_parallel_actions_unknown_tool_does_not_crash(agent_factory, llm: ScriptedLLM):
    """Unknown tool in actions=[...] returns an error observation, loop continues."""
    step = {"n": 0}

    def react(system, messages, kwargs):
        step["n"] += 1
        if step["n"] == 1:
            return {
                "thought": "try ghost + echo",
                "actions": [
                    {"tool": "ghost", "args": {}},
                    {"tool": "echo", "args": {"message": "ok"}},
                ],
            }
        return {"thought": "done", "action": "finish", "answer": "handled", "confidence": 0.8}

    llm.routes = {"react": react}
    config = AgentConfig(
        agent_id="par4",
        role="parallel",
        system_prompt="ReAct.",
        allowed_tools=["echo"],
        max_steps=3,
    )
    agent = agent_factory(config, tools={"echo": EchoTool()})
    result = await agent.run("ghost + echo")

    assert result["answer"] == "handled"
    assert result["steps"] == 2
