"""Vision support tests — FetchImage tool, multimodal WorkingMemory, agent routing."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.working import _IMAGE_TOKEN_ESTIMATE, WorkingMemory, _count_content, count_tokens

# ── _count_content ────────────────────────────────────────────────────────────


def test_count_content_string():
    assert _count_content("hello world", count_tokens) == count_tokens("hello world")


def test_count_content_text_block():
    blocks = [{"type": "text", "text": "hello world"}]
    assert _count_content(blocks, count_tokens) == count_tokens("hello world")


def test_count_content_image_block_uses_estimate():
    blocks = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}]
    assert _count_content(blocks, count_tokens) == _IMAGE_TOKEN_ESTIMATE


def test_count_content_mixed():
    blocks = [
        {"type": "text", "text": "look at this image:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
    ]
    expected = count_tokens("look at this image:") + _IMAGE_TOKEN_ESTIMATE
    assert _count_content(blocks, count_tokens) == expected


# ── WorkingMemory multimodal ──────────────────────────────────────────────────


class ConstantLLM:
    async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
        return {"text": "summary"}


@pytest.mark.asyncio
async def test_get_messages_passes_list_content_through():
    wm = WorkingMemory(llm=ConstantLLM(), max_tokens=10_000)
    image_block = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
    content = [{"type": "text", "text": "what is this?"}, image_block]
    await wm.append("user", content)
    msgs = wm.get_messages()
    assert msgs[0]["content"] == content


@pytest.mark.asyncio
async def test_image_content_counts_as_estimate():
    wm = WorkingMemory(llm=ConstantLLM(), max_tokens=10_000)
    image_block = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
    await wm.append("user", [image_block])
    assert wm.token_count() == _IMAGE_TOKEN_ESTIMATE


@pytest.mark.asyncio
async def test_summarize_renders_image_blocks_as_placeholder():
    """_summarize must strip image blocks to [image] so the text-only summarizer LLM can proceed."""
    received_content: list[str] = []

    class CaptureLLM:
        async def complete(self, system: Any, messages: Any, **_: Any) -> dict:
            received_content.append(messages[0]["content"])
            return {"text": "summary"}

    # Use token_counter=len and a tight budget to force eviction.
    wm = WorkingMemory(llm=CaptureLLM(), max_tokens=30, token_counter=len)
    await wm.append("system", "s", pinned=True)
    image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 10}}
    # Append image content — costs _IMAGE_TOKEN_ESTIMATE (500) >> budget of 30 → eviction fires.
    await wm.append("user", [{"type": "text", "text": "look:"}, image_block])

    assert received_content, "expected summarization LLM to be called"
    # The content passed to the summarizer should contain [image], not raw base64.
    summarize_input = received_content[0]
    assert "[image]" in summarize_input
    assert "base64" not in summarize_input


# ── FetchImage tool ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_image_returns_image_url_block():
    raw = b"\xff\xd8\xff"  # minimal JPEG magic bytes
    b64 = base64.b64encode(raw).decode()

    mock_response = MagicMock()
    mock_response.content = raw
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.builtin.fetch_image import FetchImage

        tool = FetchImage()
        result = await tool.execute("https://example.com/img.jpg")

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == f"data:image/jpeg;base64,{b64}"
    assert result["image_url"]["detail"] == "auto"


@pytest.mark.asyncio
async def test_fetch_image_detail_override():
    raw = b"\x89PNG"
    mock_response = MagicMock()
    mock_response.content = raw
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.builtin.fetch_image import FetchImage

        result = await FetchImage(detail="low").execute(
            "https://example.com/img.png", detail="high"
        )

    assert result["image_url"]["detail"] == "high"


@pytest.mark.asyncio
async def test_fetch_image_network_error_returns_error_dict():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.builtin.fetch_image import FetchImage

        result = await FetchImage().execute("https://example.com/img.jpg")

    assert "error" in result
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_fetch_image_too_large_returns_error():
    raw = b"x" * (21 * 1024 * 1024)  # 21 MiB — over the 20 MiB default cap
    mock_response = MagicMock()
    mock_response.content = raw
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        from tools.builtin.fetch_image import FetchImage

        result = await FetchImage().execute("https://example.com/huge.jpg")

    assert "error" in result
    assert "too large" in result["error"]


# ── Agent observation routing ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_routes_image_observation_as_content_block(agent_factory):
    """When a tool returns an image block, the agent appends it as a content list in WorkingMemory."""
    from agents.base import AgentConfig

    image_block = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,abc123"},
    }

    class ImageTool:
        name = "fetch_image"

        async def execute(self, url: str = "") -> dict:
            return image_block

    from tests.conftest import ScriptedLLM

    calls: list[list[dict]] = []

    class CaptureLLM(ScriptedLLM):
        async def complete(self, system, messages, **kwargs):
            calls.append(list(messages))
            return await super().complete(system, messages, **kwargs)

    llm = CaptureLLM(
        routes={
            "default finish": lambda s, m, kw: {
                "thought": "I see the image",
                "action": "finish",
                "answer": "done",
                "confidence": 0.9,
            }
        }
    )

    # First call: return fetch_image action; second call: finish
    step = 0

    async def _complete(system, messages, **kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return {
                "thought": "fetching image",
                "action": "fetch_image",
                "args": {"url": "https://example.com/img.jpg"},
            }
        return {"thought": "done", "action": "finish", "answer": "saw image", "confidence": 0.9}

    llm.complete = _complete

    config = AgentConfig(
        agent_id="vision_agent",
        role="vision test",
        system_prompt="You are a vision agent.",
        allowed_tools=["fetch_image"],
    )
    agent = agent_factory(config, tools={"fetch_image": ImageTool()})
    agent._llm = llm

    result = await agent.run("describe the image at https://example.com/img.jpg")
    assert result["answer"] == "saw image"

    # Inspect working memory messages for an image content block.
    msgs = agent._working_memory.get_messages()
    image_msgs = [m for m in msgs if isinstance(m["content"], list)]
    assert image_msgs, "expected at least one message with list content (image block)"
    blocks = image_msgs[0]["content"]
    image_found = any(isinstance(b, dict) and b.get("type") == "image_url" for b in blocks)
    assert image_found, f"no image_url block found in content: {blocks}"
