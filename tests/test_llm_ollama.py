from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.llm import LLMUnavailable
from app.llm.ollama import OllamaClient

CFG = Settings(ollama_host="http://ollama.test:11434", ollama_model="llama3.1:8b")


def _client(handler) -> OllamaClient:
    return OllamaClient(CFG, transport=httpx.MockTransport(handler))


async def test_posts_chat_payload_and_returns_message_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "the answer"}})

    out = await _client(handler).generate(system="rules", prompt="q", temperature=0.0)
    assert out == "the answer"
    assert seen["url"] == "http://ollama.test:11434/api/chat"
    assert seen["body"]["model"] == "llama3.1:8b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"][0] == {"role": "system", "content": "rules"}
    assert seen["body"]["options"]["temperature"] == 0.0


async def test_stop_sequences_are_forwarded_only_when_present():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["options"]["stop"] == ["\n\n"]
        return httpx.Response(200, json={"message": {"content": "x"}})

    await _client(handler).generate(system="s", prompt="p", stop=["\n\n"])


async def test_connection_failure_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailable):
        await _client(handler).generate(system="s", prompt="p")


async def test_non_2xx_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        await _client(lambda r: httpx.Response(503)).generate(system="s", prompt="p")


async def test_unexpected_body_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        await _client(lambda r: httpx.Response(200, json={"oops": True})).generate(
            system="s", prompt="p"
        )


async def test_ping_is_true_on_200_and_false_on_error():
    assert await _client(lambda r: httpx.Response(200)).ping() is True

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert await _client(boom).ping() is False
