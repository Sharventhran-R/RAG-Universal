"""LLM boundary: the ``LLMClient`` protocol and its implementations."""

from app.llm.base import LLMClient, LLMError, LLMUnavailable
from app.llm.fake import FakeLLM

__all__ = ["LLMClient", "LLMError", "LLMUnavailable", "FakeLLM", "get_llm"]


def get_llm(*, fake: bool = False) -> LLMClient:
    if fake:
        return FakeLLM()
    from app.llm.ollama import OllamaClient

    return OllamaClient()
