"""LLM boundary: the ``LLMClient`` protocol and its implementations."""

from app.llm.base import LLMClient, LLMError, LLMUnavailable
from app.llm.fake import FakeLLM

__all__ = ["LLMClient", "LLMError", "LLMUnavailable", "FakeLLM", "get_llm"]


def get_llm(*, fake: bool | None = None) -> LLMClient:
    """``fake`` (or ``FAKE_MODELS=1``) → deterministic offline LLM; otherwise
    the real Ollama client."""
    if fake is None:
        from app.config import get_settings

        fake = get_settings().fake_models
    if fake:
        return FakeLLM()
    from app.llm.ollama import OllamaClient

    return OllamaClient()
