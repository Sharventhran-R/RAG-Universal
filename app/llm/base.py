"""The ``LLMClient`` interface.

The only place a model client is imported is ``app/llm/``. The query pipeline
depends on this protocol, nothing else.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


class LLMError(Exception):
    """Base for anything the LLM layer raises."""


class LLMUnavailable(LLMError):
    """The model backend could not be reached or refused the request. The API
    maps this to HTTP 503 (not 500) -- it is an environment problem, not a bug."""


@runtime_checkable
class LLMClient(Protocol):
    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
    ) -> str: ...
