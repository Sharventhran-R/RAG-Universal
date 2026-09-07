"""Ollama-backed ``LLMClient``.

Talks to ``{OLLAMA_HOST}/api/chat`` with ``stream=false``. Any transport-level
failure (host down, timeout, non-2xx) surfaces as :class:`LLMUnavailable` so the
API can answer 503 rather than 500.
"""

from __future__ import annotations

from typing import Optional

import httpx

from app.config import Settings, get_settings
from app.llm.base import LLMUnavailable

_GENERATE_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_PING_TIMEOUT = httpx.Timeout(3.0)


class OllamaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        s = settings or get_settings()
        self._host = s.ollama_host.rstrip("/")
        self._model = s.ollama_model
        self._transport = transport  # test seam

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        options: dict = {"temperature": temperature}
        if stop:
            options["stop"] = stop
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": options,
        }
        try:
            async with httpx.AsyncClient(
                timeout=_GENERATE_TIMEOUT, transport=self._transport
            ) as client:
                resp = await client.post(f"{self._host}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailable(
                f"Ollama {self._host} returned {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama {self._host} unreachable: {exc}") from exc

        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMUnavailable(f"Ollama {self._host} sent an unexpected body") from exc

    async def ping(self) -> bool:
        """Best-effort reachability check for startup logging. Never raises."""
        try:
            async with httpx.AsyncClient(
                timeout=_PING_TIMEOUT, transport=self._transport
            ) as client:
                return (await client.get(f"{self._host}/api/tags")).status_code == 200
        except httpx.HTTPError:
            return False
