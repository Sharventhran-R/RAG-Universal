"""Deterministic offline ``LLMClient`` for the CI test tier.

``response`` may be:

* ``None``  -- the default heuristic: echo a grounded-looking sentence citing
  the first ``[chunk_id | ...]`` header found in the prompt, or the
  insufficient-context sentinel if there are no passages;
* ``str``   -- always return this exact string;
* callable ``(system, prompt) -> str`` -- compute the reply per call.

Every call is recorded in ``.calls``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Optional

from app.config import get_settings

_HEADER_RE = re.compile(r"\[([A-Za-z0-9_-]+) \|")


class FakeLLM:
    def __init__(self, response: str | Callable[[str, str], str] | None = None) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        self.calls.append(
            {"system": system, "prompt": prompt, "temperature": temperature, "stop": stop}
        )
        if callable(self._response):
            return self._response(system, prompt)
        if self._response is not None:
            return self._response
        match = _HEADER_RE.search(prompt)
        if match is None:
            return get_settings().insufficient_sentinel
        return f"According to the documents the figure is 42 [{match.group(1)}]."
