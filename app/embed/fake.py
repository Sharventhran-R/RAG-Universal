"""Deterministic, offline stand-in for the real embedder.

Feature-hashing bag of words: each token is hashed to a dimension and a sign and
summed, then the vector is L2-normalized. Same text always yields the same
vector; texts that share vocabulary land closer in cosine space than texts that
don't. Good enough for wiring, filter, and retrieval-ordering tests; it is not a
semantic model.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.config import get_settings

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class FakeEmbedder:
    max_tokens = 512

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or get_settings().embed_dim

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            v[idx] += 1.0 if h[4] & 1 else -1.0
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            v[0] = 1.0
            return v
        return v / norm
