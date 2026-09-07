"""Real local embedder: sentence-transformers ``BAAI/bge-small-en-v1.5``.

The only module in the codebase that imports sentence-transformers, and it does
so lazily (the import + model load happen on first use, not at construction) so
importing ``app.embed`` stays cheap and offline-friendly.

Not exercised by the default test tier; ``tests/test_embed_bge.py`` is gated on
``-m local_llm``.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from app.config import Settings, get_settings


class BgeEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.dim = self._settings.embed_dim
        self._query_prefix = self._settings.embed_query_prefix

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._settings.embed_model)
        actual = int(model.get_sentence_embedding_dimension())
        if actual != self.dim:
            raise RuntimeError(
                f"EMBED_DIM={self.dim} but {self._settings.embed_model!r} "
                f"produces {actual}-dim vectors — fix EMBED_DIM or the model."
            )
        return model

    @property
    def max_tokens(self) -> int:
        return int(self._model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        ids = self._model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(ids["input_ids"])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)
