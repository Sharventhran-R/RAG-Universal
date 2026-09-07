"""The ``Embedder`` interface.

The rest of the system depends on this protocol, never on
sentence-transformers directly (that import lives only in ``app/embed/bge.py``).
Two implementations: :class:`~app.embed.bge.BgeEmbedder` (real, local) and
:class:`~app.embed.fake.FakeEmbedder` (deterministic, offline, for CI).

Contract:

* ``embed_documents`` / ``embed_query`` return ``float32`` arrays,
  **L2-normalized** (so FAISS inner product == cosine). Shapes: ``(n, dim)``
  and ``(dim,)``.
* ``embed_query`` applies the asymmetric bge instruction prefix; callers pass
  the bare query.
* ``count_tokens`` uses the model's own tokenizer so the chunker can respect
  ``max_tokens``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int
    max_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: list[str]) -> "np.ndarray": ...

    def embed_query(self, text: str) -> "np.ndarray": ...
