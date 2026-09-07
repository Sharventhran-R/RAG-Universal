"""Embedding: the ``Embedder`` protocol and its two implementations."""

from app.embed.base import Embedder
from app.embed.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder", "get_embedder"]


def get_embedder(*, fake: bool = False) -> Embedder:
    """Default wiring. ``fake=True`` (or tests) → deterministic offline embedder;
    otherwise the real local bge model (imported lazily here)."""
    if fake:
        return FakeEmbedder()
    from app.embed.bge import BgeEmbedder

    return BgeEmbedder()
