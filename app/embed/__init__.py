"""Embedding: the ``Embedder`` protocol and its two implementations."""

from app.embed.base import Embedder
from app.embed.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder", "get_embedder"]


def get_embedder(*, fake: bool | None = None) -> Embedder:
    """Default wiring. ``fake`` (or ``FAKE_MODELS=1``) → deterministic offline
    embedder; otherwise the real local bge model (imported lazily here)."""
    if fake is None:
        from app.config import get_settings

        fake = get_settings().fake_models
    if fake:
        return FakeEmbedder()
    from app.embed.bge import BgeEmbedder

    return BgeEmbedder()
