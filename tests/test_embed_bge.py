"""Real bge-small embedder. Gated: `pytest -m local_llm` (downloads a model)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.local_llm

pytest.importorskip("sentence_transformers")

from app.embed.bge import BgeEmbedder


def test_dim_matches_config_and_vectors_are_normalized():
    e = BgeEmbedder()
    v = e.embed_documents(["hello world", "a second sentence"])
    assert v.shape == (2, e.dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_query_prefix_makes_asymmetric_pair_agree():
    e = BgeEmbedder()
    passage = e.embed_documents(["The capital of France is Paris."])[0]
    q = e.embed_query("What is the capital of France?")
    assert float(q @ passage) > 0.5


def test_count_tokens_is_monotonic():
    e = BgeEmbedder()
    assert e.count_tokens("short") < e.count_tokens("a considerably longer sentence than the first")
