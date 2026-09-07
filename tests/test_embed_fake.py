from __future__ import annotations

import numpy as np

from app.embed import Embedder, FakeEmbedder


def test_satisfies_protocol_and_reports_dim():
    e = FakeEmbedder(dim=32)
    assert isinstance(e, Embedder)
    assert e.dim == 32
    assert e.max_tokens == 512


def test_vectors_are_float32_and_l2_normalized():
    e = FakeEmbedder(dim=64)
    q = e.embed_query("revenue grew in the third quarter")
    docs = e.embed_documents(["quarterly revenue report", "unrelated cooking recipe"])
    assert q.dtype == np.float32 and docs.dtype == np.float32
    assert q.shape == (64,) and docs.shape == (2, 64)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-5
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)


def test_deterministic():
    a, b = FakeEmbedder(dim=48), FakeEmbedder(dim=48)
    assert np.array_equal(a.embed_query("same text here"), b.embed_query("same text here"))


def test_lexical_overlap_ranks_higher_than_unrelated():
    e = FakeEmbedder(dim=256)
    q = e.embed_query("annual revenue by region")
    close = e.embed_query("revenue per region for the year")
    far = e.embed_query("photosynthesis in aquatic plants")
    assert float(q @ close) > float(q @ far)


def test_empty_and_whitespace_text_stay_unit_vectors():
    e = FakeEmbedder(dim=16)
    for text in ("", "   \n\t"):
        v = e.embed_query(text)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5
        assert not np.isnan(v).any()


def test_embed_documents_empty_list_returns_empty_matrix():
    e = FakeEmbedder(dim=8)
    out = e.embed_documents([])
    assert out.shape == (0, 8) and out.dtype == np.float32


def test_count_tokens_counts_words_and_punctuation():
    e = FakeEmbedder(dim=8)
    assert e.count_tokens("one two three") == 3
    assert e.count_tokens("a, b.") == 4  # a , b .
