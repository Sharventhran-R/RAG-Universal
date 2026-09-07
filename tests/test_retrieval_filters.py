from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("faiss")

from app.db import repo
from app.db.connection import transaction
from app.embed import FakeEmbedder
from app.index.store import IndexStore
from app.paths import DataPaths
from app.retrieval.filters import ChunkFilter, resolve_allowed_ids, retrieve, search

DIM = 16


@pytest.fixture
def env(conn: sqlite3.Connection, tmp_path):
    """A session with two ready documents (pdf + csv), their chunks in SQLite
    and their vectors in a real FAISS index."""
    paths = DataPaths(tmp_path)
    paths.ensure()
    emb = FakeEmbedder(dim=DIM)
    store = IndexStore(paths, DIM, cache_size=4)

    session = repo.create_session(conn, "s")
    pdf = repo.create_document(conn, session_id=session.id, filename="report.pdf",
                               mimetype=None, file_type="pdf", file_sha256="h1", byte_size=1)
    csv = repo.create_document(conn, session_id=session.id, filename="sales.csv",
                               mimetype=None, file_type="csv", file_sha256="h2", byte_size=1)
    repo.set_document_status(conn, pdf.id, "ready")
    repo.set_document_status(conn, csv.id, "ready")

    pdf_texts = ["annual revenue grew in every region",
                 "the weather forecast is cloudy",
                 "revenue by region and quarter"]
    csv_texts = ["regional sales totals for the year"]
    with transaction(conn):
        pdf_chunks = repo.replace_chunks(conn, pdf.id, [
            dict(session_id=session.id, ord=i, text=t, embed_input=t, token_count=5,
                 embedded=True, section_path=[]) for i, t in enumerate(pdf_texts)
        ])
    with transaction(conn):
        csv_chunks = repo.replace_chunks(conn, csv.id, [
            dict(session_id=session.id, ord=0, text=csv_texts[0], embed_input=csv_texts[0],
                 token_count=5, embedded=True, section_path=[])
        ])

    index = store.get_or_create(session.id)
    index.add([c.faiss_id for c in pdf_chunks], emb.embed_documents(pdf_texts))
    index.add([c.faiss_id for c in csv_chunks], emb.embed_documents(csv_texts))

    return dict(conn=conn, session=session, pdf=pdf, csv=csv,
                pdf_chunks=pdf_chunks, csv_chunks=csv_chunks, index=index, emb=emb)


# --- resolve_allowed_ids -------------------------------------------------


def test_session_isolation_is_enforced_in_the_resolver(env):
    all_ids = resolve_allowed_ids(env["conn"], ChunkFilter(session_id=env["session"].id))
    assert set(all_ids) == {c.faiss_id for c in env["pdf_chunks"] + env["csv_chunks"]}
    assert resolve_allowed_ids(env["conn"], ChunkFilter(session_id="someone-else")) == []


def test_only_ready_and_embedded_chunks_are_eligible(env):
    conn, session = env["conn"], env["session"]
    # a third document that never finished ingesting
    draft = repo.create_document(conn, session_id=session.id, filename="wip.pdf",
                                 mimetype=None, file_type="pdf", file_sha256="h3", byte_size=1)
    with transaction(conn):
        repo.replace_chunks(conn, draft.id, [
            dict(session_id=session.id, ord=0, text="pending", embed_input="pending",
                 token_count=1, embedded=True, section_path=[]),
        ])
    ids = resolve_allowed_ids(conn, ChunkFilter(session_id=session.id))
    assert all(i not in ids for i in repo.document_faiss_ids(conn, session.id, draft.id))

    # a non-embeddable chunk on a ready doc is excluded too
    with transaction(conn):
        extra = repo.replace_chunks(conn, env["pdf"].id, [
            dict(session_id=session.id, ord=0, text="summary", embed_input="s",
                 token_count=1, embedded=True, section_path=[]),
            dict(session_id=session.id, ord=1, text="rows", embed_input="r",
                 token_count=1, embedded=False, section_path=[]),
        ])
    ids = resolve_allowed_ids(conn, ChunkFilter(session_id=session.id))
    assert extra[0].faiss_id in ids and extra[1].faiss_id not in ids


def test_file_type_and_filename_filters(env):
    conn, session = env["conn"], env["session"]
    pdf_ids = {c.faiss_id for c in env["pdf_chunks"]}
    assert set(resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, file_type="pdf"))) == pdf_ids
    assert set(resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, filename="sales.csv"))) == {
        env["csv_chunks"][0].faiss_id
    }


def test_document_ids_filter_and_empty_list_short_circuits(env):
    conn, session = env["conn"], env["session"]
    got = resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, document_ids=[env["pdf"].id]))
    assert set(got) == {c.faiss_id for c in env["pdf_chunks"]}
    assert resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, document_ids=[])) == []


def test_uploaded_before_and_after(env):
    conn, session = env["conn"], env["session"]
    future = datetime.now(timezone.utc) + timedelta(days=1)
    past = datetime.now(timezone.utc) - timedelta(days=1)
    assert resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, uploaded_after=future)) == []
    assert set(resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, uploaded_before=future))) != set()
    assert resolve_allowed_ids(conn, ChunkFilter(session_id=session.id, uploaded_before=past)) == []


# --- retrieve (resolve -> prefiltered search) --------------------------


def test_retrieve_honours_the_filter_and_ranks_by_similarity(env):
    hits = retrieve(
        env["conn"], env["index"], env["emb"].embed_query("revenue by region"),
        ChunkFilter(session_id=env["session"].id, file_type="pdf"), top_k=5,
    )
    pdf_ids = {c.faiss_id for c in env["pdf_chunks"]}
    assert hits and {h.faiss_id for h in hits} <= pdf_ids
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)
    # the "revenue by region and quarter" chunk should top it
    assert hits[0].faiss_id == env["pdf_chunks"][2].faiss_id


def test_retrieve_top_k_is_applied_to_the_filtered_set(env):
    hits = retrieve(
        env["conn"], env["index"], env["emb"].embed_query("anything"),
        ChunkFilter(session_id=env["session"].id, file_type="csv"), top_k=8,
    )
    assert len(hits) <= 1  # only one csv chunk exists, despite top_k=8


def test_retrieve_returns_empty_when_session_has_no_index(env):
    assert retrieve(env["conn"], None, env["emb"].embed_query("x"),
                    ChunkFilter(session_id=env["session"].id), top_k=5) == []


def test_search_short_circuits_on_empty_allowed(env):
    assert search(env["index"], env["emb"].embed_query("x"), [], top_k=5) == []
