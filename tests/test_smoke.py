"""End-to-end smoke test: ingest the demo corpus, then prove the three query
shapes work -- pure-semantic, metadata-filtered, and delete-then-unretrievable.

Fully offline: FakeEmbedder (lexical hash) + FakeLLM (cites the first context
header). No Ollama, no model download.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("pymupdf")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from build_fixtures import build  # noqa: E402

from app.db import repo  # noqa: E402
from app.db.connection import transaction  # noqa: E402
from app.embed import FakeEmbedder  # noqa: E402
from app.index.store import IndexStore  # noqa: E402
from app.ingest import process_document  # noqa: E402
from app.llm import FakeLLM  # noqa: E402
from app.paths import DataPaths  # noqa: E402
from app.query import QueryRequest, answer_query  # noqa: E402

DIM = 96


@pytest.fixture
def corpus(conn: sqlite3.Connection, tmp_path: Path):
    paths = DataPaths(tmp_path)
    paths.ensure()
    embedder = FakeEmbedder(dim=DIM)
    store = IndexStore(paths, DIM, 8)
    session = repo.create_session(conn, "demo")
    files = build(tmp_path / "fixtures")

    docs: dict[str, str] = {}
    for kind, path in files.items():
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        blob = paths.blob_path(session.id, sha)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        doc = repo.create_document(
            conn, session_id=session.id, filename=path.name,
            mimetype=None, file_type=None, file_sha256=sha, byte_size=len(data),
        )
        job = repo.enqueue_job(conn, doc.id, session.id)
        claimed = repo.claim_job(conn, "test", claim_timeout_s=300, max_attempts=3)
        status = process_document(
            conn, claimed, paths=paths, embedder=embedder, index_store=store,
        )
        assert status == "ready", (kind, status, repo.get_document(conn, doc.id).error)
        docs[kind] = doc.id

    return dict(conn=conn, paths=paths, session=session, store=store,
               embedder=embedder, docs=docs)


def _ask(corpus, question: str, **filters):
    return answer_query(
        QueryRequest(session_id=corpus["session"].id, question=question, **filters),
        conn=corpus["conn"],
        index=corpus["store"].get(corpus["session"].id),
        embedder=corpus["embedder"],
        llm=FakeLLM(),  # default heuristic: cite the first context header
    )


def test_ingest_populates_catalog(corpus):
    conn, session = corpus["conn"], corpus["session"]
    docs = repo.list_documents(conn, session.id)
    assert {d.filename for d in docs} == {"handbook.pdf", "sales.xlsx", "notes.docx"}
    assert all(d.status == "ready" for d in docs)
    assert all(d.chunk_count and d.chunk_count > 0 for d in docs)
    assert {d.file_type for d in docs} == {"pdf", "xlsx", "docx"}


async def test_pure_semantic_query_is_grounded_and_cited(corpus):
    res = await _ask(corpus, "In what city was Northwind founded?")
    assert res.insufficient_context is False
    assert res.citations, res.answer
    cite = res.citations[0]
    assert cite.filename == "handbook.pdf"
    assert "Rotterdam" in cite.snippet


async def test_metadata_filtered_query_only_touches_the_spreadsheet(corpus):
    res = await _ask(corpus, "Which regions are in the sales figures?", file_type="xlsx")
    assert res.insufficient_context is False
    assert res.citations
    assert {c.filename for c in res.citations} == {"sales.xlsx"}


async def test_deleted_document_becomes_unretrievable(corpus):
    conn, session, store = corpus["conn"], corpus["session"], corpus["store"]
    doc_id = corpus["docs"]["docx"]

    # sanity: the docx is reachable first
    before = await _ask(corpus, "When is the Rotterdam office open?")
    assert any(c.filename == "notes.docx" for c in before.citations)

    # delete: vectors first, persist, then metadata (ARCHITECTURE section 13)
    faiss_ids = repo.document_faiss_ids(conn, session.id, doc_id)
    assert faiss_ids
    index = store.get(session.id)
    index.remove(faiss_ids)
    index.persist()
    with transaction(conn):
        removed = repo.delete_document_rows(conn, session.id, doc_id)
    assert removed == faiss_ids

    assert repo.get_document(conn, doc_id) is None
    assert index.id_set().isdisjoint(faiss_ids)

    after = await _ask(corpus, "When is the Rotterdam office open?")
    assert all(c.filename != "notes.docx" for c in after.citations)
