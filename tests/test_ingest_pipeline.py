from __future__ import annotations

import hashlib
import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("pymupdf")
import pymupdf

from app.db import repo
from app.embed import FakeEmbedder
from app.index.store import IndexStore
from app.ingest import process_document, reconcile
from app.paths import DataPaths

DIM = 48


@pytest.fixture
def rig(conn: sqlite3.Connection, tmp_path: Path):
    paths = DataPaths(tmp_path)
    paths.ensure()
    store = IndexStore(paths, DIM, 8)
    session = repo.create_session(conn, "s")
    embedder = FakeEmbedder(dim=DIM)

    def ingest(filename: str, data: bytes, *, mimetype: str | None = None) -> str:
        sha = hashlib.sha256(data).hexdigest()
        blob = paths.blob_path(session.id, sha)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        doc = repo.create_document(
            conn, session_id=session.id, filename=filename, mimetype=mimetype,
            file_type=None, file_sha256=sha, byte_size=len(data),
        )
        repo.enqueue_job(conn, doc.id, session.id)
        job = repo.claim_job(conn, "t", claim_timeout_s=300, max_attempts=3)
        status = process_document(
            conn, job, paths=paths, embedder=embedder, index_store=store,
        )
        return doc.id, status

    return dict(conn=conn, paths=paths, store=store, session=session, embedder=embedder, ingest=ingest)


def test_text_document_reaches_ready_with_vectors(rig):
    doc_id, status = rig["ingest"]("n.txt", b"A short note about ingestion pipelines.")
    assert status == "ready"
    d = repo.get_document(rig["conn"], doc_id)
    assert d.chunk_count == 1 and d.block_count == 1
    assert len(rig["store"].get(rig["session"].id)) == 1


def test_unknown_type_is_marked_unsupported_not_failed(rig):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.bin", b"data")
    doc_id, status = rig["ingest"]("archive.zip", buf.getvalue(), mimetype="application/zip")
    assert status == "unsupported"
    d = repo.get_document(rig["conn"], doc_id)
    assert d.status == "unsupported" and d.error is None


def test_pdf_with_no_text_layer_is_empty_no_text(rig):
    doc = pymupdf.open()
    doc.new_page()  # blank
    doc_id, status = rig["ingest"]("scan.pdf", doc.tobytes(), mimetype="application/pdf")
    assert status == "empty_no_text"
    d = repo.get_document(rig["conn"], doc_id)
    assert d.status == "empty_no_text" and d.chunk_count == 0
    assert "pdf_no_text_layer" in d.extraction_flags


def test_ingest_error_marks_failed_and_requeues_the_job(rig):
    conn, session = rig["conn"], rig["session"]
    sha = hashlib.sha256(b"whatever").hexdigest()
    doc = repo.create_document(
        conn, session_id=session.id, filename="ghost.txt", mimetype="text/plain",
        file_type=None, file_sha256=sha, byte_size=7,
    )  # note: no blob written
    repo.enqueue_job(conn, doc.id, session.id)
    job = repo.claim_job(conn, "t", claim_timeout_s=300, max_attempts=3)
    status = process_document(
        conn, job, paths=rig["paths"], embedder=rig["embedder"], index_store=rig["store"],
    )
    assert status == "failed"
    d = repo.get_document(conn, doc.id)
    assert d.status == "failed" and d.error
    assert repo.get_job(conn, job.id).status == "queued"  # fail_job requeues


def test_reingest_replaces_vectors_rather_than_duplicating(rig):
    conn, session, store = rig["conn"], rig["session"], rig["store"]
    doc_id, _ = rig["ingest"]("n.txt", b"Reindex me please, twice over.")
    first = len(store.get(session.id))

    repo.enqueue_job(conn, doc_id, session.id)
    job = repo.claim_job(conn, "t", claim_timeout_s=300, max_attempts=3)
    status = process_document(
        conn, job, paths=rig["paths"], embedder=rig["embedder"], index_store=store,
    )
    assert status == "ready"
    assert len(store.get(session.id)) == first  # not doubled


def test_extraction_flags_persist_to_the_document_row(rig):
    pytest.importorskip("docx")
    from docx import Document

    doc = Document()
    doc.add_paragraph("Intro")
    t = doc.add_table(rows=1, cols=1)
    t.cell(0, 0).text = "a boxed sentence used only for layout"
    buf = io.BytesIO()
    doc.save(buf)
    doc_id, status = rig["ingest"](
        "layout.docx", buf.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert status == "ready"
    assert "low_confidence_table" in repo.get_document(rig["conn"], doc_id).extraction_flags


def test_reconcile_drops_chunk_rows_with_no_vector(rig):
    conn, session, store = rig["conn"], rig["session"], rig["store"]
    doc_id, _ = rig["ingest"]("n.txt", b"content that will be half deleted")
    faiss_ids = repo.document_faiss_ids(conn, session.id, doc_id)
    assert faiss_ids

    # simulate a crash: vectors gone from the index, rows still in SQLite
    store.get(session.id).remove(faiss_ids)
    dropped = reconcile(conn, store)
    assert dropped == len(faiss_ids)
    assert repo.count_chunks(conn, session.id, doc_id, embedded_only=False) == 0
