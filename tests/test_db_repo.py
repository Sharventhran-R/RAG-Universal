from __future__ import annotations

import sqlite3
import threading

import pytest

from app.db.connection import connect, transaction
from app.db import repo


@pytest.fixture
def session(conn: sqlite3.Connection):
    return repo.create_session(conn, "s1")


@pytest.fixture
def document(conn: sqlite3.Connection, session):
    return repo.create_document(
        conn,
        session_id=session.id,
        filename="a.pdf",
        mimetype="application/pdf",
        file_type="pdf",
        file_sha256="abc123",
        byte_size=10,
    )


def test_session_and_document_crud(conn, session, document):
    assert repo.get_session(conn, session.id) == session
    assert repo.get_document(conn, document.id).status == "queued"
    assert [d.id for d in repo.list_documents(conn, session.id)] == [document.id]


def test_set_document_status_and_counts(conn, document):
    repo.set_document_status(conn, document.id, "failed", detail="parse", error="boom")
    d = repo.get_document(conn, document.id)
    assert (d.status, d.status_detail, d.error) == ("failed", "parse", "boom")
    repo.set_document_counts(conn, document.id, block_count=5, chunk_count=3)
    d = repo.get_document(conn, document.id)
    assert (d.block_count, d.chunk_count) == (5, 3)


def test_extraction_flags_default_empty_and_round_trip(conn, document):
    assert repo.get_document(conn, document.id).extraction_flags == []
    repo.set_document_extraction_flags(conn, document.id, ["low_confidence_table", "sheet_empty"])
    assert repo.get_document(conn, document.id).extraction_flags == [
        "low_confidence_table",
        "sheet_empty",
    ]


def test_two_workers_never_double_claim(conn, db_path, document, session):
    repo.enqueue_job(conn, document.id, session.id)
    other = connect(db_path)
    try:
        a = repo.claim_job(conn, "w1", claim_timeout_s=300, max_attempts=3)
        b = repo.claim_job(other, "w2", claim_timeout_s=300, max_attempts=3)
    finally:
        other.close()
    assert a is not None and b is None


def test_concurrent_claim_is_exclusive_under_real_contention(conn, db_path, session):
    """8 threads, each its own sqlite3 connection, all release from one barrier
    and race claim_job against 3 queued jobs. BEGIN IMMEDIATE + WAL must yield
    exactly 3 winners with 3 distinct job ids -- no mocks, no call ordering."""
    n_jobs, n_threads = 3, 8
    for i in range(n_jobs):
        doc = repo.create_document(
            conn, session_id=session.id, filename=f"{i}.pdf", mimetype="application/pdf",
            file_type="pdf", file_sha256=f"sha{i}", byte_size=1,
        )
        repo.enqueue_job(conn, doc.id, session.id)

    barrier = threading.Barrier(n_threads)
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        c = connect(db_path)
        try:
            barrier.wait()
            job = repo.claim_job(c, f"w{wid}", claim_timeout_s=300, max_attempts=3)
            if job is not None:
                with lock:
                    claimed.append(job.id)
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(claimed) == n_jobs
    assert len(set(claimed)) == n_jobs  # every winner got a different job


def test_fail_requeues_and_increments_attempts(conn, document, session):
    job = repo.enqueue_job(conn, document.id, session.id)
    first = repo.claim_job(conn, "w1", claim_timeout_s=300, max_attempts=3)
    assert first.attempts == 1
    repo.fail_job(conn, job.id, "kaboom")
    assert repo.get_job(conn, job.id).status == "queued"
    second = repo.claim_job(conn, "w1", claim_timeout_s=300, max_attempts=3)
    assert second.attempts == 2
    assert second.last_error == "kaboom"


def test_stale_claim_is_reaped(conn, document, session):
    job = repo.enqueue_job(conn, document.id, session.id)
    repo.claim_job(conn, "w1", claim_timeout_s=300, max_attempts=3)
    assert repo.claim_job(conn, "w2", claim_timeout_s=300, max_attempts=3) is None
    # negative timeout => every running job is already stale
    assert repo.reap_stale_jobs(conn, claim_timeout_s=-1) == 1
    reclaimed = repo.claim_job(conn, "w2", claim_timeout_s=300, max_attempts=3)
    assert reclaimed is not None and reclaimed.attempts == 2


def test_attempt_budget_fails_the_job(conn, document, session):
    job = repo.enqueue_job(conn, document.id, session.id)
    seen = []
    for i in range(6):
        claimed = repo.claim_job(conn, "w1", claim_timeout_s=300, max_attempts=3)
        seen.append(None if claimed is None else claimed.attempts)
        if claimed is not None:
            repo.fail_job(conn, claimed.id, f"e{i}")
    assert seen == [1, 2, 3, None, None, None]
    assert repo.get_job(conn, job.id).status == "failed"


def test_replace_chunks_is_idempotent_and_counts_embedded_only(conn, document, session):
    rows = [
        dict(session_id=session.id, ord=0, text="summary", embed_input="x",
             token_count=3, embedded=True, sheet="Q3", section_path=["Q3"]),
        dict(session_id=session.id, ord=1, text="rows 1-50", embed_input="y",
             token_count=9, embedded=False, sheet="Q3", section_path=["Q3", "rows 1-50"]),
    ]
    with transaction(conn):
        first = repo.replace_chunks(conn, document.id, rows)
    with transaction(conn):
        second = repo.replace_chunks(conn, document.id, rows)

    assert repo.count_chunks(conn, session.id, document.id, embedded_only=True) == 1
    assert repo.count_chunks(conn, session.id, document.id, embedded_only=False) == 2
    # re-chunk replaced, not appended
    assert {c.faiss_id for c in first}.isdisjoint({c.faiss_id for c in second})


def test_get_chunks_by_faiss_ids_preserves_request_order(conn, document, session):
    rows = [
        dict(session_id=session.id, ord=i, text=f"t{i}", embed_input="x",
             token_count=1, embedded=True, section_path=[])
        for i in range(3)
    ]
    with transaction(conn):
        chunks = repo.replace_chunks(conn, document.id, rows)
    ids = [chunks[2].faiss_id, chunks[0].faiss_id]
    assert [c.text for c in repo.get_chunks_by_faiss_ids(conn, session.id, ids)] == ["t2", "t0"]


def test_get_chunks_by_faiss_ids_will_not_cross_sessions(conn, document, session):
    with transaction(conn):
        chunks = repo.replace_chunks(
            conn, document.id,
            [dict(session_id=session.id, ord=0, text="secret", embed_input="x",
                  token_count=1, embedded=True, section_path=[])],
        )
    other = repo.create_session(conn, "other")
    assert repo.get_chunks_by_faiss_ids(conn, other.id, [chunks[0].faiss_id]) == []


def test_delete_document_rows_returns_embedded_faiss_ids_and_cascades(conn, document, session):
    rows = [
        dict(session_id=session.id, ord=0, text="a", embed_input="x",
             token_count=1, embedded=True, section_path=[]),
        dict(session_id=session.id, ord=1, text="b", embed_input="y",
             token_count=1, embedded=False, section_path=[]),
    ]
    with transaction(conn):
        chunks = repo.replace_chunks(conn, document.id, rows)
    repo.enqueue_job(conn, document.id, session.id)

    with transaction(conn):
        removed = repo.delete_document_rows(conn, session.id, document.id)

    assert removed == [chunks[0].faiss_id]
    assert repo.get_document(conn, document.id) is None
    assert repo.count_chunks(conn, session.id, document.id, embedded_only=False) == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM ingest_jobs WHERE document_id = ?", (document.id,)
    ).fetchone()[0] == 0
