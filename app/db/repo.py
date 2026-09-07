"""Data-access functions. Every function takes an open ``sqlite3.Connection``
as its first argument and never opens or closes one itself. Writers wrap their
work in :func:`app.db.connection.transaction`.

Single-purpose by design: the ingest worker composes job + document updates
into one transaction itself (slice 6); this module does not couple them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone

from app.db.connection import transaction
from app.db.models import Chunk, Document, IngestJob, Session

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


def create_session(conn: sqlite3.Connection, name: str) -> Session:
    sess = Session(id=_id("sess"), name=name, created_at=now_iso())
    with transaction(conn):
        conn.execute(
            "INSERT INTO sessions (id, name, created_at) VALUES (?, ?, ?)",
            (sess.id, sess.name, sess.created_at),
        )
    return sess


def get_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return Session.from_row(row) if row else None


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------


def create_document(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    filename: str,
    mimetype: str | None,
    file_type: str | None,
    file_sha256: str,
    byte_size: int,
) -> Document:
    ts = now_iso()
    doc_id = _id("doc")
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO documents
                (id, session_id, filename, mimetype, file_type, file_sha256,
                 byte_size, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """,
            (doc_id, session_id, filename, mimetype, file_type, file_sha256, byte_size, ts, ts),
        )
    got = get_document(conn, doc_id)
    assert got is not None
    return got


def get_document(conn: sqlite3.Connection, document_id: str) -> Document | None:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return Document.from_row(row) if row else None


def list_documents(conn: sqlite3.Connection, session_id: str) -> list[Document]:
    rows = conn.execute(
        "SELECT * FROM documents WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    return [Document.from_row(r) for r in rows]


def set_document_status(
    conn: sqlite3.Connection,
    document_id: str,
    status: str,
    *,
    detail: str | None = None,
    error: str | None = None,
) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE documents
               SET status = ?, status_detail = ?, error = ?, updated_at = ?
             WHERE id = ?
            """,
            (status, detail, error, now_iso(), document_id),
        )


def set_document_extractor(
    conn: sqlite3.Connection, document_id: str, name: str, version: str
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE documents SET extractor_name = ?, extractor_version = ?, updated_at = ? WHERE id = ?",
            (name, version, now_iso(), document_id),
        )


def set_document_counts(
    conn: sqlite3.Connection, document_id: str, *, block_count: int, chunk_count: int
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE documents SET block_count = ?, chunk_count = ?, updated_at = ? WHERE id = ?",
            (block_count, chunk_count, now_iso(), document_id),
        )


def set_document_extraction_flags(
    conn: sqlite3.Connection, document_id: str, flags: list[str]
) -> None:
    """Persist ``ParseResult.extraction_flags`` (the parser's "this happened"
    markers) on the document row as a JSON array."""
    with transaction(conn):
        conn.execute(
            "UPDATE documents SET extraction_flags = ?, updated_at = ? WHERE id = ?",
            (json.dumps(list(flags)), now_iso(), document_id),
        )


def delete_document_rows(
    conn: sqlite3.Connection, session_id: str, document_id: str
) -> list[int]:
    """Delete the document and its chunks + jobs, returning the embedded
    ``faiss_id``s that were removed so the caller can drop them from the index.

    ``session_id`` is required and scopes every statement -- a chunk row is
    never touched without its session, here as everywhere else.

    NOT wrapped in a transaction here -- the DELETE endpoint owns the boundary
    (remove vectors -> persist index -> these deletes -> COMMIT), see
    ARCHITECTURE section 13.
    """
    ids = [
        r["faiss_id"]
        for r in conn.execute(
            "SELECT faiss_id FROM chunks "
            "WHERE session_id = ? AND document_id = ? AND embedded = 1",
            (session_id, document_id),
        ).fetchall()
    ]
    conn.execute(
        "DELETE FROM chunks WHERE session_id = ? AND document_id = ?",
        (session_id, document_id),
    )
    conn.execute(
        "DELETE FROM ingest_jobs WHERE session_id = ? AND document_id = ?",
        (session_id, document_id),
    )
    conn.execute(
        "DELETE FROM documents WHERE session_id = ? AND id = ?",
        (session_id, document_id),
    )
    return ids


# --------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------


def replace_chunks(conn: sqlite3.Connection, document_id: str, chunks: Iterable[dict]) -> list[Chunk]:
    """Idempotent re-chunk: delete this document's chunks, insert the new set.
    Caller wraps in a transaction. Each dict carries every ``chunks`` column
    except ``faiss_id`` (assigned by AUTOINCREMENT) and ``created_at``.
    """
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    ts = now_iso()
    out: list[Chunk] = []
    for c in chunks:
        cur = conn.execute(
            """
            INSERT INTO chunks
                (id, document_id, session_id, ord, text, embed_input, token_count,
                 embedded, page_start, page_end, sheet, slide, char_start, char_end,
                 section_path, created_at)
            VALUES
                (:id, :document_id, :session_id, :ord, :text, :embed_input, :token_count,
                 :embedded, :page_start, :page_end, :sheet, :slide, :char_start, :char_end,
                 :section_path, :created_at)
            RETURNING *
            """,
            {
                "id": c.get("id") or _id("c"),
                "document_id": document_id,
                "session_id": c["session_id"],
                "ord": c["ord"],
                "text": c["text"],
                "embed_input": c["embed_input"],
                "token_count": c["token_count"],
                "embedded": 1 if c.get("embedded", True) else 0,
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "sheet": c.get("sheet"),
                "slide": c.get("slide"),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
                "section_path": json.dumps(c.get("section_path") or []),
                "created_at": ts,
            },
        )
        out.append(Chunk.from_row(cur.fetchone()))
    return out


def get_chunks_by_faiss_ids(
    conn: sqlite3.Connection, session_id: str, faiss_ids: Sequence[int]
) -> list[Chunk]:
    """Resolve chunks for citation display. ``session_id`` is required and is
    part of the WHERE clause: even though these ids come out of an
    already-session-scoped search, no chunk-read path in this module runs
    without its session (ARCHITECTURE section 9)."""
    if not faiss_ids:
        return []
    marks = ",".join("?" * len(faiss_ids))
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE session_id = ? AND faiss_id IN ({marks})",
        (session_id, *faiss_ids),
    ).fetchall()
    by_id = {r["faiss_id"]: Chunk.from_row(r) for r in rows}
    return [by_id[fid] for fid in faiss_ids if fid in by_id]


def count_chunks(
    conn: sqlite3.Connection,
    session_id: str,
    document_id: str,
    *,
    embedded_only: bool = True,
) -> int:
    sql = "SELECT COUNT(*) FROM chunks WHERE session_id = ? AND document_id = ?"
    if embedded_only:
        sql += " AND embedded = 1"
    return conn.execute(sql, (session_id, document_id)).fetchone()[0]


def document_faiss_ids(
    conn: sqlite3.Connection, session_id: str, document_id: str
) -> list[int]:
    """The embedded ``faiss_id``s currently belonging to a document -- the
    worker removes these from the session index before a re-embed, and the
    DELETE endpoint uses :func:`delete_document_rows` (which returns the same
    set). ``session_id`` scopes the read, as on every chunk-read path."""
    rows = conn.execute(
        "SELECT faiss_id FROM chunks "
        "WHERE session_id = ? AND document_id = ? AND embedded = 1 "
        "ORDER BY faiss_id",
        (session_id, document_id),
    ).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------
# ingest_jobs  (SQLite-as-queue -- see ARCHITECTURE section 6)
# --------------------------------------------------------------------------


def enqueue_job(conn: sqlite3.Connection, document_id: str, session_id: str) -> IngestJob:
    ts = now_iso()
    job_id = _id("job")
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO ingest_jobs
                (id, document_id, session_id, status, attempts, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', 0, ?, ?)
            """,
            (job_id, document_id, session_id, ts, ts),
        )
    got = get_job(conn, job_id)
    assert got is not None
    return got


def get_job(conn: sqlite3.Connection, job_id: str) -> IngestJob | None:
    row = conn.execute("SELECT * FROM ingest_jobs WHERE id = ?", (job_id,)).fetchone()
    return IngestJob.from_row(row) if row else None


def claim_job(
    conn: sqlite3.Connection,
    worker_id: str,
    *,
    claim_timeout_s: int,
    max_attempts: int,
) -> IngestJob | None:
    """Atomically claim the oldest runnable job: one that is ``queued`` or a
    ``running`` job whose claim went stale. Returns ``None`` when there is
    nothing to do.

    ``BEGIN IMMEDIATE`` + WAL means two workers can never claim the same row.
    A job that has now exhausted ``max_attempts`` is marked ``failed`` and
    ``None`` is returned (the worker also fails the document).
    """
    now = now_iso()
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(seconds=claim_timeout_s))
    with transaction(conn):
        row = conn.execute(
            """
            UPDATE ingest_jobs
               SET status = 'running',
                   claimed_at = :now,
                   claimed_by = :worker,
                   attempts = attempts + 1,
                   updated_at = :now
             WHERE id = (
                 SELECT id FROM ingest_jobs
                  WHERE status = 'queued'
                     OR (status = 'running'
                         AND (claimed_at IS NULL OR claimed_at < :cutoff))
                  ORDER BY created_at
                  LIMIT 1
             )
            RETURNING *
            """,
            {"now": now, "worker": worker_id, "cutoff": cutoff},
        ).fetchone()
        if row is None:
            return None
        job = IngestJob.from_row(row)
        if job.attempts > max_attempts:
            conn.execute(
                """
                UPDATE ingest_jobs
                   SET status = 'failed', updated_at = :now,
                       last_error = COALESCE(last_error, 'exceeded max attempts')
                 WHERE id = :id
                """,
                {"now": now, "id": job.id},
            )
            return None
    return job


def complete_job(conn: sqlite3.Connection, job_id: str) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE ingest_jobs SET status = 'done', updated_at = ? WHERE id = ?",
            (now_iso(), job_id),
        )


def fail_job(conn: sqlite3.Connection, job_id: str, error: str) -> None:
    """Record an error and release the job. It stays ``queued`` for another
    attempt unless :func:`claim_job` later finds it over the attempt budget."""
    with transaction(conn):
        conn.execute(
            """
            UPDATE ingest_jobs
               SET status = 'queued', claimed_at = NULL, claimed_by = NULL,
                   last_error = ?, updated_at = ?
             WHERE id = ?
            """,
            (error[:4000], now_iso(), job_id),
        )


def reap_stale_jobs(conn: sqlite3.Connection, claim_timeout_s: int) -> int:
    """Explicit sweep (claim_job also reaps opportunistically). Returns the
    number of jobs released back to ``queued``."""
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(seconds=claim_timeout_s))
    with transaction(conn):
        cur = conn.execute(
            """
            UPDATE ingest_jobs
               SET status = 'queued', claimed_at = NULL, claimed_by = NULL,
                   updated_at = ?
             WHERE status = 'running'
               AND (claimed_at IS NULL OR claimed_at < ?)
            """,
            (now_iso(), cutoff),
        )
        return cur.rowcount
