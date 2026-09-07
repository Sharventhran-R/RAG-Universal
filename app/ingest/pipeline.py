"""The per-document ingest pipeline, run inside the worker process.

``queued -> extracting -> chunking -> embedding -> ready``
with ``unsupported`` / ``empty_no_text`` / ``failed`` as the visible dead ends.

Every stage is idempotent: ``extract`` is cache-backed, ``replace_chunks``
deletes-then-inserts, and the document's prior vectors are removed from the
session index before re-adding. A job that is re-run after a crash converges.
"""

from __future__ import annotations

import sqlite3
import traceback

from app.config import Settings, get_settings
from app.chunk import chunk_document
from app.db import repo
from app.db.connection import transaction
from app.db.models import IngestJob
from app.embed.base import Embedder
from app.extract import extract
from app.index.store import IndexStore
from app.parsers.base import EMPTY_NO_TEXT, UnsupportedFormatError
from app.paths import DataPaths


def process_document(
    conn: sqlite3.Connection,
    job: IngestJob,
    *,
    paths: DataPaths,
    embedder: Embedder,
    index_store: IndexStore,
    settings: Settings | None = None,
) -> str:
    """Run one job to a terminal document status and mark the job done/failed.
    Returns the terminal document status."""
    s = settings or get_settings()
    doc = repo.get_document(conn, job.document_id)
    if doc is None:
        repo.fail_job(conn, job.id, f"document {job.document_id} vanished")
        return "failed"

    try:
        repo.set_document_status(conn, doc.id, "extracting")
        blob = paths.blob_path(doc.session_id, doc.file_sha256)

        try:
            name, version, file_type, result = extract(
                document_id=doc.id,
                filename=doc.filename,
                mimetype=doc.mimetype,
                file_sha256=doc.file_sha256,
                blob_path=blob,
                paths=paths,
            )
        except UnsupportedFormatError as exc:
            repo.set_document_status(conn, doc.id, "unsupported", detail=str(exc))
            repo.complete_job(conn, job.id)
            return "unsupported"

        repo.set_document_extractor(conn, doc.id, name, version, file_type)
        repo.set_document_extraction_flags(conn, doc.id, result.extraction_flags)

        if result.status_hint == EMPTY_NO_TEXT or not result.blocks:
            repo.set_document_counts(
                conn, doc.id, block_count=len(result.blocks), chunk_count=0
            )
            repo.set_document_status(conn, doc.id, "empty_no_text")
            repo.complete_job(conn, job.id)
            return "empty_no_text"

        repo.set_document_status(conn, doc.id, "chunking")
        stale_ids = repo.document_faiss_ids(conn, doc.session_id, doc.id)
        drafts = chunk_document(
            result.blocks,
            filename=doc.filename,
            count_tokens=embedder.count_tokens,
            settings=s,
        )
        with transaction(conn):
            chunks = repo.replace_chunks(
                conn, doc.id, [d.to_row(doc.session_id) for d in drafts]
            )

        repo.set_document_status(conn, doc.id, "embedding")
        index = index_store.get_or_create(doc.session_id)
        if stale_ids:
            index.remove(stale_ids)
        to_embed = [c for c in chunks if c.embedded]
        if to_embed:
            vectors = embedder.embed_documents([c.embed_input for c in to_embed])
            index.add([c.faiss_id for c in to_embed], vectors)
        index.persist()

        repo.set_document_counts(
            conn, doc.id, block_count=len(result.blocks), chunk_count=len(to_embed)
        )
        repo.set_document_status(conn, doc.id, "ready")
        repo.complete_job(conn, job.id)
        return "ready"

    except Exception:  # noqa: BLE001 -- any failure is a job failure, recorded verbatim
        tb = traceback.format_exc(limit=8)
        repo.set_document_status(conn, doc.id, "failed", detail="ingest error", error=tb)
        repo.fail_job(conn, job.id, tb)
        return "failed"
