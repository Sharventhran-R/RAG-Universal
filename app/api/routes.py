"""All six endpoints. Ingestion is only ever *enqueued* here."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.api.schemas import (
    DocumentOut,
    QueryIn,
    QueryOut,
    SessionCreate,
    SessionDetailOut,
    SessionOut,
    UploadOut,
)
from app.db import repo
from app.db.connection import connect, transaction
from app.llm.base import LLMUnavailable
from app.paths import atomic_write_bytes
from app.query import QueryRequest, answer_query

router = APIRouter()


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.paths.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _doc_out(d) -> DocumentOut:
    return DocumentOut(
        id=d.id, session_id=d.session_id, filename=d.filename, file_type=d.file_type,
        status=d.status, status_detail=d.status_detail, error=d.error,
        block_count=d.block_count, chunk_count=d.chunk_count,
        extraction_flags=d.extraction_flags, created_at=d.created_at, updated_at=d.updated_at,
    )


def _require_session(conn: sqlite3.Connection, session_id: str):
    session = repo.get_session(conn, session_id)
    if session is None:
        raise HTTPException(404, f"session {session_id} not found")
    return session


def _require_document(conn: sqlite3.Connection, document_id: str):
    doc = repo.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(404, f"document {document_id} not found")
    return doc


@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate, conn=Depends(get_conn)) -> SessionOut:
    s = repo.create_session(conn, body.name)
    return SessionOut(id=s.id, name=s.name, created_at=s.created_at)


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(session_id: str, conn=Depends(get_conn)) -> SessionDetailOut:
    s = _require_session(conn, session_id)
    docs = repo.list_documents(conn, session_id)
    return SessionDetailOut(
        id=s.id, name=s.name, created_at=s.created_at,
        documents=[_doc_out(d) for d in docs],
    )


@router.post("/sessions/{session_id}/documents", response_model=UploadOut, status_code=202)
async def upload_document(
    session_id: str, file: UploadFile, request: Request, conn=Depends(get_conn)
) -> UploadOut:
    _require_session(conn, session_id)
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    sha = hashlib.sha256(data).hexdigest()
    atomic_write_bytes(
        request.app.state.paths.blob_path(session_id, sha),
        data,
        tmp_dir=request.app.state.paths.tmp_dir,
    )
    doc = repo.create_document(
        conn,
        session_id=session_id,
        filename=file.filename or "upload.bin",
        mimetype=file.content_type,
        file_type=None,  # the worker sets the authoritative value after parser resolve
        file_sha256=sha,
        byte_size=len(data),
    )
    job = repo.enqueue_job(conn, doc.id, session_id)
    return UploadOut(document_id=doc.id, job={"id": job.id, "status": job.status})


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, conn=Depends(get_conn)) -> DocumentOut:
    return _doc_out(_require_document(conn, document_id))


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request, conn=Depends(get_conn)) -> None:
    doc = _require_document(conn, document_id)
    paths = request.app.state.paths
    store = request.app.state.index_store

    faiss_ids = repo.document_faiss_ids(conn, doc.session_id, document_id)
    index = store.get(doc.session_id)
    if index is not None and faiss_ids:
        index.remove(faiss_ids)
        index.persist()
    with transaction(conn):
        repo.delete_document_rows(conn, doc.session_id, document_id)
    paths.blob_path(doc.session_id, doc.file_sha256).unlink(missing_ok=True)


@router.post("/sessions/{session_id}/query", response_model=QueryOut)
async def query(
    session_id: str, body: QueryIn, request: Request, conn=Depends(get_conn)
) -> QueryOut:
    _require_session(conn, session_id)
    state = request.app.state
    req = QueryRequest(
        session_id=session_id,
        question=body.question,
        file_type=body.file_type,
        filename=body.filename,
        document_ids=body.document_ids,
        uploaded_after=body.uploaded_after,
        uploaded_before=body.uploaded_before,
    )
    try:
        result = await answer_query(
            req,
            conn=conn,
            index=state.index_store.get(session_id),
            embedder=state.embedder,
            llm=state.llm,
            settings=state.settings,
        )
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return QueryOut.from_result(result)
