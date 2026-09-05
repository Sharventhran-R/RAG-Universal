"""Row dataclasses. Thin mirrors of the tables in ``schema.sql`` -- no
behaviour, just typed access and ``from_row``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal, Optional

DocumentStatus = Literal[
    "queued", "extracting", "chunking", "embedding", "ready",
    "failed", "unsupported", "empty_no_text",
]
JobStatus = Literal["queued", "running", "done", "failed"]

TERMINAL_DOCUMENT_STATUS: frozenset[str] = frozenset(
    ("ready", "failed", "unsupported", "empty_no_text")
)
TERMINAL_JOB_STATUS: frozenset[str] = frozenset(("done", "failed"))


@dataclass(slots=True)
class Session:
    id: str
    name: str
    created_at: str

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Session":
        return cls(id=r["id"], name=r["name"], created_at=r["created_at"])


@dataclass(slots=True)
class Document:
    id: str
    session_id: str
    filename: str
    mimetype: Optional[str]
    file_type: Optional[str]
    file_sha256: str
    byte_size: int
    status: DocumentStatus
    status_detail: Optional[str]
    error: Optional[str]
    extractor_name: Optional[str]
    extractor_version: Optional[str]
    block_count: Optional[int]
    chunk_count: Optional[int]
    extraction_flags: list[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Document":
        raw_flags = r["extraction_flags"]
        return cls(
            id=r["id"],
            session_id=r["session_id"],
            filename=r["filename"],
            mimetype=r["mimetype"],
            file_type=r["file_type"],
            file_sha256=r["file_sha256"],
            byte_size=r["byte_size"],
            status=r["status"],
            status_detail=r["status_detail"],
            error=r["error"],
            extractor_name=r["extractor_name"],
            extractor_version=r["extractor_version"],
            block_count=r["block_count"],
            chunk_count=r["chunk_count"],
            extraction_flags=json.loads(raw_flags) if raw_flags else [],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )


@dataclass(slots=True)
class Chunk:
    faiss_id: int
    id: str
    document_id: str
    session_id: str
    ord: int
    text: str
    embed_input: str
    token_count: int
    embedded: bool
    page_start: Optional[int]
    page_end: Optional[int]
    sheet: Optional[str]
    slide: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    section_path: list[str]
    created_at: str

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Chunk":
        return cls(
            faiss_id=r["faiss_id"],
            id=r["id"],
            document_id=r["document_id"],
            session_id=r["session_id"],
            ord=r["ord"],
            text=r["text"],
            embed_input=r["embed_input"],
            token_count=r["token_count"],
            embedded=bool(r["embedded"]),
            page_start=r["page_start"],
            page_end=r["page_end"],
            sheet=r["sheet"],
            slide=r["slide"],
            char_start=r["char_start"],
            char_end=r["char_end"],
            section_path=json.loads(r["section_path"]) if r["section_path"] else [],
            created_at=r["created_at"],
        )


@dataclass(slots=True)
class IngestJob:
    id: str
    document_id: str
    session_id: str
    status: JobStatus
    attempts: int
    claimed_at: Optional[str]
    claimed_by: Optional[str]
    last_error: Optional[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "IngestJob":
        return cls(
            id=r["id"],
            document_id=r["document_id"],
            session_id=r["session_id"],
            status=r["status"],
            attempts=r["attempts"],
            claimed_at=r["claimed_at"],
            claimed_by=r["claimed_by"],
            last_error=r["last_error"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
