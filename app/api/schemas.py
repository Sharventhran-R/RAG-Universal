"""Request/response models for the HTTP edge (pydantic lives only here + config)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.query import QueryResult


class SessionCreate(BaseModel):
    name: str = Field(default="session", max_length=200)


class SessionOut(BaseModel):
    id: str
    name: str
    created_at: str


class DocumentOut(BaseModel):
    id: str
    session_id: str
    filename: str
    file_type: str | None
    status: str
    status_detail: str | None = None
    error: str | None = None
    block_count: int | None = None
    chunk_count: int | None = None
    extraction_flags: list[str] = []
    created_at: str
    updated_at: str


class JobOut(BaseModel):
    id: str
    status: str


class UploadOut(BaseModel):
    document_id: str
    job: JobOut


class SessionDetailOut(SessionOut):
    documents: list[DocumentOut] = []


class QueryIn(BaseModel):
    question: str
    file_type: str | None = None
    filename: str | None = None
    document_ids: list[str] | None = None
    uploaded_after: datetime | None = None
    uploaded_before: datetime | None = None


class CitationOut(BaseModel):
    chunk_id: str
    filename: str
    page: int | None
    sheet: str | None
    slide: int | None
    snippet: str


class QueryOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    chunks_used: list[str]
    insufficient_context: bool
    citation_warning: str | None

    @classmethod
    def from_result(cls, r: QueryResult) -> "QueryOut":
        return cls(
            answer=r.answer,
            citations=[CitationOut(**vars(c)) for c in r.citations],
            chunks_used=r.chunks_used,
            insufficient_context=r.insufficient_context,
            citation_warning=r.citation_warning,
        )
