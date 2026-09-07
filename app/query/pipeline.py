"""``answer_query`` -- embed → retrieve → generate → validate → resolve.

Straight line, no tool loop (agentic retrieval is out of scope). Every number
in the returned answer traces to a cited chunk id that was actually in the
retrieved context; invented ids are stripped, and an answer left with no valid
citation either becomes the insufficient-context response (``strict``) or is
returned with a warning banner (``flag``).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from app.config import Settings, get_settings
from app.db import repo
from app.db.models import Chunk, Document
from app.query.context import build_context, snippet
from app.query.prompt import system_prompt, user_prompt
from app.retrieval.filters import ChunkFilter, retrieve

if TYPE_CHECKING:
    from app.embed.base import Embedder
    from app.index.store import SessionIndex
    from app.llm.base import LLMClient

_CITE_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_INSUFFICIENT_MESSAGE = (
    "The uploaded documents do not contain enough information to answer that."
)
_FLAG_WARNING = "This answer could not be tied to a specific passage and may not be grounded."


@dataclass(frozen=True)
class QueryRequest:
    session_id: str
    question: str
    file_type: str | None = None
    filename: str | None = None
    document_ids: Sequence[str] | None = None
    uploaded_after: datetime | None = None
    uploaded_before: datetime | None = None


@dataclass
class Citation:
    chunk_id: str
    filename: str
    page: int | None
    sheet: str | None
    slide: int | None
    snippet: str


@dataclass
class QueryResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks_used: list[str] = field(default_factory=list)
    insufficient_context: bool = False
    citation_warning: str | None = None


async def answer_query(
    req: QueryRequest,
    *,
    conn: sqlite3.Connection,
    index: "SessionIndex | None",
    embedder: "Embedder",
    llm: "LLMClient",
    settings: Settings | None = None,
) -> QueryResult:
    s = settings or get_settings()

    if not req.question.strip():
        return _insufficient()

    hits = retrieve(
        conn,
        index,
        embedder.embed_query(req.question),
        ChunkFilter(
            session_id=req.session_id,
            file_type=req.file_type,
            filename=req.filename,
            document_ids=req.document_ids,
            uploaded_after=req.uploaded_after,
            uploaded_before=req.uploaded_before,
        ),
        s.top_k,
    )
    if not hits:
        return _insufficient()

    chunks = repo.get_chunks_by_faiss_ids(conn, req.session_id, [h.faiss_id for h in hits])
    docs = _documents(conn, chunks)
    context = build_context([(c, docs[c.document_id]) for c in chunks])

    raw = await llm.generate(
        system=system_prompt(s.insufficient_sentinel),
        prompt=user_prompt(context, req.question),
        temperature=0.0,
    )
    answer = raw.strip()
    if answer == s.insufficient_sentinel:
        return _insufficient()

    retrieved_ids = {c.id for c in chunks}
    cited_ids = _dedupe(cid for cid in _CITE_RE.findall(answer) if cid in retrieved_ids)
    cleaned = _drop_invented_citations(answer, retrieved_ids)

    if not cited_ids:
        if s.citation_enforcement == "strict":
            return _insufficient()
        return QueryResult(answer=cleaned, citation_warning=_FLAG_WARNING)

    by_id = {c.id: c for c in chunks}
    citations = [_citation(by_id[cid], docs[by_id[cid].document_id]) for cid in cited_ids]
    return QueryResult(
        answer=cleaned,
        citations=citations,
        chunks_used=[c.chunk_id for c in citations],
    )


# --------------------------------------------------------------------------


def _insufficient() -> QueryResult:
    return QueryResult(answer=_INSUFFICIENT_MESSAGE, insufficient_context=True)


def _documents(conn: sqlite3.Connection, chunks: list[Chunk]) -> dict[str, Document]:
    out: dict[str, Document] = {}
    for cid in {c.document_id for c in chunks}:
        doc = repo.get_document(conn, cid)
        if doc is not None:
            out[cid] = doc
    return out


def _dedupe(ids) -> list[str]:
    seen: dict[str, None] = {}
    for i in ids:
        seen.setdefault(i, None)
    return list(seen)


def _drop_invented_citations(answer: str, valid: set[str]) -> str:
    def repl(m: re.Match) -> str:
        return m.group(0) if m.group(1) in valid else ""

    stripped = _CITE_RE.sub(repl, answer)
    stripped = re.sub(r" {2,}", " ", stripped)
    stripped = re.sub(r"\s+([.,;:)])", r"\1", stripped)
    return stripped.strip()


def _citation(chunk: Chunk, doc: Document) -> Citation:
    return Citation(
        chunk_id=chunk.id,
        filename=doc.filename,
        page=chunk.page_start,
        sheet=chunk.sheet,
        slide=chunk.slide,
        snippet=snippet(chunk.text),
    )
