"""The metadata-filter chokepoint.

FAISS has no metadata filtering of its own. Every retrieval in the system goes
through this module: resolve filters in SQLite -> get the allowed ``faiss_id``
set -> hand it to FAISS as a pre-filter. Nothing else may touch a session
index for search.

**Session isolation lives here and nowhere else.** ``ChunkFilter.session_id``
is mandatory and is always part of the SQL. No endpoint builds its own WHERE
clause against ``chunks``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # keep faiss/numpy out of import time
    import numpy as np

    from app.index.store import SessionIndex


@dataclass(frozen=True)
class ChunkFilter:
    """Resolved against ``chunks`` JOIN ``documents`` in SQLite.

    ``session_id`` is never optional. Everything else narrows further. Only
    documents with status ``ready`` are ever eligible. Datetimes should be
    tz-aware UTC; a naive value is read as UTC.
    """

    session_id: str
    file_type: Optional[str] = None                 # e.g. "pdf", "docx", "xlsx"
    filename: Optional[str] = None                   # exact match
    uploaded_after: Optional[datetime] = None        # documents.created_at >
    uploaded_before: Optional[datetime] = None       # documents.created_at <
    document_ids: Optional[Sequence[str]] = None     # restrict to these documents


@dataclass(frozen=True)
class Hit:
    faiss_id: int
    score: float                                     # inner product on L2-normalized vectors == cosine


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def resolve_allowed_ids(conn: sqlite3.Connection, flt: ChunkFilter) -> list[int]:
    """Resolve ``flt`` to the sorted list of ``faiss_id`` values permitted for
    this search.

    - ``session_id`` and ``documents.status = 'ready'`` and ``chunks.embedded = 1``
      are always in the WHERE clause.
    - Returns ``[]`` when nothing matches. Callers MUST treat ``[]`` as "zero
      results" -- never as "no filter, search everything".
    """
    where = [
        "c.session_id = ?",
        "c.embedded = 1",
        "d.status = 'ready'",
    ]
    params: list[object] = [flt.session_id]

    if flt.file_type is not None:
        where.append("d.file_type = ?")
        params.append(flt.file_type)
    if flt.filename is not None:
        where.append("d.filename = ?")
        params.append(flt.filename)
    if flt.uploaded_after is not None:
        where.append("d.created_at > ?")
        params.append(_iso(flt.uploaded_after))
    if flt.uploaded_before is not None:
        where.append("d.created_at < ?")
        params.append(_iso(flt.uploaded_before))
    if flt.document_ids is not None:
        ids = list(flt.document_ids)
        if not ids:
            return []
        where.append(f"c.document_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)

    sql = (
        "SELECT c.faiss_id FROM chunks c "
        "JOIN documents d ON d.id = c.document_id "
        f"WHERE {' AND '.join(where)} ORDER BY c.faiss_id"
    )
    return [row[0] for row in conn.execute(sql, tuple(params)).fetchall()]


def search(
    index: "SessionIndex | None",
    query_vector: "np.ndarray",
    allowed_ids: list[int],
    top_k: int,
) -> list[Hit]:
    """Pre-filtered similarity search over a single session's index.

    ``SessionIndex.search`` builds ``faiss.IDSelectorBatch(allowed_ids)`` into
    ``faiss.SearchParameters`` -- a true pre-filter, so ``top_k`` is honoured
    against the filtered set (never top-k-then-post-filter). ``index is None``
    (session never had a vector) or ``allowed_ids == []`` -> ``[]``.
    """
    if index is None or not allowed_ids:
        return []
    return [Hit(fid, score) for fid, score in index.search(query_vector, allowed_ids, top_k)]


def retrieve(
    conn: sqlite3.Connection,
    index: "SessionIndex | None",
    query_vector: "np.ndarray",
    flt: ChunkFilter,
    top_k: int,
) -> list[Hit]:
    """The only entry point endpoints and query code may call:
    ``resolve_allowed_ids`` -> ``search``. Do not bypass."""
    return search(index, query_vector, resolve_allowed_ids(conn, flt), top_k)
