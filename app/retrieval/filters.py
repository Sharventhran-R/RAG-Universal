"""The metadata-filter chokepoint.

FAISS has no metadata filtering of its own. Every retrieval in the system goes
through this module: resolve filters in SQLite -> get the allowed ``faiss_id``
set -> hand it to FAISS as a pre-filter. Nothing else may touch a session
index for search.

**Session isolation lives here and nowhere else.** ``ChunkFilter.session_id``
is mandatory and is always part of the SQL. No endpoint builds its own WHERE
clause against ``chunks``.

This file is an interface sketch for review -- bodies are ``NotImplementedError``
until the retrieval slice.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # heavy deps kept out of import time
    import faiss
    import numpy as np


@dataclass(frozen=True)
class ChunkFilter:
    """Resolved against ``chunks`` JOIN ``documents`` in SQLite.

    ``session_id`` is never optional. Everything else narrows further. Only
    documents with status ``ready`` are ever eligible.
    """

    session_id: str
    file_type: Optional[str] = None                 # e.g. "pdf", "xlsx", "docx"
    filename: Optional[str] = None                   # exact match
    uploaded_after: Optional[datetime] = None        # documents.created_at >
    uploaded_before: Optional[datetime] = None       # documents.created_at <
    document_ids: Optional[Sequence[str]] = None     # restrict to these documents


@dataclass(frozen=True)
class Hit:
    faiss_id: int
    score: float                                     # inner product on L2-normalized vectors == cosine


def resolve_allowed_ids(conn: sqlite3.Connection, flt: ChunkFilter) -> list[int]:
    """Resolve ``flt`` to the sorted list of ``faiss_id`` values permitted for
    this search.

    - ``session_id`` is always in the WHERE clause; ``documents.status = 'ready'``
      is always in the WHERE clause.
    - Returns ``[]`` when nothing matches. Callers MUST treat ``[]`` as "zero
      results" -- never as "no filter, search everything".
    """
    raise NotImplementedError


def search(
    index: "faiss.IndexIDMap2",
    query_vector: "np.ndarray",
    allowed_ids: list[int],
    top_k: int,
) -> list[Hit]:
    """Pre-filtered similarity search over a single session's index.

    Builds ``faiss.IDSelectorArray(allowed_ids)`` inside
    ``faiss.SearchParameters(sel=...)`` and passes it to ``index.search``. This
    is a true pre-filter: vectors outside ``allowed_ids`` are never scored, so
    ``top_k`` is honoured against the filtered set (no silent shortfall from
    top-k-then-post-filter).

    ``allowed_ids == []`` returns ``[]`` without calling FAISS.

    NOTE: the numpy buffer backing ``IDSelectorArray`` must stay alive for the
    duration of the ``search`` call -- FAISS holds it by pointer.
    """
    raise NotImplementedError


def retrieve(
    conn: sqlite3.Connection,
    index: "faiss.IndexIDMap2",
    query_vector: "np.ndarray",
    flt: ChunkFilter,
    top_k: int,
) -> list[Hit]:
    """The only entry point endpoints and query code may call.

    ``resolve_allowed_ids(conn, flt)`` -> ``search(index, query_vector, ids, top_k)``.
    Do not bypass this to call ``index.search`` directly.
    """
    raise NotImplementedError
