"""Startup reconcile: drop `chunks` rows whose vector is missing from the
session index.

A crash between "index persisted" and "SQLite committed" (delete or re-embed)
can leave an embedded chunk row with no matching vector. Such a row is already
unreachable via retrieval (its faiss_id is not in the index), but we clean it up
so counts and reconciliation stay honest.
"""

from __future__ import annotations

import sqlite3

from app.db.connection import transaction
from app.index.store import IndexStore


def reconcile(conn: sqlite3.Connection, index_store: IndexStore) -> int:
    """Returns the number of orphan chunk rows removed."""
    removed = 0
    session_ids = [r[0] for r in conn.execute("SELECT id FROM sessions").fetchall()]
    for session_id in session_ids:
        index = index_store.get(session_id)
        present = index.id_set() if index is not None else set()
        rows = conn.execute(
            "SELECT faiss_id FROM chunks WHERE session_id = ? AND embedded = 1",
            (session_id,),
        ).fetchall()
        orphans = [(r[0],) for r in rows if r[0] not in present]
        if orphans:
            with transaction(conn):
                conn.executemany("DELETE FROM chunks WHERE faiss_id = ?", orphans)
            removed += len(orphans)
    return removed
