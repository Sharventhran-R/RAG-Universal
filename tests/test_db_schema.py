from __future__ import annotations

from pathlib import Path

from app.db.connection import SCHEMA_VERSION, connect, init_db

_V1_DOCUMENTS = """
CREATE TABLE sessions (id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
CREATE TABLE documents (
    id TEXT PRIMARY KEY, session_id TEXT, filename TEXT, mimetype TEXT,
    file_type TEXT, file_sha256 TEXT, byte_size INTEGER, status TEXT,
    status_detail TEXT, error TEXT, extractor_name TEXT, extractor_version TEXT,
    block_count INTEGER, chunk_count INTEGER, created_at TEXT, updated_at TEXT
);
"""


def _columns(conn, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_db_is_at_head_with_extraction_flags(tmp_path: Path):
    p = tmp_path / "fresh.db"
    init_db(p)
    conn = connect(p)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert "extraction_flags" in _columns(conn, "documents")
    finally:
        conn.close()


def test_v1_database_is_migrated_in_place(tmp_path: Path):
    p = tmp_path / "old.db"
    conn = connect(p)
    conn.executescript(_V1_DOCUMENTS)
    conn.execute("PRAGMA user_version = 1")
    conn.close()

    init_db(p)

    conn = connect(p)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert "extraction_flags" in _columns(conn, "documents")
    finally:
        conn.close()


def test_init_db_is_idempotent(tmp_path: Path):
    p = tmp_path / "x.db"
    init_db(p)
    init_db(p)  # must not raise
    conn = connect(p)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()
