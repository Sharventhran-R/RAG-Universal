"""SQLite connection setup and explicit transactions.

WAL mode is what lets the API and one or more worker processes share the file.
Connections run in autocommit mode (``isolation_level=None``); every write goes
through :func:`transaction` so lock acquisition is explicit -- ``BEGIN
IMMEDIATE`` for the claim protocol and deletes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Forward migrations from (version-1) -> version. schema.sql always describes the
# HEAD schema, so a fresh database jumps straight to SCHEMA_VERSION and these run
# only for databases created by an older build.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: ("ALTER TABLE documents ADD COLUMN extraction_flags TEXT",),
}

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with the standard pragmas and ``sqlite3.Row`` rows.

    ``check_same_thread=False``: a connection may move between threads, but must
    not be used by two threads at once. One connection per request / per worker
    loop.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        conn.execute(pragma)
    return conn


def init_db(db_path: str | Path) -> None:
    """Bring the database up to ``SCHEMA_VERSION``. Idempotent and safe to call
    from every process at startup.

    * brand-new file (``user_version == 0``): apply ``schema.sql`` (HEAD) and
      stamp ``SCHEMA_VERSION``. ``CREATE ... IF NOT EXISTS`` keeps a concurrent
      first run harmless.
    * older file: run ``_MIGRATIONS`` under ``BEGIN IMMEDIATE`` so only one
      process migrates; the loser re-reads ``user_version`` and does nothing.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        (current,) = conn.execute("PRAGMA user_version").fetchone()
        if current >= SCHEMA_VERSION:
            return
        if current == 0:
            # executescript() issues its own COMMIT, so keep it out of transaction().
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        with transaction(conn):
            (current,) = conn.execute("PRAGMA user_version").fetchone()
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for stmt in _MIGRATIONS.get(version, ()):
                    conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    finally:
        conn.close()


@contextmanager
def transaction(
    conn: sqlite3.Connection, mode: str = "IMMEDIATE"
) -> Iterator[sqlite3.Connection]:
    """Explicit transaction. ``mode`` is ``DEFERRED`` | ``IMMEDIATE`` |
    ``EXCLUSIVE``. Use ``IMMEDIATE`` for anything that writes so the reserved
    lock is taken up front and concurrent writers serialize cleanly instead of
    racing to an ``SQLITE_BUSY`` at COMMIT."""
    conn.execute(f"BEGIN {mode}")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
