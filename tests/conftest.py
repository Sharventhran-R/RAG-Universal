from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.db.connection import connect, init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "app.db"
    init_db(p)
    return p


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(db_path)
    try:
        yield c
    finally:
        c.close()
