"""SQLite storage layer: connection setup, schema, row models, data access."""

from app.db.connection import SCHEMA_VERSION, connect, init_db, transaction
from app.db.models import (
    TERMINAL_DOCUMENT_STATUS,
    TERMINAL_JOB_STATUS,
    Chunk,
    Document,
    DocumentStatus,
    IngestJob,
    JobStatus,
    Session,
)

__all__ = [
    "SCHEMA_VERSION",
    "connect",
    "init_db",
    "transaction",
    "Session",
    "Document",
    "Chunk",
    "IngestJob",
    "DocumentStatus",
    "JobStatus",
    "TERMINAL_DOCUMENT_STATUS",
    "TERMINAL_JOB_STATUS",
]
