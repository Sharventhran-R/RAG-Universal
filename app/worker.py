"""Ingestion worker. Run one or more:  ``python -m app.worker``

Claims jobs from SQLite (``UPDATE ... RETURNING`` under ``BEGIN IMMEDIATE``),
runs the pipeline, sleeps when idle. The API process never does this.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time

import app.parsers  # noqa: F401  -- populate the parser registry
from app.config import get_settings
from app.db import repo
from app.db.connection import connect, init_db
from app.embed import get_embedder
from app.index.store import IndexStore, verify_index_dimensions
from app.ingest import process_document, reconcile
from app.paths import get_paths

log = logging.getLogger("app.worker")


def run(*, once: bool = False) -> None:
    settings = get_settings()
    paths = get_paths(settings)
    paths.ensure()
    init_db(paths.db_path)
    verify_index_dimensions(paths, settings.embed_dim)

    embedder = get_embedder()
    index_store = IndexStore.from_settings(settings)
    conn = connect(paths.db_path)
    worker_id = f"{socket.gethostname()}:{os.getpid()}"

    dropped = reconcile(conn, index_store)
    if dropped:
        log.warning("reconcile removed %d orphan chunk rows", dropped)
    log.info("worker %s up (poll=%.1fs)", worker_id, settings.ingest_poll_interval)

    stopping = False

    def _stop(_sig, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stopping:
            job = repo.claim_job(
                conn,
                worker_id,
                claim_timeout_s=settings.ingest_claim_timeout,
                max_attempts=settings.ingest_max_attempts,
            )
            if job is None:
                if once:
                    break
                time.sleep(settings.ingest_poll_interval)
                continue
            log.info("job %s -> document %s (attempt %d)", job.id, job.document_id, job.attempts)
            status = process_document(
                conn, job, paths=paths, embedder=embedder, index_store=index_store, settings=settings
            )
            log.info("document %s -> %s", job.document_id, status)
    finally:
        index_store.persist_all()
        conn.close()
        log.info("worker %s stopped", worker_id)


def drain() -> int:
    """Process every currently-queued job and return, for `seed` and tests.
    Returns how many jobs were handled."""
    settings = get_settings()
    paths = get_paths(settings)
    paths.ensure()
    init_db(paths.db_path)
    verify_index_dimensions(paths, settings.embed_dim)

    embedder = get_embedder()
    index_store = IndexStore.from_settings(settings)
    conn = connect(paths.db_path)
    worker_id = f"seed:{os.getpid()}"
    handled = 0
    try:
        while True:
            job = repo.claim_job(
                conn, worker_id,
                claim_timeout_s=settings.ingest_claim_timeout,
                max_attempts=settings.ingest_max_attempts,
            )
            if job is None:
                return handled
            process_document(
                conn, job, paths=paths, embedder=embedder, index_store=index_store, settings=settings
            )
            handled += 1
    finally:
        index_store.persist_all()
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run()
