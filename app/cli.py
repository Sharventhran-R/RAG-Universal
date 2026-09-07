"""Command line: ``python -m app.cli seed``

Ingests ``./fixtures`` (building it first if missing) through the real queue +
pipeline in this process, then prints the catalog. ``--fake`` swaps in the
offline embedder so it runs without a model download.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import app.parsers  # noqa: F401  -- register parsers
from app.config import get_settings
from app.db import repo
from app.db.connection import connect, init_db
from app.index.store import IndexStore, verify_index_dimensions
from app.ingest import process_document
from app.paths import atomic_write_bytes, get_paths

FIXTURES = Path("fixtures")


def seed(*, fake: bool) -> int:
    settings = get_settings()
    paths = get_paths(settings)
    paths.ensure()
    init_db(paths.db_path)
    verify_index_dimensions(paths, settings.embed_dim)

    files = _fixture_files()
    if not files:
        sys.path.insert(0, str(FIXTURES.resolve()))
        from build_fixtures import build

        files = list(build().values())

    if fake:
        from app.embed import FakeEmbedder

        embedder = FakeEmbedder()
    else:
        from app.embed import get_embedder

        embedder = get_embedder()

    store = IndexStore.from_settings(settings)
    conn = connect(paths.db_path)
    session = repo.create_session(conn, "seed")
    print(f"session {session.id}")

    for path in files:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        atomic_write_bytes(paths.blob_path(session.id, sha), data, tmp_dir=paths.tmp_dir)
        doc = repo.create_document(
            conn, session_id=session.id, filename=path.name, mimetype=None,
            file_type=None, file_sha256=sha, byte_size=len(data),
        )
        repo.enqueue_job(conn, doc.id, session.id)

    handled = 0
    while True:
        job = repo.claim_job(
            conn, "seed",
            claim_timeout_s=settings.ingest_claim_timeout,
            max_attempts=settings.ingest_max_attempts,
        )
        if job is None:
            break
        process_document(conn, job, paths=paths, embedder=embedder, index_store=store, settings=settings)
        handled += 1
    store.persist_all()

    print(f"\ningested {handled} document(s):\n")
    print(f"  {'filename':22} {'type':6} {'status':13} {'blocks':>7} {'chunks':>7}  flags")
    for d in repo.list_documents(conn, session.id):
        flags = ",".join(d.extraction_flags) or "-"
        print(
            f"  {d.filename:22} {d.file_type or '-':6} {d.status:13} "
            f"{d.block_count if d.block_count is not None else '-':>7} "
            f"{d.chunk_count if d.chunk_count is not None else '-':>7}  {flags}"
        )
    conn.close()
    return 0


def _fixture_files() -> list[Path]:
    if not FIXTURES.is_dir():
        return []
    return sorted(
        p for p in FIXTURES.iterdir()
        if p.suffix.lower() in {".pdf", ".xlsx", ".csv", ".docx", ".pptx", ".txt", ".md", ".html"}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed_p = sub.add_parser("seed", help="ingest ./fixtures and print the catalog")
    seed_p.add_argument("--fake", action="store_true", help="use the offline fake embedder")
    args = parser.parse_args(argv)
    if args.cmd == "seed":
        return seed(fake=args.fake)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
