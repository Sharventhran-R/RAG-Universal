from __future__ import annotations

import mimetypes
from types import SimpleNamespace

import pytest

pytest.importorskip("faiss")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.db import repo
from app.db.connection import connect
from app.embed import FakeEmbedder
from app.index.store import IndexStore
from app.ingest import process_document
from app.llm import FakeLLM
from app.paths import get_paths

DIM = 64


@pytest.fixture
def api(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=DIM)
    paths = get_paths(settings)
    store = IndexStore(paths, DIM, 8)
    app = create_app(
        settings=settings, embedder=FakeEmbedder(dim=DIM), llm=FakeLLM(), index_store=store
    )
    with TestClient(app) as http:
        yield SimpleNamespace(http=http, settings=settings, paths=paths, store=store)


def _drain(api) -> int:
    """Run the worker pipeline against the API's db + index store."""
    conn = connect(api.paths.db_path)
    handled = 0
    try:
        while True:
            job = repo.claim_job(conn, "test", claim_timeout_s=300, max_attempts=3)
            if job is None:
                return handled
            process_document(
                conn, job, paths=api.paths,
                embedder=FakeEmbedder(dim=DIM), index_store=api.store, settings=api.settings,
            )
            handled += 1
    finally:
        conn.close()


def _new_session(api) -> str:
    r = api.http.post("/sessions", json={"name": "s"})
    assert r.status_code == 201
    return r.json()["id"]


def _upload(api, session_id, name="notes.txt", content=b"Redis was dropped in favour of SQLite as the job queue."):
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return api.http.post(
        f"/sessions/{session_id}/documents",
        files={"file": (name, content, ctype)},
    )


def test_builtin_ui_is_served_and_api_still_routes(api):
    root = api.http.get("/", follow_redirects=False)
    assert root.status_code in (307, 308)
    assert root.headers["location"] == "/ui/"
    page = api.http.get("/ui/")
    assert page.status_code == 200
    assert "<title>" in page.text.lower()
    # mounting the UI must not shadow the JSON API
    assert api.http.post("/sessions", json={"name": "x"}).status_code == 201


def test_session_crud_and_404s(api):
    sid = _new_session(api)
    detail = api.http.get(f"/sessions/{sid}")
    assert detail.status_code == 200 and detail.json()["documents"] == []
    assert api.http.get("/sessions/nope").status_code == 404
    assert api.http.get("/documents/nope").status_code == 404
    assert api.http.delete("/documents/nope").status_code == 404


def test_upload_enqueues_and_stays_queued_until_a_worker_runs(api):
    sid = _new_session(api)
    r = _upload(api, sid)
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert r.json()["job"]["status"] == "queued"

    doc = api.http.get(f"/documents/{doc_id}").json()
    assert doc["status"] == "queued" and doc["file_type"] is None

    # query before ingestion completes -> insufficient, nothing indexed yet
    q = api.http.post(f"/sessions/{sid}/query", json={"question": "what replaced redis?"})
    assert q.status_code == 200 and q.json()["insufficient_context"] is True


def test_upload_to_missing_session_and_empty_upload(api):
    assert _upload(api, "ghost").status_code == 404
    sid = _new_session(api)
    assert api.http.post(
        f"/sessions/{sid}/documents", files={"file": ("x.txt", b"", "text/plain")}
    ).status_code == 400


def test_full_flow_upload_ingest_query_cite(api):
    sid = _new_session(api)
    doc_id = _upload(api, sid).json()["document_id"]
    assert _drain(api) == 1

    doc = api.http.get(f"/documents/{doc_id}").json()
    assert doc["status"] == "ready"
    assert doc["file_type"] == "txt"
    assert doc["chunk_count"] >= 1

    q = api.http.post(f"/sessions/{sid}/query", json={"question": "what replaced redis?"})
    body = q.json()
    assert body["insufficient_context"] is False
    assert body["citations"][0]["filename"] == "notes.txt"
    assert "SQLite" in body["citations"][0]["snippet"]


def test_delete_makes_the_document_unretrievable(api):
    sid = _new_session(api)
    doc_id = _upload(api, sid).json()["document_id"]
    _drain(api)

    assert api.http.delete(f"/documents/{doc_id}").status_code == 204
    assert api.http.get(f"/documents/{doc_id}").status_code == 404

    q = api.http.post(f"/sessions/{sid}/query", json={"question": "what replaced redis?"})
    assert q.json()["insufficient_context"] is True


def test_metadata_filter_flows_through_the_query_endpoint(api):
    sid = _new_session(api)
    _upload(api, sid, name="a.txt", content=b"The alpha subsystem handles ingestion.")
    _upload(api, sid, name="b.md", content=b"# Beta\n\nThe beta subsystem handles retrieval.")
    _drain(api)

    q = api.http.post(
        f"/sessions/{sid}/query",
        json={"question": "which subsystem handles retrieval?", "file_type": "md"},
    )
    body = q.json()
    assert body["insufficient_context"] is False
    assert {c["filename"] for c in body["citations"]} == {"b.md"}
