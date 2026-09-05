from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.paths import DataPaths, atomic_write_text


def test_settings_defaults():
    s = Settings()
    assert s.embed_dim == 384
    assert s.chunk_tokens == 512
    assert s.citation_enforcement == "strict"


def test_settings_rejects_overlap_ge_chunk(monkeypatch):
    monkeypatch.setenv("CHUNK_OVERLAP", "512")
    with pytest.raises(ValueError):
        Settings()


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "768")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    s = Settings()
    assert s.embed_dim == 768
    assert s.ollama_model == "mistral"


def test_datapaths_layout(tmp_path: Path):
    p = DataPaths(tmp_path)
    assert p.db_path == tmp_path / "app.db"
    assert p.blob_path("s1", "sha") == tmp_path / "blobs" / "s1" / "sha"
    assert p.faiss_path("s1") == tmp_path / "faiss" / "s1.index"
    assert p.extraction_path("sha", "v3") == tmp_path / "extractions" / "sha" / "v3.json"
    p.ensure()
    assert p.blobs_dir.is_dir() and p.faiss_dir.is_dir()


def test_atomic_write_is_all_or_nothing(tmp_path: Path):
    target = tmp_path / "nested" / "out.json"
    atomic_write_text(target, json.dumps({"ok": 1}))
    assert json.loads(target.read_text()) == {"ok": 1}
    # no leftover temp files
    assert [x.name for x in (tmp_path / "nested").iterdir()] == ["out.json"]
