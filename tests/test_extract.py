from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
import pymupdf

from app.extract import cache, extract
from app.ir import IR_VERSION
from app.parsers.base import UnsupportedFormatError
from app.paths import DataPaths


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    p = DataPaths(tmp_path)
    p.ensure()
    return p


def _pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from a one line PDF.", fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_first_call_parses_and_caches_second_call_is_a_cache_hit(paths):
    data = _pdf_bytes()
    blob = paths.blobs_dir / "doc" / "sha"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)

    name, version, file_type, result = extract(
        document_id="d1", filename="a.pdf", mimetype="application/pdf",
        file_sha256="sha", blob_path=blob, paths=paths,
    )
    assert file_type == "pdf" and result.blocks
    assert cache.load(paths, "sha", version) is not None

    # delete the source: a cache hit must not need it
    blob.unlink()
    again = extract(
        document_id="d1", filename="a.pdf", mimetype="application/pdf",
        file_sha256="sha", blob_path=blob, paths=paths,
    )
    assert [b.content for b in again[3].blocks] == [b.content for b in result.blocks]


def test_cache_entry_from_a_different_ir_version_is_ignored(paths):
    path = paths.extraction_path("sha", "1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"ir_version": 999, "blocks": []}', encoding="utf-8")
    assert cache.load(paths, "sha", "1") is None


def test_cache_roundtrip_preserves_flags_and_hint(paths):
    from app.parsers.base import EMPTY_NO_TEXT, ParseResult

    src = ParseResult(blocks=[], extraction_flags=["low_confidence_table"], status_hint=EMPTY_NO_TEXT)
    cache.store(paths, "sha9", "pdf-pymupdf", "2", src)
    got = cache.load(paths, "sha9", "2")
    assert got.extraction_flags == ["low_confidence_table"]
    assert got.status_hint == EMPTY_NO_TEXT


def test_unsupported_format_raises(paths):
    blob = paths.blobs_dir / "d" / "s"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(UnsupportedFormatError):
        extract(
            document_id="d", filename="archive.zip", mimetype="application/zip",
            file_sha256="s", blob_path=blob, paths=paths,
        )


def test_ir_version_is_stamped_on_store(paths):
    import json

    from app.parsers.base import ParseResult

    cache.store(paths, "shaX", "x", "1", ParseResult(blocks=[]))
    data = json.loads(paths.extraction_path("shaX", "1").read_text(encoding="utf-8"))
    assert data["ir_version"] == IR_VERSION
