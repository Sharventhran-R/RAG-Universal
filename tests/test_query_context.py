from __future__ import annotations

from app.db.models import Chunk, Document
from app.query.context import build_context, chunk_header, format_locator, snippet


def _chunk(**over) -> Chunk:
    base = dict(
        faiss_id=1, id="c_1", document_id="d1", session_id="s", ord=0,
        text="body text", embed_input="x", token_count=2, embedded=True,
        page_start=None, page_end=None, sheet=None, slide=None,
        char_start=None, char_end=None, section_path=[], created_at="t",
    )
    base.update(over)
    return Chunk(**base)


def _doc(filename="report.pdf") -> Document:
    return Document(
        id="d1", session_id="s", filename=filename, mimetype=None, file_type="pdf",
        file_sha256="h", byte_size=1, status="ready", status_detail=None, error=None,
        extractor_name=None, extractor_version=None, block_count=None, chunk_count=None,
        extraction_flags=[], created_at="t", updated_at="t",
    )


def test_format_locator_covers_every_axis():
    assert format_locator(_chunk(page_start=3, page_end=3)) == "p.3"
    assert format_locator(_chunk(page_start=3, page_end=5)) == "pp.3-5"
    assert format_locator(_chunk(sheet="Q3")) == "sheet Q3"
    assert format_locator(_chunk(slide=7)) == "slide 7"
    assert format_locator(_chunk()) == "—"


def test_chunk_header_and_context_layout():
    c1, c2 = _chunk(id="c_a", page_start=1, page_end=1), _chunk(id="c_b", sheet="Data")
    d = _doc("10k.pdf")
    assert chunk_header(c1, d) == "[c_a | 10k.pdf | p.1]"
    ctx = build_context([(c1, d), (c2, d)])
    assert ctx == "[c_a | 10k.pdf | p.1]\nbody text\n\n[c_b | 10k.pdf | sheet Data]\nbody text"


def test_snippet_truncates_long_text_with_ellipsis():
    short = "a short passage"
    assert snippet(short) == short
    long = "x" * 500
    out = snippet(long, radius=10)
    assert out.endswith("…") and len(out) <= 21
