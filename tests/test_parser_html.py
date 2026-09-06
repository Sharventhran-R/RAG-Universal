from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("selectolax")

from app.parsers.base import EMPTY_NO_TEXT, ParseInput
from app.parsers.html import HtmlParser


def _src(html: str, filename: str = "page.html", version: str = "1") -> ParseInput:
    data = html.encode("utf-8")
    return ParseInput(
        document_id="doc1",
        filename=filename,
        mimetype="text/html",
        file_sha256="sha",
        extractor_version=version,
        open_stream=lambda: io.BytesIO(data),
        local_path=lambda: Path("unused"),
    )


PAGE = """<html><head><title>My Page</title></head><body>
<nav>site nav, drop me</nav>
<h1>Main Heading</h1>
<p>A paragraph with 9,800 in it.</p>
<div><section>
  <h2>Sub</h2>
  <p>Nested paragraph.</p>
  <ul><li>one</li><li>two</li></ul>
  <table>
    <tr><th>City</th><th>Pop</th></tr>
    <tr><td>NYC</td><td>8</td></tr>
    <tr><td>LA</td><td>4</td></tr>
  </table>
</section></div>
<script>var x = 1;</script>
<style>.x{color:red}</style>
</body></html>"""


def test_identity():
    assert HtmlParser.file_type == "html"
    assert "text/html" in HtmlParser.mimetypes


def test_script_style_nav_are_stripped():
    blocks = HtmlParser().parse(_src(PAGE)).blocks
    joined = " ".join(b.content for b in blocks)
    assert "var x = 1" not in joined
    assert "color:red" not in joined
    assert "site nav" not in joined


def test_title_is_a_persistent_root_above_headings():
    blocks = HtmlParser().parse(_src(PAGE)).blocks
    assert blocks[0].type == "heading" and blocks[0].content == "My Page"

    h1 = next(b for b in blocks if b.content == "Main Heading")
    assert h1.section_path == ["My Page"]

    para = next(b for b in blocks if "9,800" in b.content)
    assert para.section_path == ["My Page", "Main Heading"]

    nested = next(b for b in blocks if b.content == "Nested paragraph.")
    assert nested.section_path == ["My Page", "Main Heading", "Sub"]


def test_nested_containers_are_recursed_lists_and_tables_extracted():
    blocks = HtmlParser().parse(_src(PAGE)).blocks
    assert [b.content for b in blocks if b.type == "list_item"] == ["one", "two"]

    tbl = next(b for b in blocks if b.type == "table")
    assert "| City | Pop |" in tbl.content and "NYC" in tbl.content and "LA" in tbl.content
    assert tbl.section_path == ["My Page", "Main Heading", "Sub"]
    assert tbl.flag is None


def test_char_spans_ordered_and_ids_deterministic():
    a = HtmlParser().parse(_src(PAGE)).blocks
    b = HtmlParser().parse(_src(PAGE)).blocks
    assert [x.id for x in a] == [x.id for x in b]
    prev = -1
    for blk in a:
        assert blk.locator.char_start >= prev
        prev = blk.locator.char_end


def test_unstructured_body_falls_back_to_one_paragraph():
    res = HtmlParser().parse(_src("<html><body>just loose text, no tags at all</body></html>"))
    assert "html_unstructured" in res.extraction_flags
    assert [b.type for b in res.blocks] == ["paragraph"]
    assert res.blocks[0].content == "just loose text, no tags at all"


def test_layout_table_is_flattened_and_flagged():
    html = "<html><body><table><tr><td>just one boxed cell</td></tr></table></body></html>"
    res = HtmlParser().parse(_src(html))
    assert "low_confidence_table" in res.extraction_flags
    assert not any(b.type == "table" for b in res.blocks)
    assert any(b.content == "just one boxed cell" for b in res.blocks)


def test_empty_body_signals_empty_no_text():
    res = HtmlParser().parse(_src("<html><head></head><body>   </body></html>"))
    assert res.blocks == []
    assert res.status_hint == EMPTY_NO_TEXT
    assert "html_no_content" in res.extraction_flags
