from __future__ import annotations

import io
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from app.parsers._common import rows_to_markdown, table_verdict
from app.parsers.base import EMPTY_NO_TEXT, ParseInput
from app.parsers.pdf import PdfParser


def _src(data: bytes, *, filename: str = "annual_report.pdf", document_id: str = "doc1") -> ParseInput:
    return ParseInput(
        document_id=document_id,
        filename=filename,
        mimetype="application/pdf",
        file_sha256="deadbeef",
        extractor_version=PdfParser.version,
        open_stream=lambda: io.BytesIO(data),
        local_path=lambda: Path("unused"),
    )


def _sample_pdf() -> bytes:
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), "Annual Report", fontsize=24)
    p1.insert_text((72, 120), "Revenue", fontsize=16)
    p1.insert_text((72, 150), "Total revenue was 1,234 million in 2023.", fontsize=11)
    p1.insert_text((72, 175), "- first point", fontsize=11)
    p1.insert_text((72, 195), "- second point", fontsize=11)

    p2 = doc.new_page()
    p2.insert_text((72, 72), "Financials", fontsize=16)
    x0, y0, rowh, colw, nrows, ncols = 72, 110, 22, 130, 3, 2
    for i in range(nrows + 1):
        p2.draw_line((x0, y0 + i * rowh), (x0 + ncols * colw, y0 + i * rowh))
    for j in range(ncols + 1):
        p2.draw_line((x0 + j * colw, y0), (x0 + j * colw, y0 + nrows * rowh))
    grid = [["Region", "Amount"], ["North", "500"], ["South", "734"]]
    for ri, row in enumerate(grid):
        for ci, val in enumerate(row):
            p2.insert_text((x0 + ci * colw + 6, y0 + ri * rowh + 15), val, fontsize=10)

    return doc.tobytes()


@pytest.fixture
def sample_blocks():
    data = _sample_pdf()
    return lambda document_id="doc1": PdfParser().parse(_src(data, document_id=document_id)).blocks


def test_parser_identity():
    assert PdfParser.file_type == "pdf"
    assert "application/pdf" in PdfParser.mimetypes


def test_returns_parse_result_with_no_flags_for_clean_input():
    res = PdfParser().parse(_src(_sample_pdf()))
    assert res.blocks
    assert res.extraction_flags == []
    assert res.status_hint is None


def test_blocks_have_ids_document_id_and_monotonic_order(sample_blocks):
    blocks = sample_blocks()
    assert [b.order for b in blocks] == list(range(len(blocks)))
    assert all(b.document_id == "doc1" for b in blocks)
    assert len({b.id for b in blocks}) == len(blocks)


def test_ids_are_deterministic_across_runs(sample_blocks):
    assert [b.id for b in sample_blocks()] == [b.id for b in sample_blocks()]


def test_headings_paragraph_list_and_table_all_present(sample_blocks):
    kinds = {b.type for b in sample_blocks()}
    assert {"heading", "paragraph", "list_item", "table"} <= kinds


def test_first_block_is_the_title_on_page_one(sample_blocks):
    first = sample_blocks()[0]
    assert first.type == "heading"
    assert first.content == "Annual Report"
    assert first.locator.page == 1


def test_numeric_paragraph_is_captured(sample_blocks):
    assert any("1,234" in b.content for b in sample_blocks() if b.type == "paragraph")


def test_table_block_is_markdown_on_page_two(sample_blocks):
    tables = [b for b in sample_blocks() if b.type == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.locator.page == 2
    assert "| Region | Amount |" in t.content
    assert "North" in t.content and "734" in t.content
    assert t.embeddable is True
    assert t.flag is None


def test_section_path_tracks_heading_hierarchy(sample_blocks):
    blocks = sample_blocks()
    para = next(b for b in blocks if b.type == "paragraph" and "1,234" in b.content)
    assert para.section_path == ["Annual Report", "Revenue"]
    table = next(b for b in blocks if b.type == "table")
    assert table.section_path == ["Annual Report", "Financials"]


def test_char_spans_are_ordered_and_non_overlapping(sample_blocks):
    prev_end = -1
    for b in sample_blocks():
        cs, ce = b.locator.char_start, b.locator.char_end
        assert cs is not None and ce is not None
        assert cs < ce
        assert cs >= prev_end
        prev_end = ce


def test_scanned_pdf_with_no_text_layer_signals_empty_no_text():
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text
    doc.new_page()
    res = PdfParser().parse(_src(doc.tobytes(), filename="scan.pdf"))
    assert res.blocks == []
    assert res.status_hint == EMPTY_NO_TEXT
    assert "pdf_no_text_layer" in res.extraction_flags


# --- shared table helpers (the copyable judgment-call pattern) --------------


def test_table_verdict_three_way():
    assert table_verdict([["a", "b"], ["1", "2"], ["3", "4"]]) == "solid"
    # tabular but one row is ragged -> column consistency 3/4
    assert table_verdict(
        [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"], ["7", "8"]]
    ) == "low_confidence"
    assert table_verdict([["only one row", "x"]]) == "reject"          # < 2 rows
    assert table_verdict([["a"], ["b"], ["c"]]) == "reject"            # < 2 cols
    assert table_verdict([["a", "b"], [None, None], [None, None]]) == "reject"  # mostly empty


def test_rows_to_markdown_escapes_pipes_and_pads_ragged_rows():
    md = _md = rows_to_markdown([["h1", "h2"], ["a|b"], ["x", "y"]])
    lines = _md.splitlines()
    assert lines[0] == "| h1 | h2 |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == r"| a\|b |  |"
    assert lines[3] == "| x | y |"
