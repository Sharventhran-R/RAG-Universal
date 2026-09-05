from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("pptx")
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from app.parsers.base import EMPTY_NO_TEXT, ParseInput
from app.parsers.pptx import PptxParser


def _src(data: bytes, document_id: str = "doc1") -> ParseInput:
    return ParseInput(
        document_id=document_id,
        filename="deck.pptx",
        mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        file_sha256="sha",
        extractor_version=PptxParser.version,
        open_stream=lambda: io.BytesIO(data),
        local_path=lambda: Path("unused"),
    )


@pytest.fixture
def sample() -> bytes:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Market Overview"
    tf = slide.placeholders[1].text_frame
    tf.text = "Total sales reached 9,800 units."
    bullet = tf.add_paragraph()
    bullet.text = "Region breakdown follows"
    bullet.level = 1

    table = slide.shapes.add_table(3, 2, Inches(1), Inches(3), Inches(4), Inches(2)).table
    cells = {(0, 0): "Region", (0, 1): "Units",
             (1, 0): "North", (1, 1): "500",
             (2, 0): "South", (2, 1): "734"}
    for (r, c), v in cells.items():
        table.cell(r, c).text = v

    chart_data = CategoryChartData()
    chart_data.categories = ["a", "b"]
    chart_data.add_series("s", (1, 2))
    slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(2), Inches(2), chart_data
    )

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_identity():
    assert PptxParser.file_type == "pptx"


def test_title_emitted_once_as_heading_with_slide_locator(sample):
    blocks = PptxParser().parse(_src(sample)).blocks
    headings = [b for b in blocks if b.type == "heading"]
    assert [h.content for h in headings] == ["Market Overview"]
    assert all(b.locator.slide == 1 for b in blocks)


def test_body_paragraph_and_bullet_under_title(sample):
    blocks = PptxParser().parse(_src(sample)).blocks
    para = next(b for b in blocks if b.type == "paragraph")
    assert "9,800" in para.content
    assert para.section_path == ["Market Overview"]
    assert any(b.type == "list_item" and b.section_path == ["Market Overview"] for b in blocks)


def test_table_block_present_and_unflagged(sample):
    tbl = next(b for b in PptxParser().parse(_src(sample)).blocks if b.type == "table")
    assert "| Region | Units |" in tbl.content and "734" in tbl.content
    assert tbl.section_path == ["Market Overview"]
    assert tbl.flag is None


def test_chart_is_recorded_as_a_skip(sample):
    res = PptxParser().parse(_src(sample))
    assert "pptx_chart_skipped" in res.extraction_flags


def test_char_spans_ordered_and_ids_deterministic(sample):
    a = PptxParser().parse(_src(sample)).blocks
    b = PptxParser().parse(_src(sample)).blocks
    assert [x.id for x in a] == [x.id for x in b]
    prev = -1
    for blk in a:
        assert blk.locator.char_start >= prev
        prev = blk.locator.char_end


def test_deck_with_no_slides_signals_empty_no_text():
    buf = io.BytesIO()
    Presentation().save(buf)
    res = PptxParser().parse(_src(buf.getvalue()))
    assert res.blocks == []
    assert res.status_hint == EMPTY_NO_TEXT
    assert "pptx_no_text" in res.extraction_flags
