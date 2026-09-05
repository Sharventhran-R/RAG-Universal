from __future__ import annotations

import io
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from app.parsers.base import EMPTY_NO_TEXT, ParseInput
from app.parsers.spreadsheet import CsvParser, XlsxParser


def _src(data: bytes, filename: str, document_id: str = "doc1") -> ParseInput:
    return ParseInput(
        document_id=document_id,
        filename=filename,
        mimetype="x",
        file_sha256="sha",
        extractor_version="1",
        open_stream=lambda: io.BytesIO(data),
        local_path=lambda: Path("unused"),
    )


def _xlsx(**sheets: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return buf.getvalue()


DF = pd.DataFrame(
    {
        "Region": ["North", "South", "East", "West", "North"],
        "Units": [120, 340, 55, 700, 90],
        "Revenue": [1200.5, 3400.0, 550.25, 7000.0, 900.0],
    }
)


# --- the summary block: the retrieval unit -------------------------------


def test_exactly_one_embeddable_summary_block_per_sheet():
    res = XlsxParser(rows_per_block=2).parse(_src(_xlsx(Q3=DF), "s.xlsx"))
    embeddable = [b for b in res.blocks if b.embeddable]
    assert len(embeddable) == 1
    summary = embeddable[0]
    assert summary.type == "table"
    assert summary.section_path == ["Q3"]


def test_summary_block_content_shape():
    res = XlsxParser().parse(_src(_xlsx(Q3=DF), "s.xlsx"))
    summary = next(b for b in res.blocks if b.embeddable)
    text = summary.content

    assert text.startswith('Sheet "Q3" — 5 rows × 3 columns')
    assert "| column | dtype | non-null | distinct | min | max | samples |" in text

    lines = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("| ")}
    assert "string" in lines["Region"] and "North; South; East" in lines["Region"]
    assert "integer" in lines["Units"] and "| 55 | 700 |" in lines["Units"]
    assert "float" in lines["Revenue"] and "550.25" in lines["Revenue"]


def test_row_windows_are_stored_but_not_embeddable():
    res = XlsxParser(rows_per_block=2).parse(_src(_xlsx(Q3=DF), "s.xlsx"))
    windows = [b for b in res.blocks if not b.embeddable]
    assert [w.section_path for w in windows] == [
        ["Q3", "rows 1-2"],
        ["Q3", "rows 3-4"],
        ["Q3", "rows 5-5"],
    ]
    assert all(w.type == "table" for w in windows)
    assert "| Region | Units | Revenue |" in windows[0].content
    assert "North" in windows[0].content and "South" in windows[0].content
    assert "East" not in windows[0].content  # row 3 belongs to the next window


def test_char_spans_ordered_and_ids_deterministic():
    data = _xlsx(Q3=DF)
    a = XlsxParser(rows_per_block=2).parse(_src(data, "s.xlsx")).blocks
    b = XlsxParser(rows_per_block=2).parse(_src(data, "s.xlsx")).blocks
    assert [x.id for x in a] == [x.id for x in b]
    prev = -1
    for blk in a:
        assert blk.locator.sheet == "Q3"
        assert blk.locator.char_start >= prev
        prev = blk.locator.char_end


# --- judgment-call flags ----------------------------------------------------


def test_clean_single_sheet_has_no_flags():
    res = XlsxParser().parse(_src(_xlsx(Q3=DF), "s.xlsx"))
    assert res.extraction_flags == []
    assert res.status_hint is None


def test_headers_only_sheet_is_flagged_empty_but_still_summarised():
    empty = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="float64")})
    res = XlsxParser().parse(_src(_xlsx(Data=DF, Blank=empty), "s.xlsx"))
    assert "sheet_empty" in res.extraction_flags
    summaries = [b for b in res.blocks if b.embeddable]
    assert {s.section_path[0] for s in summaries} == {"Data", "Blank"}
    assert not any(b.section_path[:1] == ["Blank"] and not b.embeddable for b in res.blocks)


def test_all_string_sheet_is_flagged():
    df = pd.DataFrame({"Name": ["a", "b"], "City": ["x", "y"]})
    res = XlsxParser().parse(_src(_xlsx(S=df), "s.xlsx"))
    assert "sheet_all_string_columns" in res.extraction_flags


# --- csv --------------------------------------------------------------------


def test_csv_uses_file_stem_as_sheet_name():
    res = CsvParser(rows_per_block=50).parse(_src(b"Region,Units\nNorth,120\nSouth,340\n", "sales.csv"))
    assert CsvParser.file_type == "csv"
    summary = next(b for b in res.blocks if b.embeddable)
    assert summary.section_path == ["sales"]
    assert res.extraction_flags == []


def test_empty_csv_signals_empty_no_text():
    res = CsvParser().parse(_src(b"", "empty.csv"))
    assert res.blocks == []
    assert res.status_hint == EMPTY_NO_TEXT
    assert "spreadsheet_no_data" in res.extraction_flags


def test_single_column_csv_survives_delimiter_sniff_failure():
    res = CsvParser(rows_per_block=50).parse(_src(b"Name\nAlice\nBob\n", "names.csv"))
    assert res.status_hint is None
    assert sum(b.embeddable for b in res.blocks) == 1
