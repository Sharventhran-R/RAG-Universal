from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("markdown_it")

from app.parsers.base import EMPTY_NO_TEXT, ParseInput
from app.parsers.text import MarkdownParser, TextParser


def _src(text: str, filename: str, version: str = "1") -> ParseInput:
    data = text.encode("utf-8")
    return ParseInput(
        document_id="doc1",
        filename=filename,
        mimetype="x",
        file_sha256="sha",
        extractor_version=version,
        open_stream=lambda: io.BytesIO(data),
        local_path=lambda: Path("unused"),
    )


# --- text/plain -----------------------------------------------------------

TXT = "Hello world.\nSecond line, same paragraph.\n\n- alpha\n- beta\n- gamma\n\nClosing paragraph."


def test_txt_identity():
    assert TextParser.file_type == "txt"


def test_txt_blank_lines_split_blocks_and_wrapped_lines_join():
    blocks = TextParser().parse(_src(TXT, "n.txt")).blocks
    assert blocks[0].type == "paragraph"
    assert blocks[0].content == "Hello world. Second line, same paragraph."
    assert [b.type for b in blocks if b.type == "list_item"] == ["list_item"] * 3
    assert blocks[-1].content == "Closing paragraph."
    assert all(b.section_path == [] for b in blocks)  # plain text has no headings


def test_txt_char_spans_ordered_and_ids_deterministic():
    a = TextParser().parse(_src(TXT, "n.txt")).blocks
    b = TextParser().parse(_src(TXT, "n.txt")).blocks
    assert [x.id for x in a] == [x.id for x in b]
    prev = -1
    for blk in a:
        assert blk.locator.page is None
        assert blk.locator.char_start >= prev
        prev = blk.locator.char_end


def test_txt_empty_signals_empty_no_text():
    res = TextParser().parse(_src("   \n\n\t\n", "e.txt"))
    assert res.blocks == []
    assert res.status_hint == EMPTY_NO_TEXT
    assert "text_no_content" in res.extraction_flags


# --- text/markdown ------------------------------------------------------

MD = """# Report

Intro paragraph with 1,234 in it.

## Section A

- first
- second

```python
x = 1
```

| Region | Amount |
| --- | --- |
| North | 500 |
| South | 734 |

> a block quote
"""


def test_md_identity():
    assert MarkdownParser.file_type == "md"


def test_md_block_types_and_section_path():
    blocks = MarkdownParser().parse(_src(MD, "r.md")).blocks
    by_type = {}
    for b in blocks:
        by_type.setdefault(b.type, []).append(b)

    assert by_type["heading"][0].content == "Report"
    assert by_type["heading"][0].section_path == []

    para = next(b for b in blocks if b.type == "paragraph" and "1,234" in b.content)
    assert para.section_path == ["Report"]

    assert [li.content for li in by_type["list_item"]] == ["first", "second"]
    assert by_type["list_item"][0].section_path == ["Report", "Section A"]

    assert by_type["code"][0].content == "x = 1"

    tbl = by_type["table"][0]
    assert "| Region | Amount |" in tbl.content and "734" in tbl.content
    assert tbl.section_path == ["Report", "Section A"]
    assert tbl.flag is None

    assert any(b.type == "paragraph" and b.content == "a block quote" for b in blocks)


def test_md_degenerate_table_is_flagged():
    md = "| only |\n| --- |\n| one |\n"
    res = MarkdownParser().parse(_src(md, "d.md"))
    assert "low_confidence_table" in res.extraction_flags
    assert not any(b.type == "table" for b in res.blocks)


def test_md_empty_signals_empty_no_text():
    res = MarkdownParser().parse(_src("", "e.md"))
    assert res.status_hint == EMPTY_NO_TEXT
    assert "markdown_no_content" in res.extraction_flags
