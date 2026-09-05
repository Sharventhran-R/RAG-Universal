"""DOCX parser (python-docx).

Follows the pattern set by :mod:`app.parsers.pdf`:

* **Locator** -- no page (python-docx exposes no reliable page numbers), so
  citations lean on ``section_path``. ``char_start``/``char_end`` are offsets
  into the :class:`~app.parsers._common.PlainText` rendering.
* **section_path** -- from paragraph style: ``Heading 1``..``Heading 9`` and
  ``Title`` (level 0 -> treated as level 1).
* **Tables** -- a ``w:tbl`` is explicit, but Word tables are often used for page
  layout. :func:`~app.parsers._common.table_verdict` decides: ``reject`` ->
  the cell text is emitted as paragraphs and ``low_confidence_table`` is noted;
  ``low_confidence`` -> emitted as a ``table`` block flagged ``low_confidence``;
  ``solid`` -> a plain ``table`` block.
* **Empty** -- a document with no non-whitespace text -> ``status_hint``
  ``EMPTY_NO_TEXT``.
"""

from __future__ import annotations

import io
import re

from docx import Document as open_docx
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ir import Locator
from app.parsers._common import (
    FLAG_LOW_CONFIDENCE_TABLE,
    BlockBuilder,
    PlainText,
    SectionPath,
    rows_to_markdown,
    table_verdict,
)
from app.parsers.base import EMPTY_NO_TEXT, ParseInput, ParseResult, register_parser

_HEADING_RE = re.compile(r"^Heading (\d)$")
_LIST_STYLE_RE = re.compile(r"List (Bullet|Number|Paragraph)", re.IGNORECASE)


class DocxParser:
    name = "docx-python-docx"
    version = "1"
    file_type = "docx"
    mimetypes = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    def parse(self, src: ParseInput) -> ParseResult:
        doc = open_docx(io.BytesIO(src.open_stream().read()))
        text = PlainText()
        sect = SectionPath()
        builder = BlockBuilder(src=src)

        for item in _iter_body(doc):
            if isinstance(item, Paragraph):
                _emit_paragraph(item, text, sect, builder)
            else:
                _emit_table(item, text, sect, builder)

        if not builder.blocks:
            builder.note("docx_no_text")
            return builder.result(status_hint=EMPTY_NO_TEXT)
        return builder.result()


register_parser(DocxParser())


def _iter_body(doc):
    """Yield ``Paragraph`` / ``Table`` objects in document order (python-docx
    has no built-in ordered iterator across both)."""
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _emit_paragraph(
    para: Paragraph, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    content = para.text.strip()
    if not content:
        return
    style = (para.style.name if para.style is not None else "") or ""

    heading = _HEADING_RE.match(style)
    if heading or style == "Title":
        level = int(heading.group(1)) if heading else 1
        start, end = text.add(content)
        builder.add(
            "heading", content,
            locator=Locator(char_start=start, char_end=end),
            section_path=sect.current(),
        )
        sect.enter(level, content)
        return

    block_type = "list_item" if _LIST_STYLE_RE.search(style) else "paragraph"
    start, end = text.add(content)
    builder.add(
        block_type, content,
        locator=Locator(char_start=start, char_end=end),
        section_path=sect.current(),
    )


def _emit_table(
    table: Table, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    verdict = table_verdict(rows)

    if verdict == "reject":
        builder.note(FLAG_LOW_CONFIDENCE_TABLE)
        for row in rows:  # keep the text -- it just isn't a table
            for cell in row:
                if cell.strip():
                    start, end = text.add(cell.strip())
                    builder.add(
                        "paragraph", cell.strip(),
                        locator=Locator(char_start=start, char_end=end),
                        section_path=sect.current(),
                    )
        return

    flag = "low_confidence" if verdict == "low_confidence" else None
    if flag:
        builder.note(FLAG_LOW_CONFIDENCE_TABLE)
    md = rows_to_markdown(rows)
    start, end = text.add(md)
    builder.add(
        "table", md,
        locator=Locator(char_start=start, char_end=end),
        section_path=sect.current(),
        flag=flag,
    )
