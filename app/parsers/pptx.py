"""PPTX parser (python-pptx).

* **Locator** -- ``slide`` is 1-based; ``char_*`` offsets into the
  :class:`~app.parsers._common.PlainText` rendering.
* **section_path** -- the slide title is a level-1 heading; everything on the
  slide sits under ``[title]``. Speaker notes go under ``[title, "Notes"]``.
* **Tables** -- graphic-frame tables run through
  :func:`~app.parsers._common.table_verdict` like pdf/docx.
* **Silent skips get flags** -- charts (``chart_skipped``) and pictures carry
  data/meaning that would otherwise vanish without a trace; grouped shapes are
  not recursed in Phase 1 (``pptx_group_not_recursed``).
* **Empty** -- no text on any slide -> ``status_hint`` ``EMPTY_NO_TEXT``.
"""

from __future__ import annotations

import io

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

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


class PptxParser:
    name = "pptx-python-pptx"
    version = "1"
    file_type = "pptx"
    mimetypes = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    def parse(self, src: ParseInput) -> ParseResult:
        prs = Presentation(io.BytesIO(src.open_stream().read()))
        text = PlainText()
        sect = SectionPath()
        builder = BlockBuilder(src=src)

        for idx, slide in enumerate(prs.slides, start=1):
            _emit_slide(slide, idx, text, sect, builder)

        if not builder.blocks:
            builder.note("pptx_no_text")
            return builder.result(status_hint=EMPTY_NO_TEXT)
        return builder.result()


register_parser(PptxParser())


def _emit_slide(
    slide, idx: int, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    title_shape = slide.shapes.title
    title = (title_shape.text.strip() if title_shape is not None else "") or f"Slide {idx}"
    title_id = title_shape.shape_id if title_shape is not None else None

    start, end = text.add(title)
    builder.add(
        "heading", title,
        locator=Locator(slide=idx, char_start=start, char_end=end),
        section_path=sect.current(),
    )
    sect.enter(1, title)

    for shape in slide.shapes:
        if title_id is not None and shape.shape_id == title_id:
            continue

        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            builder.note("pptx_group_not_recursed")
            continue
        if shape.has_chart:
            builder.note("pptx_chart_skipped")
            continue
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            continue

        if shape.has_table:
            _emit_table(shape.table, idx, text, sect, builder)
        elif shape.has_text_frame:
            _emit_text_frame(shape.text_frame, idx, text, sect, builder)

    if slide.has_notes_slide:
        note = slide.notes_slide.notes_text_frame.text.strip()
        if note:
            start, end = text.add(note)
            builder.add(
                "paragraph", note,
                locator=Locator(slide=idx, char_start=start, char_end=end),
                section_path=sect.current() + ["Notes"],
            )


def _emit_text_frame(
    frame, idx: int, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    for para in frame.paragraphs:
        content = "".join(run.text for run in para.runs).strip() or para.text.strip()
        if not content:
            continue
        block_type = "list_item" if (para.level or 0) > 0 else "paragraph"
        start, end = text.add(content)
        builder.add(
            block_type, content,
            locator=Locator(slide=idx, char_start=start, char_end=end),
            section_path=sect.current(),
        )


def _emit_table(
    table, idx: int, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    verdict = table_verdict(rows)
    if verdict == "reject":
        builder.note(FLAG_LOW_CONFIDENCE_TABLE)
        for row in rows:
            for cell in row:
                if cell.strip():
                    start, end = text.add(cell.strip())
                    builder.add(
                        "paragraph", cell.strip(),
                        locator=Locator(slide=idx, char_start=start, char_end=end),
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
        locator=Locator(slide=idx, char_start=start, char_end=end),
        section_path=sect.current(),
        flag=flag,
    )
