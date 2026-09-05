"""PDF parser (PyMuPDF).

Reference implementation of the parser pattern the other five copy:

* **Locator semantics** -- ``page`` is 1-based; ``char_start``/``char_end`` are
  offsets into the single plaintext rendering built by
  :class:`~app.parsers._common.PlainText`. Spans never overlap and always widen.
* **section_path** -- a font-size heuristic promotes short single-line runs to
  headings; :class:`~app.parsers._common.SectionPath` turns the heading stack
  into the breadcrumb. A heading block's own ``section_path`` is its ancestors,
  not itself.
* **Tables** -- ``page.find_tables()`` is noisy. Each hit goes through
  :func:`~app.parsers._common.table_verdict` (structural, score-free). ``reject``
  -> dropped, ``low_confidence_table`` flag on the document. ``low_confidence``
  -> emitted with ``Block.flag = "low_confidence"`` plus the document flag.
  There is **no** confidence field in the IR (the promotion / 0.7-threshold
  design was dropped in the v2 rewrite).
* **No text layer** -- pages exist but every extracted run is empty (scanned, no
  OCR) -> ``status_hint = EMPTY_NO_TEXT``; the document never reaches ``ready``.

Known Phase 1 limitations: no code-block detection (rare in PDFs).
"""

from __future__ import annotations

import re
from collections import Counter

import pymupdf

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

_MAX_HEADING_CHARS = 160
_LIST_RE = re.compile(r"^\s*(?:[•◦▪·–\-\*]\s+|\(?\d+[.)]\s+)")
_HEADING_MIN_DELTA = 0.5          # a heading size must exceed body size by this
_INSIDE_RATIO = 0.7              # text block counts as "in" a table at this overlap


class PdfParser:
    name = "pdf-pymupdf"
    version = "2"                # BUMP on any change to emitted blocks (busts the extraction cache)
    file_type = "pdf"
    mimetypes = ("application/pdf",)

    def parse(self, src: ParseInput) -> ParseResult:
        data = src.open_stream().read()
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            body = _body_size(doc)
            heading_sizes = _heading_sizes(doc, body)

            text = PlainText()
            sect = SectionPath()
            builder = BlockBuilder(src=src)

            for pno in range(1, doc.page_count + 1):
                _parse_page(doc[pno - 1], pno, heading_sizes, text, sect, builder)

            if not builder.blocks and doc.page_count > 0:
                builder.note("pdf_no_text_layer")
                return builder.result(status_hint=EMPTY_NO_TEXT)
            return builder.result()
        finally:
            doc.close()


register_parser(PdfParser())


# --------------------------------------------------------------------------
# font-size analysis
# --------------------------------------------------------------------------


def _round2(x: float) -> float:
    return round(x * 2) / 2


def _size_histogram(doc: "pymupdf.Document") -> Counter:
    hist: Counter = Counter()
    for page in doc:
        for blk in page.get_text("dict").get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        hist[_round2(span.get("size", 0.0))] += len(txt)
    return hist


def _body_size(doc: "pymupdf.Document") -> float:
    hist = _size_histogram(doc)
    return hist.most_common(1)[0][0] if hist else 12.0


def _heading_sizes(doc: "pymupdf.Document", body: float) -> list[float]:
    """Distinct sizes clearly larger than body text, largest first. A run at
    ``heading_sizes[i]`` is heading level ``i + 1``."""
    hist = _size_histogram(doc)
    sizes = {s for s, weight in hist.items() if s >= body + _HEADING_MIN_DELTA and weight >= 3}
    return sorted(sizes, reverse=True)


# --------------------------------------------------------------------------
# per-page extraction
# --------------------------------------------------------------------------


def _parse_page(
    page: "pymupdf.Page",
    pno: int,
    heading_sizes: list[float],
    text: PlainText,
    sect: SectionPath,
    builder: BlockBuilder,
) -> None:
    table_rects: list[pymupdf.Rect] = []
    units: list[tuple[float, str, str, object]] = []  # (y_top, kind, content, meta)

    try:
        found = list(page.find_tables().tables)
    except Exception:  # pragma: no cover - PyMuPDF can raise on pathological pages
        found = []
    for tab in found:
        rect = pymupdf.Rect(tab.bbox)
        verdict = table_verdict(tab.extract())
        if verdict == "reject":
            builder.note(FLAG_LOW_CONFIDENCE_TABLE)
            continue
        table_rects.append(rect)
        flag = "low_confidence" if verdict == "low_confidence" else None
        if flag:
            builder.note(FLAG_LOW_CONFIDENCE_TABLE)
        units.append((rect.y0, "table", rows_to_markdown(tab.extract()), flag))

    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        line_texts: list[str] = []
        max_size = 0.0
        for line in blk.get("lines", []):
            joined = "".join(s.get("text", "") for s in line.get("spans", []))
            if joined.strip():
                line_texts.append(joined.strip())
            for span in line.get("spans", []):
                max_size = max(max_size, span.get("size", 0.0))
        content = " ".join(line_texts).strip()
        if not content:
            continue
        rect = pymupdf.Rect(blk["bbox"])
        if any(_mostly_inside(rect, tr) for tr in table_rects):
            continue
        units.append((rect.y0, "text", content, (len(line_texts), max_size)))

    units.sort(key=lambda u: u[0])

    for _y, kind, content, meta in units:
        if kind == "table":
            start, end = text.add(content)
            builder.add(
                "table",
                content,
                locator=Locator(page=pno, char_start=start, char_end=end),
                section_path=sect.current(),
                flag=meta,  # type: ignore[arg-type]
            )
            continue

        n_lines, max_size = meta  # type: ignore[misc]
        rsize = _round2(max_size)
        if n_lines == 1 and rsize in heading_sizes and len(content) <= _MAX_HEADING_CHARS:
            start, end = text.add(content)
            builder.add(
                "heading",
                content,
                locator=Locator(page=pno, char_start=start, char_end=end),
                section_path=sect.current(),
            )
            sect.enter(heading_sizes.index(rsize) + 1, content)
            continue

        block_type = "list_item" if _LIST_RE.match(content) else "paragraph"
        start, end = text.add(content)
        builder.add(
            block_type,
            content,
            locator=Locator(page=pno, char_start=start, char_end=end),
            section_path=sect.current(),
        )


def _mostly_inside(inner: "pymupdf.Rect", outer: "pymupdf.Rect") -> bool:
    clip = inner & outer
    if not clip or inner.get_area() == 0:
        return False
    return clip.get_area() / inner.get_area() >= _INSIDE_RATIO
