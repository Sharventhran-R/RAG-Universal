"""HTML parser (selectolax).

* ``script``/``style``/``nav``/``header``/``footer``/``form``/``svg``/
  ``noscript``/``template`` are stripped before anything else.
* ``<title>`` (if any) seeds ``section_path`` as a level-1 heading, like the
  pptx slide title.
* The body is walked depth-first; a block element is emitted and **not**
  recursed into (headings, ``p``, ``blockquote``, ``pre`` -> ``code``, ``li``,
  ``table``). Containers (``div``/``section``/``article``/...) are recursed.
* ``table`` runs through :func:`~app.parsers._common.table_verdict`; a layout
  table is flattened to paragraphs and flagged, same as docx/pptx.
* If the walk finds no block elements at all but the body has text, the whole
  body text is emitted as one paragraph with ``html_unstructured``.
* No text anywhere -> ``status_hint = EMPTY_NO_TEXT``.
"""

from __future__ import annotations

from selectolax.parser import HTMLParser, Node

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

_DROP = ("script", "style", "nav", "header", "footer", "form", "svg", "noscript", "template")
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARAGRAPHS = {"p", "blockquote"}
_STOP = _HEADINGS | _PARAGRAPHS | {"pre", "li", "table"}


class HtmlParser:
    name = "html-selectolax"
    version = "1"
    file_type = "html"
    mimetypes = ("text/html",)

    def parse(self, src: ParseInput) -> ParseResult:
        tree = HTMLParser(src.open_stream().read().decode("utf-8", errors="replace"))
        for sel in _DROP:
            for node in tree.css(sel):
                node.decompose()

        text = PlainText()
        sect = SectionPath()
        builder = BlockBuilder(src=src)

        if tree.head is not None:
            title_node = tree.head.css_first("title")
            title = title_node.text(strip=True) if title_node is not None else ""
            if title:
                start, end = text.add(title)
                builder.add(
                    "heading", title,
                    locator=Locator(char_start=start, char_end=end),
                    section_path=sect.current(),
                )
                sect.enter(0, title)  # level 0: a persistent root above any <h1>..<h6>

        if tree.body is not None:
            for child in tree.body.iter(include_text=False):
                _walk(child, text, sect, builder)

            if not any(b.type != "heading" for b in builder.blocks):
                body_text = tree.body.text(separator=" ", strip=True)
                if body_text:
                    builder.note("html_unstructured")
                    start, end = text.add(body_text)
                    builder.add(
                        "paragraph", body_text,
                        locator=Locator(char_start=start, char_end=end),
                        section_path=sect.current(),
                    )

        if not builder.blocks:
            builder.note("html_no_content")
            return builder.result(status_hint=EMPTY_NO_TEXT)
        return builder.result()


register_parser(HtmlParser())


def _walk(node: Node, text: PlainText, sect: SectionPath, builder: BlockBuilder) -> None:
    tag = node.tag

    if tag in _HEADINGS:
        content = node.text(separator=" ", strip=True)
        if content:
            start, end = text.add(content)
            builder.add(
                "heading", content,
                locator=Locator(char_start=start, char_end=end),
                section_path=sect.current(),
            )
            sect.enter(int(tag[1]), content)
        return

    if tag in _PARAGRAPHS:
        _emit_text(node, "paragraph", text, sect, builder)
        return

    if tag == "pre":
        _emit_text(node, "code", text, sect, builder)
        return

    if tag in ("ul", "ol"):
        for li in node.css("li"):
            _emit_text(li, "list_item", text, sect, builder)
        return

    if tag == "table":
        _emit_table(node, text, sect, builder)
        return

    for child in node.iter(include_text=False):
        _walk(child, text, sect, builder)


def _emit_text(
    node: Node, block_type: str, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    content = node.text(separator=" ", strip=True)
    if not content:
        return
    start, end = text.add(content)
    builder.add(
        block_type, content,
        locator=Locator(char_start=start, char_end=end),
        section_path=sect.current(),
    )


def _emit_table(
    node: Node, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    rows: list[list[str]] = []
    for tr in node.css("tr"):
        cells = tr.css("th, td")
        if cells:
            rows.append([c.text(separator=" ", strip=True) for c in cells])

    verdict = table_verdict(rows)
    if verdict == "reject":
        builder.note(FLAG_LOW_CONFIDENCE_TABLE)
        for row in rows:
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
