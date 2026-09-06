"""Plain-text and Markdown parsers.

``TextParser`` (``text/plain``) is deliberately dumb: blank lines separate
blocks; a block whose lines all look like list items becomes ``list_item``
blocks, otherwise one ``paragraph``. No heading detection -- plain text has no
reliable heading signal.

``MarkdownParser`` (``text/markdown``) tokenises with markdown-it-py:
``#``..``######`` -> headings (with ``section_path``), fenced/indented code ->
``code``, list items -> ``list_item``, GFM tables -> ``table`` via
:func:`~app.parsers._common.table_verdict`, everything else -> ``paragraph``.

Char offsets are into the :class:`~app.parsers._common.PlainText` rendering, as
everywhere else -- not into the original file.
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from app.ir import Locator
from app.parsers._common import (
    FLAG_LOW_CONFIDENCE_TABLE,
    LIST_ITEM_RE,
    BlockBuilder,
    PlainText,
    SectionPath,
    rows_to_markdown,
    table_verdict,
)
from app.parsers.base import EMPTY_NO_TEXT, ParseInput, ParseResult, register_parser

_MD = MarkdownIt("commonmark").enable("table")


# --------------------------------------------------------------------------
# text/plain
# --------------------------------------------------------------------------


class TextParser:
    name = "text-plain"
    version = "1"
    file_type = "txt"
    mimetypes = ("text/plain",)

    def parse(self, src: ParseInput) -> ParseResult:
        raw = src.open_stream().read().decode("utf-8", errors="replace")
        text = PlainText()
        builder = BlockBuilder(src=src)

        for chunk in _split_blank_lines(raw):
            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            if not lines:
                continue
            if all(LIST_ITEM_RE.match(ln) for ln in lines):
                for ln in lines:
                    start, end = text.add(ln)
                    builder.add(
                        "list_item", ln,
                        locator=Locator(char_start=start, char_end=end),
                    )
            else:
                content = " ".join(lines)
                start, end = text.add(content)
                builder.add(
                    "paragraph", content,
                    locator=Locator(char_start=start, char_end=end),
                )

        if not builder.blocks:
            builder.note("text_no_content")
            return builder.result(status_hint=EMPTY_NO_TEXT)
        return builder.result()


def _split_blank_lines(raw: str) -> list[str]:
    blocks, current = [], []
    for line in raw.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


# --------------------------------------------------------------------------
# text/markdown
# --------------------------------------------------------------------------


class MarkdownParser:
    name = "markdown-it"
    version = "1"
    file_type = "md"
    mimetypes = ("text/markdown",)

    def parse(self, src: ParseInput) -> ParseResult:
        raw = src.open_stream().read().decode("utf-8", errors="replace")
        root = SyntaxTreeNode(_MD.parse(raw))
        text = PlainText()
        sect = SectionPath()
        builder = BlockBuilder(src=src)

        for node in root.children:
            _emit_node(node, text, sect, builder)

        if not builder.blocks:
            builder.note("markdown_no_content")
            return builder.result(status_hint=EMPTY_NO_TEXT)
        return builder.result()


register_parser(TextParser())
register_parser(MarkdownParser())


def _emit_node(
    node: SyntaxTreeNode, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    kind = node.type

    if kind == "heading":
        content = _inline_text(node)
        if not content:
            return
        start, end = text.add(content)
        builder.add(
            "heading", content,
            locator=Locator(char_start=start, char_end=end),
            section_path=sect.current(),
        )
        sect.enter(int(node.tag[1]), content)

    elif kind in ("fence", "code_block"):
        content = node.content.rstrip("\n")
        if not content:
            return
        start, end = text.add(content)
        builder.add(
            "code", content,
            locator=Locator(char_start=start, char_end=end),
            section_path=sect.current(),
        )

    elif kind in ("bullet_list", "ordered_list"):
        for item in node.children:
            if item.type != "list_item":
                continue
            content = " ".join(
                _inline_text(p) for p in item.children if p.type == "paragraph"
            ).strip() or _all_text(item)
            if not content:
                continue
            start, end = text.add(content)
            builder.add(
                "list_item", content,
                locator=Locator(char_start=start, char_end=end),
                section_path=sect.current(),
            )

    elif kind == "table":
        _emit_table(node, text, sect, builder)

    elif kind == "blockquote":
        for child in node.children:
            _emit_node(child, text, sect, builder)

    elif kind == "paragraph":
        content = _inline_text(node)
        if not content:
            return
        start, end = text.add(content)
        builder.add(
            "paragraph", content,
            locator=Locator(char_start=start, char_end=end),
            section_path=sect.current(),
        )


def _emit_table(
    node: SyntaxTreeNode, text: PlainText, sect: SectionPath, builder: BlockBuilder
) -> None:
    rows: list[list[str]] = []
    for section in node.children:                 # thead, tbody
        for tr in section.children:
            rows.append([_inline_text(cell) for cell in tr.children])

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


def _inline_text(node: SyntaxTreeNode) -> str:
    for child in node.children:
        if child.type == "inline":
            return (child.content or "").strip()
    return (node.content or "").strip()


def _all_text(node: SyntaxTreeNode) -> str:
    if node.type == "inline":
        return (node.content or "").strip()
    return " ".join(t for t in (_all_text(c) for c in node.children) if t).strip()
