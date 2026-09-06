"""Shared building blocks every parser uses so the IR means the same thing
regardless of source format. The six concrete parsers copy this pattern; keep
the format-specific code in the parser and the format-neutral mechanics here.

* :class:`SectionPath`   -- a heading stack that yields ``Block.section_path``
* :class:`PlainText`     -- accumulates the document's plaintext and hands back
  the ``(char_start, char_end)`` span for each block, so ``Locator.char_*`` is
  always an offset into one newline-joined rendering of the document in reading
  order
* :class:`BlockBuilder`  -- assigns ``order`` + deterministic ``id``, stamps
  ``document_id``, collects document-level ``extraction_flags``, and produces
  the final :class:`~app.parsers.base.ParseResult`
* :func:`rows_to_markdown` / :func:`table_verdict` -- table serialization and
  the "is this actually a table" judgment call, shared by pdf/docx/pptx
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.ir import Block, BlockType, Locator
from app.parsers.base import ParseInput, ParseResult

# A line that opens with a bullet glyph or an enumerator -> list_item.
LIST_ITEM_RE = re.compile(r"^\s*(?:[•◦▪·–\-\*]\s+|\(?\d+[.)]\s+)")


class SectionPath:
    """Maintain the heading breadcrumb. Call :meth:`enter` when a heading block
    is produced, read :meth:`current` for every following block. A heading
    block's own ``section_path`` is its ancestors (call ``enter`` *after*
    emitting it)."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def enter(self, level: int, title: str) -> None:
        title = title.strip()
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))

    def reset(self) -> None:
        self._stack.clear()

    def current(self) -> list[str]:
        return [t for _, t in self._stack]


class PlainText:
    r"""Accumulate the document plaintext. :meth:`add` appends one block's text
    and returns its char span; blocks are separated by a single ``\n`` so spans
    never overlap and always widen."""

    _SEP = "\n"

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0

    def add(self, text: str) -> tuple[int, int]:
        start = self._len
        self._parts.append(text)
        self._len += len(text)
        end = self._len
        self._parts.append(self._SEP)
        self._len += len(self._SEP)
        return start, end

    @property
    def value(self) -> str:
        return "".join(self._parts)


@dataclass
class BlockBuilder:
    """Turns ``(type, content, locator, section_path)`` into a fully-formed
    :class:`~app.ir.Block` with monotonic ``order`` and deterministic ``id``,
    and collects document-level flags."""

    src: ParseInput
    _order: int = 0
    blocks: list[Block] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def add(
        self,
        type: BlockType,
        content: str,
        *,
        locator: Locator | None = None,
        section_path: list[str] | None = None,
        embeddable: bool = True,
        flag: str | None = None,
    ) -> Block:
        block = Block(
            id=self.src.block_id(self._order),
            document_id=self.src.document_id,
            type=type,
            order=self._order,
            content=content,
            locator=locator or Locator(),
            section_path=list(section_path or []),
            embeddable=embeddable,
            flag=flag,
        )
        self.blocks.append(block)
        self._order += 1
        return block

    def note(self, flag: str) -> None:
        """Record a document-level extraction flag (deduped)."""
        if flag not in self.flags:
            self.flags.append(flag)

    def result(self, *, status_hint: str | None = None) -> ParseResult:
        return ParseResult(
            blocks=self.blocks,
            extraction_flags=list(self.flags),
            status_hint=status_hint,
        )


# --------------------------------------------------------------------------
# table helpers (shared by pdf / docx / pptx)
# --------------------------------------------------------------------------

TableVerdict = Literal["solid", "low_confidence", "reject"]

# The one flag string parsers append when a detected table is shaky or dropped.
FLAG_LOW_CONFIDENCE_TABLE = "low_confidence_table"


def table_verdict(rows: list[list[object]]) -> TableVerdict:
    """A structural, score-free judgment on a detected grid.

    * ``reject``          -- not tabular (< 2 rows / < 2 cols / mostly empty /
      wildly ragged). The caller drops it and appends
      :data:`FLAG_LOW_CONFIDENCE_TABLE`.
    * ``low_confidence``  -- tabular but thin or ragged. The caller emits it as a
      ``table`` block with ``Block.flag = "low_confidence"`` and also appends
      :data:`FLAG_LOW_CONFIDENCE_TABLE`.
    * ``solid``           -- emit as a plain ``table`` block.
    """
    if len(rows) < 2:
        return "reject"
    ncol = max((len(r) for r in rows), default=0)
    if ncol < 2:
        return "reject"
    cells = [c for r in rows for c in r]
    if not cells:
        return "reject"
    nonempty = sum(1 for c in cells if c is not None and str(c).strip()) / len(cells)
    consistent = sum(1 for r in rows if len(r) == ncol) / len(rows)
    if nonempty < 0.4 or consistent < 0.5:
        return "reject"
    if nonempty < 0.65 or consistent < 0.8:
        return "low_confidence"
    return "solid"


def rows_to_markdown(rows: list[list[object]]) -> str:
    """Row 0 is the header. Cells are stringified, newline-flattened, pipe-
    escaped; ragged rows are padded to the widest row."""
    clean = [
        [("" if c is None else str(c).replace("\n", " ").replace("|", r"\|").strip()) for c in r]
        for r in rows
    ]
    ncol = max((len(r) for r in clean), default=0)
    clean = [r + [""] * (ncol - len(r)) for r in clean]
    header, *body = clean
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncol) + " |",
    ]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)
