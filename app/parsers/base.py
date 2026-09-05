"""Parser registry and the ``Parser`` interface.

Invariant: adding a format is **one new module** in ``app/parsers/`` plus
**one** ``register_parser(...)`` call at the bottom of it, plus one import line
in ``app/parsers/__init__.py``. Parsers are the only format-aware code.
"""

from __future__ import annotations

import mimetypes as _mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Protocol, runtime_checkable

from app.ir import Block, new_block_id


@dataclass(slots=True)
class ParseInput:
    """Everything a parser needs. No storage-backend detail leaks in here.

    ``local_path()`` for libraries that want a filename on disk (PyMuPDF,
    python-pptx); ``open_stream()`` for streaming readers (pandas, ``csv``).
    Both are safe to call more than once.
    """

    document_id: str
    filename: str
    mimetype: str
    file_sha256: str
    extractor_version: str
    open_stream: Callable[[], BinaryIO]
    local_path: Callable[[], Path]

    def block_id(self, order: int) -> str:
        """Deterministic id for the block at ``order``. Use this for every
        block; never invent your own id scheme."""
        return new_block_id(self.file_sha256, self.extractor_version, order)


class UnsupportedFormatError(Exception):
    """Raised by ``ParserRegistry.resolve`` when nothing matches.

    The pipeline catches this and sets the document's status to ``unsupported``
    -- never ``failed``, never a silently empty list.
    """

    def __init__(self, mimetype: str, filename: str) -> None:
        super().__init__(f"no parser for mimetype={mimetype!r} filename={filename!r}")
        self.mimetype = mimetype
        self.filename = filename


# Terminal status a parser can ask the pipeline to set when it parsed the file
# fine but produced nothing indexable (e.g. a scanned PDF with no text layer).
EMPTY_NO_TEXT = "empty_no_text"


@dataclass(slots=True)
class ParseResult:
    """What a parser returns.

    ``blocks``            -- IR blocks in reading order.
    ``extraction_flags`` -- deduped document-level markers for silent judgment
                            calls the parser made (a dropped uncertain table, a
                            skipped chart, a sheet with no header). Persisted on
                            ``documents.extraction_flags``. "This happened"
                            visibility, nothing more.
    ``status_hint``      -- if set, the terminal status the pipeline should use
                            instead of ``ready``. Currently only
                            ``EMPTY_NO_TEXT``. A parser that returns no blocks
                            without a hint still lands on ``empty_no_text`` (the
                            pipeline's blanket rule) -- the hint is for saying so
                            deliberately and distinctly.
    """

    blocks: list[Block]
    extraction_flags: list[str] = field(default_factory=list)
    status_hint: Optional[str] = None


@runtime_checkable
class Parser(Protocol):
    name: str                       # log/identity name, e.g. "pdf-pymupdf"
    version: str                    # extractor_version; part of the extraction cache key.
    #                                 BUMP whenever parse() output changes for the same input.
    file_type: str                  # short canonical tag written to documents.file_type
    #                                 and matched by the `file_type` retrieval filter
    #                                 (e.g. "pdf", "docx", "xlsx", "csv").
    mimetypes: tuple[str, ...]

    def parse(self, src: ParseInput) -> ParseResult:
        """Parse ``src`` into IR.

        - ``block.id = src.block_id(order)``, ``block.document_id = src.document_id``
          (use ``app.parsers._common.BlockBuilder`` and you get this for free)
        - set the ``Locator`` fields that apply (page for pdf, sheet for xlsx,
          slide for pptx, char offsets everywhere)
        - serialize tables as GitHub-flavoured markdown into ``block.content``
        - record any silent keep/skip decision in ``extraction_flags``
        - if the file parsed but yielded nothing indexable, set
          ``status_hint = EMPTY_NO_TEXT``
        """
        ...


class ParserRegistry:
    def __init__(self) -> None:
        self._by_mime: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        for mime in parser.mimetypes:
            existing = self._by_mime.get(mime)
            if existing is not None and existing is not parser:
                raise ValueError(
                    f"parser already registered for {mime}: {existing.name}"
                )
            self._by_mime[mime] = parser

    def get(self, mimetype: str) -> Parser | None:
        return self._by_mime.get(mimetype)

    def resolve(self, mimetype: str | None, filename: str) -> Parser:
        """Resolve by mimetype, then by filename extension for callers that only
        had ``application/octet-stream``. Raises ``UnsupportedFormatError``."""
        if mimetype:
            parser = self._by_mime.get(mimetype)
            if parser is not None:
                return parser
        guessed, _ = _mimetypes.guess_type(filename)
        if guessed:
            parser = self._by_mime.get(guessed)
            if parser is not None:
                return parser
        raise UnsupportedFormatError(mimetype or guessed or "unknown", filename)

    def supported_mimetypes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_mime))


registry = ParserRegistry()


def register_parser(parser: Parser) -> Parser:
    """Call once at module load time from each parser module."""
    registry.register(parser)
    return parser
