"""Canonical intermediate representation (IR).

Parsers are the ONLY format-aware code in this system. Every parser returns
``list[Block]``; nothing downstream ever sees a file extension or a mimetype.

stdlib only. If the dataclasses below change shape in a way that invalidates
cached extractions, bump ``IR_VERSION`` and every parser's ``version``.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

IR_VERSION = 3

BlockType = Literal["heading", "paragraph", "table", "code", "list_item"]
BLOCK_TYPES: frozenset[str] = frozenset(
    ("heading", "paragraph", "table", "code", "list_item")
)


@dataclass(slots=True)
class Locator:
    """Everything needed to cite a block exactly.

    All fields optional: a CSV row has no page, a PDF paragraph has no sheet,
    a python-docx paragraph has no reliable page number (so docx citations lean
    on ``section_path`` instead).
    """

    page: Optional[int] = None          # 1-based (pdf)
    sheet: Optional[str] = None         # worksheet name (xlsx / csv stem)
    slide: Optional[int] = None         # 1-based (pptx)
    char_start: Optional[int] = None    # offset into the document plaintext
    char_end: Optional[int] = None      # exclusive


@dataclass(slots=True)
class Block:
    id: str                            # set via ParseInput.block_id(order)
    document_id: str                   # set to ParseInput.document_id
    type: BlockType
    order: int                         # 0-based, monotonic within a document
    content: str                      # markdown; tables serialized as markdown
    locator: Locator = field(default_factory=Locator)
    section_path: list[str] = field(default_factory=list)   # heading breadcrumb, outermost first
    embeddable: bool = True            # False => stored as a chunk for citation/reading,
    #                                    but its vector is NEVER added to FAISS.
    #                                    Used for spreadsheet row-window blocks
    #                                    (near-duplicate rows poison top-k); any
    #                                    parser may use it for bulk/appendix tables.
    flag: Optional[str] = None         # table blocks ONLY. A minimal "this happened"
    #                                    marker, e.g. "low_confidence" -- NOT a score,
    #                                    NOT a threshold. Document-level counterparts
    #                                    live in ParseResult.extraction_flags.

    def __post_init__(self) -> None:
        if self.type not in BLOCK_TYPES:
            raise ValueError(f"unknown block type: {self.type!r}")
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if self.flag is not None and self.type != "table":
            raise ValueError("Block.flag is only valid on table blocks")


# Fixed namespace so ids are reproducible across processes and runs.
_BLOCK_ID_NS = uuid.UUID("6f9b1d2e-0c3a-4e5b-9a7c-1f2e3d4c5b6a")


def new_block_id(file_sha256: str, extractor_version: str, order: int) -> str:
    """Deterministic block id: stable across re-extraction of identical bytes by
    the same extractor version, so a cache reload reproduces the same ids and
    downstream rows can be diffed rather than blindly replaced."""
    return str(uuid.uuid5(_BLOCK_ID_NS, f"{file_sha256}:{extractor_version}:{order}"))


# --- serialization for the extraction cache (JSON on disk) --------------------


def block_to_dict(block: Block) -> dict[str, Any]:
    return asdict(block)


def block_from_dict(data: dict[str, Any]) -> Block:
    loc = dict(data.get("locator") or {})
    return Block(
        id=data["id"],
        document_id=data["document_id"],
        type=data["type"],
        order=data["order"],
        content=data["content"],
        locator=Locator(**loc),
        section_path=list(data.get("section_path") or []),
        embeddable=bool(data.get("embeddable", True)),
        flag=data.get("flag"),
    )
