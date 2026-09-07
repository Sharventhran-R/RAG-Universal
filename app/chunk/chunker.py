"""Structure-aware sliding-window chunker.

Turns a document's ``list[Block]`` into ``ChunkDraft``s ready for
``app.db.repo.replace_chunks``. No embedding here -- the worker calls the
embedder afterwards on the drafts whose ``embedded`` is true.

Rules (ARCHITECTURE section 7):

* A window boundary only ever falls **between** blocks.
* ``table`` / ``code`` blocks are atomic -- each is its own chunk, never merged,
  never split even if it exceeds the budget.
* A ``heading`` starts a fresh chunk (no overlap carried across it) and sits at
  the top of that chunk. A heading with no content after it (``heading`` then
  another ``heading`` / ``table`` / ``code``) is never a chunk on its own -- it
  is prepended to the next block's chunk, so "Table 3: Revenue" stays attached
  to its table. (An atomic block's chunk is still never split and never merged
  with prose or another atomic block.)
* ``paragraph`` / ``list_item`` blocks accumulate up to ``CHUNK_TOKENS``; on a
  size flush, trailing blocks up to ``CHUNK_OVERLAP`` tokens are carried into
  the next window.
* ``embeddable=False`` blocks (spreadsheet row windows) become chunks with
  ``embedded=0`` -- persisted for citation, never vectorised.
* ``embed_input`` = ``"{prefix}\n\n{text}"``. The body keeps its full budget;
  the prefix (``"{filename} > {last 2 section levels}"``) is what gets trimmed,
  down to a middle-elided filename if it still won't fit ``PREFIX_MAX_TOKENS``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.ir import Block

TokenCounter = Callable[[str], int]

_BLOCK_SEP = "\n\n"
_ELLIPSIS = "…"
_ATOMIC_TYPES = ("table", "code")


@dataclass(slots=True)
class ChunkDraft:
    ord: int
    text: str
    embed_input: str
    token_count: int
    embedded: bool
    page_start: int | None
    page_end: int | None
    sheet: str | None
    slide: int | None
    char_start: int | None
    char_end: int | None
    section_path: list[str]

    def to_row(self, session_id: str) -> dict:
        return {
            "session_id": session_id,
            "ord": self.ord,
            "text": self.text,
            "embed_input": self.embed_input,
            "token_count": self.token_count,
            "embedded": self.embedded,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "sheet": self.sheet,
            "slide": self.slide,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_path": self.section_path,
        }


def chunk_document(
    blocks: list[Block],
    *,
    filename: str,
    count_tokens: TokenCounter,
    settings: Settings | None = None,
) -> list[ChunkDraft]:
    s = settings or get_settings()
    max_body, overlap, prefix_cap = s.chunk_tokens, s.chunk_overlap, s.prefix_max_tokens

    drafts: list[ChunkDraft] = []
    window: list[Block] = []

    def emit(members: list[Block], *, embedded: bool) -> None:
        drafts.append(
            _draft(len(drafts), members, filename, count_tokens, prefix_cap, embedded)
        )

    def flush(*, carry: bool) -> None:
        """Emit accumulated prose. A window that is only a heading is left in
        place (it belongs to whatever comes next), not emitted alone."""
        nonlocal window
        if not window or _is_lone_heading(window):
            window = []
            return
        emit(window, embedded=True)
        window = _overlap_tail(window, overlap, count_tokens) if carry else []

    for block in blocks:
        if not block.embeddable or block.type in _ATOMIC_TYPES:
            if _is_lone_heading(window):
                emit([window[0], block], embedded=block.embeddable)
                window = []
            else:
                flush(carry=False)
                emit([block], embedded=block.embeddable)
            continue
        if block.type == "heading":
            flush(carry=False)
            window = [block]
            continue

        # paragraph / list_item
        if not window or count_tokens(_join(window + [block])) <= max_body:
            window.append(block)
            continue
        flush(carry=True)
        if window and count_tokens(_join(window + [block])) > max_body:
            window = [block]  # a big block: drop the overlap so it gets room
        else:
            window.append(block)

    if _is_lone_heading(window) and not drafts:
        emit(window, embedded=True)  # a document that is nothing but headings
    else:
        flush(carry=False)
    return drafts


# --------------------------------------------------------------------------


def _join(blocks: list[Block]) -> str:
    return _BLOCK_SEP.join(b.content for b in blocks)


def _is_lone_heading(window: list[Block]) -> bool:
    return len(window) == 1 and window[0].type == "heading"


def _overlap_tail(window: list[Block], overlap: int, count: TokenCounter) -> list[Block]:
    tail: list[Block] = []
    total = 0
    for block in reversed(window):
        total += count(block.content)
        if total > overlap:
            break
        tail.insert(0, block)
    return tail


def _representative_section_path(blocks: list[Block]) -> list[str]:
    first = blocks[0]
    if first.type == "heading":
        return [*first.section_path, first.content]
    return list(first.section_path)


def _draft(
    ordinal: int,
    blocks: list[Block],
    filename: str,
    count: TokenCounter,
    prefix_cap: int,
    embedded: bool,
) -> ChunkDraft:
    text = _join(blocks)
    section_path = _representative_section_path(blocks)
    pages = [b.locator.page for b in blocks if b.locator.page is not None]
    return ChunkDraft(
        ord=ordinal,
        text=text,
        embed_input=_embed_input(filename, section_path, text, count, prefix_cap),
        token_count=count(text),
        embedded=embedded,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        sheet=blocks[0].locator.sheet,
        slide=blocks[0].locator.slide,
        char_start=blocks[0].locator.char_start,
        char_end=blocks[-1].locator.char_end,
        section_path=section_path,
    )


def _embed_input(
    filename: str, section_path: list[str], text: str, count: TokenCounter, cap: int
) -> str:
    prefix = _prefix(filename, section_path)
    if count(prefix) > cap:
        prefix = _prefix(_middle_elide(filename, section_path, count, cap), section_path)
    return f"{prefix}{_BLOCK_SEP}{text}"


def _prefix(filename: str, section_path: list[str]) -> str:
    if len(section_path) > 2:
        crumbs = f"{_ELLIPSIS} > " + " > ".join(section_path[-2:])
    elif section_path:
        crumbs = " > ".join(section_path)
    else:
        return filename
    return f"{filename} > {crumbs}"


def _middle_elide(
    filename: str, section_path: list[str], count: TokenCounter, cap: int
) -> str:
    for keep in range(len(filename) - 2, 4, -4):
        head = (keep + 1) // 2
        tail = keep // 2
        candidate = f"{filename[:head]}{_ELLIPSIS}{filename[-tail:]}"
        if count(_prefix(candidate, section_path)) <= cap:
            return candidate
    return f"{filename[:3]}{_ELLIPSIS}{filename[-3:]}"
