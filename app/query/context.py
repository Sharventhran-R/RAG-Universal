"""Turn retrieved chunks into the labelled context block and resolve citations."""

from __future__ import annotations

from app.db.models import Chunk, Document

SNIPPET_RADIUS = 160


def format_locator(chunk: Chunk) -> str:
    if chunk.page_start is not None:
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            return f"pp.{chunk.page_start}-{chunk.page_end}"
        return f"p.{chunk.page_start}"
    if chunk.sheet:
        return f"sheet {chunk.sheet}"
    if chunk.slide is not None:
        return f"slide {chunk.slide}"
    return "—"


def chunk_header(chunk: Chunk, doc: Document) -> str:
    return f"[{chunk.id} | {doc.filename} | {format_locator(chunk)}]"


def build_context(pairs: list[tuple[Chunk, Document]]) -> str:
    return "\n\n".join(f"{chunk_header(c, d)}\n{c.text}" for c, d in pairs)


def snippet(text: str, radius: int = SNIPPET_RADIUS) -> str:
    text = text.strip()
    if len(text) <= 2 * radius:
        return text
    return text[: 2 * radius].rstrip() + "…"
