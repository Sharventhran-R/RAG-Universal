"""Extraction cache: keyed by ``(file_sha256, extractor_version)`` + ``IR_VERSION``.

A chunking change must never re-run a parser (never re-OCR). The parsed
``ParseResult`` is written as JSON under
``{DATA_DIR}/extractions/{sha256}/{extractor_version}.json``; a cache entry
whose ``ir_version`` no longer matches is treated as a miss.
"""

from __future__ import annotations

import json

from app.ir import IR_VERSION, block_from_dict, block_to_dict
from app.parsers.base import ParseResult
from app.paths import DataPaths, atomic_write_text


def load(paths: DataPaths, sha256: str, extractor_version: str) -> ParseResult | None:
    path = paths.extraction_path(sha256, extractor_version)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("ir_version") != IR_VERSION:
        return None
    return ParseResult(
        blocks=[block_from_dict(b) for b in data["blocks"]],
        extraction_flags=list(data.get("extraction_flags") or []),
        status_hint=data.get("status_hint"),
    )


def store(
    paths: DataPaths, sha256: str, extractor_name: str, extractor_version: str, result: ParseResult
) -> None:
    payload = {
        "ir_version": IR_VERSION,
        "extractor": extractor_name,
        "extractor_version": extractor_version,
        "status_hint": result.status_hint,
        "extraction_flags": list(result.extraction_flags),
        "blocks": [block_to_dict(b) for b in result.blocks],
    }
    atomic_write_text(
        paths.extraction_path(sha256, extractor_version),
        json.dumps(payload, ensure_ascii=False),
        tmp_dir=paths.tmp_dir,
    )
