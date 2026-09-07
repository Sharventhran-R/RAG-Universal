"""Resolve a parser and extract, going through the cache.

``UnsupportedFormatError`` propagates -- the ingest pipeline turns it into
document status ``unsupported``.
"""

from __future__ import annotations

import io
from pathlib import Path

from app.extract import cache
from app.parsers.base import ParseInput, ParseResult, registry
from app.paths import DataPaths


def extract(
    *,
    document_id: str,
    filename: str,
    mimetype: str | None,
    file_sha256: str,
    blob_path: Path,
    paths: DataPaths,
) -> tuple[str, str, str, ParseResult]:
    """Returns ``(extractor_name, extractor_version, file_type, result)``. Cache
    hit skips the parser entirely."""
    parser = registry.resolve(mimetype, filename)  # -> UnsupportedFormatError

    cached = cache.load(paths, file_sha256, parser.version)
    if cached is not None:
        return parser.name, parser.version, parser.file_type, cached

    src = ParseInput(
        document_id=document_id,
        filename=filename,
        mimetype=mimetype or "",
        file_sha256=file_sha256,
        extractor_version=parser.version,
        open_stream=lambda: io.BytesIO(blob_path.read_bytes()),
        local_path=lambda: blob_path,
    )
    result = parser.parse(src)
    cache.store(paths, file_sha256, parser.name, parser.version, result)
    return parser.name, parser.version, parser.file_type, result
