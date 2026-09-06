from __future__ import annotations

import pytest

from app.ir import Block
from app.parsers.base import (
    ParseInput,
    ParserRegistry,
    UnsupportedFormatError,
)


class _FakeParser:
    def __init__(self, name: str, file_type: str, mimetypes: tuple[str, ...]):
        self.name = name
        self.version = "1"
        self.file_type = file_type
        self.mimetypes = mimetypes

    def parse(self, src: ParseInput) -> list[Block]:
        return []


@pytest.fixture
def reg() -> ParserRegistry:
    r = ParserRegistry()
    r.register(_FakeParser("pdf", "pdf", ("application/pdf",)))
    r.register(
        _FakeParser(
            "sheet",
            "xlsx",
            ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"),
        )
    )
    return r


def test_resolve_dispatches_by_mimetype(reg: ParserRegistry):
    assert reg.resolve("application/pdf", "whatever.bin").file_type == "pdf"
    assert reg.resolve("text/csv", "data").name == "sheet"


def test_resolve_falls_back_to_extension_when_mimetype_is_generic(reg: ParserRegistry):
    assert reg.resolve("application/octet-stream", "report.pdf").file_type == "pdf"
    assert reg.resolve(None, "book.pdf").file_type == "pdf"


def test_resolve_raises_typed_error_for_unknown_format(reg: ParserRegistry):
    with pytest.raises(UnsupportedFormatError) as ei:
        reg.resolve("application/zip", "archive.zip")
    assert ei.value.mimetype == "application/zip"
    assert ei.value.filename == "archive.zip"


def test_register_rejects_duplicate_mimetype(reg: ParserRegistry):
    with pytest.raises(ValueError):
        reg.register(_FakeParser("pdf2", "pdf", ("application/pdf",)))


def test_supported_mimetypes_is_sorted(reg: ParserRegistry):
    assert list(reg.supported_mimetypes()) == sorted(reg.supported_mimetypes())


@pytest.mark.parametrize(
    "mimetype, filename, expected_file_type",
    [
        ("application/pdf", "x.pdf", "pdf"),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "x.docx",
            "docx",
        ),
        (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "x.pptx",
            "pptx",
        ),
        (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "x.xlsx",
            "xlsx",
        ),
        ("text/csv", "x.csv", "csv"),
        ("text/plain", "x.txt", "txt"),
        ("text/markdown", "x.md", "md"),
        ("text/html", "x.html", "html"),
    ],
)
def test_real_registry_dispatches_every_phase1_format(mimetype, filename, expected_file_type):
    import app.parsers  # noqa: F401  triggers registration
    from app.parsers.base import registry

    assert registry.resolve(mimetype, filename).file_type == expected_file_type
