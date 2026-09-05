"""On-disk layout under ``DATA_DIR`` and helpers for atomic writes.

```
DATA_DIR/
  app.db                                  sqlite (+ -wal, -shm)
  blobs/{session_id}/{sha256}             uploaded originals
  faiss/{session_id}.index                one vector index per session
  extractions/{sha256}/{extractor_version}.json    IR extraction cache
  tmp/                                    scratch for atomic replace
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class DataPaths:
    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "app.db"

    @property
    def blobs_dir(self) -> Path:
        return self.root / "blobs"

    @property
    def faiss_dir(self) -> Path:
        return self.root / "faiss"

    @property
    def extractions_dir(self) -> Path:
        return self.root / "extractions"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def blob_path(self, session_id: str, sha256: str) -> Path:
        return self.blobs_dir / session_id / sha256

    def faiss_path(self, session_id: str) -> Path:
        return self.faiss_dir / f"{session_id}.index"

    def extraction_path(self, sha256: str, extractor_version: str) -> Path:
        return self.extractions_dir / sha256 / f"{extractor_version}.json"

    def ensure(self) -> None:
        for d in (self.blobs_dir, self.faiss_dir, self.extractions_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_paths(settings: Settings | None = None) -> DataPaths:
    return DataPaths((settings or get_settings()).data_dir)


def atomic_write_bytes(path: Path, data: bytes, *, tmp_dir: Path | None = None) -> None:
    """Write ``data`` to ``path`` via a temp file + fsync + ``os.replace`` so a
    reader never sees a partial file and a crash never corrupts an existing one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = tmp_dir or path.parent
    scratch.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = _mkstemp(scratch, path.name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        _unlink_quiet(tmp_name)
        raise


def atomic_write_text(path: Path, text: str, *, tmp_dir: Path | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), tmp_dir=tmp_dir)


def atomic_replace_from(tmp_path: Path, final_path: Path) -> None:
    """Promote an already-written temp file (e.g. ``faiss.write_index`` output)
    to its final location atomically."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, final_path)


def _mkstemp(directory: Path, stem: str) -> tuple[int, str]:
    import tempfile

    return tempfile.mkstemp(prefix=f"{stem}.", suffix=".tmp", dir=str(directory))


def _unlink_quiet(name: str) -> None:
    try:
        os.unlink(name)
    except OSError:
        pass
