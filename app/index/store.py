"""Per-session FAISS store.

One ``IndexIDMap2`` over ``IndexFlatIP`` per session, on disk at
``{DATA_DIR}/faiss/{session_id}.index``. Vectors are L2-normalized on the way in
and on every query, so inner product == cosine.

* :class:`SessionIndex` -- one session's index + its mutation lock. All of
  ``add`` / ``remove`` / ``search`` / ``persist`` take the lock; ``search`` only
  briefly.
* :class:`IndexStore` -- an LRU cache of open ``SessionIndex`` objects
  (``INDEX_CACHE_SIZE``); eviction persists first.
* :func:`verify_index_dimensions` -- the startup guard: refuse to run if any
  existing index's ``.d`` disagrees with ``EMBED_DIM``.

faiss-cpu >= 1.7.4 (selector support on ``IndexIDMap2.search``).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from app.config import Settings, get_settings
from app.paths import DataPaths, get_paths


class IndexDimMismatch(RuntimeError):
    def __init__(self, path: Path, found: int, expected: int) -> None:
        super().__init__(f"{path}: index dim {found} != EMBED_DIM {expected}")
        self.path = path
        self.found = found
        self.expected = expected


def _as_2d_f32(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


def _normalized(vectors: np.ndarray) -> np.ndarray:
    arr = _as_2d_f32(vectors).copy()
    faiss.normalize_L2(arr)
    return arr


class SessionIndex:
    def __init__(self, session_id: str, path: Path, dim: int, tmp_dir: Path) -> None:
        self.session_id = session_id
        self.path = path
        self.dim = dim
        self._tmp_dir = tmp_dir
        self._lock = threading.Lock()
        self._dirty = False
        self._index = self._load_or_create()

    def _load_or_create(self) -> "faiss.Index":
        if self.path.exists():
            index = faiss.read_index(str(self.path))
            if index.d != self.dim:
                raise IndexDimMismatch(self.path, index.d, self.dim)
            return index
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    # -- mutation ---------------------------------------------------------

    def add(self, faiss_ids: Sequence[int], vectors: np.ndarray) -> None:
        vecs = _normalized(vectors)
        ids = np.asarray(list(faiss_ids), dtype=np.int64)
        if ids.shape[0] != vecs.shape[0]:
            raise ValueError(f"{ids.shape[0]} ids vs {vecs.shape[0]} vectors")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"vector dim {vecs.shape[1]} != {self.dim}")
        with self._lock:
            self._index.add_with_ids(vecs, ids)
            self._dirty = True

    def remove(self, faiss_ids: Sequence[int]) -> int:
        ids = np.asarray(sorted({int(i) for i in faiss_ids}), dtype=np.int64)
        if ids.size == 0:
            return 0
        selector = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))
        with self._lock:
            removed = self._index.remove_ids(selector)
            if removed:
                self._dirty = True
        return int(removed)

    # -- read -----------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, allowed_ids: Sequence[int], top_k: int
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        ids = np.asarray(sorted({int(i) for i in allowed_ids}), dtype=np.int64)
        if ids.size == 0:
            return []
        query = _normalized(query_vector)
        selector = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))  # keep `ids` alive
        params = faiss.SearchParameters()
        params.sel = selector
        with self._lock:
            scores, found = self._index.search(query, min(top_k, ids.size), params=params)
        hits: list[tuple[int, float]] = []
        for fid, score in zip(found[0].tolist(), scores[0].tolist()):
            if fid != -1:
                hits.append((int(fid), float(score)))
        return hits

    def id_set(self) -> set[int]:
        """Every faiss_id currently in the index (for the startup reconcile)."""
        with self._lock:
            if self._index.ntotal == 0:
                return set()
            return {int(i) for i in faiss.vector_to_array(self._index.id_map).tolist()}

    def __len__(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    # -- persistence ---------------------------------------------------

    def persist(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._tmp_dir / f"{self.session_id}.{os.getpid()}.index.tmp"
            faiss.write_index(self._index, str(tmp))
            fd = os.open(str(tmp), os.O_RDWR)
            try:
                os.fsync(fd)
            except OSError:  # fsync is best-effort; the atomic rename is the guarantee
                pass
            finally:
                os.close(fd)
            os.replace(tmp, self.path)
            self._dirty = False


class IndexStore:
    def __init__(self, paths: DataPaths, dim: int, cache_size: int) -> None:
        self._paths = paths
        self._dim = dim
        self._cache_size = max(1, cache_size)
        self._cache: "OrderedDict[str, SessionIndex]" = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "IndexStore":
        s = settings or get_settings()
        return cls(get_paths(s), s.embed_dim, s.index_cache_size)

    def get(self, session_id: str) -> SessionIndex | None:
        """The session's index, or ``None`` if it has never held a vector."""
        with self._lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                self._cache.move_to_end(session_id)
                return cached
            if not self._paths.faiss_path(session_id).exists():
                return None
            return self._open(session_id)

    def get_or_create(self, session_id: str) -> SessionIndex:
        with self._lock:
            cached = self._cache.get(session_id)
            if cached is not None:
                self._cache.move_to_end(session_id)
                return cached
            return self._open(session_id)

    def persist_all(self) -> None:
        with self._lock:
            for index in self._cache.values():
                index.persist()

    def _open(self, session_id: str) -> SessionIndex:
        index = SessionIndex(
            session_id, self._paths.faiss_path(session_id), self._dim, self._paths.tmp_dir
        )
        self._cache[session_id] = index
        while len(self._cache) > self._cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.persist()
        return index


def verify_index_dimensions(paths: DataPaths, embed_dim: int) -> None:
    """Startup guard. Raise :class:`IndexDimMismatch` if any existing index on
    disk was built for a different dimension than the configured model."""
    if not paths.faiss_dir.exists():
        return
    for path in sorted(paths.faiss_dir.glob("*.index")):
        index = faiss.read_index(str(path))
        if index.d != embed_dim:
            raise IndexDimMismatch(path, index.d, embed_dim)
