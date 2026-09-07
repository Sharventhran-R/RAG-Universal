from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from app.index.store import (
    IndexDimMismatch,
    IndexStore,
    SessionIndex,
    verify_index_dimensions,
)
from app.paths import DataPaths


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    p = DataPaths(tmp_path)
    p.ensure()
    return p


def _index(paths: DataPaths, session_id: str = "s1", dim: int = 4) -> SessionIndex:
    return SessionIndex(session_id, paths.faiss_path(session_id), dim, paths.tmp_dir)


def test_new_index_is_empty_and_store_get_returns_none(paths):
    si = _index(paths)
    assert len(si) == 0
    assert si.id_set() == set()
    assert IndexStore(paths, 4, 8).get("s1") is None  # no file yet


def test_add_search_ranks_by_cosine_and_normalizes_inputs(paths):
    si = _index(paths)
    si.add([100, 200, 300, 400], np.eye(4) * 7.0)  # deliberately un-normalized
    hits = si.search(np.array([0.0, 5.0, 0.0, 0.1]), [100, 200, 300, 400], top_k=2)
    assert hits[0][0] == 200
    assert hits[0][1] <= 1.0 + 1e-5  # cosine, not raw dot of un-normalized vectors


def test_search_is_a_real_prefilter_not_post_filter(paths):
    si = _index(paths)
    si.add([1, 2, 3, 4], np.eye(4))
    # the best match (id 2) is excluded by the allow-list -> never scored
    hits = si.search(np.array([0.0, 1.0, 0.0, 0.0]), [1, 3, 4], top_k=5)
    assert {h[0] for h in hits} <= {1, 3, 4}
    assert 2 not in {h[0] for h in hits}


def test_empty_allowed_or_nonpositive_k_returns_nothing(paths):
    si = _index(paths)
    si.add([1], np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert si.search(np.array([1.0, 0, 0, 0]), [], 5) == []
    assert si.search(np.array([1.0, 0, 0, 0]), [1], 0) == []


def test_persist_is_atomic_and_survives_reload(paths):
    si = _index(paths)
    si.add([10, 20, 30], np.eye(3, 4))
    si.persist()
    assert si.path.exists()
    assert not any(paths.tmp_dir.iterdir())  # temp file was renamed away

    reloaded = _index(paths)
    assert reloaded.id_set() == {10, 20, 30}
    assert reloaded.remove([20]) == 1
    assert reloaded.id_set() == {10, 30}


def test_remove_ignores_unknown_ids(paths):
    si = _index(paths)
    si.add([1, 2], np.eye(2, 4))
    assert si.remove([999]) == 0
    assert si.remove([]) == 0


def test_dim_mismatch_is_raised_on_load_and_by_the_startup_guard(paths):
    si = _index(paths, dim=4)
    si.add([1], np.array([[1.0, 0, 0, 0]]))
    si.persist()

    with pytest.raises(IndexDimMismatch) as ei:
        _index(paths, dim=8)
    assert ei.value.found == 4 and ei.value.expected == 8
    assert "s1.index" in str(ei.value)

    with pytest.raises(IndexDimMismatch):
        verify_index_dimensions(paths, 8)
    verify_index_dimensions(paths, 4)  # correct dim: no raise


def test_verify_is_a_noop_when_no_indexes_exist(tmp_path: Path):
    verify_index_dimensions(DataPaths(tmp_path), 384)


def test_store_lru_evicts_oldest_and_persists_it(paths):
    store = IndexStore(paths, 4, cache_size=2)
    a = store.get_or_create("a")
    a.add([1], np.array([[1.0, 0, 0, 0]]))
    store.get_or_create("b")
    store.get_or_create("c")  # over capacity -> "a" evicted

    assert paths.faiss_path("a").exists()  # eviction persisted it
    assert store.get("a") is not None      # and it reloads from disk
    assert store.get("never-seen") is None


def test_store_get_or_create_reuses_the_cached_instance(paths):
    store = IndexStore(paths, 4, cache_size=4)
    assert store.get_or_create("x") is store.get_or_create("x")
