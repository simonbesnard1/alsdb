"""Throughput-oriented store/tiling changes must not change results.

Covers tile-level resume, targeted slice clearing, cached write_tile and the
unordered bounded pool.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from alsdb.processing._tiling import bounded_map_unordered
from alsdb.processing.chm import compute_chm
from alsdb.storage.zarr_store import ALSZarrStore

BBOX = (308000.0, 4688000.0, 309000.0, 4689000.0)
OPTS = {"resolution": 10.0, "year": 2021, "bbox": BBOX, "tile_size": 250.0}


def _count_worker_calls(monkeypatch, fail_after=None):
    """Wrap _products._worker; optionally raise once *fail_after* tiles succeeded."""
    from alsdb.processing import _products

    # Always wrap the real worker, not a wrapper left by an earlier call.
    original = getattr(_products._worker, "real_worker", _products._worker)
    calls = []

    def wrapped(*args, **kwargs):
        if fail_after is not None and len(calls) >= fail_after:
            raise RuntimeError("injected crash")
        calls.append(args[4])  # tile_index
        return original(*args, **kwargs)

    wrapped.real_worker = original
    monkeypatch.setattr(_products, "_worker", wrapped)
    return calls


def test_interrupted_run_resumes_only_missing_tiles(provider, store, tmp_path, monkeypatch):
    reference = ALSZarrStore(str(tmp_path / "reference.zarr"))
    compute_chm(provider, reference, **OPTS)

    first = _count_worker_calls(monkeypatch, fail_after=5)
    with pytest.raises(RuntimeError, match="injected"):
        compute_chm(provider, store, **OPTS)
    assert store.completed_tiles("chm", 10.0, 2021) == set(first)

    second = _count_worker_calls(monkeypatch)
    compute_chm(provider, store, **OPTS)
    n_tiles = 16  # 1000 m bbox / 250 m tiles
    assert not set(first) & set(second)
    assert len(first) + len(second) == n_tiles
    assert store._root["10m"]["chm"].attrs["processing_runs"]["2021"]["status"] == "complete"
    np.testing.assert_array_equal(store._root["10m"]["chm"][0], reference._root["10m"]["chm"][0])


def test_overwrite_after_interruption_recomputes_everything(provider, store, monkeypatch):
    _count_worker_calls(monkeypatch, fail_after=5)
    with pytest.raises(RuntimeError):
        compute_chm(provider, store, **OPTS)
    calls = _count_worker_calls(monkeypatch)
    compute_chm(provider, store, overwrite=True, **OPTS)
    assert len(calls) == 16


def test_completed_run_is_reused_without_tiles(provider, store, monkeypatch):
    compute_chm(provider, store, **OPTS)
    calls = _count_worker_calls(monkeypatch)
    compute_chm(provider, store, **OPTS)
    assert calls == []


def test_clear_slice_deletes_only_existing_chunks(tmp_path):
    from zarr.storage import LocalStore

    store = ALSZarrStore(str(tmp_path / "s.zarr"))
    store.ensure_group("chm", 1.0, (0, 0, 2000, 2000), "EPSG:25832", 500.0)  # 16 chunks
    store.write_tile("chm", 1.0, 2022, np.ones((500, 500), np.float32), (0, 1500, 500, 2000))
    store.write_tile("chm", 1.0, 2023, np.ones((500, 500), np.float32), (0, 1500, 500, 2000))
    deleted = []
    original = LocalStore.delete

    async def delete(self, key):
        deleted.append(key)
        return await original(self, key)

    LocalStore.delete = delete
    try:
        store.begin_run("chm", 1.0, 2022, {"x": 1}, ("chm",))
    finally:
        LocalStore.delete = original
    assert deleted == ["1m/chm/c/0/0/0"]  # the one chunk of 2022, nothing else
    assert np.isnan(store._root["1m"]["chm"][0]).all()
    assert np.all(store._root["1m"]["chm"][1, :500, :500] == 1)  # 2023 untouched
    assert not store.has_data("chm", 1.0, 2022)
    assert store.has_data("chm", 1.0, 2023)


def test_cached_writes_across_years_and_after_clear(tmp_path):
    store = ALSZarrStore(str(tmp_path / "s.zarr"))
    store.ensure_group("dtm", 1.0, (0, 0, 1000, 1000), "EPSG:25832", 500.0)
    tile = (500, 500, 1000, 1000)
    for year, value in ((2020, 1), (2021, 2), (2020, 3)):  # revisit a cached year
        store.write_tile("dtm", 1.0, year, np.full((500, 500), value, np.float32), tile)
    store.ensure_group("chm", 1.0, (0, 0, 1000, 1000), "EPSG:25832", 500.0)  # added late
    store.write_tile("chm", 1.0, 2021, np.full((500, 500), 7, np.float32), tile)
    reopened = ALSZarrStore(str(tmp_path / "s.zarr"))._root["1m"]
    assert list(reopened["time"][:]) == [2020, 2021]
    assert reopened["dtm"][0, 0, 500] == 3 and reopened["dtm"][1, 0, 500] == 2
    assert reopened["chm"][1, 0, 500] == 7 and np.isnan(reopened["chm"][0, 0, 500])
    assert reopened["dtm"].attrs["data_years"] == [2020, 2021]

    store.begin_run("dtm", 1.0, 2020, {"x": 1}, ("dtm",))
    store.write_tile("dtm", 1.0, 2020, np.full((500, 500), 5, np.float32), tile)
    assert store._root["1m"]["dtm"][0, 0, 500] == 5
    assert store.has_data("dtm", 1.0, 2020)


def test_concurrent_unaligned_writes_sharing_chunks(tmp_path):
    """Windows smaller than a chunk must not lose each other's pixels."""
    store = ALSZarrStore(str(tmp_path / "s.zarr"))
    store.ensure_group("chm", 1.0, (0, 0, 100, 100), "EPSG:25832", 100.0)  # one 100x100 chunk
    tiles = [(x, y, x + 10, y + 10) for x in range(0, 100, 10) for y in range(0, 100, 10)]

    def write(i):
        store.write_tile("chm", 1.0, 2022, np.full((10, 10), i, np.float32), tiles[i])

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(write, range(len(tiles))))
    out = store._root["1m"]["chm"][0]
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        assert np.all(out[100 - y1 : 100 - y0, x0:x1] == i)


def test_unordered_pool_processes_all_and_stays_bounded():
    in_flight, peak, lock = 0, 0, threading.Lock()

    def work(i):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05 if i % 7 == 0 else 0.002)
        with lock:
            in_flight -= 1
        return i

    assert sorted(bounded_map_unordered(work, range(60), 4)) == list(range(60))
    assert peak <= 4  # never more running than workers
    with pytest.raises(ZeroDivisionError):
        list(bounded_map_unordered(lambda i: 1 / (i - 30), range(60), 4))


def test_unordered_pool_does_not_idle_behind_slow_item():
    def work(i):
        time.sleep(0.3 if i == 0 else 0.01)
        return i

    t = time.time()
    list(bounded_map_unordered(work, range(80), 4))
    # Ordered consumption would stall the pool behind item 0; unordered
    # finishes in about max(0.3, 80 * 0.01 / 4) seconds.
    assert time.time() - t < 0.6
