# SPDX-License-Identifier: EUPL-1.2
"""Trimmed schemas, integer Z, tile-by-index and coordinated (multi-job) product runs."""

from pathlib import Path

import numpy as np
import pytest
import tiledb

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.methods._grid import GridSpec
from alsdb.products._tiling import _retry_transient as _ORIG_RETRY
from alsdb.products._tiling import query_to_array
from alsdb.products.chm import compute_all, compute_dtm
from alsdb.storage.zarr_store import ALSZarrStore
from alsdb.utils.schema import TileDBSchemaConfig
from tests.conftest import BBOX, CRS, SCHEMA_CFG, YEAR, _make_synthetic_points

KEEP = ("Z", "Intensity", "ReturnNumber", "NumberOfReturns", "Classification", "ScanAngleRank",
        "Withheld", "Overlap")  # fmt: skip


def _array(tmp_path, name, **schema):
    cfg = TileDBSchemaConfig(**{**SCHEMA_CFG.__dict__, **schema})
    db = ALSDatabase(storage_type="local", uri=str(tmp_path / name), schema_cfg=cfg)
    x, y, attrs = _make_synthetic_points(np.random.default_rng(0))
    attrs["Z"] = np.round(attrs["Z"] / 0.01) * 0.01  # LAS coordinates: multiples of the scale
    attrs["GpsTime"] = np.arange(len(x), dtype=float)
    db.write(x, y, YEAR, attrs, crs=CRS)
    return db, x, y, attrs


def test_trimmed_schema_stores_only_the_kept_fields_and_reads_the_rest_as_zeros(tmp_path):
    db, x, _, _ = _array(tmp_path, "trim", attributes=KEEP)
    with tiledb.open(db.array_uri) as arr:
        assert [arr.schema.attr(i).name for i in range(arr.schema.nattr)] == list(KEEP)
    pts = query_to_array(db, BBOX, YEAR, attributes=("Z", "Classification", "GpsTime"))
    assert len(pts) == len(x)
    assert (pts["GpsTime"] == 0).all()  # not stored
    assert np.isin(pts["Classification"], (2, 3)).all()
    with pytest.raises(ValueError, match="Unknown LAS attributes"):
        TileDBSchemaConfig(attributes=("Z", "Nonsense"))  # caught when the schema is built
        from alsdb.utils.schema import create_schema

        create_schema(TileDBSchemaConfig(attributes=("Z", "Nonsense")))


def test_integer_z_reads_back_bit_identical(tmp_path):
    db, *_ = _array(tmp_path, "zint", attributes=KEEP, z_scale=0.001)
    full, *_ = _array(tmp_path, "full")
    a = np.sort(query_to_array(db, BBOX, YEAR, attributes=("Z",))["Z"])
    b = np.sort(query_to_array(full, BBOX, YEAR, attributes=("Z",))["Z"])
    assert np.array_equal(a, b)


def test_products_are_identical_on_a_trimmed_array(tmp_path):
    full, *_ = _array(tmp_path, "full")
    trim, *_ = _array(tmp_path, "trim", attributes=KEEP, z_scale=0.001)
    out = {}
    for name, db in (("full", full), ("trim", trim)):
        store = ALSZarrStore(str(tmp_path / f"{name}.zarr"))
        compute_all(db, store, resolution=1.0, bbox=BBOX, year=YEAR, tile_size=250.0, method="spikefree"
                    if _spikefree() else "pitfree")  # fmt: skip
        ds = store.to_dataset(1.0)
        out[name] = {v: ds[v].values for v in ("dtm", "chm")}
    for v in ("dtm", "chm"):
        assert np.array_equal(out["full"][v], out["trim"][v], equal_nan=True)


def _spikefree():
    try:
        from alsdb.methods import _spikefree_native  # noqa: F401
    except ImportError:
        return False
    return True


def test_tile_by_index_matches_the_tile_listing():
    grid = GridSpec.from_bbox((0.0, 0.0, 1234.0, 987.0), 1.0)
    listed = list(grid.tiles(250.0, 50.0))
    assert grid.tile_layout(250.0) == (4, 5) and len(listed) == 20
    for i, t in enumerate(listed):
        assert grid.tile(i, 250.0, 50.0) == t
    with pytest.raises(IndexError):
        grid.tile(20, 250.0)


def test_coordinated_jobs_fill_one_store_like_a_single_run(provider, tmp_path):
    common = {"resolution": 1.0, "bbox": BBOX, "year": YEAR, "tile_size": 250.0}
    single = ALSZarrStore(str(tmp_path / "single.zarr"))
    compute_dtm(provider, single, **common)
    shared = ALSZarrStore(str(tmp_path / "shared.zarr"))
    with pytest.raises(RuntimeError, match="no registered run"):
        compute_dtm(provider, shared, **common, coordination="worker", tile_indices=[0])
    compute_dtm(provider, shared, **common, coordination="register")
    # two "jobs" on separate store handles, as on two nodes; one tile done twice
    for indices in ([0, 1, 2, 3, 4, 5, 6, 7], [7, 8, 9, 10, 11, 12, 13, 14, 15]):
        job = ALSZarrStore(str(tmp_path / "shared.zarr"))
        compute_dtm(provider, job, **common, coordination="worker", tile_indices=indices)
    record = shared._root["1m"]["dtm"].attrs["processing_runs"][str(YEAR)]
    assert record["status"] == "running"  # workers never finish the run
    compute_dtm(
        provider, ALSZarrStore(str(tmp_path / "shared.zarr")), **common, coordination="finish"
    )
    final = ALSZarrStore(str(tmp_path / "shared.zarr"))
    assert final._root["1m"]["dtm"].attrs["processing_runs"][str(YEAR)]["status"] == "complete"
    assert YEAR in final._root["1m"]["dtm"].attrs["data_years"]
    a = single.to_dataset(1.0)["dtm"].values
    b = final.to_dataset(1.0)["dtm"].values
    assert np.isfinite(a).any() and np.array_equal(a, b, equal_nan=True)
    assert final.completed_tiles("dtm", 1.0, YEAR) == set(range(16))
    # a rerun of a finished job resumes: nothing is recomputed, nothing is cleared
    compute_dtm(
        provider, ALSZarrStore(str(tmp_path / "shared.zarr")), **common, coordination="register"
    )
    assert np.array_equal(ALSZarrStore(str(tmp_path / "shared.zarr")).to_dataset(1.0)["dtm"].values, a,
                          equal_nan=True)  # fmt: skip


def test_worker_settings_must_match_the_registered_run(provider, tmp_path):
    common = {"resolution": 1.0, "bbox": BBOX, "year": YEAR, "tile_size": 250.0}
    store = ALSZarrStore(str(tmp_path / "s.zarr"))
    compute_dtm(provider, store, **common, coordination="register")
    with pytest.raises(RuntimeError, match="no registered run"):
        compute_dtm(
            provider, store, **common, coordination="worker", tile_indices=[0], dtm_method="idw"
        )


def test_array_exists_lists_the_schema_instead_of_opening_the_array(tmp_path, monkeypatch):
    """On S3 every open reads all unconsolidated fragment metadata (HTTP 429 at scale)."""
    db, *_ = _array(tmp_path, "exists")
    fresh = ALSDatabase(storage_type="local", uri=db.array_uri)
    opened = []
    real = tiledb.open
    monkeypatch.setattr(tiledb, "open", lambda *a, **k: opened.append(a) or real(*a, **k))
    assert fresh.array_exists() and fresh.array_exists()
    assert opened == []
    assert not ALSDatabase(storage_type="local", uri=str(tmp_path / "missing")).array_exists()


def test_ingest_without_consolidation_leaves_the_metadata_unconsolidated(tmp_path):
    import glob

    files = sorted(glob.glob("data/example_als/PNOA_2021_*.laz"))
    if not files:
        pytest.skip("sample PNOA tile not available")
    from alsdb.core.alstile import ALSTile

    db = ALSDatabase(storage_type="local", uri=str(tmp_path / "noc"), schema_cfg=SCHEMA_CFG)
    db.create(crs=ALSTile(files[0]).target_crs)  # the sample reports EPSG:3042
    db.ingest_many(files, consolidate=False)
    commits = tmp_path / "noc" / "__commits"
    assert not list(commits.glob("*.con")), "no consolidated commits written"
    db.ingest_many(files, overwrite=False)  # already ingested: skipped, nothing written
    assert db.load_manifest()[Path(files[0]).name]["status"] == "ok"


def test_registering_again_replaces_an_unfinished_run_but_not_a_completed_one(tmp_path):
    db, *_ = _array(tmp_path, "rereg")
    common = {"resolution": 1.0, "bbox": BBOX, "year": YEAR, "tile_size": 250.0}
    store = ALSZarrStore(str(tmp_path / "s.zarr"))
    compute_dtm(db, store, **common, coordination="register")
    compute_dtm(db, store, **common, coordination="worker", tile_indices=[0, 1])
    assert store.completed_tiles("dtm", 1.0, YEAR) == {0, 1}
    x, y, attrs = _make_synthetic_points(np.random.default_rng(1))  # more input: new fragments
    db.write(x[:10], y[:10], YEAR, {k: v[:10] for k, v in attrs.items()})
    with pytest.raises(RuntimeError, match="no registered run"):  # workers notice the change
        compute_dtm(db, store, **common, coordination="worker", tile_indices=[2])
    compute_dtm(
        db, store, **common, coordination="register"
    )  # unfinished: replaced, stale tiles cleared
    assert ALSZarrStore(str(tmp_path / "s.zarr")).completed_tiles("dtm", 1.0, YEAR) == set()
    compute_dtm(db, store, **common, coordination="worker", tile_indices=list(range(16)))
    compute_dtm(db, store, **common, coordination="finish")
    db.write(x[10:20], y[10:20], YEAR, {k: v[10:20] for k, v in attrs.items()})
    with pytest.raises(ValueError, match="different processing configuration"):
        compute_dtm(
            db, store, **common, coordination="register"
        )  # completed: kept unless overwrite


def test_tile_queries_retry_s3_throttling_but_not_other_errors(tmp_path, monkeypatch):
    import alsdb.products._tiling as tiling

    db, x, *_ = _array(tmp_path, "throttle")
    real_open = db.open
    failures = {"n": 2}

    def flaky(mode="r"):
        if failures["n"]:
            failures["n"] -= 1
            raise tiledb.TileDBError("S3: Cannot retrieve S3 object size [HTTP Response Code: 429]")
        return real_open(mode)

    monkeypatch.setattr(db, "open", flaky)
    monkeypatch.setattr(
        tiling,
        "_retry_transient",
        lambda fn, **k: tiling.__dict__["_retry_transient_orig"](fn, first_wait=0.01, **k),
    )
    monkeypatch.setitem(tiling.__dict__, "_retry_transient_orig", _ORIG_RETRY)
    pts = query_to_array(db, BBOX, YEAR, attributes=("Z",))
    assert len(pts) == len(x) and failures["n"] == 0

    monkeypatch.setattr(
        db, "open", lambda mode="r": (_ for _ in ()).throw(tiledb.TileDBError("schema mismatch"))
    )
    with pytest.raises(tiledb.TileDBError, match="schema mismatch"):
        query_to_array(db, BBOX, YEAR, attributes=("Z",))


def test_input_revision_lists_fragment_names_instead_of_reading_their_metadata(
    tmp_path, monkeypatch
):
    """array_fragments reads every fragment's metadata: 546 s for 63,118 fragments on S3."""
    from alsdb.products._surfaces import input_revision

    db, x, y, attrs = _array(tmp_path, "revision")
    monkeypatch.setattr(
        tiledb, "array_fragments", lambda *a, **k: pytest.fail("reads fragment metadata")
    )
    before = input_revision(db)
    db.consolidate(mode="fragment_meta")
    db.consolidate(mode="commits")
    assert input_revision(db) == before  # metadata consolidation: same input
    db.write(x[:10], y[:10], YEAR, {k: v[:10] for k, v in attrs.items()})
    assert input_revision(db) != before  # a new fragment


def test_a_worker_that_never_exits_does_not_block_the_pool_shutdown():
    """A job whose tiles were all written waited 2.5 h for one worker to exit."""
    import multiprocessing
    import multiprocessing.connection
    import time
    from concurrent.futures import ProcessPoolExecutor

    from alsdb.products._tiling import _shutdown_pool

    pool = ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"))
    pool.submit(time.sleep, 3600)  # a worker stuck for an hour
    assert pool.submit(abs, -1).result() == 1  # the other one answers
    sentinels = [p.sentinel for p in pool._processes.values()]
    t0 = time.monotonic()
    _shutdown_pool(pool, grace=1.0)
    assert time.monotonic() - t0 < 30
    exited = multiprocessing.connection.wait(sentinels, timeout=10)  # the pool's thread reaps them
    assert len(exited) == len(sentinels) == 2
