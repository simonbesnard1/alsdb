# SPDX-License-Identifier: EUPL-1.2
"""Trimmed schemas, integer Z, tile-by-index and coordinated (multi-job) product runs."""

from pathlib import Path

import numpy as np
import pytest
import tiledb

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.methods._grid import GridSpec
from alsdb.products._tiling import query_to_array
from alsdb.products.chm import compute_all, compute_dtm
from alsdb.storage.zarr_store import ALSZarrStore
from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig

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
    db, x, y, attrs = _array(tmp_path, "trim", attributes=KEEP)
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
    db, x, y, attrs = _array(tmp_path, "zint", attributes=KEEP, z_scale=0.001)
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
    common = dict(resolution=1.0, bbox=BBOX, year=YEAR, tile_size=250.0)
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
    compute_dtm(provider, ALSZarrStore(str(tmp_path / "shared.zarr")), **common, coordination="finish")
    final = ALSZarrStore(str(tmp_path / "shared.zarr"))
    assert final._root["1m"]["dtm"].attrs["processing_runs"][str(YEAR)]["status"] == "complete"
    assert YEAR in final._root["1m"]["dtm"].attrs["data_years"]
    a = single.to_dataset(1.0)["dtm"].values
    b = final.to_dataset(1.0)["dtm"].values
    assert np.isfinite(a).any() and np.array_equal(a, b, equal_nan=True)
    assert final.completed_tiles("dtm", 1.0, YEAR) == set(range(16))
    # a rerun of a finished job resumes: nothing is recomputed, nothing is cleared
    compute_dtm(provider, ALSZarrStore(str(tmp_path / "shared.zarr")), **common, coordination="register")
    assert np.array_equal(ALSZarrStore(str(tmp_path / "shared.zarr")).to_dataset(1.0)["dtm"].values, a,
                          equal_nan=True)  # fmt: skip


def test_worker_settings_must_match_the_registered_run(provider, tmp_path):
    common = dict(resolution=1.0, bbox=BBOX, year=YEAR, tile_size=250.0)
    store = ALSZarrStore(str(tmp_path / "s.zarr"))
    compute_dtm(provider, store, **common, coordination="register")
    with pytest.raises(RuntimeError, match="no registered run"):
        compute_dtm(provider, store, **common, coordination="worker", tile_indices=[0], dtm_method="idw")


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
