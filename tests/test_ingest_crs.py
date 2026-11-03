"""Ingesting LAS files whose header has no CRS into an array created with one."""

import json

import numpy as np
import pytest

from alsdb.core.alsdatabase import ALSDatabase


def _write_las(path, srs=None, n=500):
    import pdal

    rng = np.random.default_rng(0)
    pts = np.zeros(n, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")])
    pts["X"], pts["Y"] = 500_000 + rng.uniform(0, 50, n), 4_600_000 + rng.uniform(0, 50, n)
    pts["Z"] = 100 + rng.uniform(0, 20, n)
    pts["Classification"] = 2
    writer = {"type": "writers.las", "filename": str(path), "scale_x": 0.01, "scale_y": 0.01,
              "scale_z": 0.01, "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}  # fmt: skip
    if srs:
        writer["a_srs"] = srs
    pdal.Pipeline(json.dumps([writer]), arrays=[pts]).execute()
    return path


def test_tile_without_crs_takes_the_arrays_crs(tmp_path):
    las = _write_las(tmp_path / "SITE_2017_tile.las")
    db = ALSDatabase(storage_type="local", uri=str(tmp_path / "db"))
    db.create(crs="EPSG:25831")
    assert db.ingest_many([las]) == {las.name: 500}
    entry = db.load_manifest()[las.name]
    assert entry["status"] == "ok" and entry["crs"] == "EPSG:25831"
    assert db.stored_crs() == "EPSG:25831"
    db.ingest(las, overwrite=True)  # recreating the array keeps its CRS
    assert db.stored_crs() == "EPSG:25831"
    assert db.load_manifest()[las.name]["crs"] == "EPSG:25831"


def test_tile_with_a_different_crs_is_still_rejected(tmp_path):
    las = _write_las(tmp_path / "SITE_2017_tile.las", srs="EPSG:25830")
    db = ALSDatabase(storage_type="local", uri=str(tmp_path / "db"))
    db.create(crs="EPSG:25831")
    assert db.ingest_many([las]) == {las.name: 0}
    entry = db.load_manifest()[las.name]
    assert entry["status"] == "failed" and "CRS mismatch" in entry["error"]
    with pytest.raises(ValueError, match="CRS mismatch"):
        db.ingest(las)


def test_ingest_many_optimize_is_optional_and_data_identical(tmp_path):
    """Compaction only merges fragments: same points with or without it."""
    files = [_write_las(tmp_path / f"SITE_2017_t{i}.las") for i in range(3)]
    data = {}
    for optimize in (True, False):
        db = ALSDatabase(storage_type="local", uri=str(tmp_path / f"db_{optimize}"))
        db.create(crs="EPSG:25831")
        assert sum(db.ingest_many(files, optimize=optimize).values()) == 1500
        if not optimize:
            assert db.fragment_count() == 3  # one per file until optimized
            assert db.optimize() == 1
        with db.open("r") as arr:
            out = arr[:]
        order = np.lexsort((out["Z"], out["Y"], out["X"]))
        data[optimize] = {k: v[order] for k, v in out.items()}
    for name, values in data[True].items():
        np.testing.assert_array_equal(values, data[False][name], err_msg=name)
