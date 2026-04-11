# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Unit tests for ALSZarrStore."""

import numpy as np
import pytest

from alsdb.storage.zarr_store import ALSZarrStore

# Small bbox for fast grid arithmetic
BBOX = (0.0, 0.0, 1000.0, 1000.0)
RES = 10.0          # → 100 × 100 grid
YEAR = 2021
CRS = "EPSG:25830"


@pytest.fixture()
def store(tmp_path):
    return ALSZarrStore(str(tmp_path / "store.zarr"))


@pytest.fixture()
def initialised_store(store):
    store.ensure_group("chm", RES, BBOX, CRS)
    return store


# ---------------------------------------------------------------------------
# ensure_group
# ---------------------------------------------------------------------------

def test_ensure_group_creates_resolution_group(store):
    store.ensure_group("chm", RES, BBOX, CRS)
    assert "10m" in store._root


def test_ensure_group_creates_coordinate_arrays(initialised_store):
    grp = initialised_store._root["10m"]
    assert "x" in grp
    assert "y" in grp
    assert "time" in grp


def test_ensure_group_creates_variable(initialised_store):
    grp = initialised_store._root["10m"]
    assert "chm" in grp


def test_ensure_group_idempotent(store):
    store.ensure_group("chm", RES, BBOX, CRS)
    store.ensure_group("chm", RES, BBOX, CRS)  # second call must not raise
    assert "chm" in store._root["10m"]


def test_ensure_group_adds_second_variable(initialised_store):
    initialised_store.ensure_group("dtm", RES, BBOX, CRS)
    grp = initialised_store._root["10m"]
    assert "chm" in grp
    assert "dtm" in grp


def test_ensure_group_fractional_resolution(store):
    store.ensure_group("chm", 0.5, BBOX, CRS)
    assert "0.5m" in store._root


# ---------------------------------------------------------------------------
# has_data
# ---------------------------------------------------------------------------

def test_has_data_false_before_write(initialised_store):
    assert not initialised_store.has_data("chm", RES, YEAR)


def test_has_data_false_missing_variable(initialised_store):
    assert not initialised_store.has_data("dtm", RES, YEAR)


def test_has_data_false_missing_resolution(store):
    assert not store.has_data("chm", RES, YEAR)


# ---------------------------------------------------------------------------
# write_tile
# ---------------------------------------------------------------------------

def _tile(ny=100, nx=100, fill=5.0):
    return np.full((ny, nx), fill, dtype=np.float32)


def test_write_tile_sets_has_data(initialised_store):
    initialised_store.write_tile("chm", RES, YEAR, _tile(), BBOX)
    assert initialised_store.has_data("chm", RES, YEAR)


def test_write_tile_does_not_affect_other_year(initialised_store):
    initialised_store.write_tile("chm", RES, YEAR, _tile(), BBOX)
    assert not initialised_store.has_data("chm", RES, 2022)


def test_write_tile_values_round_trip(initialised_store):
    data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    initialised_store.write_tile("chm", RES, YEAR, data, BBOX)
    stored = initialised_store._root["10m"]["chm"][0]
    np.testing.assert_allclose(stored, data)


def test_write_tile_sub_tile(initialised_store):
    """Write a tile covering only the first quadrant."""
    sub_bbox = (0.0, 500.0, 500.0, 1000.0)
    sub_data = np.ones((50, 50), dtype=np.float32) * 3.0
    initialised_store.write_tile("chm", RES, YEAR, sub_data, sub_bbox)
    assert initialised_store.has_data("chm", RES, YEAR)


def test_write_tile_multiple_years(initialised_store):
    initialised_store.write_tile("chm", RES, 2020, _tile(fill=1.0), BBOX)
    initialised_store.write_tile("chm", RES, 2021, _tile(fill=2.0), BBOX)
    assert initialised_store.has_data("chm", RES, 2020)
    assert initialised_store.has_data("chm", RES, 2021)
    # time axis should have two entries
    assert initialised_store._root["10m"]["time"].shape[0] == 2


def test_write_tile_overwrite_year(initialised_store):
    initialised_store.write_tile("chm", RES, YEAR, _tile(fill=1.0), BBOX)
    initialised_store.write_tile("chm", RES, YEAR, _tile(fill=9.0), BBOX)
    # Only one time slice for the same year
    assert initialised_store._root["10m"]["time"].shape[0] == 1
    # Value should be updated
    assert float(initialised_store._root["10m"]["chm"][0, 0, 0]) == pytest.approx(9.0)


def test_write_tile_empty_slice_is_skipped(initialised_store):
    """write_tile with row0 >= row1 must not raise."""
    bad_bbox = (0.0, 0.0, 0.0, 0.0)   # zero-area tile
    initialised_store.write_tile("chm", RES, YEAR, np.zeros((1, 1)), bad_bbox)
    # No data written; has_data should still be False
    assert not initialised_store.has_data("chm", RES, YEAR)


# ---------------------------------------------------------------------------
# resolutions and variables
# ---------------------------------------------------------------------------

def test_resolutions_empty(store):
    assert store.resolutions == []


def test_resolutions_after_ensure(store):
    store.ensure_group("chm", 1.0, BBOX, CRS)
    store.ensure_group("gap", 10.0, BBOX, CRS)
    assert sorted(store.resolutions) == [1.0, 10.0]


def test_variables_returns_data_vars_only(store):
    store.ensure_group("chm", RES, BBOX, CRS)
    store.ensure_group("dtm", RES, BBOX, CRS)
    vs = store.variables(RES)
    assert "chm" in vs
    assert "dtm" in vs
    # coordinate arrays must not appear
    assert "x" not in vs
    assert "y" not in vs
    assert "time" not in vs


# ---------------------------------------------------------------------------
# create factory
# ---------------------------------------------------------------------------

def test_create_factory(tmp_path):
    path = tmp_path / "created.zarr"
    s = ALSZarrStore.create(
        path,
        bbox=BBOX,
        crs_wkt=CRS,
        variables={"10m": ["chm", "dtm"], "1m": ["dsm"]},
        tile_size=500.0,
    )
    assert 10.0 in s.resolutions
    assert 1.0 in s.resolutions
    assert "chm" in s.variables(10.0)
    assert "dsm" in s.variables(1.0)


# ---------------------------------------------------------------------------
# to_dataset
# ---------------------------------------------------------------------------

def test_to_dataset_returns_xarray_dataset(initialised_store):
    xr = pytest.importorskip("xarray")
    initialised_store.write_tile("chm", RES, YEAR, _tile(), BBOX)
    ds = initialised_store.to_dataset(RES)
    assert isinstance(ds, xr.Dataset)
    assert "chm" in ds


def test_to_dataset_has_correct_coords(initialised_store):
    xr = pytest.importorskip("xarray")
    initialised_store.write_tile("chm", RES, YEAR, _tile(), BBOX)
    ds = initialised_store.to_dataset(RES)
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords


def test_to_dataset_coord_lengths(initialised_store):
    xr = pytest.importorskip("xarray")
    initialised_store.write_tile("chm", RES, YEAR, _tile(), BBOX)
    ds = initialised_store.to_dataset(RES)
    # BBOX = (0,0,1000,1000) at 10m → 100 cells each axis
    assert ds.sizes["x"] == 100
    assert ds.sizes["y"] == 100


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_path(store):
    store.ensure_group("chm", RES, BBOX, CRS)
    r = repr(store)
    assert "store.zarr" in r
    assert "chm" in r
