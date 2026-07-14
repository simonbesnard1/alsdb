# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for chm.py — _rasterise (unit) and compute_* (integration)."""

import numpy as np
import pytest

from alsdb.processing.chm import (
    _dtm_idw,
    _nn_fill,
    _fill_pits,
    _rasterise,
    compute_all,
    compute_chm,
    compute_dsm,
    compute_dtm,
)

BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
RES = 10.0
YEAR = 2021


# ---------------------------------------------------------------------------
# _rasterise — pure numpy, no PDAL
# ---------------------------------------------------------------------------


def _scatter(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 100.0, n)
    y = rng.uniform(0.0, 100.0, n)
    v = rng.uniform(1.0, 10.0, n)
    return x, y, v


def test_rasterise_output_shape():
    x, y, v = _scatter()
    grid = _rasterise(x, y, v, (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert grid.shape == (10, 10)


def test_rasterise_dtype_is_float32():
    x, y, v = _scatter()
    grid = _rasterise(x, y, v, (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert grid.dtype == np.float32


def test_rasterise_empty_bins_are_nan():
    # Single point at (5, 5); all other cells should be NaN
    grid = _rasterise(
        np.array([5.0]),
        np.array([5.0]),
        np.array([7.0]),
        (0.0, 0.0, 100.0, 100.0),
        resolution=10.0,
    )
    assert np.isnan(grid).sum() == grid.size - 1
    assert float(grid[~np.isnan(grid)][0]) == pytest.approx(7.0)


def test_rasterise_max_statistic():
    # Two points in the same cell; max should win
    x = np.array([5.0, 5.0])
    y = np.array([5.0, 5.0])
    v = np.array([3.0, 9.0])
    grid = _rasterise(x, y, v, (0.0, 0.0, 100.0, 100.0), resolution=10.0, statistic="max")
    assert float(grid[~np.isnan(grid)][0]) == pytest.approx(9.0)


def test_rasterise_north_up_orientation():
    """Top row of the grid should correspond to the highest y values."""
    x = np.array([5.0, 5.0])
    y = np.array([5.0, 95.0])
    v = np.array([1.0, 99.0])
    grid = _rasterise(x, y, v, (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    # row 0 = north (y≈95), row 9 = south (y≈5)
    assert float(grid[0, 0]) == pytest.approx(99.0)
    assert float(grid[9, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration — requires real TileDB + PDAL (uses session provider)
# ---------------------------------------------------------------------------


def test_compute_dtm_writes_data(provider, store):
    store.ensure_group("dtm", RES, BBOX, "EPSG:25830")
    compute_dtm(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("dtm", RES, YEAR)


def test_compute_dsm_writes_data(provider, store):
    store.ensure_group("dsm", RES, BBOX, "EPSG:25830")
    compute_dsm(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("dsm", RES, YEAR)


def test_compute_chm_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("chm", RES, YEAR)


def test_compute_all_writes_all_three(provider, store):
    compute_all(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("dtm", RES, YEAR)
    assert store.has_data("dsm", RES, YEAR)
    assert store.has_data("chm", RES, YEAR)


def test_compute_dtm_overwrite_false_skips(provider, store):
    """Second call with overwrite=False must not re-compute."""
    compute_dtm(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    # Corrupt the written data to detect if it gets rewritten
    store._root["10m"]["dtm"][0] = 999.0
    compute_dtm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False)
    # Value should still be 999 (skipped)
    assert float(store._root["10m"]["dtm"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_chm_overwrite_true_rewrites(provider, store):
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["chm"][0] = 0.0
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=True)
    # At least some values should be non-zero after recompute
    arr = np.array(store._root["10m"]["chm"][0])
    valid = arr[~np.isnan(arr)]
    assert len(valid) > 0


def test_compute_all_skips_when_all_present(provider, store):
    """compute_all with overwrite=False must no-op when all three present."""
    compute_all(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["chm"][0] = 999.0
    compute_all(provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False)
    assert float(store._root["10m"]["chm"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_dtm_out_of_year_skips(provider, store):
    """Year outside the stored range should silently return without writing."""
    compute_dtm(provider, store, resolution=RES, bbox=BBOX, year=1900)
    assert not store.has_data("dtm", RES, 1900)


def test_compute_dtm_non_overlapping_bbox_skips(provider, store):
    far_bbox = (0.0, 0.0, 1.0, 1.0)
    compute_dtm(provider, store, resolution=RES, bbox=far_bbox, year=YEAR)
    assert not store.has_data("dtm", RES, YEAR)


# ---------------------------------------------------------------------------
# _fill_pits — unit tests (pure numpy)
# ---------------------------------------------------------------------------


def _solid_chm(ny: int = 20, nx: int = 20, fill: float = 15.0) -> np.ndarray:
    return np.full((ny, nx), fill, dtype=np.float32)


def test_fill_pits_all_valid_unchanged():
    grid = _solid_chm()
    result = _fill_pits(grid)
    np.testing.assert_allclose(result, grid, atol=1e-4)


def test_fill_pits_output_shape_preserved():
    grid = _solid_chm(10, 15)
    assert _fill_pits(grid).shape == (10, 15)


def test_fill_pits_output_dtype_float32():
    grid = _solid_chm()
    assert _fill_pits(grid).dtype == np.float32


def test_fill_pits_fills_interior_nan_pit():
    grid = _solid_chm()
    grid[10, 10] = np.nan  # single interior NaN
    result = _fill_pits(grid)
    assert not np.isnan(result[10, 10]), "Interior NaN pit should be filled"


def test_fill_pits_edge_nan_may_remain():
    """NaN cells at the very edge of the array have no valid neighbours — may stay NaN."""
    grid = np.full((10, 10), 15.0, dtype=np.float32)
    grid[0, :] = np.nan  # entire top edge NaN
    result = _fill_pits(grid)
    # Interior should be unaffected
    np.testing.assert_allclose(result[5, 5], 15.0, atol=1e-3)


def test_fill_pits_leaves_isolated_peak_untouched():
    """A genuine tall, isolated tree crown is a valid (non-NaN) cell and must
    survive unchanged — only NaN cells are ever filled."""
    grid = _solid_chm(30, 30, fill=10.0)
    grid[15, 15] = 200.0
    result = _fill_pits(grid)
    assert float(result[15, 15]) == pytest.approx(200.0)


def test_fill_pits_all_nan_returns_all_nan():
    grid = np.full((10, 10), np.nan, dtype=np.float32)
    result = _fill_pits(grid)
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# _dtm_idw — unit tests (pure numpy / scipy)
# ---------------------------------------------------------------------------


def _ground_points(n: int = 50, z_mean: float = 500.0, seed: int = 0) -> np.ndarray:
    """Minimal structured array of ground points for IDW tests."""
    rng = np.random.default_rng(seed)
    dtype = [("X", np.float64), ("Y", np.float64), ("Z", np.float64)]
    arr = np.zeros(n, dtype=dtype)
    arr["X"] = rng.uniform(0.0, 100.0, n)
    arr["Y"] = rng.uniform(0.0, 100.0, n)
    arr["Z"] = rng.normal(z_mean, 0.5, n)
    return arr


def test_dtm_idw_output_shape():
    pts = _ground_points()
    grid = _dtm_idw(pts, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert grid.shape == (10, 10)


def test_dtm_idw_dtype_float32():
    pts = _ground_points()
    grid = _dtm_idw(pts, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert grid.dtype == np.float32


def test_dtm_idw_values_near_input_mean():
    z_mean = 500.0
    pts = _ground_points(z_mean=z_mean)
    grid = _dtm_idw(pts, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
    valid = grid[~np.isnan(grid)]
    assert len(valid) > 0
    assert np.abs(valid.mean() - z_mean) < 2.0


def test_dtm_idw_no_nan_with_sufficient_points():
    pts = _ground_points(n=100)
    grid = _dtm_idw(pts, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert not np.any(np.isnan(grid))


def test_dtm_idw_max_distance_creates_nans():
    pts = _ground_points(n=10, seed=5)
    # Force all points into a small corner; large max_distance should leave far cells NaN
    pts["X"] = np.linspace(0.0, 5.0, 10)
    pts["Y"] = np.linspace(0.0, 5.0, 10)
    grid = _dtm_idw(pts, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0, max_distance=5.0)
    # Far cells should be NaN
    assert np.any(np.isnan(grid))


# ---------------------------------------------------------------------------
# _nn_fill — unit tests
# ---------------------------------------------------------------------------


def _ground_arr_nn(n: int = 30, z: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = [("X", np.float64), ("Y", np.float64), ("Z", np.float64)]
    arr = np.zeros(n, dtype=dtype)
    arr["X"] = rng.uniform(0.0, 100.0, n)
    arr["Y"] = rng.uniform(0.0, 100.0, n)
    arr["Z"] = z
    return arr


def test_nn_fill_fills_nan_cells():
    grid = np.full((10, 10), 100.0, dtype=np.float32)
    grid[5, 5] = np.nan
    gnd = _ground_arr_nn()
    result = _nn_fill(grid, gnd, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert not np.isnan(result[5, 5])


def test_nn_fill_no_nan_unchanged():
    grid = np.full((10, 10), 100.0, dtype=np.float32)
    gnd = _ground_arr_nn()
    result = _nn_fill(grid, gnd, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
    np.testing.assert_array_equal(result, grid)


def test_nn_fill_empty_ground_unchanged():
    grid = np.full((5, 5), np.nan, dtype=np.float32)
    gnd = np.zeros(0, dtype=[("X", np.float64), ("Y", np.float64), ("Z", np.float64)])
    result = _nn_fill(grid, gnd, crop_bbox=(0.0, 0.0, 50.0, 50.0), resolution=10.0)
    assert np.all(np.isnan(result))
