# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for chm.py — _rasterise (unit) and compute_* (integration)."""

import numpy as np
import pytest

from alsdb.processing.chm import _rasterise, compute_all, compute_chm, compute_dsm, compute_dtm

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
        np.array([5.0]), np.array([5.0]), np.array([7.0]),
        (0.0, 0.0, 100.0, 100.0), resolution=10.0,
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
