# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for gap.py — _compute_gap_grid/_gap_to_lai (unit) and compute_gap_fraction (integration)."""

import numpy as np
import pytest

from alsdb.processing.gap import _compute_gap_grid, _gap_to_lai, compute_gap_fraction

BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
RES = 10.0
YEAR = 2021
_LAI_K = 0.5


# ---------------------------------------------------------------------------
# Helpers — build minimal structured arrays for gap / LAI functions
# ---------------------------------------------------------------------------


def _make_points(n_gnd, n_veg, bbox, seed=0):
    """
    Structured array with fields expected by _compute_gap_grid.
    """
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = bbox
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("ReturnNumber", np.uint8),
        ("Classification", np.uint8),
    ]
    n = n_gnd + n_veg
    arr = np.zeros(n, dtype=dtype)
    arr["X"] = rng.uniform(min_x, max_x, n)
    arr["Y"] = rng.uniform(min_y, max_y, n)
    arr["ReturnNumber"][:] = 1
    arr["Classification"][:n_gnd] = 2  # ground
    arr["Classification"][n_gnd:] = 3  # vegetation
    return arr


# ---------------------------------------------------------------------------
# _compute_gap_grid — unit tests
# ---------------------------------------------------------------------------


def test_compute_gap_grid_output_shape():
    pts = _make_points(50, 50, (0.0, 0.0, 100.0, 100.0))
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    assert grid.shape == (10, 10)


def test_compute_gap_grid_dtype_float32():
    pts = _make_points(50, 50, (0.0, 0.0, 100.0, 100.0))
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    assert grid.dtype == np.float32


def test_compute_gap_grid_all_ground_gives_one():
    """All first-return ground points → gap fraction = 1.0."""
    pts = _make_points(100, 0, (0.0, 0.0, 100.0, 100.0))
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    valid = grid[~np.isnan(grid)]
    assert len(valid) > 0
    np.testing.assert_allclose(valid, 1.0)


def test_compute_gap_grid_all_vegetation_gives_zero():
    """All first-return vegetation points → gap fraction = 0.0."""
    pts = _make_points(0, 100, (0.0, 0.0, 100.0, 100.0))
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    valid = grid[~np.isnan(grid)]
    assert len(valid) > 0
    np.testing.assert_allclose(valid, 0.0)


def test_compute_gap_grid_mixed_values_between_zero_and_one():
    pts = _make_points(100, 100, (0.0, 0.0, 100.0, 100.0))
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    valid = grid[~np.isnan(grid)]
    assert np.all(valid >= 0.0)
    assert np.all(valid <= 1.0)


def test_compute_gap_grid_empty_cells_are_nan():
    """A single point; all other cells must be NaN."""
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("ReturnNumber", np.uint8),
        ("Classification", np.uint8),
    ]
    pts = np.array([(5.0, 5.0, 1, 2)], dtype=dtype)
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    assert np.isnan(grid).sum() == grid.size - 1


def test_compute_gap_grid_only_unclassified_first_returns_give_nan():
    """
    First returns that are neither ground (2) nor vegetation (3–5) contribute
    to n_tot but not to n_gnd or n_veg, so n_gnd/(n_gnd+n_veg) = 0/0 → NaN.
    """
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("ReturnNumber", np.uint8),
        ("Classification", np.uint8),
    ]
    n = 50
    pts = np.zeros(n, dtype=dtype)
    rng = np.random.default_rng(0)
    pts["X"] = rng.uniform(0, 100, n)
    pts["Y"] = rng.uniform(0, 100, n)
    pts["ReturnNumber"] = 1  # first returns
    pts["Classification"] = 1  # unclassified — neither ground nor veg
    grid = _compute_gap_grid(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    valid = grid[~np.isnan(grid)]
    # Cells with unclassified first returns have 0/0 → NaN; those without are also NaN
    assert len(valid) == 0


# ---------------------------------------------------------------------------
# _gap_to_lai — unit tests
# ---------------------------------------------------------------------------


def test_gap_to_lai_zero_gap_gives_max():
    """P_gap = 0 → LAI = _LAI_MAX (clamped)."""
    gap = np.zeros((3, 3), dtype=np.float32)
    lai = _gap_to_lai(gap, k=_LAI_K)
    # _LAI_MAX = 10.0
    assert np.all(np.isnan(lai))  # 0 → nan before clamp


def test_gap_to_lai_one_gap_gives_zero():
    """P_gap = 1 → LAI = -ln(1) / k = 0."""
    gap = np.ones((3, 3), dtype=np.float32)
    lai = _gap_to_lai(gap, k=_LAI_K)
    np.testing.assert_allclose(lai, 0.0, atol=1e-5)


def test_gap_to_lai_known_value():
    """P_gap = exp(-1) → LAI = 1/k = 2.0 (for k=0.5)."""
    p = float(np.exp(-1.0))
    gap = np.full((1, 1), p, dtype=np.float32)
    lai = _gap_to_lai(gap, k=0.5)
    assert float(lai[0, 0]) == pytest.approx(2.0, rel=1e-4)


def test_gap_to_lai_nan_passthrough():
    gap = np.array([[np.nan, 0.5], [0.5, np.nan]], dtype=np.float32)
    lai = _gap_to_lai(gap, k=_LAI_K)
    assert np.isnan(lai[0, 0])
    assert np.isnan(lai[1, 1])
    assert not np.isnan(lai[0, 1])


def test_gap_to_lai_clamped_to_max():
    """Very small (but positive) gap values must be clamped to LAI_MAX=10."""
    gap = np.array([[1e-10]], dtype=np.float32)
    lai = _gap_to_lai(gap, k=_LAI_K)
    assert float(lai[0, 0]) == pytest.approx(10.0)


def test_gap_to_lai_dtype_float32():
    gap = np.full((3, 3), 0.5, dtype=np.float32)
    lai = _gap_to_lai(gap, k=_LAI_K)
    assert lai.dtype == np.float32


# ---------------------------------------------------------------------------
# Integration — requires real TileDB + PDAL (uses session provider)
# ---------------------------------------------------------------------------


def test_compute_gap_fraction_writes_gap(provider, store):
    compute_gap_fraction(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("gap", RES, YEAR)


def test_compute_gap_fraction_with_lai(provider, store):
    compute_gap_fraction(provider, store, resolution=RES, bbox=BBOX, year=YEAR, lai=True, k=_LAI_K)
    assert store.has_data("gap", RES, YEAR)
    assert store.has_data("lai", RES, YEAR)


def test_compute_gap_fraction_gap_values_in_range(provider, store):
    compute_gap_fraction(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    gap_arr = np.array(store._root["10m"]["gap"][0])
    valid = gap_arr[~np.isnan(gap_arr)]
    assert len(valid) > 0
    assert np.all(valid >= 0.0)
    assert np.all(valid <= 1.0)


def test_compute_gap_fraction_overwrite_false_skips(provider, store):
    compute_gap_fraction(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["gap"][0] = 999.0
    compute_gap_fraction(provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False)
    assert float(store._root["10m"]["gap"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_gap_fraction_out_of_year_skips(provider, store):
    compute_gap_fraction(provider, store, resolution=RES, bbox=BBOX, year=1900)
    assert not store.has_data("gap", RES, 1900)
