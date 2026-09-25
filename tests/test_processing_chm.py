# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for chm.py — _rasterise (unit) and compute_* (integration)."""

import numpy as np
import pytest

from alsdb.processing.chm import (
    _adaptive_max_distance,
    _cap_height,
    _classification_ranges,
    _delaunay_raster,
    _dtm_idw,
    _exclude_classes_stages,
    _gate_by_ground_distance,
    _nn_fill,
    _fill_pits,
    _outlier_removal_stages,
    _pitfree_rasterise,
    _rasterise,
    _run,
    _spikefree_rasterise,
    _thin_highest_per_subcell,
    _validate_grid_alignment,
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


def test_compute_chm_max_ground_distance_generous_value_still_writes_data(provider, store):
    """A generous max_ground_distance shouldn't filter out real, well-supported points."""
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, max_ground_distance=1000.0)
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_max_ground_distance_tiny_value_yields_no_data(provider, store):
    """An unrealistically tiny max_ground_distance should gate out effectively everything."""
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, max_ground_distance=0.001)
    assert not store.has_data("chm", RES, YEAR)


def test_compute_chm_veg_classes_custom_still_writes_data(provider, store):
    """Including unclassified points (Class 1) alongside vegetation shouldn't break anything."""
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, veg_classes=(1, 3, 4, 5))
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_remove_outliers_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, remove_outliers=True)
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_max_height_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, max_height=60.0)
    assert store.has_data("chm", RES, YEAR)


def test_compute_dsm_tile_size_not_multiple_of_resolution_raises(provider, store):
    with pytest.raises(ValueError, match="whole multiple"):
        compute_dsm(provider, store, resolution=3.0, bbox=BBOX, year=YEAR, tile_size=250.0)


def test_compute_chm_veg_classes_nonexistent_class_yields_no_data(provider, store):
    """A class that doesn't occur in the tile should behave like 'no vegetation points'."""
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, veg_classes=(31,))
    assert not store.has_data("chm", RES, YEAR)


def test_compute_chm_pitfree_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        pitfree=True,
        pitfree_max_distance=20.0,
    )
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_pitfree_without_max_distance_raises(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    with pytest.raises(ValueError, match="pitfree_max_distance"):
        compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, pitfree=True)


def test_compute_chm_pitfree_max_distance_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        pitfree=True,
        pitfree_max_distance=20.0,
    )
    assert store.has_data("chm", RES, YEAR)


def test_compute_all_pitfree_still_writes_all_three(provider, store):
    compute_all(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        pitfree=True,
        pitfree_max_distance=20.0,
    )
    assert store.has_data("dtm", RES, YEAR)
    assert store.has_data("dsm", RES, YEAR)
    assert store.has_data("chm", RES, YEAR)


def test_compute_all_pitfree_without_max_distance_raises(provider, store):
    with pytest.raises(ValueError, match="pitfree_max_distance"):
        compute_all(provider, store, resolution=RES, bbox=BBOX, year=YEAR, pitfree=True)


def test_compute_chm_spikefree_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        spikefree=True,
        spikefree_max_distance=20.0,
    )
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_spikefree_without_max_distance_raises(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    with pytest.raises(ValueError, match="spikefree_max_distance"):
        compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR, spikefree=True)


def test_compute_chm_pitfree_and_spikefree_together_raises(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_chm(
            provider,
            store,
            resolution=RES,
            bbox=BBOX,
            year=YEAR,
            pitfree=True,
            pitfree_max_distance=20.0,
            spikefree=True,
            spikefree_max_distance=20.0,
        )


def test_compute_all_spikefree_still_writes_all_three(provider, store):
    compute_all(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        spikefree=True,
        spikefree_max_distance=20.0,
    )
    assert store.has_data("dtm", RES, YEAR)
    assert store.has_data("dsm", RES, YEAR)
    assert store.has_data("chm", RES, YEAR)


def test_compute_all_spikefree_without_max_distance_raises(provider, store):
    with pytest.raises(ValueError, match="spikefree_max_distance"):
        compute_all(provider, store, resolution=RES, bbox=BBOX, year=YEAR, spikefree=True)


def test_compute_all_writes_all_three(provider, store):
    compute_all(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("dtm", RES, YEAR)
    assert store.has_data("dsm", RES, YEAR)
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_pitfree_max_distance_auto_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        pitfree=True,
        pitfree_max_distance="auto",
    )
    assert store.has_data("chm", RES, YEAR)


def test_compute_chm_spikefree_max_distance_auto_still_writes_data(provider, store):
    store.ensure_group("chm", RES, BBOX, "EPSG:25830")
    compute_chm(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        spikefree=True,
        spikefree_max_distance="auto",
    )
    assert store.has_data("chm", RES, YEAR)


def test_compute_all_pitfree_max_distance_auto_still_writes_all_three(provider, store):
    compute_all(
        provider,
        store,
        resolution=RES,
        bbox=BBOX,
        year=YEAR,
        pitfree=True,
        pitfree_max_distance="auto",
    )
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
# _pitfree_rasterise — unit tests (real PDAL filters.delaunay/faceraster)
# ---------------------------------------------------------------------------


def _hag_points(xs, ys, heights) -> np.ndarray:
    """Minimal PDAL-compatible structured array with HeightAboveGround attached.

    ``Z`` must be present (even though _pitfree_rasterise reads
    HeightAboveGround) because _delaunay_raster's non-"Z" value_field path
    overwrites a copy's Z field in place - the dtype has to declare it.
    """
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("Z", np.float64),
        ("HeightAboveGround", np.float64),
    ]
    arr = np.zeros(len(xs), dtype=dtype)
    arr["X"] = xs
    arr["Y"] = ys
    arr["Z"] = heights
    arr["HeightAboveGround"] = heights
    return arr


def _canopy_with_one_pit() -> np.ndarray:
    """
    A grid of tall-canopy points (height 15 m) near cell centres on a 10 m
    grid, with the point near (55, 55) replaced by a spurious low return
    (height 1 m) - the exact failure mode naive per-cell max-binning can't
    recover from (nothing else competes inside that one cell), but a
    pit-free threshold layer above 1 m excludes entirely, leaving only the
    surrounding (constant 15 m) points to interpolate from.

    Positions are jittered (small, deterministic offsets) rather than a
    perfectly regular lattice: an exact square grid is a degenerate input
    for Delaunay triangulation (cocircular points, ambiguous diagonals) and
    can leave the exact cell where a point was removed as NaN instead of
    interpolated - a real point cloud is never perfectly grid-aligned, and
    the jitter avoids that pathological edge case here too.
    """
    rng = np.random.default_rng(0)
    xs, ys, hs = [], [], []
    for cx in range(5, 100, 10):
        for cy in range(5, 100, 10):
            if (cx, cy) == (55, 55):
                continue
            jx, jy = rng.uniform(-2.0, 2.0, 2)
            xs.append(cx + jx)
            ys.append(cy + jy)
            hs.append(15.0)
    xs.append(55.0)
    ys.append(55.0)
    hs.append(1.0)
    return _hag_points(xs, ys, hs)


def test_pitfree_rasterise_recovers_pit_naive_binning_creates():
    points = _canopy_with_one_pit()
    bbox = (0.0, 0.0, 100.0, 100.0)

    naive = _rasterise(
        points["X"],
        points["Y"],
        points["HeightAboveGround"],
        bbox,
        resolution=10.0,
        statistic="max",
    )
    assert float(np.nanmin(naive)) == pytest.approx(1.0, abs=0.01), (
        "sanity check: naive per-cell binning should show the spurious low cell"
    )

    pitfree = _pitfree_rasterise(points, bbox, resolution=10.0, thresholds=(0.0, 10.0))
    assert float(np.nanmin(pitfree)) > 14.0, (
        "pit-free's t=10 layer excludes the low point entirely, so the only "
        "contribution at that cell comes from interpolating the surrounding "
        "constant-15m points"
    )


def test_pitfree_rasterise_output_shape():
    points = _canopy_with_one_pit()
    grid = _pitfree_rasterise(points, (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert grid.shape == (10, 10)


def test_pitfree_rasterise_output_dtype_float32():
    points = _canopy_with_one_pit()
    grid = _pitfree_rasterise(points, (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert grid.dtype == np.float32


def test_pitfree_rasterise_too_few_points_returns_all_nan():
    points = _hag_points([1.0, 2.0], [1.0, 2.0], [10.0, 12.0])
    grid = _pitfree_rasterise(points, (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert np.all(np.isnan(grid))


def _two_clusters_far_apart() -> np.ndarray:
    """
    Two vegetation clusters far apart (30 m wide each, ~130 m gap between
    them), all at the same height. Delaunay triangulation over both
    clusters together spans the whole gap as a single convex hull,
    interpolating "canopy" across genuinely non-vegetated ground in
    between unless distance-masked - the exact coverage mismatch found on
    real data (~100% coverage where naive binning only covered ~20-30%).
    """
    rng = np.random.default_rng(1)
    xs, ys, hs = [], [], []
    for cx, cy in zip(rng.uniform(5, 35, 30), rng.uniform(5, 45, 30)):
        xs.append(cx)
        ys.append(cy)
        hs.append(15.0)
    for cx, cy in zip(rng.uniform(165, 195, 30), rng.uniform(5, 45, 30)):
        xs.append(cx)
        ys.append(cy)
        hs.append(15.0)
    return _hag_points(xs, ys, hs)


def test_pitfree_rasterise_no_max_distance_extrapolates_across_gap():
    """Sanity check: without max_distance, the gap is filled in (the failure
    mode max_distance exists to fix)."""
    points = _two_clusters_far_apart()
    grid = _pitfree_rasterise(points, (0.0, 0.0, 200.0, 50.0), resolution=10.0, thresholds=(0.0,))
    assert not np.isnan(grid[2, 10]), "middle of the gap should be interpolated without a mask"


def test_pitfree_rasterise_max_distance_masks_extrapolated_gap():
    points = _two_clusters_far_apart()
    grid = _pitfree_rasterise(
        points, (0.0, 0.0, 200.0, 50.0), resolution=10.0, thresholds=(0.0,), max_distance=20.0
    )
    assert np.isnan(grid[2, 10]), "middle of the gap should be masked out, far from either cluster"
    assert not np.isnan(grid[2, 2]), "cells near a real cluster should still be kept"


def test_pitfree_rasterise_max_distance_auto_masks_extrapolated_gap():
    """ "auto" should behave like a well-chosen fixed value on this fixture:
    the within-cluster spacing is small relative to the ~130 m gap, so the
    derived distance should mask the gap without touching near-cluster cells."""
    points = _two_clusters_far_apart()
    grid = _pitfree_rasterise(
        points, (0.0, 0.0, 200.0, 50.0), resolution=10.0, thresholds=(0.0,), max_distance="auto"
    )
    assert np.isnan(grid[2, 10]), "middle of the gap should be masked out, far from either cluster"
    assert not np.isnan(grid[2, 2]), "cells near a real cluster should still be kept"


# ---------------------------------------------------------------------------
# _adaptive_max_distance — unit tests (pure numpy/scipy)
# ---------------------------------------------------------------------------


def test_adaptive_max_distance_matches_known_grid_spacing():
    dtype = [("X", np.float64), ("Y", np.float64)]
    xs, ys = np.meshgrid(np.arange(0.0, 40.0, 2.0), np.arange(0.0, 40.0, 2.0))
    points = np.zeros(xs.size, dtype=dtype)
    points["X"], points["Y"] = xs.ravel(), ys.ravel()
    d = _adaptive_max_distance(points, percentile=95.0, multiplier=2.0)
    assert d == pytest.approx(4.0, abs=0.2)


def test_adaptive_max_distance_scales_with_density():
    """Sparser points must give a proportionally larger distance - confirms
    this actually adapts to point spacing rather than returning a constant."""
    dtype = [("X", np.float64), ("Y", np.float64)]
    xs_dense, ys_dense = np.meshgrid(np.arange(0.0, 40.0, 2.0), np.arange(0.0, 40.0, 2.0))
    dense = np.zeros(xs_dense.size, dtype=dtype)
    dense["X"], dense["Y"] = xs_dense.ravel(), ys_dense.ravel()

    xs_sparse, ys_sparse = np.meshgrid(np.arange(0.0, 100.0, 5.0), np.arange(0.0, 100.0, 5.0))
    sparse = np.zeros(xs_sparse.size, dtype=dtype)
    sparse["X"], sparse["Y"] = xs_sparse.ravel(), ys_sparse.ravel()

    d_dense = _adaptive_max_distance(dense)
    d_sparse = _adaptive_max_distance(sparse)
    assert d_sparse > d_dense


def test_adaptive_max_distance_multiplier_scales_linearly():
    dtype = [("X", np.float64), ("Y", np.float64)]
    xs, ys = np.meshgrid(np.arange(0.0, 40.0, 2.0), np.arange(0.0, 40.0, 2.0))
    points = np.zeros(xs.size, dtype=dtype)
    points["X"], points["Y"] = xs.ravel(), ys.ravel()
    d1 = _adaptive_max_distance(points, multiplier=1.0)
    d3 = _adaptive_max_distance(points, multiplier=3.0)
    assert d3 == pytest.approx(3.0 * d1)


# ---------------------------------------------------------------------------
# _thin_highest_per_subcell — unit tests (pure numpy)
# ---------------------------------------------------------------------------


def test_thin_highest_per_subcell_keeps_max_value_point():
    points = _hag_points(
        xs=[1.0, 1.2, 5.0, 5.1],
        ys=[1.0, 1.1, 5.0, 5.2],
        heights=[3.0, 9.0, 4.0, 2.0],
    )
    out = _thin_highest_per_subcell(points, subcell_resolution=2.0)
    assert len(out) == 2
    assert set(out["HeightAboveGround"].tolist()) == {9.0, 4.0}


def test_thin_highest_per_subcell_preserves_all_fields():
    points = _hag_points(xs=[1.0, 1.2], ys=[1.0, 1.1], heights=[3.0, 9.0])
    out = _thin_highest_per_subcell(points, subcell_resolution=5.0)
    assert len(out) == 1
    assert out.dtype == points.dtype
    assert float(out["Z"][0]) == pytest.approx(9.0)


def test_thin_highest_per_subcell_empty_input_unchanged():
    points = _hag_points(xs=[], ys=[], heights=[])
    out = _thin_highest_per_subcell(points, subcell_resolution=1.0)
    assert len(out) == 0


def test_thin_highest_per_subcell_absolute_grid_not_relative_to_input_extent():
    """Subcells are snapped to floor(X / subcell_resolution) directly, not
    relative to the point set's own bounding box - two points that land in
    the same absolute subcell should thin down to one regardless of where
    the point set as a whole sits."""
    points = _hag_points(xs=[100.4, 100.6], ys=[200.4, 200.6], heights=[5.0, 8.0])
    out = _thin_highest_per_subcell(points, subcell_resolution=1.0)
    assert len(out) == 1
    assert float(out["HeightAboveGround"][0]) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _spikefree_rasterise — unit tests (real PDAL filters.delaunay/faceraster)
# ---------------------------------------------------------------------------


def test_spikefree_rasterise_output_shape_and_dtype():
    points = _two_clusters_far_apart()
    grid = _spikefree_rasterise(
        points, (0.0, 0.0, 200.0, 50.0), resolution=10.0, subcell_resolution=3.0, max_distance=20.0
    )
    assert grid.shape == (5, 20)
    assert grid.dtype == np.float32


def test_spikefree_rasterise_masks_extrapolated_gap():
    """Same gap-protection contract as pitfree's max_distance, since
    _spikefree_rasterise reuses _mask_by_point_distance directly."""
    points = _two_clusters_far_apart()
    grid = _spikefree_rasterise(
        points, (0.0, 0.0, 200.0, 50.0), resolution=10.0, subcell_resolution=3.0, max_distance=20.0
    )
    assert np.isnan(grid[2, 10]), "middle of the gap should be masked out, far from either cluster"
    assert not np.isnan(grid[2, 2]), "cells near a real cluster should still be kept"


def test_spikefree_rasterise_max_distance_auto_masks_extrapolated_gap():
    points = _two_clusters_far_apart()
    grid = _spikefree_rasterise(
        points,
        (0.0, 0.0, 200.0, 50.0),
        resolution=10.0,
        subcell_resolution=3.0,
        max_distance="auto",
    )
    assert np.isnan(grid[2, 10]), "middle of the gap should be masked out, far from either cluster"
    assert not np.isnan(grid[2, 2]), "cells near a real cluster should still be kept"


def test_spikefree_rasterise_too_few_points_returns_all_nan():
    points = _hag_points([1.0, 2.0], [1.0, 2.0], [10.0, 12.0])
    grid = _spikefree_rasterise(
        points, (0.0, 0.0, 100.0, 100.0), resolution=10.0, subcell_resolution=3.0, max_distance=20.0
    )
    assert np.all(np.isnan(grid))


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


def test_dtm_idw_north_up_orientation():
    """Top row of the grid should correspond to the highest y values."""
    dtype = [("X", np.float64), ("Y", np.float64), ("Z", np.float64)]
    pts = np.zeros(2, dtype=dtype)
    pts["X"] = [5.0, 5.0]
    pts["Y"] = [5.0, 95.0]
    pts["Z"] = [1.0, 99.0]
    grid = _dtm_idw(pts, crop_bbox=(0.0, 0.0, 100.0, 100.0), resolution=10.0, k=1)
    assert float(grid[0, 0]) == pytest.approx(99.0)
    assert float(grid[9, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _delaunay_raster — unit tests (real PDAL filters.delaunay/faceraster)
# ---------------------------------------------------------------------------


def _plane_corners() -> np.ndarray:
    """Four corner points lying exactly on the plane Z = Y (flat in X).

    A TIN of these four points reproduces Z = Y exactly everywhere inside
    the convex hull regardless of which diagonal splits the two triangles,
    since both triangles lie in the same plane - giving exact expected
    values to assert against, not just an approximate/monotonic check.
    """
    dtype = [("X", np.float64), ("Y", np.float64), ("Z", np.float64)]
    pts = np.zeros(4, dtype=dtype)
    pts["X"] = [0.0, 100.0, 0.0, 100.0]
    pts["Y"] = [0.0, 0.0, 100.0, 100.0]
    pts["Z"] = [0.0, 0.0, 100.0, 100.0]
    return pts


def test_delaunay_raster_north_up_orientation():
    """Same orientation contract as _rasterise/_dtm_idw: row 0 = north (max y)."""
    grid = _delaunay_raster(_plane_corners(), (0.0, 0.0, 100.0, 100.0), resolution=10.0)
    assert float(grid[0, 0]) == pytest.approx(95.0)
    assert float(grid[9, 0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _exclude_classes_stages — unit tests (real PDAL filters.range)
# ---------------------------------------------------------------------------


def _classified_points(classes) -> np.ndarray:
    dtype = [("X", np.float64), ("Y", np.float64), ("Z", np.float64), ("Classification", np.uint8)]
    arr = np.zeros(len(classes), dtype=dtype)
    arr["X"] = np.arange(len(classes))
    arr["Y"] = np.arange(len(classes))
    arr["Classification"] = classes
    return arr


def test_exclude_classes_stages_ands_out_all_listed_classes():
    """Chained negated stages must exclude noise classes 7 *and* 18, not just
    points that are simultaneously both (which a single OR'd stage would do)."""
    pts = _classified_points([1, 2, 3, 7, 18])
    out = _run(_exclude_classes_stages((7, 18)), pts)
    assert sorted(out["Classification"].tolist()) == [1, 2, 3]


def test_exclude_classes_stages_empty_tuple_keeps_everything():
    pts = _classified_points([1, 2, 7, 18])
    out = _run(_exclude_classes_stages(()), pts)
    assert len(out) == len(pts)


# ---------------------------------------------------------------------------
# _cap_height — unit tests
# ---------------------------------------------------------------------------


def test_cap_height_none_disables_cap():
    grid = np.array([[1.0, 999.0], [np.nan, 5.0]], dtype=np.float32)
    out = _cap_height(grid, None)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(999.0)
    assert np.isnan(out[1, 0])
    assert out[1, 1] == pytest.approx(5.0)


def test_cap_height_nulls_values_above_threshold():
    grid = np.array([[1.0, 999.0], [np.nan, 61.0]], dtype=np.float32)
    out = _cap_height(grid, max_height=60.0)
    assert out[0, 0] == pytest.approx(1.0)
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 0])
    assert np.isnan(out[1, 1])


# ---------------------------------------------------------------------------
# _outlier_removal_stages — unit tests (real PDAL filters.outlier)
# ---------------------------------------------------------------------------


def _dense_canopy_lattice(spike_z: float = None) -> np.ndarray:
    """
    A dense, level canopy (~1 pt/m^2, height ~15 m, small jitter) - the small
    nearest-neighbour distances a real ALS survey has, unlike a sparse
    uniform-random scatter, where random density fluctuation alone triggers
    false positives regardless of any genuine anomaly.

    If *spike_z* is given, one extra point is added at that height, isolated
    from its neighbours in Z alone (same X/Y density as everything else) -
    the exact failure mode a height cap alone can't catch if it lands under
    the cap.
    """
    dtype = [("X", np.float64), ("Y", np.float64), ("Z", np.float64), ("Classification", np.uint8)]
    rng = np.random.default_rng(0)
    xs, ys, zs = [], [], []
    for cx in range(0, 40, 2):
        for cy in range(0, 40, 2):
            jx, jy = rng.uniform(-0.3, 0.3, 2)
            xs.append(cx + jx)
            ys.append(cy + jy)
            zs.append(15.0 + rng.uniform(-0.2, 0.2))
    if spike_z is not None:
        xs.append(20.0)
        ys.append(20.0)
        zs.append(spike_z)
    arr = np.zeros(len(xs), dtype=dtype)
    arr["X"] = xs
    arr["Y"] = ys
    arr["Z"] = zs
    arr["Classification"] = 4
    return arr


def test_outlier_removal_stages_drops_isolated_spike():
    pts = _dense_canopy_lattice(spike_z=45.0)
    out = _run(_outlier_removal_stages(mean_k=8, multiplier=2.0), pts)
    assert float(np.max(out["Z"])) < 20.0, "the isolated 45 m spike should be removed"


def test_outlier_removal_stages_mostly_keeps_uniform_canopy():
    """Statistical outlier detection has real edge effects even on uniform
    data (fewer neighbours near the boundary of a bounded point cloud) - the
    property worth asserting is that it isn't wholesale discarding a normal,
    outlier-free canopy, not that it's a no-op."""
    pts = _dense_canopy_lattice()
    out = _run(_outlier_removal_stages(mean_k=8, multiplier=2.0), pts)
    assert len(out) >= 0.9 * len(pts)
    assert float(np.max(out["Z"])) < 16.0


# ---------------------------------------------------------------------------
# _validate_grid_alignment — unit tests
# ---------------------------------------------------------------------------


def test_validate_grid_alignment_accepts_whole_multiple():
    _validate_grid_alignment(tile_size=500.0, resolution=1.0)
    _validate_grid_alignment(tile_size=250.0, resolution=25.0)


def test_validate_grid_alignment_rejects_non_multiple():
    with pytest.raises(ValueError, match="whole multiple"):
        _validate_grid_alignment(tile_size=250.0, resolution=3.0)


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
# _classification_ranges — unit tests
# ---------------------------------------------------------------------------


def test_classification_ranges_contiguous():
    assert _classification_ranges((3, 4, 5)) == "Classification[3:5]"


def test_classification_ranges_single_class():
    assert _classification_ranges((6,)) == "Classification[6:6]"


def test_classification_ranges_noncontiguous():
    assert _classification_ranges((1, 3, 4, 5)) == "Classification[1:1],Classification[3:5]"


def test_classification_ranges_deduplicates_and_sorts():
    assert _classification_ranges((5, 3, 3, 4)) == "Classification[3:5]"


def test_classification_ranges_multiple_gaps():
    assert _classification_ranges((1, 2, 3, 7)) == "Classification[1:3],Classification[7:7]"


# ---------------------------------------------------------------------------
# _gate_by_ground_distance — unit tests
# ---------------------------------------------------------------------------


def _veg_points(xs, ys) -> np.ndarray:
    dtype = [("X", np.float64), ("Y", np.float64), ("HeightAboveGround", np.float64)]
    arr = np.zeros(len(xs), dtype=dtype)
    arr["X"] = xs
    arr["Y"] = ys
    arr["HeightAboveGround"] = 10.0
    return arr


def _ground_at(xs, ys) -> np.ndarray:
    dtype = [("X", np.float64), ("Y", np.float64)]
    arr = np.zeros(len(xs), dtype=dtype)
    arr["X"] = xs
    arr["Y"] = ys
    return arr


def test_gate_by_ground_distance_none_disables_gating():
    points = _veg_points([0.0, 100.0], [0.0, 100.0])
    ground = _ground_at([0.0], [0.0])
    result = _gate_by_ground_distance(points, ground, max_distance=None)
    assert len(result) == len(points)


def test_gate_by_ground_distance_drops_far_points():
    # One point right next to the only ground point, one far away
    points = _veg_points([0.0, 500.0], [0.0, 500.0])
    ground = _ground_at([0.0], [0.0])
    result = _gate_by_ground_distance(points, ground, max_distance=5.0)
    assert len(result) == 1
    assert float(result["X"][0]) == pytest.approx(0.0)


def test_gate_by_ground_distance_keeps_near_points():
    points = _veg_points([1.0, 2.0], [1.0, 2.0])
    ground = _ground_at([0.0], [0.0])
    result = _gate_by_ground_distance(points, ground, max_distance=5.0)
    assert len(result) == 2


def test_gate_by_ground_distance_empty_ground_rejects_unsupported_points():
    points = _veg_points([0.0, 100.0], [0.0, 100.0])
    ground = _ground_at([], [])
    result = _gate_by_ground_distance(points, ground, max_distance=5.0)
    assert len(result) == 0


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
