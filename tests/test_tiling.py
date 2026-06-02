# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for shared tiling utilities (_tiling.py)."""

import pytest

import numpy as np

from alsdb.processing._tiling import (
    _filter_ground_outliers,
    _hag_stage,
    array_crs,
    array_data_bbox,
    attach_hag,
    check_bbox_overlap,
    check_year_exists,
    run_tiled,
    tile_bboxes,
)
from alsdb.utils.schema import LAS_ATTRIBUTES

# ---------------------------------------------------------------------------
# Fixtures — reuse conftest session fixtures (provider from conftest.py)
# ---------------------------------------------------------------------------

BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)


# ---------------------------------------------------------------------------
# tile_bboxes
# ---------------------------------------------------------------------------


def test_tile_bboxes_single_tile_no_buffer():
    tiles = tile_bboxes((0.0, 0.0, 500.0, 500.0), tile_size=500.0, buffer=0.0)
    assert len(tiles) == 1
    qb, cb = tiles[0]
    assert qb == cb == (0.0, 0.0, 500.0, 500.0)


def test_tile_bboxes_count_matches_grid():
    bbox = (0.0, 0.0, 1000.0, 1000.0)
    tiles = tile_bboxes(bbox, tile_size=500.0, buffer=0.0)
    assert len(tiles) == 4  # 2 × 2


def test_tile_bboxes_non_divisible_count():
    bbox = (0.0, 0.0, 1100.0, 750.0)
    tiles = tile_bboxes(bbox, tile_size=500.0, buffer=0.0)
    # ceil(1100/500)=3, ceil(750/500)=2  → 6
    assert len(tiles) == 6


def test_tile_bboxes_buffer_inflates_query():
    tiles = tile_bboxes((0.0, 0.0, 500.0, 500.0), tile_size=500.0, buffer=50.0)
    qb, cb = tiles[0]
    assert qb[0] == pytest.approx(-50.0)  # min_x inflated
    assert qb[1] == pytest.approx(-50.0)  # min_y inflated
    assert qb[2] == pytest.approx(550.0)  # max_x inflated
    assert qb[3] == pytest.approx(550.0)  # max_y inflated


def test_tile_bboxes_crop_bbox_is_not_inflated():
    tiles = tile_bboxes((0.0, 0.0, 500.0, 500.0), tile_size=500.0, buffer=50.0)
    _, cb = tiles[0]
    assert cb == (0.0, 0.0, 500.0, 500.0)


def test_tile_bboxes_last_tile_clips_to_bbox():
    bbox = (0.0, 0.0, 1100.0, 1000.0)
    tiles = tile_bboxes(bbox, tile_size=500.0, buffer=0.0)
    # Last column crop_bbox max_x should == 1100 (not 1500)
    last_col_tiles = [cb for _, cb in tiles if cb[0] == 1000.0]
    assert all(cb[2] == 1100.0 for cb in last_col_tiles)


def test_tile_bboxes_contiguous_coverage():
    """Crop bboxes must tile the full area without gaps or overlap."""
    bbox = (0.0, 0.0, 1000.0, 1000.0)
    tiles = tile_bboxes(bbox, tile_size=300.0, buffer=0.0)
    # Collect all crop x/y extents
    xs = sorted({cb[0] for _, cb in tiles} | {cb[2] for _, cb in tiles})
    ys = sorted({cb[1] for _, cb in tiles} | {cb[3] for _, cb in tiles})
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(1000.0)
    assert ys[0] == pytest.approx(0.0)
    assert ys[-1] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# run_tiled
# ---------------------------------------------------------------------------


class _DummyStore:
    pass


def _make_recorder():
    """Return a worker_fn that records its calls and a list to inspect."""
    calls = []

    def worker(provider, query_bbox, crop_bbox, store, tile_index, **kwargs):
        calls.append((tile_index, crop_bbox, kwargs))

    return worker, calls


def test_run_tiled_sequential_calls_all_tiles():
    tiles = tile_bboxes((0.0, 0.0, 1000.0, 1000.0), tile_size=500.0, buffer=0.0)
    worker, calls = _make_recorder()
    run_tiled(worker, None, tiles, _DummyStore(), n_workers=1)
    assert len(calls) == 4


def test_run_tiled_sequential_passes_kwargs():
    tiles = tile_bboxes((0.0, 0.0, 500.0, 500.0), tile_size=500.0, buffer=0.0)
    worker, calls = _make_recorder()
    run_tiled(worker, None, tiles, _DummyStore(), n_workers=1, resolution=1.0, year=2021)
    assert calls[0][2] == {"resolution": 1.0, "year": 2021}


def test_run_tiled_parallel_calls_all_tiles():
    tiles = tile_bboxes((0.0, 0.0, 1000.0, 1000.0), tile_size=500.0, buffer=0.0)
    worker, calls = _make_recorder()
    run_tiled(worker, None, tiles, _DummyStore(), n_workers=2)
    assert len(calls) == 4


def test_run_tiled_parallel_propagates_exception():
    tiles = tile_bboxes((0.0, 0.0, 500.0, 500.0), tile_size=500.0, buffer=0.0)

    def bad_worker(provider, query_bbox, crop_bbox, store, tile_index, **kwargs):
        raise ValueError("deliberate error")

    with pytest.raises(ValueError, match="deliberate error"):
        run_tiled(bad_worker, None, tiles, _DummyStore(), n_workers=2)


# ---------------------------------------------------------------------------
# array helpers — integration tests (use session provider from conftest)
# ---------------------------------------------------------------------------


def test_array_crs(provider):
    crs = array_crs(provider)
    assert crs == "EPSG:25830"


def test_array_data_bbox_within_written_extent(provider):
    mn_x, mn_y, mx_x, mx_y = array_data_bbox(provider)
    # Written points are in [308050, 308950] x [4688050, 4688950]
    assert mn_x >= 308_000.0
    assert mn_y >= 4_688_000.0
    assert mx_x <= 309_000.0
    assert mx_y <= 4_689_000.0


def test_check_year_exists_true(provider):
    assert check_year_exists(2021, provider) is True


def test_check_year_exists_false(provider):
    assert check_year_exists(1900, provider) is False


def test_check_bbox_overlap_true(provider):
    assert check_bbox_overlap(BBOX, provider) is True


def test_check_bbox_overlap_false(provider):
    far_bbox = (0.0, 0.0, 1.0, 1.0)
    assert check_bbox_overlap(far_bbox, provider) is False


# ---------------------------------------------------------------------------
# _hag_stage
# ---------------------------------------------------------------------------


def _minimal_arr(n_gnd: int, n_other: int = 5) -> np.ndarray:
    """Minimal PDAL-compatible structured array for hag_stage tests."""
    n = n_gnd + n_other
    dtype = [(name, LAS_ATTRIBUTES[name]) for name in LAS_ATTRIBUTES] + [
        ("X", np.float64),
        ("Y", np.float64),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["Classification"][:n_gnd] = 2  # ground
    arr["Classification"][n_gnd:] = 3  # vegetation
    return arr


def test_hag_stage_returns_delaunay_when_enough_ground():
    arr = _minimal_arr(n_gnd=5)
    stage = _hag_stage(arr)
    assert stage["type"] == "filters.hag_delaunay"


def test_hag_stage_falls_back_to_nn_when_too_few_ground():
    arr = _minimal_arr(n_gnd=2)
    stage = _hag_stage(arr)
    assert stage["type"] == "filters.hag_nn"


def test_hag_stage_nn_has_allow_extrapolation():
    arr = _minimal_arr(n_gnd=1)
    stage = _hag_stage(arr)
    assert stage.get("allow_extrapolation") is True


def test_hag_stage_exactly_three_ground_uses_delaunay():
    arr = _minimal_arr(n_gnd=3)
    stage = _hag_stage(arr)
    assert stage["type"] == "filters.hag_delaunay"


# ---------------------------------------------------------------------------
# _filter_ground_outliers
# ---------------------------------------------------------------------------


def _make_ground_arr(z_values: np.ndarray) -> np.ndarray:
    """Structured array of ground-only points at given Z elevations."""
    n = len(z_values)
    dtype = [(name, LAS_ATTRIBUTES[name]) for name in LAS_ATTRIBUTES] + [
        ("X", np.float64),
        ("Y", np.float64),
    ]
    arr = np.zeros(n, dtype=dtype)
    rng = np.random.default_rng(42)
    arr["X"] = rng.uniform(0.0, 100.0, n)
    arr["Y"] = rng.uniform(0.0, 100.0, n)
    arr["Z"] = z_values
    arr["Classification"] = 2
    return arr


def test_filter_ground_outliers_preserves_point_count():
    """PDAL filters reclassify in-place — total point count must never change."""
    z = np.linspace(100.0, 110.0, 100)
    arr = _make_ground_arr(z)
    result = _filter_ground_outliers(arr)
    assert len(result) == len(arr)


def test_filter_ground_outliers_spike_reclassified():
    z = np.concatenate([np.linspace(100.0, 101.0, 30), [200.0]])
    arr = _make_ground_arr(z)
    result = _filter_ground_outliers(arr)
    # The spike at Z=200 should no longer be class 2
    spike_idx = np.argmax(z)
    assert result["Classification"][spike_idx] != 2


def test_filter_ground_outliers_below_ground_reclassified():
    z = np.concatenate([np.linspace(100.0, 101.0, 30), [0.0]])
    arr = _make_ground_arr(z)
    result = _filter_ground_outliers(arr)
    pit_idx = np.argmin(z)
    assert result["Classification"][pit_idx] != 2


def test_filter_ground_outliers_too_few_points_unchanged():
    z = np.array([100.0, 101.0, 500.0])  # only 3 points, below minimum
    arr = _make_ground_arr(z)
    result = _filter_ground_outliers(arr)
    # Should return unchanged (too few for PDAL outlier detection)
    assert np.array_equal(result["Classification"], arr["Classification"])


def test_filter_ground_outliers_does_not_modify_original():
    z = np.concatenate([np.linspace(100.0, 101.0, 30), [200.0]])
    arr = _make_ground_arr(z)
    original_cls = arr["Classification"].copy()
    _filter_ground_outliers(arr)
    assert np.array_equal(arr["Classification"], original_cls)


def test_filter_ground_outliers_vegetation_untouched():
    """Vegetation points (class 3) must never be reclassified."""
    n = 40
    dtype = [(name, LAS_ATTRIBUTES[name]) for name in LAS_ATTRIBUTES] + [
        ("X", np.float64),
        ("Y", np.float64),
    ]
    arr = np.zeros(n, dtype=dtype)
    rng = np.random.default_rng(0)
    arr["X"] = rng.uniform(0, 100, n)
    arr["Y"] = rng.uniform(0, 100, n)
    arr["Z"][:20] = np.linspace(100.0, 101.0, 20)
    arr["Z"][20:] = np.linspace(110.0, 120.0, 20)
    arr["Classification"][:20] = 2  # ground
    arr["Classification"][20:] = 3  # vegetation
    result = _filter_ground_outliers(arr)
    assert np.all(result["Classification"][20:] == 3)


# ---------------------------------------------------------------------------
# attach_hag — integration
# ---------------------------------------------------------------------------


def test_attach_hag_adds_hag_field(provider):
    from alsdb.processing._tiling import query_to_array

    arr = query_to_array(provider, BBOX, year=2021)
    result = attach_hag(arr)
    assert "HeightAboveGround" in result.dtype.names


def test_attach_hag_ground_hag_near_zero(provider):
    from alsdb.processing._tiling import query_to_array

    arr = query_to_array(provider, BBOX, year=2021)
    result = attach_hag(arr)
    gnd = result[result["Classification"] == 2]
    if len(gnd) > 0:
        # Ground points should have HAG ≈ 0 (clamped from small negatives)
        assert np.all(gnd["HeightAboveGround"] >= 0.0)
        assert np.all(gnd["HeightAboveGround"] < 5.0)


def test_attach_hag_veg_hag_positive(provider):
    from alsdb.processing._tiling import query_to_array

    arr = query_to_array(provider, BBOX, year=2021)
    result = attach_hag(arr)
    veg = result[result["Classification"] == 3]
    if len(veg) > 0:
        above = veg["HeightAboveGround"][veg["HeightAboveGround"] > 0]
        assert len(above) > 0
