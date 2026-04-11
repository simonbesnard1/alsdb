# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for shared tiling utilities (_tiling.py)."""

import pytest

from alsdb.processing._tiling import (
    array_crs,
    array_data_bbox,
    check_bbox_overlap,
    check_year_exists,
    run_tiled,
    tile_bboxes,
)

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
