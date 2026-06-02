# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for trees.py — _tree_metrics unit tests and segment_trees integration."""

import numpy as np
import pandas as pd
import pytest

from alsdb.processing.trees import _tree_metrics, segment_trees

BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
YEAR = 2021


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree_points(
    n: int = 60,
    tree_id: int = 1,
    hag_range: tuple = (5.0, 20.0),
    seed: int = 0,
) -> np.ndarray:
    """
    Synthetic structured array with TreeID and HeightAboveGround.
    Mimics the output of filters.litree fed through the HAG pipeline.
    """
    rng = np.random.default_rng(seed)
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("HeightAboveGround", np.float64),
        ("TreeID", np.int64),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["X"] = rng.uniform(0.0, 15.0, n)
    arr["Y"] = rng.uniform(0.0, 15.0, n)
    arr["HeightAboveGround"] = rng.uniform(*hag_range, n)
    arr["TreeID"] = tree_id
    return arr


def _make_multi_tree_points(n_trees: int = 3, n_per_tree: int = 40, seed: int = 0):
    """Concatenated points for multiple trees with distinct spatial clusters."""
    rng = np.random.default_rng(seed)
    parts = []
    for tid in range(1, n_trees + 1):
        cx = rng.uniform(0.0, 100.0)
        cy = rng.uniform(0.0, 100.0)
        pts = _make_tree_points(n_per_tree, tree_id=tid, seed=seed + tid)
        pts["X"] += cx
        pts["Y"] += cy
        parts.append(pts)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# _tree_metrics — unit tests (no PDAL/TileDB)
# ---------------------------------------------------------------------------


def test_tree_metrics_returns_list_of_dicts():
    pts = _make_tree_points()
    records = _tree_metrics(pts)
    assert isinstance(records, list)
    assert len(records) > 0
    assert isinstance(records[0], dict)


def test_tree_metrics_no_tree_id_field_returns_empty():
    pts = np.zeros(10, dtype=[("X", np.float64), ("Y", np.float64)])
    assert _tree_metrics(pts) == []


def test_tree_metrics_excludes_tree_id_zero():
    pts = _make_tree_points(tree_id=0)
    records = _tree_metrics(pts)
    assert records == []


def test_tree_metrics_single_tree_record_count():
    pts = _make_tree_points(tree_id=1)
    records = _tree_metrics(pts)
    assert len(records) == 1


def test_tree_metrics_tree_id_in_record():
    pts = _make_tree_points(tree_id=7)
    records = _tree_metrics(pts)
    assert records[0]["tree_id"] == 7


def test_tree_metrics_height_is_max_hag():
    pts = _make_tree_points(hag_range=(5.0, 20.0))
    records = _tree_metrics(pts)
    expected_max = float(pts["HeightAboveGround"].max())
    assert records[0]["height"] == pytest.approx(expected_max, rel=1e-6)


def test_tree_metrics_n_points_correct():
    n = 55
    pts = _make_tree_points(n=n)
    records = _tree_metrics(pts)
    assert records[0]["n_points"] == n


def test_tree_metrics_centroid_within_point_bounds():
    pts = _make_tree_points()
    records = _tree_metrics(pts)
    cx = records[0]["centroid_x"]
    cy = records[0]["centroid_y"]
    assert pts["X"].min() <= cx <= pts["X"].max()
    assert pts["Y"].min() <= cy <= pts["Y"].max()


def test_tree_metrics_crown_area_positive():
    pts = _make_tree_points(n=50)
    records = _tree_metrics(pts)
    assert not np.isnan(records[0]["crown_area"])
    assert records[0]["crown_area"] > 0.0


def test_tree_metrics_crown_radius_consistent():
    """crown_radius == sqrt(crown_area / π)."""
    pts = _make_tree_points(n=50)
    records = _tree_metrics(pts)
    r = records[0]
    if not np.isnan(r["crown_area"]):
        expected_r = float(np.sqrt(r["crown_area"] / np.pi))
        assert r["crown_radius"] == pytest.approx(expected_r, rel=1e-5)


def test_tree_metrics_crown_fraction_reduces_area():
    """Upper-crown restriction (fraction=0.8) must give smaller area than full crown."""
    pts = _make_tree_points(n=100)
    records_full = _tree_metrics(pts, crown_fraction=0.0)
    records_upper = _tree_metrics(pts, crown_fraction=0.8)
    # Upper-crown area ≤ full-crown area
    a_full = records_full[0]["crown_area"]
    a_upper = records_upper[0]["crown_area"]
    if not (np.isnan(a_full) or np.isnan(a_upper)):
        assert a_upper <= a_full + 1e-6


def test_tree_metrics_too_few_points_for_hull():
    """With fewer than 3 points, crown metrics should be NaN."""
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("HeightAboveGround", np.float64),
        ("TreeID", np.int64),
    ]
    pts = np.zeros(2, dtype=dtype)
    pts["HeightAboveGround"] = [5.0, 10.0]
    pts["TreeID"] = 1
    records = _tree_metrics(pts)
    assert len(records) == 1
    assert np.isnan(records[0]["crown_area"])


def test_tree_metrics_multiple_trees():
    pts = _make_multi_tree_points(n_trees=3)
    records = _tree_metrics(pts)
    assert len(records) == 3
    ids = {r["tree_id"] for r in records}
    assert ids == {1, 2, 3}


def test_tree_metrics_required_keys():
    pts = _make_tree_points()
    records = _tree_metrics(pts)
    required = {
        "tree_id",
        "centroid_x",
        "centroid_y",
        "height",
        "min_point_height",
        "crown_area",
        "crown_radius",
        "n_points",
    }
    assert required.issubset(records[0].keys())


# ---------------------------------------------------------------------------
# segment_trees — integration (needs provider fixture)
# ---------------------------------------------------------------------------


def test_segment_trees_returns_tuple(provider):
    pts, trees = segment_trees(provider, bbox=BBOX, year=YEAR, min_height=2.0)
    assert isinstance(trees, pd.DataFrame)
    assert isinstance(pts, np.ndarray)


def test_segment_trees_dataframe_columns(provider):
    _, trees = segment_trees(provider, bbox=BBOX, year=YEAR, min_height=2.0)
    if len(trees) > 0:
        for col in ("tree_id", "height", "crown_area", "n_points"):
            assert col in trees.columns


def test_segment_trees_height_positive(provider):
    _, trees = segment_trees(provider, bbox=BBOX, year=YEAR, min_height=2.0)
    if len(trees) > 0:
        assert (trees["height"] > 0).all()


def test_segment_trees_sorted_descending_height(provider):
    _, trees = segment_trees(provider, bbox=BBOX, year=YEAR, min_height=2.0)
    if len(trees) > 1:
        assert list(trees["height"]) == sorted(trees["height"], reverse=True)


def test_segment_trees_tree_ids_positive(provider):
    pts, trees = segment_trees(provider, bbox=BBOX, year=YEAR, min_height=2.0)
    if len(trees) > 0:
        assert (trees["tree_id"] > 0).all()
        # TreeID=0 (unassigned) should not appear in the trees DataFrame
        assert 0 not in trees["tree_id"].values
