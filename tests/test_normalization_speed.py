"""Faster height normalization must not change CHM results."""

import numpy as np
import pytest

from alsdb.methods._surface import CHMConfig, reconstruct_chm, surface_mask
from alsdb.methods._terrain import TerrainModel, normalize_points


def _scene():
    from tests.test_surface_pipeline import scene

    return scene()


@pytest.mark.parametrize("method", ["max", "pitfree"])
def test_candidate_only_normalization_matches_full(method):
    arr = _scene()
    config = CHMConfig(method=method)
    bbox = (0.0, 0.0, 10.0, 10.0)
    full = reconstruct_chm(arr, bbox, 1.0, config)
    model = TerrainModel(arr[np.isin(arr["Classification"], config.ground_classes)])
    subset = normalize_points(arr[surface_mask(arr, config)], model=model)
    fast = reconstruct_chm(arr, bbox, 1.0, config, prepared=subset)
    for name in full:
        np.testing.assert_array_equal(full[name], fast[name], err_msg=name)


def test_tin_locator_matches_scipy_delaunay_on_scattered_points():
    """Delaunator TIN + grid locator give scipy's triangles and hull test.

    Random float points have a unique Delaunay triangulation, so the triangle
    containing each query must have the same three vertices in both.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(3)
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
    ground = np.zeros(4000, dtype=dtype)
    ground["X"], ground["Y"] = rng.uniform(0, 100, (2, 4000))
    ground["Z"] = 50 + 0.1 * ground["X"] + rng.normal(0, 0.2, 4000)
    model = TerrainModel(ground)
    local = model.xy - model._origin
    reference = Delaunay(local)
    assert len(model.tin.simplices) == len(reference.simplices)
    xy = rng.uniform(-5, 105, (20000, 2)) - model._origin  # includes points outside the hull
    ours, theirs = model._find_simplex(xy), reference.find_simplex(xy)
    np.testing.assert_array_equal(ours < 0, theirs < 0)
    inside = ours >= 0
    np.testing.assert_array_equal(
        np.sort(model.tin.simplices[ours[inside]], axis=1),
        np.sort(reference.simplices[theirs[inside]], axis=1),
    )
    z, _, outside = model.evaluate(xy[inside][:5] + model._origin)
    t = reference.transform[theirs[inside][:5]]
    b = np.einsum("ijk,ik->ij", t[:, :2], xy[inside][:5] - t[:, 2])
    expected = (
        np.column_stack((b, 1 - b.sum(1))) * model.z[reference.simplices[theirs[inside][:5]]]
    ).sum(1)
    np.testing.assert_allclose(z, expected, atol=1e-9)
    assert not outside.any()


def test_tin_keeps_all_ground_points_at_utm_coordinates():
    """Dense ground at real UTM offsets must not lose vertices to roundoff.

    Triangulating absolute coordinates (x ~ 6e5, y ~ 5.6e6) made Qhull treat
    most centimetre-spaced returns as coplanar and drop them, and returned an
    overlapping (invalid) triangulation.
    """
    rng = np.random.default_rng(7)
    n = 200_000
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
    ground = np.zeros(n, dtype=dtype)
    # ~10 points/m2 on a 140 m square, quantised to 1 cm like LAS coordinates
    ground["X"] = np.round(601_500 + rng.uniform(0, 140, n), 2)
    ground["Y"] = np.round(5_659_500 + rng.uniform(0, 140, n), 2)
    ground["Z"] = 350 + 0.05 * (ground["X"] - 601_500) - 0.02 * (ground["Y"] - 5_659_500)
    model = TerrainModel(ground)
    assert len(np.unique(model.tin.simplices)) == len(model.xy)

    q = np.column_stack(
        (601_500 + rng.uniform(5, 135, 5000), 5_659_500 + rng.uniform(5, 135, 5000))
    )
    z, _, outside = model.evaluate(q)
    assert not outside.any()
    expected = 350 + 0.05 * (q[:, 0] - 601_500) - 0.02 * (q[:, 1] - 5_659_500)
    np.testing.assert_allclose(z, expected, atol=1e-6)
    np.testing.assert_allclose(
        model.slope(q), np.degrees(np.arctan(np.hypot(0.05, 0.02))), atol=1e-6
    )


def _rough_ground(n=4000, seed=5):
    rng = np.random.default_rng(seed)
    ground = np.zeros(n, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    ground["X"], ground["Y"] = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    ground["Z"] = 0.1 * ground["X"] + rng.normal(0, 0.2, n)
    ground[:50] = ground[50:100]  # duplicate XY returns
    ground["Z"][:50] += 1.0
    return ground, rng


def test_cells_are_grouped_as_by_a_stable_sort(monkeypatch):
    from alsdb.methods import _terrain

    rng = np.random.default_rng(0)
    cell, values = rng.integers(0, 50, 5000), np.arange(5000)
    expected = values[np.argsort(cell, kind="stable")]
    np.testing.assert_array_equal(_terrain._group_by_cell(cell, values, 50), expected)
    monkeypatch.setattr(_terrain, "_coo_tocsr", None)  # SciPy without the private routine
    np.testing.assert_array_equal(_terrain._group_by_cell(cell, values, 50), expected)


def test_terrain_model_merges_duplicate_ground_as_numpy_unique():
    ground, _ = _rough_ground()
    model = TerrainModel(ground)
    xy = np.column_stack((ground["X"], ground["Y"]))
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    np.testing.assert_array_equal(model.xy, unique)
    np.testing.assert_array_equal(
        model.z, np.bincount(inverse.ravel(), weights=ground["Z"]) / np.bincount(inverse.ravel())
    )


def test_skipping_distances_gives_the_same_heights_and_flags():
    ground, rng = _rough_ground()
    q = rng.uniform(-20, 120, (3000, 2))  # inside and outside the TIN
    for options in ({}, {"extrapolate": True}, {"max_distance": 3.0}, {"method": "idw"}):
        z, distance, outside = TerrainModel(ground).evaluate(q, **options)
        z2, distance2, outside2 = TerrainModel(ground).evaluate(q, distances=False, **options)
        np.testing.assert_array_equal(z, z2)
        np.testing.assert_array_equal(outside, outside2)
        known = np.isfinite(distance2)
        np.testing.assert_array_equal(distance[known], distance2[known])
    assert "tree" not in TerrainModel(ground).__dict__  # built on first use only
    lazy = TerrainModel(ground)
    lazy.evaluate(q[:10] * 0 + 50.0, distances=False)
    assert "tree" not in lazy.__dict__


def test_repeated_queries_reuse_the_triangle_lookup(monkeypatch):
    ground, rng = _rough_ground()
    model = TerrainModel(ground)
    calls = []
    find = model.tin.find_simplex
    monkeypatch.setattr(model.tin, "find_simplex", lambda xy: calls.append(1) or find(xy))
    q = rng.uniform(10, 90, (500, 2))
    z, _, _ = model.evaluate(q, distances=False)
    slope = model.slope(q)
    z_again, _, _ = model.evaluate(q.copy(), distances=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(z, z_again)
    model.evaluate(q + 1.0, distances=False)
    assert len(calls) == 2 and np.isfinite(slope).any()
