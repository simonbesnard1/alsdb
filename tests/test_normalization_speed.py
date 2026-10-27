"""Faster height normalization must not change CHM results."""

import numpy as np
import pytest

from alsdb.processing._surface import CHMConfig, reconstruct_chm, surface_mask
from alsdb.processing._terrain import TerrainModel, normalize_points


def _scene():
    from tests.test_surface_pipeline import scene

    return scene()


@pytest.mark.parametrize("method", ["max", "pitfree", "highest_subcell_tin"])
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
