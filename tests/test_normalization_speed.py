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


def test_sorted_tin_lookup_matches_scipy_on_scattered_points():
    rng = np.random.default_rng(3)
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
    ground = np.zeros(4000, dtype=dtype)
    ground["X"], ground["Y"] = rng.uniform(0, 100, (2, 4000))
    ground["Z"] = 50 + 0.1 * ground["X"] + rng.normal(0, 0.2, 4000)
    model = TerrainModel(ground)
    xy = rng.uniform(-5, 105, (20000, 2))  # includes points outside the hull
    np.testing.assert_array_equal(model._find_simplex(xy), model.tin.find_simplex(xy))


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
