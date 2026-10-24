"""Faster height normalization must not change CHM results."""

import numpy as np
import pytest

from alsdb.processing._surface import CHMConfig, canopy_mask, reconstruct_chm
from alsdb.processing._terrain import TerrainModel, normalize_points


def _scene():
    from tests.test_surface_pipeline import scene

    return scene()


@pytest.mark.parametrize("method", ["max", "pitfree", "highest_subcell_tin"])
def test_candidate_only_normalization_matches_full(method):
    arr = _scene()
    extra = {
        "pitfree": {"pitfree_max_distance": "auto"},
        "highest_subcell_tin": {"spikefree_max_distance": "auto"},
    }.get(method, {})
    config = CHMConfig(method=method, **extra)
    bbox = (0.0, 0.0, 10.0, 10.0)
    full = reconstruct_chm(arr, bbox, 1.0, config)
    model = TerrainModel(arr[arr["Classification"] == 2])
    subset = normalize_points(arr[canopy_mask(arr, config)], model=model)
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
