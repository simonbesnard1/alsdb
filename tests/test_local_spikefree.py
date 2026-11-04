"""Locally adaptive spike-free freeze distances (Fischer et al. 2024)."""

import numpy as np
import pytest

from alsdb.methods._surface import CHMConfig, reconstruct_chm, surface_mask
from alsdb.methods.spikefree import (
    FreezeGrid,
    local_freeze_distance,
    local_freeze_grid,
    rasterize_spikefree,
)


def test_freeze_formula_matches_authors_workflow_and_lidr():
    # m / ln(s * pd + o) + 1 with m 3.1, s 1.75, o 2.1: 2.12 m at 8 pulses/m2
    assert local_freeze_distance(8.0) == pytest.approx(3.1 / np.log(1.75 * 8 + 2.1) + 1)
    assert local_freeze_distance(8.0) == pytest.approx(2.12, abs=0.01)
    pd = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 30.0])
    lidr = 1.42 + 1.96 / np.sqrt(pd)  # lidR lspikefree() linear form
    np.testing.assert_allclose(local_freeze_distance(pd), lidr, atol=0.1)
    # Clamped to 0.5-100 pulses/m2
    assert local_freeze_distance(0.0) == local_freeze_distance(0.5)
    assert local_freeze_distance(500.0) == local_freeze_distance(100.0)


def test_local_freeze_grid_counts_pulses_on_aligned_cells():
    pulses = np.zeros(50, dtype=[("X", "f8"), ("Y", "f8")])
    pulses["X"], pulses["Y"] = 101.0, 108.0  # all in cell x 100-105, y 105-110
    grid = local_freeze_grid(pulses, (101.0, 101.0, 109.0, 109.0))
    assert (grid.x0, grid.y1, grid.cell) == (100.0, 110.0, 5.0)
    assert grid.distance.shape == (2, 2)
    assert grid.distance[0, 0] == pytest.approx(local_freeze_distance(2.0))  # 50 / 25 m2
    assert grid.distance[1, 1] == pytest.approx(local_freeze_distance(0.0))


def _canopy_over_pit(x0):
    # Four 10 m returns 2 m apart around a 0 m return in the middle.
    return np.array(
        [[x0, 0, 10], [x0 + 2, 0, 10], [x0, 2, 10], [x0 + 2, 2, 10], [x0 + 1, 1, 0]], float
    )


def _points(xyz):
    pts = np.zeros(len(xyz), dtype=[("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")])
    pts["X"], pts["Y"], pts["HeightAboveGround"] = xyz.T
    return pts


def test_uniform_grid_equals_scalar_freeze_distance():
    pytest.importorskip("alsdb.methods._spikefree_native")
    rng = np.random.default_rng(0)
    pts = _points(np.column_stack((rng.uniform(0, 20, (300, 2)), rng.uniform(0, 25, 300))))
    scalar = rasterize_spikefree(pts, (0, 0, 20, 20), 1.0, freeze_distance=2.0)
    grid = FreezeGrid(np.full((4, 4), 2.0), 0.0, 20.0, 5.0)
    local = rasterize_spikefree(pts, (0, 0, 20, 20), 1.0, freeze_distance=grid)
    np.testing.assert_array_equal(scalar, local)


def test_each_side_uses_its_own_freeze_distance():
    pytest.importorskip("alsdb.methods._spikefree_native")
    pts = _points(np.vstack((_canopy_over_pit(0.0), _canopy_over_pit(10.0))))
    # Left: 4 m freezes the 2 m canopy triangles; right: 0.1 m freezes nothing.
    grid = FreezeGrid(np.array([[4.0, 0.1]]), 0.0, 5.0, 6.0)
    out = rasterize_spikefree(pts, (0, 0, 12, 2), 0.5, freeze_distance=grid)
    left, right = out[:, :4], out[:, 20:]
    np.testing.assert_allclose(left, 10)
    assert right.min() < 5


def test_reconstruct_chm_local_and_first_returns():
    pytest.importorskip("alsdb.methods._spikefree_native")
    from tests.test_surface_pipeline import scene

    arr = scene()
    result = reconstruct_chm(
        arr, (0, 0, 10, 10), 1, CHMConfig(method="spikefree", freeze_distance="local")
    )
    np.testing.assert_allclose(result["chm"], 10, atol=1e-5)

    arr["ReturnNumber"][-5:] = 2
    config = CHMConfig(method="spikefree", spikefree_first_returns=True)
    assert (~surface_mask(arr, config)[-5:]).all()
    assert surface_mask(arr, CHMConfig(method="spikefree"))[-5:].all()


def test_local_freeze_needs_native_spikefree():
    with pytest.raises(ValueError, match="local"):
        CHMConfig(method="lastools", freeze_distance="local")
    CHMConfig(method="spikefree", freeze_distance="local")
