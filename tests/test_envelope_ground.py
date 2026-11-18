# SPDX-License-Identifier: EUPL-1.2
"""Robust-envelope ground classification on synthetic scenes."""

import numpy as np
import pytest

from alsdb.methods import envelope
from alsdb.methods.envelope import _Grid, envelope_ground, vegetation_evidence

DTYPE = np.dtype(
    [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1"),
     ("ReturnNumber", "u1"), ("NumberOfReturns", "u1")]
)  # fmt: skip


def scene(seed=0, n=12000, size=120.0, penetration=0.3):
    """Sloping terrain with a cliff step, a shrub patch and scattered low shrubs.

    In the patch (15 m across, 1.5 m tall) a share *penetration* of the pulses
    reaches the ground as a second return; the rest end on the shrub tops.
    Returns the points, the true ground mask and the terrain function.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)

    def terrain(x, y):
        return 0.3 * x + 2.5 * (x > 80)  # 17 deg slope, a 2.5 m step at x = 80

    z = terrain(x, y) + rng.normal(0, 0.03, n)
    pts = np.zeros(n, DTYPE)
    pts["X"], pts["Y"], pts["Z"] = x, y, z
    pts["Classification"] = 1
    pts["ReturnNumber"] = 1
    pts["NumberOfReturns"] = 1
    truth = np.ones(n, bool)
    patch = np.hypot(x - 30, y - 60) < 8
    pts["Z"][patch] += 1.5  # shrub tops
    truth[patch] = False
    through = patch & (rng.uniform(size=n) < penetration)
    pts["NumberOfReturns"][through] = 2  # first of two returns
    under = pts[through].copy()  # the second return, on the ground
    under["Z"] = terrain(under["X"], under["Y"]) + rng.normal(0, 0.03, len(under))
    under["ReturnNumber"] = 2
    shrubs = (~patch) & (rng.uniform(size=n) < 0.05)  # scattered 0.3 m shrubs, two returns
    pts["Z"][shrubs] += 0.3
    pts["NumberOfReturns"][shrubs] = 2
    truth[shrubs] = False
    pts = np.concatenate([pts, under])
    truth = np.concatenate([truth, np.ones(len(under), bool)])
    return pts, truth, terrain


def dtm_error(pts, ground, terrain, where):
    """Mean absolute error of a TIN of the *ground* points against *terrain* at the points *where*."""
    from scipy.interpolate import LinearNDInterpolator

    tin = LinearNDInterpolator(np.column_stack((pts["X"][ground], pts["Y"][ground])), pts["Z"][ground])
    z = tin(pts["X"][where], pts["Y"][where])
    ok = np.isfinite(z)
    return float(np.abs(z[ok] - terrain(pts["X"][where][ok], pts["Y"][where][ok])).mean())


def test_vegetation_evidence_is_high_in_the_patch_and_low_on_bare_ground():
    pts, truth, _ = scene()
    share = vegetation_evidence(pts, cell=5.0)
    patch = np.hypot(pts["X"] - 30, pts["Y"] - 60) < 5
    bare = np.hypot(pts["X"] - 100, pts["Y"] - 30) < 10
    assert share[patch].mean() > 0.25  # 30 % penetration, diluted by the 15 m window; above the 0.2 threshold
    assert share[bare].mean() < 0.15


def test_grid_sampling_reproduces_a_plane_exactly():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 50, 500), rng.uniform(0, 50, 500)
    grid = _Grid(x, y, 2.0)
    gx = grid.x0 + np.arange(grid.nx) * grid.cell
    gy = grid.y0 + np.arange(grid.ny) * grid.cell
    nodes = (2.0 * gx[None, :] - 0.5 * gy[:, None] + 3.0).ravel()
    assert np.allclose(grid.sample(nodes, x, y), 2.0 * x - 0.5 * y + 3.0)
    assert np.allclose(grid.sampling_matrix(x, y) @ nodes, 2.0 * x - 0.5 * y + 3.0)


def test_envelope_removes_the_shrub_tops_and_keeps_the_cliff():
    pts, truth, terrain = scene()
    # a deliberately poor seed: everything but the scattered shrubs, patch tops included
    patch = np.hypot(pts["X"] - 30, pts["Y"] - 60) < 8
    seed = (pts["NumberOfReturns"] == 1) | patch
    ground = envelope_ground(pts, seed=seed)
    tops = (np.hypot(pts["X"] - 30, pts["Y"] - 60) < 5) & ~truth  # the rim is a transition zone
    assert ground[tops].mean() < 0.3, "most shrub tops go; a few stay where no return reached the ground nearby"
    assert ground[patch & truth].mean() > 0.9, "the ground returns under the patch are kept"
    # what matters: the terrain model under the patch is close to the terrain, not 1.5 m up
    assert dtm_error(pts, ground, terrain, tops) < 0.5
    assert dtm_error(pts, seed, terrain, tops) > 1.0  # the seed alone would put it on the shrubs
    near_step = (np.abs(pts["X"] - 80) < 3) & truth
    assert ground[near_step].mean() > 0.9, "bare ground on both sides of the step is kept"
    assert ground[truth].mean() > 0.95
    assert ground[~truth].mean() < 0.3  # the 0.3 m shrubs mostly go, a few sit within `keep`


def test_a_closed_patch_without_any_ground_return_is_kept():
    """Known limitation: with no return under a patch the lower envelope has nothing to settle on."""
    pts, truth, _ = scene(penetration=0.0)
    patch = np.hypot(pts["X"] - 30, pts["Y"] - 60) < 6
    ground = envelope_ground(pts, seed=truth | patch)
    assert ground[patch].mean() > 0.9


def test_envelope_without_return_fields_falls_back_to_a_lower_envelope():
    pts, truth, terrain = scene()
    plain = pts[["X", "Y", "Z", "Classification"]]
    patch = np.hypot(pts["X"] - 30, pts["Y"] - 60) < 8
    ground = envelope_ground(plain, seed=truth | patch)
    assert ground.dtype == bool and ground.shape == (len(pts),)
    # without multi-return evidence everything is treated as vegetated: lower envelope
    tops = (np.hypot(pts["X"] - 30, pts["Y"] - 60) < 5) & ~truth
    assert ground[tops].mean() < 0.3
    assert dtm_error(pts, ground, terrain, tops) < 0.5
    assert ground[truth].mean() > 0.9


def test_noise_is_never_ground_and_empty_input_is_fine():
    pts, truth, _ = scene(n=3000)
    pts["Classification"][:50] = 7
    ground = envelope_ground(pts, seed=np.ones(len(pts), bool))
    assert not ground[:50].any()
    assert envelope_ground(pts[:0], seed=np.zeros(0, bool)).shape == (0,)
    with pytest.raises(ValueError, match="unknown envelope parameters"):
        envelope_ground(pts, seed=truth, nonsense=1)


def test_blocked_fit_matches_a_single_block_fit():
    pts, truth, _ = scene(n=6000, size=90.0)
    x, y, z = (pts[k].astype(float) for k in "XYZ")
    grid = _Grid(x, y, 1.0)
    located = grid._locate(x, y)
    w = truth.astype(float)
    whole = envelope.fit_surface(grid, located, z, w, 0.1, block=10_000, anchor=0.0)
    blocked = envelope.fit_surface(grid, located, z, w, 0.1, block=40, overlap=12, workers=2, anchor=0.0)
    # free block boundaries are hidden by the overlap blending: agreement at supported points;
    # inside the unsupported patch both are extrapolations and may differ
    d = np.abs(grid.sample(whole, x, y) - grid.sample(blocked, x, y))
    assert d[w > 0].max() < 0.05
    assert d.mean() < 0.01


def test_anchor_changes_little_where_the_surface_is_supported():
    pts, truth, _ = scene(n=6000, size=90.0)
    x, y, z = (pts[k].astype(float) for k in "XYZ")
    grid = _Grid(x, y, 1.0)
    located = grid._locate(x, y)
    w = truth.astype(float)
    free = envelope.fit_surface(grid, located, z, w, 0.1, block=40, overlap=12, anchor=0.0, x=x, y=y)
    anchored = envelope.fit_surface(grid, located, z, w, 0.1, block=40, overlap=12, anchor=1e-3, x=x, y=y)
    d = np.abs(grid.sample(free, x, y) - grid.sample(anchored, x, y))
    assert d[w > 0].mean() < 0.01


def test_a_grid_split_into_blocks_is_anchored(monkeypatch):
    """Revision 2: a small grid that still spans several blocks gets the coarse anchor."""
    pts, truth, _ = scene(n=4000, size=90.0)
    x, y, z = (pts[k].astype(float) for k in "XYZ")
    grid = _Grid(x, y, 1.0)
    calls = []
    real = envelope._solve_block
    monkeypatch.setattr(envelope, "_solve_block", lambda job: calls.append(job[8] is not None) or real(job))
    envelope.fit_surface(grid, grid._locate(x, y), z, truth.astype(float), 0.1, block=40, overlap=12,
                         anchor=1e-3, x=x, y=y)
    blocks = [c for c in calls]
    assert blocks and all(blocks[1:]), "every block of the fine grid carries the anchor"


def test_non_seed_points_are_added_only_in_seed_gaps():
    """Revision 3: next to kept seed ground, low returns within `keep` are not added (grass);
    in a gap of the seed, ground returns are."""
    pts, truth, _ = scene(n=8000, size=90.0)
    seed = truth.copy()
    hole = np.hypot(pts["X"] - 60, pts["Y"] - 30) < 6  # the seed misses this ground
    seed[hole] = False
    near = (~truth) & ~hole  # the 0.3 m shrubs next to seed ground ...
    pts["NumberOfReturns"][near] = 1  # ... as single returns, like grass: last-return candidates
    ground = envelope_ground(pts, seed=seed, keep=0.4)  # a keep loose enough to reach the shrubs
    inner = np.hypot(pts["X"] - 60, pts["Y"] - 30) < 3.5  # farther than `gap` (2 m) from the rim
    assert ground[inner & truth].mean() > 0.9, "ground in the seed gap is recovered"
    assert ground[near].mean() < 0.05, "shrubs beside kept seed ground are not added"
    loose = envelope_ground(pts, seed=seed, keep=0.4, gap=0.0)
    assert loose[near].mean() > 0.5, "without the gap rule they would be"
