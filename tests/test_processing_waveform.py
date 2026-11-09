# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for waveform.py — pure-function unit tests (no TileDB required)."""

import numpy as np
import pytest

from alsdb.products.waveform import (
    WaveformResult,
    _build_histogram,
    _canopy_cover,
    _detect_ground,
    _effective_query_radius,
    _rh_metrics,
    waveform_from_points,
)

# ---------------------------------------------------------------------------
# _effective_query_radius
# ---------------------------------------------------------------------------


def test_effective_query_radius_no_weighting_returns_footprint_radius():
    r = _effective_query_radius(
        footprint_radius=12.5, gaussian_beam_weighting=False, sigma_beam=None
    )
    assert r == pytest.approx(12.5)


def test_effective_query_radius_default_sigma_beam_extends_past_footprint():
    # sigma_beam defaults to 0.44 * footprint_radius = 5.5 m (GEDI simulator);
    # beam_extent_sigma=3.0 -> 16.5 m, larger than the nominal 12.5 m footprint.
    r = _effective_query_radius(
        footprint_radius=12.5,
        gaussian_beam_weighting=True,
        sigma_beam=None,
        beam_extent_sigma=3.0,
    )
    assert r == pytest.approx(16.5)
    assert r > 12.5


def test_effective_query_radius_explicit_sigma_beam():
    # 3 * sigma_beam = 15.0, larger than footprint_radius=5.0, so it wins.
    r = _effective_query_radius(
        footprint_radius=5.0,
        gaussian_beam_weighting=True,
        sigma_beam=5.0,
        beam_extent_sigma=3.0,
    )
    assert r == pytest.approx(15.0)


def test_effective_query_radius_never_shrinks_below_footprint_radius():
    # A tiny sigma_beam shouldn't shrink the query below the nominal footprint.
    r = _effective_query_radius(
        footprint_radius=12.5,
        gaussian_beam_weighting=True,
        sigma_beam=0.1,
        beam_extent_sigma=3.0,
    )
    assert r == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# _build_histogram
# ---------------------------------------------------------------------------


def test_build_histogram_returns_centres_and_hist():
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    weights = np.ones(5)
    centres, hist = _build_histogram(z, weights, z_step=1.0)
    assert centres.shape == hist.shape
    assert len(centres) >= 1


def test_build_histogram_energy_preserved():
    z = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 3.0])
    _, hist = _build_histogram(z, weights, z_step=0.5)
    assert hist.sum() == pytest.approx(weights.sum(), rel=1e-6)


def test_build_histogram_single_point():
    z = np.array([5.0])
    weights = np.array([1.0])
    centres, hist = _build_histogram(z, weights, z_step=0.15)
    # Single point → single non-zero bin
    assert hist.sum() == pytest.approx(1.0)


def test_build_histogram_bins_ascending():
    z = np.linspace(0.0, 10.0, 50)
    centres, _ = _build_histogram(z, np.ones(50), z_step=0.15)
    assert np.all(np.diff(centres) > 0)


def test_build_histogram_uniform_weights():
    z = np.linspace(0.0, 5.0, 100)
    centres, hist = _build_histogram(z, np.ones(100), z_step=0.5)
    # All bins should have similar counts
    nonzero = hist[hist > 0]
    assert nonzero.std() / nonzero.mean() < 0.5


# ---------------------------------------------------------------------------
# _detect_ground
# ---------------------------------------------------------------------------


def _two_peak_waveform():
    """Waveform with a clear ground peak (low elevation) and canopy peak (high)."""
    z_bins = np.linspace(0.0, 30.0, 200)
    waveform = np.zeros(200)
    # Ground peak at z ≈ 2 m (index ~13)
    gnd_idx = np.argmin(np.abs(z_bins - 2.0))
    waveform[gnd_idx] = 0.5
    # Canopy peak at z ≈ 20 m
    can_idx = np.argmin(np.abs(z_bins - 20.0))
    waveform[can_idx] = 1.0
    waveform /= waveform.sum()
    return z_bins, waveform


def test_detect_ground_finds_lowest_peak():
    z_bins, waveform = _two_peak_waveform()
    z_gnd = _detect_ground(waveform, z_bins)
    # Ground peak is at ~2 m; should be well below canopy peak at 20 m
    assert z_gnd < 10.0


def test_detect_ground_fallback_on_flat_waveform():
    """When no peaks are found, falls back to the lowest non-negligible bin."""
    z_bins = np.linspace(5.0, 25.0, 100)
    waveform = np.zeros(100)
    waveform[10] = 0.01  # single small value, may not qualify as a peak
    waveform /= waveform.max()
    z_gnd = _detect_ground(waveform, z_bins)
    # Should return something within the z_bins range
    assert z_bins[0] <= z_gnd <= z_bins[-1]


def test_detect_ground_returns_float():
    z_bins = np.linspace(0.0, 20.0, 50)
    waveform = np.zeros(50)
    waveform[5] = 1.0
    waveform /= waveform.sum()
    result = _detect_ground(waveform, z_bins)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _rh_metrics
# ---------------------------------------------------------------------------


def _simple_waveform_above_ground(z_ground=0.0, n=100, height=20.0):
    """Uniform energy waveform from z_ground to z_ground+height."""
    z_bins = np.linspace(z_ground, z_ground + height, n)
    waveform = np.ones(n) / n
    return z_bins, waveform


def test_rh_metrics_rh0_near_zero():
    z_bins, waveform = _simple_waveform_above_ground(z_ground=0.0)
    rh = _rh_metrics(waveform, z_bins, z_ground=0.0, levels=(0, 50, 100))
    assert rh[0] == pytest.approx(0.0, abs=1.0)


def test_rh_metrics_rh100_near_max_height():
    height = 20.0
    z_bins, waveform = _simple_waveform_above_ground(z_ground=0.0, height=height)
    rh = _rh_metrics(waveform, z_bins, z_ground=0.0, levels=(0, 50, 100))
    assert rh[100] == pytest.approx(height, abs=1.0)


def test_rh_metrics_rh50_near_midpoint():
    height = 20.0
    z_bins, waveform = _simple_waveform_above_ground(z_ground=0.0, height=height)
    rh = _rh_metrics(waveform, z_bins, z_ground=0.0, levels=(50,))
    assert rh[50] == pytest.approx(height / 2, abs=2.0)


def test_rh_metrics_ordering():
    """RH values must be non-decreasing."""
    z_bins, waveform = _simple_waveform_above_ground()
    levels = (0, 10, 25, 50, 75, 90, 100)
    rh = _rh_metrics(waveform, z_bins, z_ground=0.0, levels=levels)
    values = [rh[lv] for lv in levels if not np.isnan(rh[lv])]
    assert values == sorted(values)


def test_rh_metrics_empty_above_ground_returns_nan():
    """If no energy above ground index, all RH should be NaN."""
    z_bins = np.array([0.0, 1.0, 2.0, 3.0])
    waveform = np.array([0.0, 0.0, 0.0, 0.0])
    rh = _rh_metrics(waveform, z_bins, z_ground=0.0, levels=(0, 50, 100))
    assert all(np.isnan(v) for v in rh.values())


def test_rh_metrics_ground_above_all_bins():
    """Ground index beyond array length → all NaN."""
    z_bins = np.linspace(0.0, 5.0, 20)
    waveform = np.ones(20) / 20.0
    rh = _rh_metrics(waveform, z_bins, z_ground=10.0, levels=(50,))
    assert np.isnan(rh[50])


def _waveform_with_energy_below_ground_peak():
    """
    A ground return spread across several bins straddling z_ground (pulse
    broadening), plus a canopy peak above - the scenario where the old
    "sum from ground_idx upward" denominator and the new "full waveform
    total" denominator actually disagree, unlike every waveform above
    (all of which have z_ground at bin 0, where the two conventions
    coincide).
    """
    z_bins = np.linspace(-2.0, 20.0, 220)  # z_step = 0.1
    waveform = np.zeros(220)
    # Ground return centred at z=0, spread across bins below AND above it.
    gnd_centre = np.argmin(np.abs(z_bins - 0.0))
    waveform[gnd_centre - 5 : gnd_centre + 5] = 0.02  # 10 bins, half below z_ground=0
    # Canopy return above ground.
    can_centre = np.argmin(np.abs(z_bins - 15.0))
    waveform[can_centre - 2 : can_centre + 2] = 0.05
    waveform /= waveform.sum()
    return z_bins, waveform


def test_rh_metrics_accounts_for_energy_below_ground_peak():
    """
    Cumulative fraction must be computed against the FULL waveform total
    (matching _canopy_cover's denominator), not just the ground-peak-and-above
    subset - otherwise RH100 would map to less than 100% of the true energy.
    """
    z_bins, waveform = _waveform_with_energy_below_ground_peak()
    z_ground = 0.0
    rh = _rh_metrics(waveform, z_bins, z_ground, levels=(0, 100))
    # RH100 must reach the true canopy-top height, not be truncated because
    # some energy was excluded from the denominator.
    assert rh[100] == pytest.approx(15.0, abs=1.0)


def test_rh_metrics_low_percentile_can_be_negative():
    """
    Real energy below the detected ground peak (pulse/slope broadening) means
    a low RH percentile can legitimately fall below the peak itself - an
    expected physical effect (also seen in real GEDI/LVIS RH products), not
    something this function should clamp to zero.
    """
    z_bins, waveform = _waveform_with_energy_below_ground_peak()
    rh = _rh_metrics(waveform, z_bins, z_ground=0.0, levels=(0, 2))
    # RH0 is the bottom of the signal, below the ground peak (GEDI rh_000)
    assert rh[0] == pytest.approx(z_bins[np.flatnonzero(waveform > 0)[0]])
    assert rh[0] <= rh[2] <= 0.0


# ---------------------------------------------------------------------------
# _canopy_cover
# ---------------------------------------------------------------------------


def test_canopy_cover_all_above_threshold():
    z_bins = np.array([5.0, 10.0, 15.0, 20.0])
    waveform = np.array([0.25, 0.25, 0.25, 0.25])
    cover = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0)
    assert cover == pytest.approx(1.0, abs=0.01)


def test_canopy_cover_none_above_threshold():
    z_bins = np.array([0.5, 1.0, 1.5])
    waveform = np.array([0.4, 0.4, 0.2])
    cover = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0)
    assert cover == pytest.approx(0.0, abs=0.01)


def test_canopy_cover_partial():
    z_bins = np.array([1.0, 3.0, 5.0, 7.0])
    waveform = np.array([0.5, 0.0, 0.25, 0.25])
    cover = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0)
    assert 0.0 < cover < 1.0


def test_canopy_cover_zero_waveform():
    z_bins = np.array([1.0, 5.0, 10.0])
    waveform = np.zeros(3)
    cover = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0)
    assert cover == pytest.approx(0.0)


def test_canopy_cover_in_unit_interval():
    rng = np.random.default_rng(42)
    z_bins = np.linspace(0.0, 30.0, 100)
    waveform = rng.uniform(0, 1, 100)
    waveform /= waveform.sum()
    cover = _canopy_cover(waveform, z_bins, z_ground=2.0, threshold=2.0)
    assert 0.0 <= cover <= 1.0


def test_canopy_cover_default_rv_rg_matches_naive_energy_ratio():
    """rv_rg=1.0 (default) must reproduce the plain energy-ratio proxy."""
    z_bins = np.array([1.0, 3.0, 5.0, 7.0])
    waveform = np.array([0.5, 0.0, 0.25, 0.25])
    cover = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0)
    naive = float(waveform[z_bins >= 2.0].sum() / waveform.sum())
    assert cover == pytest.approx(naive)


def test_canopy_cover_rv_rg_below_one_increases_cover():
    """
    rv_rg < 1 (vegetation reflects less efficiently than ground) must
    up-weight the vegetation-energy contribution relative to the naive
    ratio, increasing the reported cover for the same energy split.
    """
    z_bins = np.array([1.0, 3.0, 5.0, 7.0])
    waveform = np.array([0.5, 0.0, 0.25, 0.25])
    cover_naive = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0, rv_rg=1.0)
    cover_corrected = _canopy_cover(waveform, z_bins, z_ground=0.0, threshold=2.0, rv_rg=0.5)
    assert cover_corrected > cover_naive


def test_canopy_cover_rv_rg_in_unit_interval():
    rng = np.random.default_rng(42)
    z_bins = np.linspace(0.0, 30.0, 100)
    waveform = rng.uniform(0, 1, 100)
    waveform /= waveform.sum()
    cover = _canopy_cover(waveform, z_bins, z_ground=2.0, threshold=2.0, rv_rg=0.6)
    assert 0.0 <= cover <= 1.0


# ---------------------------------------------------------------------------
# WaveformResult
# ---------------------------------------------------------------------------


def _make_result():
    z_bins = np.linspace(0.0, 30.0, 100)
    waveform = np.zeros(100)
    waveform[10] = 0.3
    waveform[70] = 0.7
    waveform /= waveform.sum()
    rh = {lv: float(lv * 0.25) for lv in range(101)}
    return WaveformResult(
        z_bins=z_bins,
        waveform=waveform,
        z_ground=z_bins[10],
        rh=rh,
        home=rh[50],
        cover=0.7,
        n_points=500,
        center_x=308_500.0,
        center_y=4_688_500.0,
    )


def test_waveform_result_rh_array_length():
    res = _make_result()
    arr = res.rh_array(levels=tuple(range(101)))
    assert len(arr) == 101


def test_waveform_result_rh_array_values():
    res = _make_result()
    arr = res.rh_array(levels=(0, 50, 100))
    assert arr[0] == pytest.approx(0.0)
    assert arr[1] == pytest.approx(12.5)
    assert arr[2] == pytest.approx(25.0)


def test_waveform_result_rh_array_missing_level_is_nan():
    rh = {50: 10.0}
    res = WaveformResult(
        z_bins=np.zeros(1),
        waveform=np.zeros(1),
        z_ground=0.0,
        rh=rh,
        home=10.0,
        cover=0.5,
        n_points=10,
        center_x=0.0,
        center_y=0.0,
    )
    arr = res.rh_array(levels=(25, 50, 75))
    assert np.isnan(arr[0])
    assert arr[1] == pytest.approx(10.0)
    assert np.isnan(arr[2])


def test_waveform_result_to_dict_keys():
    res = _make_result()
    d = res.to_dict()
    for key in ("center_x", "center_y", "z_ground", "home", "cover", "n_points"):
        assert key in d
    assert "rh50" in d
    assert "rh100" in d


def test_waveform_result_to_dict_values_match():
    res = _make_result()
    d = res.to_dict()
    assert d["cover"] == pytest.approx(res.cover)
    assert d["home"] == pytest.approx(res.home)
    assert d["n_points"] == res.n_points


def test_waveform_result_to_dict_no_arrays():
    """to_dict must return only scalar values (no numpy arrays)."""
    res = _make_result()
    d = res.to_dict()
    for v in d.values():
        assert not isinstance(v, np.ndarray)


# ---------------------------------------------------------------------------
# GEDI simulator conventions (gediRat / libclidar)
# ---------------------------------------------------------------------------


def _points(z, x=None, y=None, nret=None):
    n = len(z)
    rng = np.random.default_rng(0)
    return {"X": rng.uniform(-3, 3, n) if x is None else x, "Y": rng.uniform(-3, 3, n) if y is None else y,
            "Z": np.asarray(z, float), "Intensity": np.ones(n), "ReturnNumber": np.ones(n, int),
            "NumberOfReturns": np.ones(n, int) if nret is None else nret,
            "Classification": np.full(n, 2), "Withheld": np.zeros(n, bool)}  # fmt: skip


def test_measured_pulse_tail_lands_below_the_return():
    # GEDI pulses have a long trailing tail: later energy comes from lower down.
    r = waveform_from_points(_points(np.zeros(50)), 0.0, 0.0, beam_id="BEAM0000",
                             gaussian_beam_weighting=False, min_points=1)  # fmt: skip
    centroid = float(np.sum(r.z_bins * r.waveform))
    assert centroid < -0.2
    assert r.z_bins[np.argmax(r.waveform)] == pytest.approx(0.0, abs=0.15)


def test_rh0_is_the_bottom_of_the_signal():
    r = waveform_from_points(_points(np.r_[np.zeros(40), np.full(40, 15.0)]), 0.0, 0.0,
                             gaussian_beam_weighting=False, min_points=1)  # fmt: skip
    assert r.rh[0] < -1.0  # the pulse spreads the ground return below its peak
    assert r.rh[0] == pytest.approx(r.z_bins[np.flatnonzero(r.waveform > 0)[0]] - r.z_ground)


def test_density_cell_removes_overlap_double_counting():
    # Duplicating the pulses of one area (a flight-line overlap) must not change
    # the waveform when each point is divided by the pulses in its cell.
    x = np.repeat(np.arange(-4.5, 5, 1.5) + 0.7, 7)
    y = np.tile(np.arange(-4.5, 5, 1.5) + 0.7, 7)
    z = np.where(x > 0, 10.0, 0.0)
    once = _points(z, x, y)
    dup = x > 0
    twice = _points(np.r_[z, z[dup]], np.r_[x, x[dup]], np.r_[y, y[dup]])
    options = {"gaussian_beam_weighting": False, "min_points": 1}
    a = waveform_from_points(once, 0.0, 0.0, **options)
    b = waveform_from_points(twice, 0.0, 0.0, **options)
    np.testing.assert_allclose(a.waveform, b.waveform, atol=1e-12)
    c = waveform_from_points(twice, 0.0, 0.0, density_cell=None, **options)
    assert not np.allclose(a.waveform, c.waveform)


def test_point_grid_disc_matches_kdtree():
    from scipy.spatial import cKDTree

    from alsdb.products._tiling import PointGrid

    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 100, 20000), rng.uniform(0, 60, 20000)
    radius = 7.0
    grid = PointGrid(x, y, np.array([-radius, -radius]), radius)
    tree = cKDTree(np.column_stack((x, y)))
    for x0, y0 in rng.uniform([-5, -5], [105, 65], (200, 2)):
        expected = np.sort(tree.query_ball_point((x0, y0), radius))
        np.testing.assert_array_equal(grid.disc(x0, y0, radius), expected)
