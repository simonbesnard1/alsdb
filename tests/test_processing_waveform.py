# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for waveform.py — pure-function unit tests (no TileDB required)."""

import numpy as np
import pytest

from alsdb.processing.waveform import (
    WaveformResult,
    _build_histogram,
    _canopy_cover,
    _detect_ground,
    _rh_metrics,
)

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
