# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for pavd.py — pure-function unit tests (no TileDB required).

Synthetic canopies are built directly as (hag, classification) arrays, the
same inputs _pavd_profile_from_hag consumes once a footprint has been
queried and ground-referenced -- mirrors gap.py's _compute_gap_grid unit
test style (test_processing_gap.py), which tests the pure math separately
from the TileDB-querying integration path.
"""

import numpy as np
import pytest

from alsdb.processing.gap import _LAI_K_DEFAULT, _gap_to_lai
from alsdb.processing.pavd import (
    _MIN_GROUND_POINTS_DEFAULT,
    _pavd_profile_from_hag,
    fit_transmittance_model,
)

_K = _LAI_K_DEFAULT


def _make_canopy(rng, pavd0, height, n_gnd=None, n_veg=60_000, gap_band=None):
    """
    Synthetic footprint with a KNOWN, uniform PAVD = pavd0 (m^2/m^3) from
    0 to `height`.

    For a canopy already ground-referenced (hag=0 is ground), Beer-Lambert
    gives the gap probability at height z as
        gap(z) = exp(-k * INTEGRAL_z^height pavd(z') dz')
               = exp(-k * pavd0 * (height - z))     [constant density]
    i.e. gap RISES from exp(-k*pavd0*height) at the ground to 1 at the top --
    the opposite direction from a plain truncated-exponential-in-z, which
    is why veg-point heights are drawn via closed-form inverse-CDF sampling
    of the survival function S(z) = [1 - exp(-r(height-z))] / [1 - exp(-r*height)]
    (r = k*pavd0), not `rng.exponential` directly.

    n_gnd defaults to the value consistent with pavd0/height (so that
    n_gnd/(n_gnd+n_veg) matches the theoretical ground gap fraction
    exp(-k*pavd0*height)) -- pass an explicit value to deliberately test
    an inconsistent ground count.

    If `gap_band` = (z0, z1) is given, veg returns are additionally excluded
    from that height band (an empty layer -- e.g. a real understorey gap).

    Returns (hag, classification) arrays ready for _pavd_profile_from_hag.
    """
    r = _K * pavd0
    denom = 1.0 - np.exp(-r * height)
    if n_gnd is None:
        n_gnd = int(round(n_veg * np.exp(-r * height) / denom))
    u = rng.uniform(0.0, 1.0, size=n_veg)
    z = height + np.log(1.0 - u * denom) / r
    z = np.clip(z, 0.0, height)
    if gap_band is not None:
        z0, z1 = gap_band
        z = z[(z < z0) | (z > z1)]
    hag = np.concatenate([np.zeros(n_gnd), z])
    classification = np.concatenate(
        [np.full(n_gnd, 2, dtype=np.uint8), np.full(len(z), 3, dtype=np.uint8)]
    )
    return hag, classification


# ---------------------------------------------------------------------------
# _pavd_profile_from_hag — guards
# ---------------------------------------------------------------------------


def test_too_few_ground_points_returns_none():
    rng = np.random.default_rng(0)
    hag, cls = _make_canopy(rng, n_gnd=_MIN_GROUND_POINTS_DEFAULT - 1, pavd0=0.3, height=20.0)
    assert _pavd_profile_from_hag(hag, cls) is None


def test_no_vegetation_returns_none():
    """All-ground footprint: no signal above ground -> canopy_height=0 -> None."""
    hag = np.zeros(100)
    cls = np.full(100, 2, dtype=np.uint8)
    assert _pavd_profile_from_hag(hag, cls) is None


def test_only_unclassified_returns_none():
    """Neither ground (2) nor vegetation (3-5): classified mask is empty."""
    hag = np.linspace(0, 20, 100)
    cls = np.full(100, 1, dtype=np.uint8)  # unclassified
    assert _pavd_profile_from_hag(hag, cls) is None


# ---------------------------------------------------------------------------
# _pavd_profile_from_hag — recovers known structure
# ---------------------------------------------------------------------------


def test_total_pai_matches_gap_to_lai_at_ground():
    """total_pai (from the vertical estimator's ground edge) must exactly match
    gap.py's own 2D _gap_to_lai on the equivalent P_gap = n_gnd/n_tot -- the
    vertical estimator is a strict extension of the horizontal one, so they
    must agree at z=0 by construction."""
    rng = np.random.default_rng(1)
    n_gnd, n_veg = 300, 20_000
    hag, cls = _make_canopy(rng, n_gnd=n_gnd, pavd0=0.25, height=25.0, n_veg=n_veg)
    profile = _pavd_profile_from_hag(hag, cls, z_step=0.25)
    assert profile is not None

    n_tot = n_gnd + len(hag) - n_gnd  # = n_gnd + n_veg_actual
    p_gap_2d = n_gnd / (n_gnd + (len(hag) - n_gnd))
    expected_total_pai = float(_gap_to_lai(np.array([[p_gap_2d]], dtype=np.float32), k=_K)[0, 0])
    assert profile.total_pai == pytest.approx(expected_total_pai, rel=1e-6)


def test_recovers_known_uniform_pavd():
    """A canopy built with a known constant PAVD should be recovered to
    within sampling noise, away from the noisy ground/top boundary bins."""
    rng = np.random.default_rng(2)
    pavd0, height = 0.30, 25.0
    hag, cls = _make_canopy(rng, pavd0=pavd0, height=height, n_veg=60_000)
    profile = _pavd_profile_from_hag(hag, cls, z_step=0.5)
    assert profile is not None

    # Interior bins only (away from ground-return floor and top-percentile edge,
    # both flagged as boundary-sensitive in the module docstring).
    interior = (profile.pavd_z > 2.0) & (profile.pavd_z < 0.9 * profile.canopy_height)
    assert interior.sum() > 5
    np.testing.assert_allclose(profile.pavd[interior], pavd0, rtol=0.25)


def test_gap_band_shows_up_as_near_zero_pavd():
    """An explicit empty layer (no veg returns in [8, 12] m) must show much
    lower PAVD there than in the fully-populated layers around it."""
    rng = np.random.default_rng(3)
    hag, cls = _make_canopy(rng, pavd0=0.30, height=25.0, n_veg=60_000, gap_band=(8.0, 12.0))
    profile = _pavd_profile_from_hag(hag, cls, z_step=0.5)
    assert profile is not None

    in_gap = (profile.pavd_z > 9.0) & (profile.pavd_z < 11.0)
    below_gap = (profile.pavd_z > 3.0) & (profile.pavd_z < 6.0)
    assert in_gap.sum() > 0 and below_gap.sum() > 0
    assert np.mean(profile.pavd[in_gap]) < 0.3 * np.mean(profile.pavd[below_gap])


def test_pai_cumulative_decreases_with_height():
    """Canopy already traversed by a downward pulse must be maximal at the
    ground edge and ~0 at the canopy-top edge (module docstring's ordering)."""
    rng = np.random.default_rng(4)
    hag, cls = _make_canopy(rng, n_gnd=300, pavd0=0.25, height=20.0, n_veg=20_000)
    profile = _pavd_profile_from_hag(hag, cls, z_step=0.5)
    assert profile is not None
    assert profile.pai_cumulative[0] == pytest.approx(profile.total_pai)
    assert profile.pai_cumulative[-1] < profile.pai_cumulative[0]
    assert np.all(np.diff(profile.pai_cumulative) <= 1e-9)  # monotonically non-increasing


def test_pavd_non_negative():
    rng = np.random.default_rng(5)
    hag, cls = _make_canopy(rng, n_gnd=300, pavd0=0.4, height=30.0, n_veg=30_000)
    profile = _pavd_profile_from_hag(hag, cls, z_step=0.3)
    assert profile is not None
    assert np.all(profile.pavd >= -1e-9)


# ---------------------------------------------------------------------------
# fit_transmittance_model
# ---------------------------------------------------------------------------


def test_transmittance_perfect_agreement_gives_beta_zero():
    """observed == als_pavd everywhere -> transmittance == 1 -> beta ~ 0."""
    c = np.linspace(0.0, 5.0, 30)
    als = np.full_like(c, 0.3)
    out = fit_transmittance_model(c, als, als)
    np.testing.assert_allclose(out["transmittance"], 1.0)
    assert out["beta"] == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(out["fitted"], 1.0, atol=1e-2)


def test_transmittance_recovers_known_beta():
    """observed = als * exp(-beta_true * c) -> fit should recover beta_true."""
    rng = np.random.default_rng(6)
    beta_true = 0.35
    c = np.linspace(0.0, 6.0, 40)
    als = 0.3 + 0.05 * rng.standard_normal(c.size) ** 2 + 0.05  # positive, noisy but away from 0
    observed = als * np.exp(-beta_true * c)
    out = fit_transmittance_model(c, als, observed)
    assert out["beta"] == pytest.approx(beta_true, rel=0.05)


def test_transmittance_zero_als_pavd_gives_nan_not_inf():
    c = np.array([0.0, 1.0, 2.0, 3.0])
    als = np.array([0.0, 0.2, 0.3, 0.0])
    observed = np.array([0.0, 0.1, 0.15, 0.0])
    out = fit_transmittance_model(c, als, observed)
    assert np.isnan(out["transmittance"][0])
    assert np.isnan(out["transmittance"][3])
    assert np.isfinite(out["transmittance"][1])


def test_transmittance_too_few_points_gives_nan_beta():
    c = np.array([0.0, 1.0])
    als = np.array([0.2, 0.3])
    observed = np.array([0.1, 0.2])
    out = fit_transmittance_model(c, als, observed)
    assert np.isnan(out["beta"])
    assert np.all(np.isnan(out["fitted"]))
