# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for biomass.py — naesset_model, wrap_sklearn_model, _extract_metrics (unit)
and compute_metrics / compute_biomass (integration)."""

import numpy as np
import pytest

from alsdb.processing.biomass import (
    _extract_metrics,
    compute_biomass,
    compute_metrics,
    naesset_model,
    wrap_sklearn_model,
)

BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
RES = 10.0
YEAR = 2021


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hag_points(n_gnd=50, n_veg=100, bbox=(0.0, 0.0, 100.0, 100.0), seed=0):
    """
    Structured array with X, Y, HeightAboveGround, Classification, ReturnNumber.
    Ground at HAG=0; vegetation at HAG=5–20 m.
    """
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = bbox
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
        ("HeightAboveGround", np.float32),
        ("Classification", np.uint8),
        ("ReturnNumber", np.uint8),
    ]
    n = n_gnd + n_veg
    arr = np.zeros(n, dtype=dtype)
    arr["X"] = rng.uniform(min_x, max_x, n)
    arr["Y"] = rng.uniform(min_y, max_y, n)
    arr["ReturnNumber"][:] = 1
    # ground
    arr["Classification"][:n_gnd] = 2
    arr["HeightAboveGround"][:n_gnd] = 0.0
    # vegetation (class 3)
    arr["Classification"][n_gnd:] = 3
    arr["HeightAboveGround"][n_gnd:] = rng.uniform(5.0, 20.0, n_veg).astype(np.float32)
    return arr


def _flat_metrics(ny=10, nx=10, h95_val=15.0, cc_val=0.7):
    return {
        "h50": np.full((ny, nx), 10.0, dtype=np.float32),
        "h75": np.full((ny, nx), 12.0, dtype=np.float32),
        "h95": np.full((ny, nx), h95_val, dtype=np.float32),
        "hmean": np.full((ny, nx), 11.0, dtype=np.float32),
        "cc": np.full((ny, nx), cc_val, dtype=np.float32),
        "density": np.full((ny, nx), 5.0, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# naesset_model — unit tests
# ---------------------------------------------------------------------------


def test_naesset_model_output_shape():
    m = _flat_metrics()
    agb = naesset_model(m)
    assert agb.shape == (10, 10)


def test_naesset_model_dtype_float32():
    m = _flat_metrics()
    agb = naesset_model(m)
    assert agb.dtype == np.float32


def test_naesset_model_known_value():
    """AGB = a * h95^b * cc^c.  With defaults a=0.8, b=1.8, c=0.5."""
    h95, cc = 10.0, 0.64
    expected = 0.8 * (h95**1.8) * (cc**0.5)
    m = {
        "h95": np.array([[h95]], dtype=np.float32),
        "cc": np.array([[cc]], dtype=np.float32),
    }
    agb = naesset_model(m)
    assert float(agb[0, 0]) == pytest.approx(expected, rel=1e-4)


def test_naesset_model_nan_for_nan_h95():
    m = _flat_metrics()
    m["h95"][0, 0] = np.nan
    agb = naesset_model(m)
    assert np.isnan(agb[0, 0])
    assert not np.isnan(agb[0, 1])


def test_naesset_model_nan_for_zero_cc():
    m = _flat_metrics(cc_val=0.0)
    agb = naesset_model(m)
    assert np.all(np.isnan(agb))


def test_naesset_model_positive_for_valid_inputs():
    m = _flat_metrics()
    agb = naesset_model(m)
    assert np.all(agb > 0)


# ---------------------------------------------------------------------------
# wrap_sklearn_model — unit tests (no sklearn import required)
# ---------------------------------------------------------------------------


class _MockEstimator:
    """Minimal sklearn-compatible estimator (just returns sum of features)."""

    def predict(self, X):
        return X.sum(axis=1).astype(np.float32)


def test_wrap_sklearn_model_output_shape():
    m = _flat_metrics()
    fn = wrap_sklearn_model(_MockEstimator())
    result = fn(m)
    assert result.shape == (10, 10)


def test_wrap_sklearn_model_dtype_float32():
    m = _flat_metrics()
    fn = wrap_sklearn_model(_MockEstimator())
    result = fn(m)
    assert result.dtype == np.float32


def test_wrap_sklearn_model_custom_features():
    m = _flat_metrics()
    fn = wrap_sklearn_model(_MockEstimator(), features=["h95", "cc"])
    result = fn(m)
    assert result.shape == (10, 10)
    # Each pixel = h95 + cc = 15 + 0.7 = 15.7
    np.testing.assert_allclose(result, 15.7, rtol=1e-4)


def test_wrap_sklearn_model_nan_pixels_stay_nan():
    """NaN pixels must not be passed to predict and must stay NaN."""
    m = _flat_metrics()
    m["h95"][5, 5] = np.nan
    fn = wrap_sklearn_model(_MockEstimator())
    result = fn(m)
    assert np.isnan(result[5, 5])
    # Other pixels must be valid
    assert not np.isnan(result[0, 0])


def test_wrap_sklearn_model_all_nan_returns_all_nan():
    m = _flat_metrics()
    for k in m:
        m[k][:] = np.nan
    fn = wrap_sklearn_model(_MockEstimator())
    result = fn(m)
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# _extract_metrics — unit tests
# ---------------------------------------------------------------------------


def test_extract_metrics_returns_all_names():
    pts = _make_hag_points(bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    for name in ["h50", "h75", "h95", "hmean", "cc", "density"]:
        assert name in metrics


def test_extract_metrics_shape():
    pts = _make_hag_points(bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    for arr in metrics.values():
        assert arr.shape == (10, 10)


def test_extract_metrics_dtype_float32():
    pts = _make_hag_points()
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    for arr in metrics.values():
        assert arr.dtype == np.float32


def test_extract_metrics_h95_positive_in_veg_cells():
    pts = _make_hag_points(n_veg=200, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    h95_valid = metrics["h95"][~np.isnan(metrics["h95"])]
    assert len(h95_valid) > 0
    assert np.all(h95_valid > 0)


def test_extract_metrics_cc_in_unit_interval():
    pts = _make_hag_points(n_veg=200, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    cc_valid = metrics["cc"][~np.isnan(metrics["cc"])]
    assert np.all(cc_valid >= 0.0)
    assert np.all(cc_valid <= 1.0)


def test_extract_metrics_density_non_negative():
    """Density (pts/m²) must be ≥ 0; cells with points will have density > 0."""
    pts = _make_hag_points(bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    dens = metrics["density"]
    # No negative values
    assert np.all(dens[~np.isnan(dens)] >= 0.0)
    # At least some cells have non-zero density
    assert np.any(dens > 0)


# ---------------------------------------------------------------------------
# Integration — requires real TileDB + PDAL (uses session provider)
# ---------------------------------------------------------------------------


def test_compute_metrics_writes_all_variables(provider, store):
    compute_metrics(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    for var in ["h50", "h75", "h95", "hmean", "cc", "density"]:
        assert store.has_data(var, RES, YEAR), f"Missing variable: {var}"


def test_compute_biomass_writes_biomass(provider, store):
    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    assert store.has_data("biomass", RES, YEAR)


def test_compute_biomass_custom_model(provider, store):
    def simple_model(metrics):
        return np.where(np.isnan(metrics["h95"]), np.nan, metrics["h95"] * 2.0)

    compute_biomass(
        provider, store, resolution=RES, bbox=BBOX, year=YEAR, model_fn=simple_model
    )
    assert store.has_data("biomass", RES, YEAR)


def test_compute_metrics_overwrite_false_skips(provider, store):
    compute_metrics(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["h95"][0] = 999.0
    compute_metrics(
        provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False
    )
    assert float(store._root["10m"]["h95"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_biomass_overwrite_false_skips(provider, store):
    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["biomass"][0] = 999.0
    compute_biomass(
        provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False
    )
    assert float(store._root["10m"]["biomass"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_biomass_out_of_year_skips(provider, store):
    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=1900)
    assert not store.has_data("biomass", RES, 1900)
