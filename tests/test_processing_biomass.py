# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for biomass.py — naesset_model, wrap_sklearn_model, _extract_metrics (unit)
and compute_metrics / compute_biomass (integration)."""

import numpy as np
import pytest

from alsdb.products.allometry import (
    calibrate_naesset,
    naesset_model,
)
from alsdb.products.biomass import (
    _extract_metrics,
    _extract_metrics_baba,
    _fhd_from_hag,
    _vci_from_hag,
    compute_biomass,
    compute_metrics,
    wrap_sklearn_model,
    wrap_torch_model,
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
        # Height percentiles
        "h50": np.full((ny, nx), 10.0, dtype=np.float32),
        "h75": np.full((ny, nx), 12.0, dtype=np.float32),
        "h95": np.full((ny, nx), h95_val, dtype=np.float32),
        "hmax": np.full((ny, nx), 20.0, dtype=np.float32),
        "hmean": np.full((ny, nx), 11.0, dtype=np.float32),
        # Canopy structure
        "cc": np.full((ny, nx), cc_val, dtype=np.float32),
        "density": np.full((ny, nx), 5.0, dtype=np.float32),
        "fhd": np.full((ny, nx), 1.2, dtype=np.float32),
        "vci": np.full((ny, nx), 0.5, dtype=np.float32),
        "crr": np.full((ny, nx), 0.6, dtype=np.float32),
        # Height stratum proportions (must sum to 1.0)
        "pv_0_2": np.full((ny, nx), 0.10, dtype=np.float32),
        "pv_2_5": np.full((ny, nx), 0.15, dtype=np.float32),
        "pv_5_10": np.full((ny, nx), 0.20, dtype=np.float32),
        "pv_10_20": np.full((ny, nx), 0.30, dtype=np.float32),
        "pv_20_40": np.full((ny, nx), 0.20, dtype=np.float32),
        "pv_above40": np.full((ny, nx), 0.05, dtype=np.float32),
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


def test_naesset_model_zero_without_cover_nan_without_returns():
    m = _flat_metrics(cc_val=0.0)
    m["h95"][0, :] = np.nan  # no vegetation returns at all
    m["cc"][1, 0] = np.nan  # no first returns
    agb = naesset_model(m, a=1.0, b=2.0, c=0.5)
    assert np.isnan(agb[1, 0])
    agb[1, 0] = 0.0
    assert np.all(agb == 0.0)


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

    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=YEAR, model_fn=simple_model)
    assert store.has_data("biomass", RES, YEAR)


def test_compute_metrics_overwrite_false_skips(provider, store):
    compute_metrics(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["h95"][0] = 999.0
    compute_metrics(provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False)
    assert float(store._root["10m"]["h95"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_biomass_overwrite_false_skips(provider, store):
    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    store._root["10m"]["biomass"][0] = 999.0
    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=YEAR, overwrite=False)
    assert float(store._root["10m"]["biomass"][0, 0, 0]) == pytest.approx(999.0)


def test_compute_biomass_out_of_year_skips(provider, store):
    compute_biomass(provider, store, resolution=RES, bbox=BBOX, year=1900)
    assert not store.has_data("biomass", RES, 1900)


# ---------------------------------------------------------------------------
# calibrate_naesset
# ---------------------------------------------------------------------------


def _synthetic_plots(n: int, seed: int = 0):
    """Synthetic field data following AGB = 0.9 * h95^1.7 * cc^0.4."""
    rng = np.random.default_rng(seed)
    h95 = rng.uniform(5.0, 30.0, n)
    cc = rng.uniform(0.3, 0.9, n)
    agb = 0.9 * h95**1.7 * cc**0.4 + rng.normal(0, 2.0, n)
    return h95, cc, np.clip(agb, 0.1, None)


def test_calibrate_naesset_raises_below_20_plots():
    h95, cc, agb = _synthetic_plots(15)
    with pytest.raises(ValueError, match="20"):
        calibrate_naesset(h95, cc, agb)


def test_calibrate_naesset_warns_below_50_plots():
    h95, cc, agb = _synthetic_plots(25)
    with pytest.warns(UserWarning, match="50"):
        calibrate_naesset(h95, cc, agb)


def test_calibrate_naesset_fits_reasonable_coefficients():
    h95, cc, agb = _synthetic_plots(60)
    a, b, c = calibrate_naesset(h95, cc, agb)
    # Coefficients should be positive and in a plausible range
    assert a > 0
    assert 0.5 < b < 4.0
    assert 0.0 < c < 2.0


def test_calibrate_naesset_return_cov_shape():
    h95, cc, agb = _synthetic_plots(60)
    _, pcov = calibrate_naesset(h95, cc, agb, return_cov=True)
    assert pcov.shape == (3, 3)


def test_calibrate_naesset_cov_diagonal_positive():
    h95, cc, agb = _synthetic_plots(60)
    _, pcov = calibrate_naesset(h95, cc, agb, return_cov=True)
    assert np.all(np.diag(pcov) > 0)


def test_calibrate_naesset_ignores_nan_plots():
    h95, cc, agb = _synthetic_plots(60)
    h95[0] = np.nan
    cc[1] = np.nan
    agb[2] = np.nan
    # Should still fit on the remaining 57 valid plots
    a, _, _ = calibrate_naesset(h95, cc, agb)
    assert a > 0


# ---------------------------------------------------------------------------
# _fhd_from_hag / _vci_from_hag
# ---------------------------------------------------------------------------


def test_fhd_from_hag_empty_returns_nan():
    assert np.isnan(_fhd_from_hag(np.array([])))


def test_fhd_from_hag_single_value_returns_zero():
    # All points in one bin → entropy = 0
    result = _fhd_from_hag(np.full(20, 5.0))
    assert result == pytest.approx(0.0, abs=1e-6)


def test_fhd_from_hag_uniform_distribution_positive():
    hag = np.arange(0.5, 40.5, 1.0)  # one point per 1 m bin
    result = _fhd_from_hag(hag)
    assert result > 0.0


def test_fhd_from_hag_more_layers_higher_entropy():
    narrow = np.arange(0.5, 5.5, 1.0)  # 5 bins
    wide = np.arange(0.5, 20.5, 1.0)  # 20 bins
    assert _fhd_from_hag(wide) > _fhd_from_hag(narrow)


def test_vci_from_hag_empty_returns_nan():
    assert np.isnan(_vci_from_hag(np.array([])))


def test_vci_from_hag_in_unit_interval():
    hag = np.arange(0.5, 30.5, 1.0)
    result = _vci_from_hag(hag)
    assert 0.0 <= result <= 1.0


def test_vci_from_hag_single_bin_is_zero():
    result = _vci_from_hag(np.full(30, 3.0))
    assert result == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _extract_metrics — new metrics (hmax, crr, strata)
# ---------------------------------------------------------------------------


def test_extract_metrics_contains_all_new_names():
    pts = _make_hag_points(n_veg=200, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    for name in (
        "hmax",
        "crr",
        "pv_0_2",
        "pv_2_5",
        "pv_5_10",
        "pv_10_20",
        "pv_20_40",
        "pv_above40",
    ):
        assert name in metrics, f"Missing metric: {name}"


def test_extract_metrics_hmax_geq_h95():
    pts = _make_hag_points(n_veg=200, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    valid = ~np.isnan(metrics["hmax"]) & ~np.isnan(metrics["h95"])
    assert np.all(metrics["hmax"][valid] >= metrics["h95"][valid] - 1e-4)


def test_extract_metrics_crr_in_unit_interval():
    pts = _make_hag_points(n_veg=200, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    crr = metrics["crr"]
    valid = crr[~np.isnan(crr)]
    if len(valid) > 0:
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)


def test_extract_metrics_strata_sum_to_one():
    pts = _make_hag_points(n_veg=300, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics = _extract_metrics(pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0))
    strata_names = ["pv_0_2", "pv_2_5", "pv_5_10", "pv_10_20", "pv_20_40", "pv_above40"]
    total = np.zeros_like(metrics["pv_0_2"])
    for name in strata_names:
        total = np.where(np.isnan(metrics[name]), np.nan, total + metrics[name])
    valid = total[~np.isnan(total)]
    if len(valid) > 0:
        np.testing.assert_allclose(valid, 1.0, atol=1e-5)


def test_extract_metrics_min_density_masks_sparse_cells():
    """With a very high min_density, most cells should become NaN."""
    pts = _make_hag_points(n_gnd=5, n_veg=5, bbox=(0.0, 0.0, 100.0, 100.0))
    metrics_no_guard = _extract_metrics(
        pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0), min_density=0.0
    )
    metrics_with_guard = _extract_metrics(
        pts, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0), min_density=100.0
    )
    # Without the threshold some cells hold values; with an impossibly high
    # one, every metric of every cell is NaN
    assert np.isfinite(metrics_no_guard["density"]).any()
    assert np.isfinite(metrics_no_guard["h95"]).any()
    for name, values in metrics_with_guard.items():
        assert np.all(np.isnan(values)), name


def test_compute_metrics_writes_new_variables(provider, store):
    compute_metrics(provider, store, resolution=RES, bbox=BBOX, year=YEAR)
    for var in ("hmax", "crr", "pv_0_2", "pv_5_10", "pv_above40"):
        assert store.has_data(var, RES, YEAR), f"Missing new variable: {var}"


# ---------------------------------------------------------------------------
# Canopy cover from classified first returns; zero-biomass plots
# ---------------------------------------------------------------------------


def _with_buildings(arr, n=60, seed=3):
    """Turn *n* ground returns into building roofs 8 m above ground."""
    out = arr.copy()
    idx = np.random.default_rng(seed).choice(np.flatnonzero(out["Classification"] == 2), n, False)
    out["Classification"][idx] = 6
    out["HeightAboveGround"][idx] = 8.0
    return out


@pytest.mark.parametrize("baba", [False, True])
def test_cover_ignores_buildings_and_unclassified(baba):
    arr = _make_hag_points(n_gnd=400, n_veg=400, seed=4)
    bbox = (0.0, 0.0, 100.0, 100.0)

    def cover(points):
        if baba:
            return _extract_metrics_baba(points, 50.0, bbox, 40.0, metrics=("cc",))["cc"]
        return _extract_metrics(points, 50.0, bbox, metrics=("cc",))["cc"]

    base = cover(arr)
    with_roofs = _with_buildings(arr)
    # roofs replace ground returns: the classified denominator shrinks, the canopy count stays
    expected = cover(with_roofs[with_roofs["Classification"] != 6])
    np.testing.assert_allclose(cover(with_roofs), expected)
    assert np.all(cover(with_roofs) >= base - 1e-6)
    unclassified = with_roofs.copy()
    unclassified["Classification"][unclassified["Classification"] == 6] = 1
    np.testing.assert_allclose(cover(unclassified), expected)


def test_baba_matches_grid_metrics_when_the_disc_holds_all_points():
    arr = _make_hag_points(n_gnd=300, n_veg=700, bbox=(0.0, 0.0, 10.0, 10.0), seed=5)
    grid = _extract_metrics(arr, 10.0, (0.0, 0.0, 10.0, 10.0))
    baba = _extract_metrics_baba(arr, 10.0, (0.0, 0.0, 10.0, 10.0), 50.0)
    for name in grid:
        if name == "density":  # per cell area vs per disc area
            continue
        np.testing.assert_allclose(baba[name], grid[name], rtol=1e-6, err_msg=name)


def test_calibrate_naesset_keeps_plots_without_biomass():
    rng = np.random.default_rng(6)
    h95 = rng.uniform(5, 30, 80)
    cc = rng.uniform(0.2, 1.0, 80)
    agb = 1.2 * h95**1.5 * cc**0.7
    h95[:20], cc[:20], agb[:20] = np.nan, 0.0, 0.0  # open plots
    a, b, c = calibrate_naesset(h95, cc, agb)
    assert (a, b, c) == pytest.approx((1.2, 1.5, 0.7), rel=1e-3)


# ---------------------------------------------------------------------------
# Machine-learning models through wrap_sklearn_model
# ---------------------------------------------------------------------------


def _fitted_forest():
    import pandas as pd

    ensemble = pytest.importorskip("sklearn.ensemble")
    RandomForestRegressor = ensemble.RandomForestRegressor

    rng = np.random.default_rng(7)
    X = pd.DataFrame({"cc": rng.uniform(0, 1, 300), "h95": rng.uniform(2, 30, 300)})
    return RandomForestRegressor(30, random_state=0).fit(X, 2 * X.h95 * X.cc)


def test_wrapper_uses_the_fitted_feature_names_and_rejects_another_order():
    forest = _fitted_forest()
    model = wrap_sklearn_model(forest)
    assert model.required_metrics == ("cc", "h95")
    m = _flat_metrics(h95_val=20.0, cc_val=0.5)
    assert float(model(m)[0, 0]) == pytest.approx(20.0, rel=0.15)
    with pytest.raises(ValueError, match="differ"):
        wrap_sklearn_model(forest, ["h95", "cc"])


def test_wrapper_rejects_a_feature_count_mismatch():
    class Fitted:
        n_features_in_ = 2

        def predict(self, X):
            return X[:, 0]

    with pytest.raises(ValueError, match="expects 2 features"):
        wrap_sklearn_model(Fitted())


def test_wrapper_accepts_column_predictions_and_unpicklable_models():
    import threading

    class Network:  # Keras/PyTorch style: (n, 1) output, holds an unpicklable handle
        lock = threading.Lock()

        def predict(self, X):
            return (X[:, 0] * X[:, 1])[:, None]

    with pytest.warns(UserWarning, match="model_id"):
        model = wrap_sklearn_model(Network(), ["cc", "h95"])
    assert model.model_id is None
    m = _flat_metrics(h95_val=10.0, cc_val=0.5)
    m["h95"][0, 0] = np.nan  # covered but missing height: NaN
    m["h95"][0, 1], m["cc"][0, 1] = np.nan, 0.0  # no vegetation returns: 0
    agb = model(m)
    assert np.isnan(agb[0, 0]) and agb[0, 1] == 0.0 and agb[1, 1] == pytest.approx(5.0)


def test_torch_network_predicts_in_batches_with_scaling_and_cover_rules():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = torch.nn.Linear(2, 1)
    with torch.no_grad():  # AGB = 2 * (h95 / 10) + 3 * cc
        net.weight[:] = torch.tensor([[2.0, 3.0]])
        net.bias.zero_()
    scale = np.array([10.0, 1.0])
    model = wrap_torch_model(net, ["h95", "cc"], batch_size=7)
    m = _flat_metrics(ny=4, nx=5, h95_val=20.0, cc_val=0.5)
    m["h95"] = m["h95"] * 0 + 2.0  # already scaled: 2 * 2 + 3 * 0.5
    m["h95"][0, 0] = np.nan
    m["h95"][0, 1], m["cc"][0, 1] = np.nan, 0.0
    agb = model(m)
    assert agb.dtype == np.float32 and agb.shape == (4, 5)
    assert np.isnan(agb[0, 0]) and agb[0, 1] == 0.0
    np.testing.assert_allclose(agb[1:], 5.5, rtol=1e-6)
    assert model.required_metrics == ("h95", "cc")
    assert model.model_id == wrap_torch_model(net, ["h95", "cc"]).model_id

    def scaled(X):
        return X / scale

    m["h95"][1:] = 20.0
    np.testing.assert_allclose(
        wrap_torch_model(net, ["h95", "cc"], transform=scaled)(m)[1:], 5.5, rtol=1e-6
    )
    with pytest.warns(UserWarning, match="model_id"):
        assert wrap_torch_model(net, ["h95", "cc"], transform=lambda X: X).model_id is None
    assert wrap_torch_model(net, ["h95", "cc"], model_id="mlp-v1").model_id == "mlp-v1"


def test_torch_network_with_its_own_normalisation():
    torch = pytest.importorskip("torch")

    class AGBNet(torch.nn.Module):  # the example of doc/area_products.md
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
            self.body = torch.nn.Linear(len(mean), 1)
            with torch.no_grad():
                self.body.weight[:] = torch.tensor([[2.0, 3.0]])
                self.body.bias.zero_()

        def forward(self, x):
            return self.body((x - self.mean) / self.std)

    net = AGBNet([10.0, 0.0], [5.0, 1.0])
    model = wrap_torch_model(net, ["h95", "cc"])
    m = _flat_metrics(ny=2, nx=2, h95_val=20.0, cc_val=0.5)  # raw metrics in
    np.testing.assert_allclose(model(m), 2 * (20 - 10) / 5 + 3 * 0.5, rtol=1e-6)
    # the normalisation is part of the weights, so of the model identity
    other = wrap_torch_model(AGBNet([12.0, 0.0], [5.0, 1.0]), ["h95", "cc"])
    assert other.model_id != model.model_id


def test_torch_network_on_gpu_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import copy

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    m = _flat_metrics(ny=30, nx=40, h95_val=20.0, cc_val=0.5)
    m["h95"] = m["h95"] + np.linspace(0, 5, 40, dtype=np.float32)
    m["h95"][0, 0] = np.nan
    m["h95"][0, 1], m["cc"][0, 1] = np.nan, 0.0
    on_cpu = wrap_torch_model(copy.deepcopy(net), ["h95", "cc"], batch_size=500)(m)
    on_gpu = wrap_torch_model(copy.deepcopy(net), ["h95", "cc"], device="cuda", batch_size=500)(m)
    assert on_gpu.dtype == np.float32 and on_gpu.shape == (30, 40)
    assert np.isnan(on_gpu[0, 0]) and on_gpu[0, 1] == 0.0
    np.testing.assert_allclose(on_gpu, on_cpu, rtol=1e-5, atol=1e-5)
