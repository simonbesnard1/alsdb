# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Biomass models in the GEDI L4A functional forms (alsdb.products.allometry)."""

import pickle

import numpy as np
import pytest

from alsdb.products.allometry import (
    GEDI_L4A_MODELS,
    GEDI_L4A_STRATA,
    LinearAGBModel,
    cross_validate,
    fit_linear_agb_model,
    gedi_l4a_model,
)


def _plots(n=400, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    h50, h95 = rng.uniform(1, 20, n), rng.uniform(5, 35, n)
    root = -2.0 + 1.5 * np.sqrt(h50) + 2.5 * np.sqrt(h95) + rng.normal(0, noise, n)
    return {"h50": h50, "h95": h95}, np.maximum(root, 0) ** 2, rng


def test_fit_recovers_a_square_root_model_and_its_errors():
    metrics, agb, _ = _plots()
    model = fit_linear_agb_model(metrics, agb, ("h50", "h95"))
    assert model.coefficients == pytest.approx((-2.0, 1.5, 2.5), abs=0.25)
    assert model.rse == pytest.approx(0.4, rel=0.1) and model.dof == 400 - 3
    # Snowdon: the corrected predictions sum to the field biomass
    assert model(metrics).sum() == pytest.approx(agb.sum(), rel=1e-5)
    # ordinary least squares in transformed units
    x = np.column_stack((np.ones(400), np.sqrt(metrics["h50"]), np.sqrt(metrics["h95"])))
    beta, rss, *_ = np.linalg.lstsq(x, np.sqrt(agb), rcond=None)
    assert model.coefficients == pytest.approx(tuple(beta))
    np.testing.assert_allclose(model.vcov, rss[0] / 397 * np.linalg.inv(x.T @ x), rtol=1e-8)


def test_prediction_interval_follows_the_l4a_formula_and_covers_new_plots():
    metrics, agb, _ = _plots()
    model = fit_linear_agb_model(metrics, agb, ("h50", "h95"))
    new, new_agb, _ = _plots(n=2000, seed=1)
    out = model.predict(new, level=0.9)
    x = np.column_stack((np.ones(2000), np.sqrt(new["h50"]), np.sqrt(new["h95"])))
    se = np.sqrt(model.rse**2 + np.einsum("ij,jk,ik->i", x, np.asarray(model.vcov), x))
    np.testing.assert_allclose(out["se"], se)
    inside = (new_agb >= out["lower"]) & (new_agb <= out["upper"])
    assert 0.86 < inside.mean() < 0.94
    np.testing.assert_allclose(out["agb"], model(new), rtol=1e-6)
    assert np.isnan(gedi_l4a_model("DBT").predict({"rh60": 5.0, "rh98": 20.0})["se"])


@pytest.mark.parametrize("bias_method", ["snowdon", "baskerville"])
def test_log_log_fit_is_a_power_law(bias_method):
    rng = np.random.default_rng(2)
    h95, cc = rng.uniform(5, 35, 500), rng.uniform(0.2, 1.0, 500)
    agb = 1.2 * h95**1.5 * cc**0.7 * np.exp(rng.normal(0, 0.2, 500))
    model = fit_linear_agb_model(
        {"h95": h95, "cc": cc}, agb, ("h95", "cc"), y_transform="log", x_transform="log",
        bias_method=bias_method,
    )  # fmt: skip
    assert model.coefficients == pytest.approx((np.log(1.2), 1.5, 0.7), abs=0.12)
    assert model.bias_correction == pytest.approx(np.exp(0.2**2 / 2), abs=0.02)
    assert model({"h95": h95, "cc": cc}).mean() == pytest.approx(agb.mean(), rel=0.02)
    with pytest.raises(ValueError, match="Baskerville"):
        fit_linear_agb_model({"h95": h95}, agb, ("h95",), bias_method="baskerville")


def test_weights_and_untransformed_fit():
    rng = np.random.default_rng(3)
    h = rng.uniform(5, 30, 60)
    agb = 4.0 * h + 10 + rng.normal(0, 1, 60)
    agb[:30] += 50  # a group fitted away with zero-ish weight
    weights = np.r_[np.full(30, 1e-9), np.ones(30)]
    model = fit_linear_agb_model(
        {"h": h}, agb, ("h",), y_transform="none", x_transform="none", weights=weights
    )
    assert model.coefficients == pytest.approx((10.0, 4.0), abs=1.5)
    assert model.bias_correction == 1.0 and model.bias_method == "none"


def test_cells_without_predictors_follow_the_cover_rule_and_negative_roots_give_zero():
    model = LinearAGBModel(("h50", "h95"), (-3.0, 1.0, 1.0), bias_correction=1.1)
    assert model.required_metrics == ("h50", "h95", "cc")
    grid = {
        "h50": np.array([[4.0, np.nan, np.nan], [0.25, 4.0, 4.0]]),
        "h95": np.array([[16.0, np.nan, np.nan], [0.25, -1.0, 16.0]]),
        "cc": np.array([[0.5, 0.0, 0.3], [0.1, 0.5, np.nan]]),
    }
    agb = model(grid)
    assert agb.dtype == np.float32 and agb.shape == (2, 3)
    assert agb[0, 0] == pytest.approx(1.1 * 9.0)  # (-3 + 2 + 4)^2
    assert agb[0, 1] == 0.0 and np.isnan(agb[0, 2])  # no cover: 0; cover but no heights: NaN
    assert agb[1, 0] == 0.0  # root -2: no biomass, not 1.1 * 4
    assert np.isnan(agb[1, 1])  # square root of a negative height
    assert agb[1, 2] == pytest.approx(1.1 * 9.0)  # all predictors present: predicted
    strict = LinearAGBModel(("h50", "h95"), (-3.0, 1.0, 1.0), zero_without_cover=False)
    assert strict.required_metrics == ("h50", "h95") and np.isnan(strict(grid)[0, 1])


def test_model_is_picklable_with_a_stable_identity():
    metrics, agb, _ = _plots()
    model = fit_linear_agb_model(metrics, agb, ("h50", "h95"), name="plots-v1")
    again = pickle.loads(pickle.dumps(model))
    assert again == model and again.model_id == model.model_id
    other = fit_linear_agb_model(metrics, agb, ("h95",), name="plots-v1")
    assert other.model_id != model.model_id


def test_fit_rejects_too_few_plots_and_collinear_predictors():
    metrics, agb, _ = _plots(n=4)
    with pytest.raises(ValueError, match="usable plots"):
        fit_linear_agb_model(metrics, agb, ("h50", "h95"))
    metrics, agb, _ = _plots()
    metrics["copy"] = metrics["h50"]
    with pytest.raises(ValueError, match="collinear"):
        fit_linear_agb_model(metrics, agb, ("h50", "copy"))
    with pytest.raises(ValueError, match="transforms"):
        LinearAGBModel(("h",), (0.0, 1.0), y_transform="cube")


def test_cross_validation_holds_out_whole_groups():
    metrics, agb, rng = _plots()
    groups = rng.integers(0, 10, 400)
    result = cross_validate(
        lambda m, a: fit_linear_agb_model(m, a, ("h50", "h95")), metrics, agb, groups
    )
    assert result["n_valid"] == 400 and set(result["fold"]) == {0, 1, 2, 3, 4}
    for group in range(10):
        assert len(set(result["fold"][groups == group])) == 1
    assert result["r2"] > 0.9 and abs(result["bias"]) < 2
    with pytest.raises(ValueError, match="spatial groups"):
        cross_validate(lambda m, a: None, metrics, agb, np.zeros(400))


def test_gedi_l4a_models_match_the_published_table():
    # Kellner et al. (2023), Table 3
    model = gedi_l4a_model("DBT x Europe")
    rh = {"rh70": np.array([10.0]), "rh98": np.array([25.0])}
    expected = 0.963 * (-96.531 + 7.175 * np.sqrt(110.0) + 2.921 * np.sqrt(125.0)) ** 2
    assert float(model(rh)[0]) == pytest.approx(expected, rel=1e-6)
    assert model.offset == 100.0 and (model.y_transform, model.x_transform) == ("sqrt", "sqrt")
    assert gedi_l4a_model("GSW").predictors == ("rh98",)
    # strata resolve to their model (Table 2); the multiplication sign is accepted
    assert gedi_l4a_model("EBT × Europe").name == "GEDI L4A DBT x Europe"
    assert gedi_l4a_model("DNT x North America").name == "GEDI L4A ENT x North America"
    assert set(GEDI_L4A_STRATA.values()) <= set(GEDI_L4A_MODELS) and len(GEDI_L4A_MODELS) == 15
    # negative RH (below the waveform ground) stays defined thanks to the offset
    assert np.isfinite(gedi_l4a_model("EBT")({"rh50": np.array([-3.0]), "rh98": np.array([9.0])}))
    with pytest.raises(ValueError, match="unknown GEDI L4A"):
        gedi_l4a_model("GSW x Oceania")


@pytest.mark.parametrize(
    "stratum, rh, agbd",
    [  # real GEDI shots (gedidb: L2A RH, GEDI04_A agbd), 2020-2021
        ("ENT x Europe", {"rh70": 6.92, "rh98": 13.36}, 74.03),
        ("ENT x North America", {"rh70": 5.53, "rh98": 24.47}, 87.068),
        ("GSW x Australia", {"rh50": 0.11, "rh80": 2.83, "rh98": 13.54}, 65.972),
        ("DBT x South America", {"rh50": 8.68, "rh98": 31.0}, 136.972),
    ],
)
def test_gedi_l4a_models_reproduce_the_product(stratum, rh, agbd):
    # the four entries where the product differs from the document's Table 3
    value = gedi_l4a_model(stratum)({k: np.array([v]) for k, v in rh.items()})
    assert float(value[0]) == pytest.approx(agbd, rel=0.01)


def test_fitted_model_runs_in_compute_biomass(provider, store):
    from alsdb.products.biomass import compute_biomass, compute_metrics

    bbox = (308000.0, 4688000.0, 309000.0, 4689000.0)
    compute_metrics(provider, store, 20.0, bbox, 2021, metrics=("h50", "h95", "cc"))
    model = LinearAGBModel(("h50", "h95"), (-1.0, 1.0, 2.0), bias_correction=1.05)
    compute_biomass(provider, store, 20.0, model, bbox, 2021, n_workers=2, parallel="processes")
    group = store._root["20m"]
    expected = model({name: group[name][0] for name in ("h50", "h95", "cc")})
    np.testing.assert_array_equal(group["biomass"][0], expected)
    assert np.isfinite(expected).any()
    runs = group["biomass"].attrs["processing_runs"]["2021"]["configuration"]
    assert runs["model_id"] == model.model_id and runs["features"] == ["h50", "h95", "cc"]
