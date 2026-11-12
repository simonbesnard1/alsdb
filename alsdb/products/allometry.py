# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Aboveground biomass models in the functional forms of GEDI L4A.

GEDI's footprint biomass product (L4A; Kellner et al. 2023, Duncanson et al.
2022) predicts aboveground biomass density (AGBD, Mg/ha) with linear models
on transformed variables::

    h(AGBD) = b0 + b1 * g(x1 + offset) + b2 * g(x2 + offset) + ...

with ``h`` and ``g`` a square root, a natural logarithm or no transformation
(a logarithm on both sides is a power law, as the Næsset-style model of
:mod:`alsdb.products.biomass`). Predictions are transformed back with a bias
correction: the ratio estimator of Snowdon (1991), ``C = sum(AGBD) /
sum(back-transformed predictions)``, or for logarithms that of Baskerville
(1972). The standard error of a prediction in transformed units is
``sqrt(MSE + x Cov(b) x')``, which gives prediction intervals.

:class:`LinearAGBModel` holds such a model and is a ``model_fn`` for
:func:`~alsdb.products.biomass.compute_biomass`; :func:`fit_linear_agb_model`
fits one to field plots with any metrics; :func:`cross_validate` holds out
spatial groups. :func:`gedi_l4a_model` returns the published L4A models.

The L4A models take the relative height (RH) metrics of 25 m GEDI
waveforms, which include the ground return (RH50 is often near zero), with
100 added before the square root. They apply to simulated waveforms
(:func:`alsdb.products.waveform.simulate_batch`, columns ``rh0`` to
``rh100``), as L4A was trained on waveforms simulated from ALS over field
plots, not to the point-cloud height percentiles of
:func:`~alsdb.products.biomass.compute_metrics`.

References
----------
Baskerville, G. L. (1972). Use of logarithmic regression in the estimation of
    plant biomass. Canadian Journal of Forest Research, 2(1), 49-53.
    https://doi.org/10.1139/x72-009
Dubayah, R. O., Armston, J., Kellner, J. R., Duncanson, L., Healey, S. P.,
    Patterson, P. L., Hancock, S., Tang, H., Bruening, J. M., Hofton, M. A.,
    Blair, J. B., & Luthcke, S. B. (2022). GEDI L4A Footprint Level Aboveground
    Biomass Density, Version 2.1. ORNL DAAC.
    https://doi.org/10.3334/ORNLDAAC/2056
Duncanson, L., et al. (2022). Aboveground biomass density models for NASA's
    Global Ecosystem Dynamics Investigation (GEDI) lidar mission. Remote
    Sensing of Environment, 270, 112845.
    https://doi.org/10.1016/j.rse.2021.112845
Kellner, J. R., Armston, J., & Duncanson, L. (2023). Algorithm theoretical
    basis document for GEDI footprint aboveground biomass density. Earth and
    Space Science, 10, e2022EA002516. https://doi.org/10.1029/2022EA002516
Snowdon, P. (1991). A ratio estimator for bias correction in logarithmic
    regressions. Canadian Journal of Forest Research, 21(5), 720-724.
    https://doi.org/10.1139/x91-101
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

_TRANSFORMS = ("none", "sqrt", "log")


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply *transform*; NaN where it is undefined (negative roots, non-positive logs)."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if transform == "sqrt":
            return np.where(values >= 0, np.sqrt(values), np.nan)
        if transform == "log":
            return np.where(values > 0, np.log(values), np.nan)
    return values


def _back(transformed: np.ndarray, transform: str) -> np.ndarray:
    """Return values in original units, before the bias correction."""
    if transform == "sqrt":
        # A negative root has no biomass: its square would turn it positive.
        return np.maximum(transformed, 0.0) ** 2
    if transform == "log":
        return np.exp(transformed)
    return transformed


@dataclass(frozen=True)
class LinearAGBModel:
    """Linear biomass model on transformed variables, as GEDI L4A.

    ``AGB = bias_correction * back(b0 + sum(b_k * g(x_k + offset)))``, with
    ``back`` the inverse of the response transformation. Instances are
    callable on a mapping of metric arrays (the ``model_fn`` of
    :func:`~alsdb.products.biomass.compute_biomass`, or a table of footprint
    metrics), hashable and picklable.

    Parameters
    ----------
    predictors : tuple of str
        Metric names, e.g. ``("h50", "h95")`` or ``("rh50", "rh98")``.
    coefficients : tuple of float
        Intercept, then one coefficient per predictor.
    y_transform, x_transform : str
        ``"sqrt"``, ``"log"`` or ``"none"``, of the response and of the
        predictors.
    offset : float
        Added to every predictor before its transformation (100 in L4A, so
        negative RH values have a square root).
    bias_correction : float
        Multiplier of the back-transformed prediction.
    bias_method : str
        How it was obtained: ``"snowdon"``, ``"baskerville"`` or ``"none"``.
    rse : float, optional
        Residual standard error, in transformed units.
    dof : int, optional
        Residual degrees of freedom.
    vcov : tuple of tuple of float, optional
        Covariance matrix of the coefficients, in transformed units.
    zero_without_cover : bool
        Give 0 where predictors are missing and the canopy cover ``cc`` is 0
        (cells with returns but no vegetation), as
        :func:`~alsdb.products.biomass.naesset_model`; ``cc`` is then
        requested with the predictors.
    name : str
        Label, part of the model identity.

    Notes
    -----
    With a square-root response, transformed predictions below zero give 0
    (their square would be positive biomass); the L4A document does not state
    how the product handles them.
    """

    predictors: tuple[str, ...]
    coefficients: tuple[float, ...]
    y_transform: str = "sqrt"
    x_transform: str = "sqrt"
    offset: float = 0.0
    bias_correction: float = 1.0
    bias_method: str = "snowdon"
    rse: float | None = None
    dof: int | None = None
    vcov: tuple[tuple[float, ...], ...] | None = None
    zero_without_cover: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the model specification."""
        if self.y_transform not in _TRANSFORMS or self.x_transform not in _TRANSFORMS:
            raise ValueError(f"transforms must be one of {_TRANSFORMS}")
        if not self.predictors or len(self.coefficients) != len(self.predictors) + 1:
            raise ValueError("coefficients must hold an intercept and one value per predictor")
        if not np.isfinite([*self.coefficients, self.offset, self.bias_correction]).all():
            raise ValueError("coefficients, offset and bias correction must be finite")
        if self.bias_correction <= 0:
            raise ValueError("bias_correction must be positive")
        if self.vcov is not None and np.shape(self.vcov) != (len(self.coefficients),) * 2:
            raise ValueError("vcov must be square with one row per coefficient")

    @property
    def required_metrics(self) -> tuple[str, ...]:
        """Metrics :func:`~alsdb.products.biomass.compute_biomass` has to provide."""
        extra = ("cc",) if self.zero_without_cover and "cc" not in self.predictors else ()
        return (*self.predictors, *extra)

    @property
    def model_id(self) -> str:
        """Stable identifier of this model (a hash of its definition)."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return "linear-agb-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _design(self, metrics: Mapping) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """Return the input shape, the design matrix and the rows it is defined for."""
        columns = [np.asarray(metrics[name], dtype=np.float64) for name in self.predictors]
        shape = columns[0].shape
        design = np.column_stack(
            [np.ones(columns[0].size)]
            + [_transform(c.ravel() + self.offset, self.x_transform) for c in columns]
        )
        return shape, design, np.isfinite(design).all(axis=1)

    def _without_cover(self, metrics: Mapping, valid: np.ndarray) -> np.ndarray:
        """Rows without predictors and without canopy cover (0 biomass)."""
        if not self.zero_without_cover or "cc" not in metrics:
            return np.zeros(len(valid), dtype=bool)
        return ~valid & (np.asarray(metrics["cc"], dtype=np.float64).ravel() == 0)

    def predict(self, metrics: Mapping, level: float | None = 0.95) -> dict[str, np.ndarray]:
        """Predict biomass with its standard error and prediction interval.

        Parameters
        ----------
        metrics : mapping
            Arrays (any common shape) for every predictor, and ``cc`` if the
            zero rule applies.
        level : float, optional
            Coverage of the prediction interval; None skips it.

        Returns
        -------
        dict
            ``agb`` (Mg/ha), ``se`` (standard error of the prediction, in
            transformed units: ``sqrt(rse^2 + x Cov(b) x')``), ``lower`` and
            ``upper`` (interval bounds back-transformed with the bias
            correction). The error terms are NaN for models without ``rse``,
            ``dof`` and ``vcov``, and where the zero rule applies.
        """
        shape, design, valid = self._design(metrics)
        nan = np.full(len(valid), np.nan)
        agb, se, lower, upper = nan.copy(), nan.copy(), nan.copy(), nan.copy()
        transformed = design[valid] @ np.asarray(self.coefficients)
        agb[valid] = self.bias_correction * _back(transformed, self.y_transform)
        if self.rse is not None and self.vcov is not None and self.dof:
            x = design[valid]
            se[valid] = np.sqrt(self.rse**2 + np.einsum("ij,jk,ik->i", x, np.asarray(self.vcov), x))
            if level is not None:
                from scipy.stats import t

                if not 0 < level < 1:
                    raise ValueError("level must be between 0 and 1")
                half = t.ppf(1 - (1 - level) / 2, self.dof) * se[valid]
                lower[valid] = self.bias_correction * _back(transformed - half, self.y_transform)
                upper[valid] = self.bias_correction * _back(transformed + half, self.y_transform)
        agb[self._without_cover(metrics, valid)] = 0.0
        return {k: v.reshape(shape) for k, v in zip(("agb", "se", "lower", "upper"), (agb, se, lower, upper))}  # fmt: skip

    def __call__(self, metrics: Mapping) -> np.ndarray:
        """Return biomass (Mg/ha, float32) for the metric arrays; NaN where undefined."""
        shape, design, valid = self._design(metrics)
        agb = np.full(len(valid), np.nan, dtype=np.float32)
        transformed = design[valid] @ np.asarray(self.coefficients)
        agb[valid] = self.bias_correction * _back(transformed, self.y_transform)
        agb[self._without_cover(metrics, valid)] = 0.0
        return agb.reshape(shape)


def fit_linear_agb_model(
    metrics: Mapping,
    agb,
    predictors: tuple[str, ...],
    *,
    y_transform: str = "sqrt",
    x_transform: str = "sqrt",
    offset: float = 0.0,
    weights=None,
    bias_method: str = "snowdon",
    zero_without_cover: bool = True,
    name: str = "",
) -> LinearAGBModel:
    """Fit a :class:`LinearAGBModel` to field plots by (weighted) least squares.

    The four functional forms GEDI L4A considers are ``y_transform="sqrt"``
    or ``"log"`` with ``x_transform="none"`` or the same transformation; all
    selected L4A models use the square root on both sides (Kellner et al.
    2023). A logarithm on both sides is a power law.

    Parameters
    ----------
    metrics : mapping
        One value per plot for every predictor (a DataFrame or a dict of
        arrays), computed at the support of the plots.
    agb : array-like
        Field biomass per plot (Mg/ha).
    predictors : tuple of str
        Metric names to use.
    y_transform, x_transform, offset
        As :class:`LinearAGBModel`.
    weights : array-like, optional
        Plot weights (L4A weights plots inversely to the number in their
        5 degree cell, so regions count equally); default equal weights.
    bias_method : str
        ``"snowdon"`` (ratio of the summed field to the summed
        back-transformed predictions), ``"baskerville"`` (logarithms only:
        ``exp(mean squared residual / 2)``) or ``"none"``.
    zero_without_cover, name
        As :class:`LinearAGBModel`.

    Returns
    -------
    LinearAGBModel
        With its residual standard error, degrees of freedom and coefficient
        covariance, so predictions carry standard errors and intervals.

    Raises
    ------
    ValueError
        With too few usable plots, or collinear predictors.
    """
    if bias_method not in ("snowdon", "baskerville", "none"):
        raise ValueError("bias_method must be 'snowdon', 'baskerville' or 'none'")
    if bias_method == "baskerville" and y_transform != "log":
        raise ValueError("the Baskerville correction applies to a logarithmic response")
    predictors = tuple(predictors)
    draft = LinearAGBModel(
        predictors, (0.0,) * (len(predictors) + 1), y_transform, x_transform, float(offset)
    )
    _, design, valid = draft._design({name: np.asarray(metrics[name]) for name in predictors})
    agb = np.asarray(agb, dtype=np.float64).ravel()
    if agb.shape != valid.shape:
        raise ValueError("agb must hold one value per plot")
    response = _transform(agb, y_transform)
    valid &= np.isfinite(response)
    w = np.ones(len(agb)) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != valid.shape:
        raise ValueError("weights must hold one value per plot")
    valid &= np.isfinite(w) & (w > 0)
    n, p = int(valid.sum()), design.shape[1]
    if n <= p + 1:
        raise ValueError(f"need more than {p + 1} usable plots to fit {p} coefficients; got {n}")
    x, y = design[valid], response[valid]
    w = w[valid] / w[valid].mean()  # mean 1: equal weights reduce to ordinary least squares
    root = np.sqrt(w)[:, None]
    normal = (x * root).T @ (x * root)
    if np.linalg.matrix_rank(normal) < p:
        raise ValueError("predictors are collinear (or constant): the fit is not unique")
    coefficients = np.linalg.solve(normal, (x * root).T @ (y * root[:, 0]))
    residual = y - x @ coefficients
    mse = float((w * residual**2).sum() / (n - p))
    raw = _back(x @ coefficients, y_transform)
    if bias_method == "snowdon" and y_transform != "none":
        correction = float(agb[valid].sum() / raw.sum())
    elif bias_method == "baskerville":
        correction = float(np.exp(np.mean(residual**2) / 2))
    else:
        bias_method, correction = "none", 1.0
    vcov = mse * np.linalg.inv(normal)
    return LinearAGBModel(
        predictors=predictors,
        coefficients=tuple(float(c) for c in coefficients),
        y_transform=y_transform,
        x_transform=x_transform,
        offset=float(offset),
        bias_correction=correction,
        bias_method=bias_method,
        rse=float(np.sqrt(mse)),
        dof=n - p,
        vcov=tuple(tuple(float(v) for v in row) for row in vcov),
        zero_without_cover=zero_without_cover,
        name=name,
    )


def cross_validate(
    fit: Callable[[dict, np.ndarray], Callable],
    metrics: Mapping,
    agb,
    spatial_groups,
    *,
    n_splits: int = 5,
) -> dict:
    """Spatially grouped cross-validation of any biomass model.

    Whole spatial groups are held out together (plots near each other are
    not independent). L4A selected its models by such geographic
    cross-validation (Kellner et al. 2023).

    Parameters
    ----------
    fit : callable
        ``fit(metrics, agb) -> model``, with ``model(metrics) -> AGB``, e.g.
        ``lambda m, a: fit_linear_agb_model(m, a, ("h50", "h95"))``.
    metrics : mapping
        One value per plot for every metric the model uses.
    agb : array-like
        Field biomass per plot (Mg/ha).
    spatial_groups : array-like
        Group label per plot (e.g. a block or region id).
    n_splits : int
        Number of folds; at most the number of groups.

    Returns
    -------
    dict
        ``prediction`` (out-of-fold, NaN where the model is undefined),
        ``fold``, ``n_valid``, ``rmse``, ``bias`` (mean prediction minus
        field biomass), ``r2`` and ``residual_quantiles`` (2.5, 50 and 97.5 %
        of field minus prediction).
    """
    agb = np.asarray(agb, dtype=np.float64).ravel()
    groups = np.asarray(spatial_groups).ravel()
    columns = {name: np.asarray(metrics[name]).ravel() for name in metrics}
    if groups.shape != agb.shape or any(c.shape != agb.shape for c in columns.values()):
        raise ValueError("metrics, agb and spatial_groups must hold one value per plot")
    unique = np.unique(groups)
    if not isinstance(n_splits, int) or n_splits < 2 or len(unique) < n_splits:
        raise ValueError("Need at least n_splits distinct spatial groups and n_splits >= 2")
    prediction = np.full(len(agb), np.nan)
    folds = np.full(len(agb), -1, dtype=int)
    for fold, held_out in enumerate(np.array_split(unique, n_splits)):
        test = np.isin(groups, held_out)
        model = fit({k: v[~test] for k, v in columns.items()}, agb[~test])
        prediction[test] = model({k: v[test] for k, v in columns.items()})
        folds[test] = fold
    valid = np.isfinite(prediction) & np.isfinite(agb)
    residual = agb[valid] - prediction[valid]
    total = ((agb[valid] - agb[valid].mean()) ** 2).sum()
    return {
        "prediction": prediction,
        "fold": folds,
        "n_valid": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(-np.mean(residual)),
        "r2": float(1 - (residual**2).sum() / total) if total > 0 else np.nan,
        "residual_quantiles": np.quantile(residual, [0.025, 0.5, 0.975]),
    }


# ---------------------------------------------------------------------------
# GEDI L4A models (Kellner et al. 2023, Table 3; releases 1 and 2 of the product)
# ---------------------------------------------------------------------------

# name: (bias correction, intercept, {RH percentile: coefficient}).
# AGBD = C * (b0 + sum(b_k * sqrt(RH_k + 100)))^2, RH in metres above the
# waveform ground. Plant functional types: DBT deciduous broadleaf, EBT
# evergreen broadleaf, ENT evergreen needleleaf trees, GSW grasses, shrubs and
# woodlands. "GSW x Australia and Oceania" is left out: the sign of its RH80
# term is printed as a plus-minus in the document.
GEDI_L4A_MODELS: dict[str, tuple[float, float, dict[int, float]]] = {
    "DBT": (1.017, -110.059, {60: 5.134, 98: 6.172}),
    "EBT": (1.113, -104.965, {50: 6.802, 98: 3.955}),
    "ENT": (1.018, -118.411, {60: 7.777, 98: 4.378}),
    "GSW": (1.118, -124.832, {98: 12.426}),
    "DBT x Africa": (1.092, -118.408, {50: 1.957, 98: 9.962}),
    "DBT x Europe": (0.963, -96.531, {70: 7.175, 98: 2.921}),
    "DBT x North America": (1.052, -120.777, {50: 5.508, 98: 6.808}),
    "EBT x Africa": (1.113, -104.965, {50: 6.802, 98: 3.955}),
    "EBT x Australia": (1.018, -155.414, {70: 7.817, 98: 7.710}),
    "EBT x South America": (1.056, -134.770, {50: 6.654, 98: 6.687}),
    "ENT x Australia": (0.898, -101.984, {40: 6.397, 98: 4.259}),
    "ENT x Europe": (1.018, -96.531, {70: 7.175, 98: 2.921}),
    "ENT x North America": (1.013, -114.355, {70: 9.401, 98: 3.346}),
}

# Prediction stratum (plant functional type x world region) -> model
# (Kellner et al. 2023, Table 2). DNT: deciduous needleleaf trees.
GEDI_L4A_STRATA: dict[str, str] = {
    "DBT x North Asia": "DBT",
    "EBT x North Asia": "DBT",
    "DBT x South Asia": "EBT",
    "EBT x South Asia": "EBT",
    "DNT x North Asia": "ENT",
    "DNT x South Asia": "ENT",
    "ENT x Africa": "ENT",
    "ENT x North Asia": "ENT",
    "ENT x South America": "ENT",
    "ENT x South Asia": "ENT",
    "GSW x Africa": "GSW",
    "GSW x Europe": "GSW",
    "GSW x North America": "GSW",
    "GSW x North Asia": "GSW",
    "GSW x South America": "GSW",
    "GSW x South Asia": "GSW",
    "DBT x Africa": "DBT x Africa",
    "DBT x Europe": "DBT x Europe",
    "EBT x Europe": "DBT x Europe",
    "DBT x North America": "DBT x North America",
    "EBT x North America": "DBT x North America",
    "EBT x Africa": "EBT x Africa",
    "DBT x Australia": "EBT x Australia",
    "EBT x Australia": "EBT x Australia",
    "DBT x South America": "EBT x South America",
    "EBT x South America": "EBT x South America",
    "DNT x Australia": "ENT x Australia",
    "ENT x Australia": "ENT x Australia",
    "DNT x Europe": "ENT x Europe",
    "ENT x Europe": "ENT x Europe",
    "DNT x North America": "ENT x North America",
    "ENT x North America": "ENT x North America",
}


def gedi_l4a_model(stratum: str) -> LinearAGBModel:
    """Return the GEDI L4A model of a prediction stratum or model name.

    Coefficients are those of Table 3 of Kellner et al. (2023), for releases
    1 and 2 of the product, transcribed from the algorithm document; check
    them against the ``model_data`` group of the L4A files before relying on
    exact agreement with the product. The document gives no residual errors
    or covariances, so these models predict without standard errors.

    The models take RH metrics of 25 m waveforms in metres (columns ``rh40``
    to ``rh98`` of :func:`alsdb.products.waveform.simulate_batch`), not
    point-cloud height percentiles.

    Parameters
    ----------
    stratum : str
        A prediction stratum, ``"<PFT> x <region>"`` (see
        :data:`GEDI_L4A_STRATA`, e.g. ``"ENT x Europe"`` or
        ``"EBT x Europe"``), or a model name of :data:`GEDI_L4A_MODELS`.

    Returns
    -------
    LinearAGBModel
        Square root on both sides, offset 100, Snowdon bias correction.
    """
    key = " ".join(stratum.replace("×", "x").split())
    name = GEDI_L4A_STRATA.get(key, key)
    if name not in GEDI_L4A_MODELS:
        raise ValueError(
            f"unknown GEDI L4A stratum or model {stratum!r}; "
            f"strata: {sorted(GEDI_L4A_STRATA)}, models: {sorted(GEDI_L4A_MODELS)}"
        )
    correction, intercept, terms = GEDI_L4A_MODELS[name]
    return LinearAGBModel(
        predictors=tuple(f"rh{level}" for level in terms),
        coefficients=(intercept, *terms.values()),
        y_transform="sqrt",
        x_transform="sqrt",
        offset=100.0,
        bias_correction=correction,
        bias_method="snowdon",
        zero_without_cover=False,
        name=f"GEDI L4A {name}",
    )
