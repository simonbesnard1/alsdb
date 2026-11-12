# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Aboveground biomass models: power law, GEDI L4A forms, machine learning.

Every model here is a ``model_fn`` for
:func:`~alsdb.products.biomass.compute_biomass`: a callable taking the metric
arrays and returning biomass (Mg/ha).

- :func:`naesset_model`: power law of height and cover, ``AGB = a * h95^b *
  cc^c`` (Næsset 2002), with :func:`calibrate_naesset` and
  :func:`validate_naesset`. The default coefficients are placeholders.
- :class:`LinearAGBModel`: linear models on transformed variables, the
  functional forms of GEDI L4A, with :func:`fit_linear_agb_model`,
  :func:`cross_validate` and the published models (:func:`gedi_l4a_model`).
- :func:`wrap_sklearn_model` and :func:`wrap_torch_model`: fitted
  machine-learning models.

GEDI's footprint biomass product (L4A; Kellner et al. 2023, Duncanson et al.
2022) predicts aboveground biomass density (AGBD, Mg/ha) with linear models
on transformed variables::

    h(AGBD) = b0 + b1 * g(x1 + offset) + b2 * g(x2 + offset) + ...

with ``h`` and ``g`` a square root, a natural logarithm or no transformation
(a logarithm on both sides is a power law, as the Næsset-style model).
Predictions are transformed back with a bias
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
Næsset, E. (2002). Predicting forest stand characteristics with airborne
    scanning laser using a practical two-stage procedure and field data.
    Remote Sensing of Environment, 80(1), 88-99.
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
        :func:`naesset_model`; ``cc`` is then
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
# Næsset-style power law
# ---------------------------------------------------------------------------

_NAESSET_DEFAULTS = (0.8, 1.8, 0.5)


def _naesset(h95, cc, a, b, c):
    """Return ``a * h95^b * cc^c``, 0 without cover (h95 is then ignored)."""
    h95, cc = np.asarray(h95, dtype=np.float64), np.asarray(cc, dtype=np.float64)
    covered = cc > 0
    with np.errstate(invalid="ignore"):
        return np.where(
            covered, a * np.power(np.where(covered, h95, 1.0), b) * np.power(cc, c), 0.0
        )


def _naesset_inputs(h95, cc):
    """Return the cells or plots the model applies to: cover in [0, 1], positive h95 if covered."""
    h95, cc = np.asarray(h95, dtype=np.float64), np.asarray(cc, dtype=np.float64)
    return np.isfinite(cc) & (cc >= 0) & (cc <= 1) & ((cc == 0) | (np.isfinite(h95) & (h95 > 0)))


def naesset_model(
    metrics: dict[str, np.ndarray],
    a: float = 0.8,
    b: float = 1.8,
    c: float = 0.5,
) -> np.ndarray:
    """Næsset-style power-law AGB model (Mg/ha): ``AGB = a * h95^b * cc^c``.

    Cells with returns but no canopy cover (``cc == 0``, including cells
    without vegetation returns, whose h95 is NaN) get 0; cells without
    first returns (``cc`` NaN) get NaN.

    Parameters
    ----------
    metrics:
        Metric dict from :func:`_extract_metrics`.
    a, b, c:
        Coefficients; the defaults are generic, calibrate them with
        :func:`calibrate_naesset`.
    """
    import warnings

    if (a, b, c) == _NAESSET_DEFAULTS:
        warnings.warn(
            "naesset_model is using uncalibrated placeholder coefficients "
            f"(a={a}, b={b}, c={c}). Results are not scientifically valid "
            "without calibration. Call calibrate_naesset(h95, cc, agb_field) "
            "with field inventory data and pass the returned coefficients explicitly.",
            UserWarning,
            stacklevel=2,
        )

    h95 = np.asarray(metrics["h95"], dtype=np.float64)
    cc = np.asarray(metrics["cc"], dtype=np.float64)
    agb = np.where(_naesset_inputs(h95, cc), _naesset(h95, cc, a, b, c), np.nan)
    return agb.astype(np.float32)


def calibrate_naesset(
    h95: np.ndarray,
    cc: np.ndarray,
    agb_field: np.ndarray,
    p0: tuple[float, float, float] = (0.8, 1.8, 0.5),
    return_cov: bool = False,
) -> tuple[float, float, float] | tuple[tuple[float, float, float], np.ndarray]:
    """Fit the coefficients of :func:`naesset_model` to field plots.

    Parameters
    ----------
    h95:
        95th height percentile per plot (m).
    cc:
        Canopy cover per plot (0-1); plots without cover (0) are kept, with
        any h95, as the model predicts 0 there.
    agb_field:
        Field AGB per plot (Mg/ha); plots with 0 are kept.
    p0:
        Initial ``(a, b, c)``.
    return_cov:
        Also return the parameter covariance matrix.

    Returns
    -------
    tuple[float, float, float]
        ``(a, b, c)``, or ``((a, b, c), pcov)`` with ``return_cov=True``.

    Raises
    ------
    ValueError
        With fewer than 20 valid plots (a warning is issued below 50).
    """
    import warnings

    from scipy.optimize import curve_fit

    h95 = np.asarray(h95, dtype=np.float64)
    cc = np.asarray(cc, dtype=np.float64)
    agb_field = np.asarray(agb_field, dtype=np.float64)

    valid = _naesset_inputs(h95, cc) & np.isfinite(agb_field) & (agb_field >= 0)
    n_valid = int(valid.sum())

    if n_valid < 20:
        raise ValueError(
            f"Need at least 20 valid field plots to fit a 3-parameter model; "
            f"got {n_valid} (after removing NaN / non-positive values). "
            "With fewer plots the fit is severely overfit (near-zero degrees of freedom)."
        )
    if n_valid < 50:
        warnings.warn(
            f"calibrate_naesset: only {n_valid} valid field plots. "
            "Reliable coefficient estimation typically requires ≥ 50 independent plots "
            "(Næsset 2002; Andersen et al. 2011). Treat results with caution.",
            UserWarning,
            stacklevel=2,
        )

    def _model(X, a, b, c):
        h, cv = X
        return _naesset(h, cv, a, b, c)

    h95 = np.where(cc == 0, 1.0, h95)  # ignored without cover; curve_fit rejects NaN
    popt, pcov = curve_fit(
        _model,
        (h95[valid], cc[valid]),
        agb_field[valid],
        p0=list(p0),
        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        maxfev=10_000,
    )
    coeffs = (float(popt[0]), float(popt[1]), float(popt[2]))
    if return_cov:
        return coeffs, pcov
    return coeffs


def validate_naesset(h95, cc, agb_field, spatial_groups, *, n_splits=5):
    """Spatially grouped cross-validation of :func:`naesset_model`.

    Whole spatial groups are held out together. Returns out-of-fold
    predictions, residual RMSE, bias (mean prediction minus field AGB) and
    residual quantiles (prediction error on the plots, not pixel confidence
    intervals). Plots are selected as in :func:`calibrate_naesset`.
    """
    h95, cc, agb = (np.asarray(a, dtype=float) for a in (h95, cc, agb_field))
    groups = np.asarray(spatial_groups)
    if not (h95.shape == cc.shape == agb.shape == groups.shape) or h95.ndim != 1:
        raise ValueError("Plot inputs must have matching one-dimensional shapes")
    valid = _naesset_inputs(h95, cc) & np.isfinite(agb) & (agb >= 0)
    unique = np.unique(groups[valid])
    if not isinstance(n_splits, int) or n_splits < 2 or len(unique) < n_splits:
        raise ValueError("Need at least n_splits distinct spatial groups and n_splits >= 2")
    prediction = np.full(len(h95), np.nan)
    folds = np.full(len(h95), -1, dtype=int)
    for fold, held_out in enumerate(np.array_split(unique, n_splits)):
        test = valid & np.isin(groups, held_out)
        train = valid & ~test
        a, b, c = calibrate_naesset(h95[train], cc[train], agb[train])
        prediction[test] = _naesset(h95[test], cc[test], a, b, c)
        folds[test] = fold
    residual = agb[valid] - prediction[valid]
    return {
        "prediction": prediction,
        "fold": folds,
        "n_valid": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(prediction[valid] - agb[valid])),
        "residual_quantiles": np.quantile(residual, [0.025, 0.5, 0.975]),
    }


# ---------------------------------------------------------------------------
# Machine-learning models
# ---------------------------------------------------------------------------


def wrap_sklearn_model(
    estimator,
    features: list[str] | None = None,
    zero_without_cover: bool = True,
    model_id: str | None = None,
) -> Callable:
    """Wrap a fitted model with ``predict(X)`` as a ``model_fn`` for :func:`compute_biomass`.

    Works for scikit-learn estimators and anything with the same ``predict``
    interface (XGBoost, LightGBM, a Keras model; PyTorch networks go through
    :func:`wrap_torch_model`):
    ``X`` is (cells, features) in *features* order, predictions of shape
    (cells,) or (cells, 1). Models trained on a DataFrame receive one with
    the same column names. Tiles run in threads, so ``predict`` may be called
    concurrently when ``n_workers > 1``.

    Parameters
    ----------
    estimator:
        Fitted model with ``predict(X)``.
    features:
        Metric names in training order. Default: the estimator's
        ``feature_names_in_`` (fitted on a DataFrame), otherwise all metrics
        in :data:`_METRIC_NAMES` order. Given together with
        ``feature_names_in_``, they must match it.
    zero_without_cover:
        Give 0 to cells without canopy cover (``cc == 0``, needs ``cc`` among
        the features) whose other features are missing, e.g. height metrics
        of cells without vegetation returns, as :func:`naesset_model`. Cells
        with all features are always predicted.
    model_id:
        Identifier of the fitted model for safe resume; default a hash of
        the model and features.

    Returns
    -------
    Callable
        ``model_fn(metrics) -> np.ndarray``; cells with missing features are
        NaN (see *zero_without_cover*). Its ``model_id`` is a hash of the
        model and features, or None if the model cannot be pickled: pass
        ``model_id`` to :func:`compute_biomass` then, or it is recomputed.

    Raises
    ------
    ValueError
        If *features* are unknown or disagree with the estimator's feature
        names or feature count.
    """
    import warnings

    from alsdb.products.biomass import _METRIC_NAMES

    names = getattr(estimator, "feature_names_in_", None)
    names = None if names is None else [str(n) for n in names]
    if features is None:
        feat = list(names) if names is not None else list(_METRIC_NAMES)
    else:
        feat = list(features)
        if names is not None and feat != names:
            raise ValueError(
                f"features {feat} differ from the names the estimator was fitted with {names}"
            )
    unknown = set(feat) - set(_METRIC_NAMES)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and expected != len(feat):
        raise ValueError(
            f"The estimator expects {expected} features, got {len(feat)}: pass features="
        )
    zero = zero_without_cover and "cc" in feat

    def _model(metrics: dict[str, np.ndarray]) -> np.ndarray:
        shape = metrics[feat[0]].shape
        # Stack into (n_pixels, n_features); ravel preserves north-up order
        X = np.column_stack([np.asarray(metrics[k], dtype=np.float64).ravel() for k in feat])
        valid = np.isfinite(X).all(axis=1)
        result = np.full(X.shape[0], np.nan, dtype=np.float32)
        if valid.any():
            inputs = X[valid]
            if names is not None:
                import pandas as pd

                inputs = pd.DataFrame(inputs, columns=names)
            predicted = np.asarray(estimator.predict(inputs), dtype=np.float32)
            if predicted.size != int(valid.sum()):
                raise ValueError(
                    f"predict returned {predicted.shape} for {int(valid.sum())} cells; "
                    "expected one value per cell"
                )
            result[valid] = predicted.reshape(-1)
        if zero:
            cover = np.asarray(metrics["cc"], dtype=np.float64).ravel()
            result[~valid & (cover == 0)] = 0.0
        return result.reshape(shape)

    _model.required_metrics = tuple(feat)
    import joblib

    if model_id is not None:
        _model.model_id = model_id
        return _model
    try:
        _model.model_id = joblib.hash((estimator, feat, zero))
    except Exception as exc:  # noqa: BLE001 - pickling failures vary by framework
        warnings.warn(
            f"Cannot hash the model ({type(exc).__name__}); pass model_id to reuse results",
            UserWarning,
            stacklevel=2,
        )
        _model.model_id = None
    return _model


class _TorchPredictor:
    """``predict(X)`` for a PyTorch network: scaled float32 batches in evaluation mode."""

    def __init__(self, net, device, transform, batch_size):
        import threading

        self.net = net.to(device).eval()
        self.device, self.transform, self.batch_size = device, transform, batch_size
        # one forward pass at a time: tiles run in threads and share the network
        self.lock = threading.Lock()

    def predict(self, X):
        import torch

        X = np.asarray(self.transform(X) if self.transform is not None else X, np.float32)
        outputs = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(X[start : start + self.batch_size]).to(self.device)
                outputs.append(self.net(batch).float().cpu().numpy().reshape(-1))
        return np.concatenate(outputs) if outputs else np.empty(0, np.float32)


def wrap_torch_model(
    net,
    features: list[str],
    *,
    transform: Callable | None = None,
    device: str = "cpu",
    batch_size: int = 65_536,
    model_id: str | None = None,
    zero_without_cover: bool = True,
) -> Callable:
    """Wrap a trained PyTorch network as a ``model_fn`` for :func:`compute_biomass`.

    The network gets a float32 tensor of shape (cells, features), in
    *features* order, and returns one value per cell, (cells,) or (cells, 1).
    It runs in evaluation mode without gradients, in batches, one forward
    pass at a time (tiles run in threads and share it). Requires ``torch``.

    Preferably the network normalises its inputs itself, with the training
    mean and standard deviation as buffers (``register_buffer``): they are
    saved with the weights, follow the network to *device* and enter the
    default ``model_id``. ``transform`` is for networks trained with an
    external scaler. See doc/area_products.md for an example.

    Parameters
    ----------
    net:
        Trained ``torch.nn.Module`` (moved to *device*, set to eval mode).
    features:
        Metric names in training order; a network does not record them, so
        a wrong order is not detected.
    transform:
        External input scaling used in training, applied to the (cells,
        features) array before the network, e.g. a fitted scikit-learn
        ``StandardScaler().transform``; None (default) when the network
        normalises its inputs. Predictions are returned as the network gives
        them: undo any target scaling inside the network.
    device:
        Torch device, e.g. ``"cpu"`` or ``"cuda"``.
    batch_size:
        Cells per forward pass.
    model_id:
        Identifier of the trained model for safe resume. Default: a hash of
        the network class, its weights, *features* and *transform*; None
        (recomputed each run, with a warning) if the transform cannot be
        pickled, e.g. a lambda: pass one then.
    zero_without_cover:
        As :func:`wrap_sklearn_model`.

    Returns
    -------
    Callable
        ``model_fn(metrics) -> np.ndarray``, as :func:`wrap_sklearn_model`.
    """
    import warnings

    import joblib

    if not features:
        raise ValueError("features must name the network's inputs in training order")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = wrap_sklearn_model(
        _TorchPredictor(net, device, transform, int(batch_size)),
        list(features),
        zero_without_cover=zero_without_cover,
        model_id="pending",
    )
    if model_id is None:
        try:
            weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            model_id = joblib.hash(
                (type(net).__qualname__, weights, list(features), transform, zero_without_cover)
            )
        except Exception as exc:  # noqa: BLE001 - e.g. a lambda transform cannot be pickled
            warnings.warn(
                f"Cannot hash the network or transform ({type(exc).__name__}, e.g. a lambda); "
                "pass model_id to reuse results",
                UserWarning,
                stacklevel=2,
            )
    model.model_id = model_id
    return model


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
