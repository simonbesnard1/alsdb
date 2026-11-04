# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Above-ground biomass (AGB) from ALS point clouds

Points are normalised with a buffered terrain TIN, per-cell metrics are
computed (height percentiles, canopy cover, density, foliage height
diversity, height-stratum proportions) and an allometric model turns them
into AGB (Mg/ha), written to an :class:`~alsdb.storage.ALSZarrStore`.

Height metrics use all vegetation returns; canopy cover (``cc``) uses first
returns of any class.

The default model is a Næsset-style power law, ``AGB = a * h95^b * cc^c``
with generic coefficients ``a=0.8, b=1.8, c=0.5``: calibrate it against
field plots (:func:`calibrate_naesset`) before using the results.

Example::

    compute_metrics(provider, store, resolution=10.0, year=2021)
    compute_biomass(provider, store, resolution=10.0, year=2021)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from alsdb.methods._grid import flip_to_north_up
from alsdb.methods._terrain import VEG_CLASSES as _VEG_CLASSES
from alsdb.products._tiling import baba_neighbourhoods

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)

_DEFAULT_CC_THRESHOLD = 2.0  # m — first returns above this count as "canopy"

_METRIC_NAMES = [
    # Height percentiles
    "h50",
    "h75",
    "h95",
    "hmax",
    "hmean",
    # Canopy structure
    "cc",
    "density",
    "fhd",
    "vci",
    "crr",
    # Height stratum proportions (fraction of veg returns per layer)
    "pv_0_2",
    "pv_2_5",
    "pv_5_10",
    "pv_10_20",
    "pv_20_40",
    "pv_above40",
]

# Height strata for pv_* metrics: (lower, upper) bounds in metres.
# The last stratum is open-ended (upper = ∞).
_HEIGHT_STRATA: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, np.inf),
)
_HEIGHT_STRATA_NAMES: tuple[str, ...] = (
    "pv_0_2",
    "pv_2_5",
    "pv_5_10",
    "pv_10_20",
    "pv_20_40",
    "pv_above40",
)

# FHD/VCI vertical binning: 1 m bands up to _FHD_MAX_H. VCI divides by
# log(_FHD_N_BINS), the total bin count (Schneider et al. 2017), so it grows
# with canopy height as well as evenness.
_FHD_BIN_SIZE: float = 1.0
_FHD_MAX_H: float = 80.0  # raised from 60 m to cover tall tropical/boreal forests
_FHD_BINS = np.arange(0.0, _FHD_MAX_H + _FHD_BIN_SIZE, _FHD_BIN_SIZE)
_FHD_N_BINS: int = len(_FHD_BINS) - 1  # number of 1 m bands
_VCI_MAX_ENTROPY: float = np.log(_FHD_N_BINS)


def _fhd_from_hag(hag_vals: np.ndarray) -> float:
    """Return the foliage height diversity (Shannon entropy of the height distribution)."""
    if len(hag_vals) == 0:
        return np.nan
    counts, _ = np.histogram(hag_vals, bins=_FHD_BINS)
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        return np.nan
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def _vci_from_hag(hag_vals: np.ndarray) -> float:
    """Return the vegetation complexity index (FHD normalised by its maximum); it grows with canopy height."""
    fhd = _fhd_from_hag(hag_vals)
    if np.isnan(fhd) or _VCI_MAX_ENTROPY == 0:
        return np.nan
    return float(fhd / _VCI_MAX_ENTROPY)


# ---------------------------------------------------------------------------
# Per-cell metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-cell structural metrics over *bbox*

    Parameters
    ----------
    min_density:
        Cells with fewer points per m2 get NaN; 0 disables the check.
    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``); check the
        classes present in your survey.

    Returns a dict ``{name: (ny, nx) float32 array}``, north-up, NaN where empty.
    """
    from alsdb.methods._grid import GridSpec

    requested = tuple(_METRIC_NAMES if metrics is None else dict.fromkeys(metrics))
    unknown = set(requested) - set(_METRIC_NAMES)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    grid = GridSpec.from_bbox(bbox, resolution)
    inside, cells = grid.point_bins(points)
    n = grid.nx * grid.ny
    density = np.bincount(cells, minlength=n) / resolution**2
    hag = points["HeightAboveGround"][inside]
    valid = np.isfinite(hag)
    vegetation = valid & np.isin(points["Classification"][inside], veg_classes) & (hag > 0)
    cv, hv = cells[vegetation], hag[vegetation].astype(np.float64)
    nv = np.bincount(cv, minlength=n)
    out = {}
    if "density" in requested:
        out["density"] = density
    if "cc" in requested:
        first = valid & (points["ReturnNumber"][inside] == 1)
        denominator = np.bincount(cells[first], minlength=n)
        numerator = np.bincount(cells[first], weights=hag[first] > cc_threshold, minlength=n)
        out["cc"] = np.divide(numerator, denominator, out=np.full(n, np.nan), where=denominator > 0)
    if {"hmean", "crr"} & set(requested):
        mean = np.divide(
            np.bincount(cv, weights=hv, minlength=n), nv, out=np.full(n, np.nan), where=nv > 0
        )
        out["hmean"] = mean
    for name, (lo, hi) in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA):
        if name in requested:
            counts = np.bincount(cv, weights=(hv > lo) & (hv <= hi), minlength=n)
            out[name] = np.divide(counts, nv, out=np.full(n, np.nan), where=nv > 0)
    if {"h50", "h75", "h95", "hmax", "crr"} & set(requested):
        order = np.lexsort((hv, cv))
        heights = hv[order]
        occupied, starts, counts = np.unique(cv[order], return_index=True, return_counts=True)
        for name, q in (("h50", 0.5), ("h75", 0.75), ("h95", 0.95), ("hmax", 1.0)):
            if name in requested or (name == "hmax" and "crr" in requested):
                rank = q * (counts - 1)
                low = np.floor(rank).astype(int)
                high = np.ceil(rank).astype(int)
                values = np.full(n, np.nan)
                values[occupied] = heights[starts + low] + (rank - low) * (
                    heights[starts + high] - heights[starts + low]
                )
                out[name] = values
        if "crr" in requested:
            minimum = np.full(n, np.nan)
            minimum[occupied] = heights[starts]
            spread = out["hmax"] - minimum
            out["crr"] = np.divide(mean - minimum, spread, out=np.full(n, np.nan), where=spread > 0)
    if {"fhd", "vci"} & set(requested):
        # Sparse occupied (cell, height-band) counts avoid a cells × bands cube.
        keep = hv <= _FHD_MAX_H
        bands = np.minimum((hv[keep] / _FHD_BIN_SIZE).astype(int), _FHD_N_BINS - 1)
        keys, counts = np.unique(cv[keep] * _FHD_N_BINS + bands, return_counts=True)
        cell = keys // _FHD_N_BINS
        totals = np.bincount(cell, weights=counts, minlength=n)
        probabilities = counts / totals[cell]
        entropy = np.bincount(
            cell, weights=-probabilities * np.log(probabilities), minlength=n
        ).astype(float)
        entropy[totals == 0] = np.nan
        out["fhd"] = entropy
        out["vci"] = entropy / _VCI_MAX_ENTROPY
    result = {}
    for name in requested:
        values = out[name].astype(np.float32).reshape(grid.shape)
        if min_density > 0:
            values[density.reshape(grid.shape) < min_density] = np.nan
        result[name] = values
    return result


# ---------------------------------------------------------------------------
# Allometric model
# ---------------------------------------------------------------------------


_NAESSET_DEFAULTS = (0.8, 1.8, 0.5)


def naesset_model(
    metrics: dict[str, np.ndarray],
    a: float = 0.8,
    b: float = 1.8,
    c: float = 0.5,
) -> np.ndarray:
    """Næsset-style power-law AGB model (Mg/ha): ``AGB = a * h95^b * cc^c``

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

    h95 = metrics["h95"]
    cc = metrics["cc"]
    with np.errstate(invalid="ignore"):
        agb = np.where(
            ~np.isfinite(h95) | ~np.isfinite(cc) | (h95 <= 0) | (cc <= 0) | (cc > 1),
            np.nan,
            a * np.power(h95, b) * np.power(cc, c),
        )
    return agb.astype(np.float32)


# ---------------------------------------------------------------------------
# Model calibration
# ---------------------------------------------------------------------------


def calibrate_naesset(
    h95: np.ndarray,
    cc: np.ndarray,
    agb_field: np.ndarray,
    p0: tuple[float, float, float] = (0.8, 1.8, 0.5),
    return_cov: bool = False,
) -> tuple[float, float, float] | tuple[tuple[float, float, float], np.ndarray]:
    """Fit the coefficients of :func:`naesset_model` to field plots

    Parameters
    ----------
    h95:
        95th height percentile per plot (m).
    cc:
        Canopy cover per plot (0-1).
    agb_field:
        Field AGB per plot (Mg/ha).
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

    valid = (
        np.isfinite(h95)
        & np.isfinite(cc)
        & np.isfinite(agb_field)
        & (cc > 0)
        & (cc <= 1)
        & (h95 > 0)
        & (agb_field > 0)
    )
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
        return a * np.power(h, b) * np.power(cv, c)

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


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def wrap_sklearn_model(
    estimator,
    features: list[str] | None = None,
) -> Callable:
    """Wrap a fitted scikit-learn estimator as a ``model_fn`` for :func:`compute_biomass`

    Parameters
    ----------
    estimator:
        Fitted estimator with ``predict(X)``.
    features:
        Metric names in training order; default all metrics.

    Returns
    -------
    Callable
        ``model_fn(metrics) -> np.ndarray``; NaN cells are not predicted.
    """
    feat = list(features) if features is not None else _METRIC_NAMES

    def _model(metrics: dict[str, np.ndarray]) -> np.ndarray:
        shape = metrics[feat[0]].shape
        # Stack into (n_pixels, n_features); ravel preserves north-up order
        X = np.column_stack([metrics[k].ravel() for k in feat])
        valid = np.isfinite(X).all(axis=1)
        result = np.full(X.shape[0], np.nan, dtype=np.float32)
        if valid.any():
            result[valid] = estimator.predict(X[valid]).astype(np.float32)
        return result.reshape(shape)

    _model.required_metrics = tuple(feat)
    import joblib

    _model.model_id = joblib.hash((estimator, feat))
    return _model


# ---------------------------------------------------------------------------
# BABA (Buffered Area-Based Approach) metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics_baba(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    baba_radius: float,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-cell metrics from all points within *baba_radius* of each cell centre

    Buffered area-based approach: robust at fine resolutions. The points must
    extend at least *baba_radius* beyond *bbox*.

    Parameters
    ----------
    min_density:
        Neighbourhoods with fewer points per m2 get NaN.
    """
    requested = tuple(_METRIC_NAMES if metrics is None else dict.fromkeys(metrics))
    if set(requested) - set(_METRIC_NAMES):
        raise ValueError("Unknown structural metric")
    nx, ny, neighborhoods, area = baba_neighbourhoods(points, resolution, bbox, baba_radius)
    out = {name: np.full(nx * ny, np.nan, np.float32) for name in requested}
    requested_set = set(requested)
    quantiles = tuple(
        (name, q) for name, q in (("h50", 50), ("h75", 75), ("h95", 95)) if name in requested_set
    )
    strata = tuple(
        (name, bounds)
        for name, bounds in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA)
        if name in requested_set
    )
    needs_entropy = bool({"fhd", "vci"} & requested_set)
    hag = points["HeightAboveGround"]
    veg = np.isfinite(hag) & (hag > 0) & np.isin(points["Classification"], veg_classes)
    first = np.isfinite(hag) & (points["ReturnNumber"] == 1)
    for cell, idx in enumerate(neighborhoods):
        density = len(idx) / area
        if density < min_density:
            continue
        if "density" in out:
            out["density"][cell] = density
        if not len(idx):
            continue
        h = hag[idx]
        if "cc" in out and first[idx].any():
            out["cc"][cell] = np.mean(h[first[idx]] > cc_threshold)
        hv = h[veg[idx]]
        if not len(hv):
            continue
        if quantiles:
            for (name, _), value in zip(quantiles, np.percentile(hv, [q for _, q in quantiles])):
                out[name][cell] = value
        height_mean = hv.mean() if {"hmean", "crr"} & requested_set else None
        height_max = hv.max() if {"hmax", "crr"} & requested_set else None
        if "hmean" in requested_set:
            out["hmean"][cell] = height_mean
        if "hmax" in requested_set:
            out["hmax"][cell] = height_max
        if "crr" in requested_set:
            height_min = hv.min()
            if height_max > height_min:
                out["crr"][cell] = (height_mean - height_min) / (height_max - height_min)
        if needs_entropy:
            entropy = _fhd_from_hag(hv)
            if "fhd" in requested_set:
                out["fhd"][cell] = entropy
            if "vci" in requested_set:
                out["vci"][cell] = entropy / _VCI_MAX_ENTROPY
        for name, (lo, hi) in strata:
            out[name][cell] = np.mean((hv > lo) & (hv <= hi))
    return {name: flip_to_north_up(values.reshape(ny, nx)) for name, values in out.items()}


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------


def compute_metrics(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 10.0,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
    baba_radius: float = 0.0,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    metrics: tuple[str, ...] | None = None,
    ground_outlier_removal: bool = True,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = False,
) -> None:
    """Compute structural metrics and write them to *store*

    Writes h50, h75, h95, hmax, hmean, cc, density, fhd, vci, crr and the
    height-stratum proportions pv_0_2 ... pv_above40.

    Parameters
    ----------
    provider:
        Point cloud provider.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Cell size (m).
    bbox:
        Optional ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year.
    cc_threshold:
        Height (m) defining canopy for cover.
    min_density:
        Cells with fewer points per m2 get NaN; 0 disables the check.
    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``); check the
        classes present in your survey.
    tile_size:
        Processing tile size (m).
    tile_buffer:
        Buffer for terrain interpolation (m).
    n_workers:
        Parallel workers.
    """
    from alsdb.products._area import compute_area_products

    compute_area_products(
        provider,
        store,
        resolution,
        bbox,
        year,
        cc_threshold=cc_threshold,
        baba_radius=baba_radius,
        min_density=min_density,
        veg_classes=veg_classes,
        overwrite=overwrite,
        tile_size=tile_size,
        tile_buffer=tile_buffer,
        n_workers=n_workers,
        ground_outlier_removal=ground_outlier_removal,
        max_ground_distance=max_ground_distance,
        ground_extrapolation=ground_extrapolation,
        metrics=metrics,
    )


def compute_biomass(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 10.0,
    model_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
    baba_radius: float = 0.0,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    model_features: tuple[str, ...] | None = None,
    model_id: str | None = None,
    ground_outlier_removal: bool = True,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = False,
) -> None:
    """Estimate above-ground biomass and write it to *store*

    Parameters
    ----------
    provider:
        Point cloud provider.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Cell size (m).
    model_fn:
        ``model_fn(metrics) -> AGB grid`` (Mg/ha); default :func:`naesset_model`.
    bbox:
        Optional ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year.
    cc_threshold:
        Height (m) defining canopy for cover.
    min_density:
        Cells with fewer points per m2 get NaN.
    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``); check the
        classes present in your survey.
    overwrite:
        Recompute when biomass for *year* exists.
    tile_size:
        Processing tile size (m).
    tile_buffer:
        Buffer for terrain interpolation (m).
    n_workers:
        Parallel workers.
    """
    from alsdb.products._area import compute_area_products

    compute_area_products(
        provider,
        store,
        resolution,
        bbox,
        year,
        cc_threshold=cc_threshold,
        baba_radius=baba_radius,
        min_density=min_density,
        veg_classes=veg_classes,
        overwrite=overwrite,
        tile_size=tile_size,
        tile_buffer=tile_buffer,
        n_workers=n_workers,
        ground_outlier_removal=ground_outlier_removal,
        max_ground_distance=max_ground_distance,
        ground_extrapolation=ground_extrapolation,
        metrics=(),
        biomass=True,
        model_fn=model_fn,
        model_features=model_features,
        model_id=model_id,
    )


def validate_naesset(h95, cc, agb_field, spatial_groups, *, n_splits=5):
    """Spatially grouped cross-validation of :func:`naesset_model`

    Whole spatial groups are held out together. Returns out-of-fold
    predictions, residual RMSE and bias, and residual quantiles (prediction
    error on the plots, not pixel confidence intervals).
    """
    h95, cc, agb = (np.asarray(a, dtype=float) for a in (h95, cc, agb_field))
    groups = np.asarray(spatial_groups)
    if not (h95.shape == cc.shape == agb.shape == groups.shape) or h95.ndim != 1:
        raise ValueError("Plot inputs must have matching one-dimensional shapes")
    valid = (
        np.isfinite(h95)
        & np.isfinite(cc)
        & np.isfinite(agb)
        & (h95 > 0)
        & (cc > 0)
        & (cc <= 1)
        & (agb > 0)
    )
    unique = np.unique(groups[valid])
    if not isinstance(n_splits, int) or n_splits < 2 or len(unique) < n_splits:
        raise ValueError("Need at least n_splits distinct spatial groups and n_splits >= 2")
    prediction = np.full(len(h95), np.nan)
    folds = np.full(len(h95), -1, dtype=int)
    for fold, held_out in enumerate(np.array_split(unique, n_splits)):
        test = valid & np.isin(groups, held_out)
        train = valid & ~test
        a, b, c = calibrate_naesset(h95[train], cc[train], agb[train])
        prediction[test] = a * h95[test] ** b * cc[test] ** c
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
