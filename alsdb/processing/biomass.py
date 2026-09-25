# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Above-Ground Biomass (AGB) estimation from TileDB ALS point clouds.

Pipeline
--------
1. Query TileDB → numpy structured array.
2. Interpolate a buffered terrain TIN to attach ``HeightAboveGround``;
   unsupported heights remain NaN unless extrapolation is explicitly enabled.
3. Compute per-cell LiDAR metrics in Python/scipy:

   ========  ===============================================================
   h50       50th-percentile HAG of vegetation points (m)
   h75       75th-percentile HAG of vegetation points (m)
   h95       95th-percentile HAG of vegetation points (m)
   hmean     Mean HAG of vegetation points (m)
   cc        Canopy cover — fraction of first returns with HAG > threshold
   density   Total point density (points m⁻²)
   ========  ===============================================================

   ``cc`` and the height metrics are deliberately computed on different
   point populations, not an oversight: ``cc`` uses *first returns only*
   and *no classification filter* (any point with ``ReturnNumber == 1``
   with supported height counts, matching the first-return-cover definition used in
   Næsset-style ABA studies), while ``h50``/``h75``/``h95``/``hmean``/etc.
   use *all returns*, restricted to vegetation classes (see ``VEG_CLASSES``).
   This means ``cc`` and ``h95`` respond differently to point density -
   worth knowing before treating them as directly comparable, and worth
   reconsidering if your survey's return/classification conventions differ
   from what this was designed against.

4. Apply an allometric model ``AGB = f(metrics)`` → Mg ha⁻¹.
5. Write results directly to an :class:`~alsdb.storage.ALSZarrStore`.

Default model
-------------
A Næsset-style power law::

    AGB = a × h95^b × cc^c

with default coefficients ``a=0.8, b=1.8, c=0.5``.  These are approximate
generic values — **calibrate against field inventory plots** for your region
and species composition before using the output scientifically.

Usage::

    from alsdb import ALSProvider
    from alsdb.storage import ALSZarrStore
    from alsdb.processing.biomass import compute_biomass, compute_metrics, wrap_sklearn_model

    provider = ALSProvider(storage_type="local", uri="array_")
    store = ALSZarrStore("output/spain.zarr")

    # Structural metrics
    compute_metrics(provider, store, resolution=10.0, year=2021)

    # AGB with default model
    compute_biomass(provider, store, resolution=10.0, year=2021)

    # AGB with a custom model
    def my_model(metrics):
        return 1.2 * metrics["h95"] ** 2.1 * metrics["cc"] ** 0.6

    compute_biomass(provider, store, resolution=10.0, year=2021,
                    model_fn=my_model)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from alsdb.processing._tiling import (
    VEG_CLASSES as _VEG_CLASSES,
)
from alsdb.processing._tiling import (
    baba_neighbourhoods,
    flip_to_north_up,
)

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

# FHD/VCI vertical binning: 1 m bands up to _FHD_MAX_H.
# Normalisation divides by log(_FHD_N_BINS) - the TOTAL bin count, not the
# number actually occupied - a fixed reference denominator convention (in
# the vein of Schneider et al. 2017's normalize-to-[0,1] approach), not a
# height-independent one: a cell's *maximum achievable* entropy is bounded
# by log(occupied bins), which can never exceed log(bins its own canopy
# height actually spans). A perfectly even 5 m canopy can occupy at most 5
# of the 80 bins, capping its VCI near log(5)/log(80) ~= 0.37, regardless of
# how even its distribution is - so VCI conflates vertical evenness with
# absolute canopy height, it does not factor height out. If a pure evenness
# metric (independent of height) is what's needed instead, normalise by
# log(occupied bins) or log(bins up to that cell's own hmax) rather than
# log(_FHD_N_BINS).
_FHD_BIN_SIZE: float = 1.0
_FHD_MAX_H: float = 80.0  # raised from 60 m to cover tall tropical/boreal forests
_FHD_BINS = np.arange(0.0, _FHD_MAX_H + _FHD_BIN_SIZE, _FHD_BIN_SIZE)
_FHD_N_BINS: int = len(_FHD_BINS) - 1  # number of 1 m bands
_VCI_MAX_ENTROPY: float = np.log(_FHD_N_BINS)


def _fhd_from_hag(hag_vals: np.ndarray) -> float:
    """Shannon entropy of the vertical HAG distribution (Foliage Height Diversity)."""
    if len(hag_vals) == 0:
        return np.nan
    counts, _ = np.histogram(hag_vals, bins=_FHD_BINS)
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        return np.nan
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def _vci_from_hag(hag_vals: np.ndarray) -> float:
    """Vegetation Complexity Index — FHD normalised by log(_FHD_N_BINS).

    Not height-independent: this rewards tall *and* evenly-filled canopies,
    it does not isolate evenness from height (see the comment above
    _FHD_BIN_SIZE) - a short canopy cannot reach the same VCI as a tall one
    even if both are perfectly even, since a short canopy structurally
    cannot occupy as many of the fixed bins.
    """
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
    """
    Compute per-cell LiDAR structural metrics over *bbox*.

    Returns a dict ``{name: (ny, nx) float32 array}`` in north-up
    orientation.  Empty cells are ``np.nan``.

    Parameters
    ----------
    min_density:
        Minimum total point density (pts m⁻²) required for a cell to receive
        metric values.  Cells below this threshold are set to ``np.nan`` for
        all metrics.  Default ``0.0`` disables the guard.
    veg_classes:
        LAS classification codes treated as vegetation (default ``(3, 4,
        5)``, standard ASPRS). Not universal across surveys -- see
        ``pavd.py``'s module docstring for a real dataset that uses a
        different code entirely; verify with
        ``np.unique(points["Classification"], return_counts=True)`` before
        trusting the default on a new array.
    """
    from alsdb.processing._grid import GridSpec

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
    """
    Næsset-style power-law AGB model (Mg ha⁻¹).

    ``AGB = a × h95^b × cc^c``

    Parameters
    ----------
    metrics:
        Dict as returned by :func:`_extract_metrics`.
    a, b, c:
        Model coefficients.  Defaults are placeholder generic values —
        **always calibrate against field inventory plots** before using
        results scientifically.  Use :func:`calibrate_naesset` to fit
        region-specific coefficients.
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
    """
    Fit Næsset power-law AGB coefficients (a, b, c) to field-plot data.

    Solves ``AGB = a × h95^b × cc^c`` via nonlinear least-squares
    (``scipy.optimize.curve_fit``).

    Parameters
    ----------
    h95:
        P95 canopy height from ALS (m), one value per field plot.
    cc:
        Canopy cover fraction (0–1), one value per field plot.
    agb_field:
        Measured AGB from field inventory (Mg ha⁻¹), one value per field plot.
    p0:
        Initial parameter guess ``(a, b, c)``.  Defaults to the generic priors.
    return_cov:
        If ``True``, also return the 3×3 parameter covariance matrix from
        ``curve_fit`` so callers can assess calibration uncertainty.  Diagonal
        elements are the variance of each coefficient; off-diagonals are
        cross-covariances.  Infinite values indicate a poorly constrained fit.

    Returns
    -------
    tuple[float, float, float]
        Fitted ``(a, b, c)`` coefficients for use in :func:`naesset_model`.
        When ``return_cov=True``, returns ``((a, b, c), pcov)`` instead.

    Raises
    ------
    ValueError
        If fewer than 20 valid (non-NaN, non-zero) field plots are provided.
        A 3-parameter power-law model fit on fewer plots has nearly zero
        degrees of freedom and will be severely overfit.

    Warns
    -----
    UserWarning
        If valid plot count is below 50, a calibration-quality warning is
        emitted.  Reliable coefficient estimation typically requires ≥ 50
        independent plots (Næsset 2002; Andersen et al. 2011).

    Example
    -------
    ::

        a, b, c = calibrate_naesset(h95_plots, cc_plots, agb_plots)
        compute_biomass(provider, store, resolution=10.0, year=2021,
                        model_fn=lambda m: naesset_model(m, a=a, b=b, c=c))

        # With uncertainty:
        (a, b, c), pcov = calibrate_naesset(h95_plots, cc_plots, agb_plots,
                                             return_cov=True)
        a_std, b_std, c_std = np.sqrt(np.diag(pcov))
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
    """
    Wrap a fitted scikit-learn estimator as a ``model_fn`` for
    :func:`compute_biomass`.

    Handles the reshaping between the per-cell metric dict used internally
    and the ``(n_samples, n_features)`` matrix expected by sklearn, and
    masks NaN pixels so they are never passed to ``predict()``.

    Parameters
    ----------
    estimator:
        Any fitted sklearn-compatible estimator that exposes a
        ``predict(X)`` method (e.g. ``RandomForestRegressor``,
        ``GradientBoostingRegressor``, ``Pipeline``, …).
    features:
        Ordered list of metric names to use as model features.
        Defaults to all sixteen standard metrics (``_METRIC_NAMES``):
        height percentiles, canopy cover, density, FHD, VCI, CRR, and the
        six height-stratum proportions.  The order must match the feature
        order used during training.  Pass an explicit list (e.g.
        ``["h50", "h95", "cc"]``) when the model was trained on a subset.

    Returns
    -------
    Callable
        A function ``model_fn(metrics) → np.ndarray`` compatible with
        the ``model_fn`` parameter of :func:`compute_biomass`.

    Examples
    --------
    ::

        from sklearn.ensemble import RandomForestRegressor
        from alsdb.processing.biomass import compute_biomass, wrap_sklearn_model

        rf = RandomForestRegressor(n_estimators=200)
        rf.fit(X_train, y_train)  # X columns must match _METRIC_NAMES order

        model_fn = wrap_sklearn_model(rf)
        compute_biomass(provider, store, resolution=10.0, year=2021,
                        model_fn=model_fn)
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
    """
    Compute per-cell LiDAR metrics using a circular neighbourhood of radius
    *baba_radius* around each cell centre (Buffered Area-Based Approach).

    Each output cell's metrics are derived from all points within *baba_radius*
    metres of the cell centre, not just points within the cell itself.  This
    gives statistically robust estimates even at fine output resolutions where
    individual cells may contain very few points.

    The caller must ensure the queried point array extends at least *baba_radius*
    beyond *bbox* on all sides (i.e. ``tile_buffer >= baba_radius``).

    Parameters
    ----------
    min_density:
        Minimum point density (pts m⁻²) for a neighbourhood to receive metric
        values.  Cells below this threshold are set to ``np.nan``.
    """
    requested = tuple(_METRIC_NAMES if metrics is None else dict.fromkeys(metrics))
    if set(requested) - set(_METRIC_NAMES):
        raise ValueError("Unknown structural metric")
    nx, ny, neighborhoods, area = baba_neighbourhoods(points, resolution, bbox, baba_radius)
    out = {name: np.full(nx * ny, np.nan, np.float32) for name in requested}
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
        quantiles = [
            (name, q) for name, q in (("h50", 50), ("h75", 75), ("h95", 95)) if name in out
        ]
        if quantiles:
            for (name, _), value in zip(quantiles, np.percentile(hv, [q for _, q in quantiles])):
                out[name][cell] = value
        if "hmean" in out:
            out["hmean"][cell] = hv.mean()
        if "hmax" in out:
            out["hmax"][cell] = hv.max()
        if "crr" in out and hv.max() > hv.min():
            out["crr"][cell] = (hv.mean() - hv.min()) / (hv.max() - hv.min())
        if {"fhd", "vci"} & set(out):
            entropy = _fhd_from_hag(hv)
            if "fhd" in out:
                out["fhd"][cell] = entropy
            if "vci" in out:
                out["vci"][cell] = entropy / _VCI_MAX_ENTROPY
        for name, (lo, hi) in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA):
            if name in out:
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
    """
    Compute LiDAR structural metrics and write them into *store*.

    Metrics written: ``h50``, ``h75``, ``h95``, ``hmax``, ``hmean``, ``cc``,
    ``density``, ``fhd``, ``vci``, ``crr``, ``pv_0_2``, ``pv_2_5``,
    ``pv_5_10``, ``pv_10_20``, ``pv_20_40``, ``pv_above40`` — each as a
    separate variable at *resolution*.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Grid cell size in metres (10–25 m typical for biomass).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.
    cc_threshold:
        HAG threshold (m) used to define "canopy" for the cover metric.
    min_density:
        Minimum total point density (pts m⁻²) for a cell to receive metric
        values.  Cells below this threshold are set to ``np.nan`` for all
        metrics.  Typical values: 0.5 (sparse survey), 1.0 (moderate),
        4.0 (dense modern ALS).  Default ``0.0`` disables the guard.
    veg_classes:
        LAS classification codes treated as vegetation (default ``(3, 4,
        5)``, standard ASPRS). Not universal across surveys -- see
        ``pavd.py``'s module docstring for a real dataset that uses a
        different code entirely; verify with
        ``np.unique(classification, return_counts=True)`` before trusting
        the default on a new array.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for terrain interpolation (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    from alsdb.processing.forest import compute_forest_products

    compute_forest_products(
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
    """
    Estimate Above-Ground Biomass (AGB) and write into *store*.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Grid cell size in metres.
    model_fn:
        Callable ``model_fn(metrics) → np.ndarray`` mapping the metric dict
        to an AGB grid (Mg ha⁻¹).  Defaults to :func:`naesset_model`.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.
    cc_threshold:
        HAG threshold (m) for the canopy cover metric.
    min_density:
        Minimum total point density (pts m⁻²) required before AGB is
        estimated for a cell.  See :func:`compute_metrics` for guidance.
    veg_classes:
        LAS classification codes treated as vegetation (default ``(3, 4,
        5)``, standard ASPRS). See :func:`compute_metrics` for the same
        parameter's full caveat -- not universal across surveys.
    overwrite:
        If ``False`` (default) and biomass data for *year* already exists
        in the store, the computation is skipped.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for terrain interpolation (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    from alsdb.processing.forest import compute_forest_products

    compute_forest_products(
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
    """Spatially grouped cross-validation of the three-parameter biomass model.

    Entire supplied spatial blocks are held out together. Returns out-of-fold
    predictions, residual RMSE/bias and empirical residual quantiles in biomass
    units. These describe predictive error on the supplied plots, not just
    coefficient uncertainty; they are not calibrated pixel confidence intervals.
    The caller must choose blocks larger than the relevant spatial dependence.
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
