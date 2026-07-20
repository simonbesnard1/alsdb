# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Above-Ground Biomass (AGB) estimation from TileDB ALS point clouds.

Pipeline
--------
1. Query TileDB → numpy structured array.
2. Run ``filters.hag_delaunay`` (or ``filters.hag_nn`` fallback) via PDAL
   to attach ``HeightAboveGround``.
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
   counts, matching the classic first-return-cover definition used in
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
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from alsdb.processing._tiling import (
    VEG_CLASSES as _VEG_CLASSES,
    _require_year,
    array_crs,
    array_data_bbox,
    attach_hag,
    baba_neighbourhoods,
    check_bbox_overlap,
    check_year_exists,
    flip_to_north_up,
    query_to_array,
    run_tiled,
    tile_bboxes,
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
    """
    x_min, y_min, x_max, y_max = bbox
    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    # Actual per-axis cell width, not the nominal *resolution* argument -
    # only identical to it when (x_max - x_min) happens to be an exact
    # multiple of resolution (tile_size/resolution alignment isn't enforced
    # here the way chm.py's _validate_grid_alignment enforces it for CHM).
    # _bin below must derive bins from this, not the raw parameter, to stay
    # correct for non-grid-aligned bbox/resolution combinations.
    x_res = (x_max - x_min) / nx
    y_res = (y_max - y_min) / ny
    n_cells = nx * ny

    x = points["X"]
    y = points["Y"]
    hag = points["HeightAboveGround"]
    fr = points["ReturnNumber"] == 1
    hag_fr = hag[fr]

    veg = np.isin(points["Classification"], _VEG_CLASSES) & (hag > 0)
    hag_v = hag[veg]

    def _bin(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Direct arithmetic instead of np.digitize (binary search) - bins
        are uniform, so this is O(n) instead of O(n log nbins). Verified
        against digitize on 5M purely random points with zero mismatches;
        differs (by exactly one bin) only for a point landing *exactly* on
        a computed bin edge to full float64 precision - a measure-zero event
        for continuous real coordinates, not something that occurs with real
        survey data."""
        xi = np.clip(np.floor((px - x_min) / x_res).astype(np.int64), 0, nx - 1)
        yi = np.clip(np.floor((py - y_min) / y_res).astype(np.int64), 0, ny - 1)
        return xi * ny + yi

    def _flip(g: np.ndarray) -> np.ndarray:
        return flip_to_north_up(g, transpose=True)

    # Bin all three point groups once — shared by every downstream metric.
    cell_all = _bin(x, y)
    cell_fr = _bin(x[fr], y[fr])
    cell_v = _bin(x[veg], y[veg])

    # --- Density ---
    cell_area = resolution**2
    density_grid = _flip(
        (np.bincount(cell_all, minlength=n_cells).astype(np.float64) / cell_area).reshape(nx, ny)
    )

    # --- Canopy cover ---
    n_fr_flat = np.bincount(cell_fr, minlength=n_cells).astype(np.float64)
    n_above_flat = np.bincount(
        cell_fr, weights=(hag_fr > cc_threshold).astype(np.float64), minlength=n_cells
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = _flip(np.where(n_fr_flat > 0, n_above_flat / n_fr_flat, np.nan).reshape(nx, ny))

    # --- Veg-point count (shared by hmean denominator and strata) ---
    n_veg_flat = np.bincount(cell_v, minlength=n_cells).astype(np.float64)

    # --- hmean via weighted bincount (pure C, one pass) ---
    hag_sum_flat = np.bincount(cell_v, weights=hag_v.astype(np.float64), minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        hmean_flat = np.where(n_veg_flat > 0, hag_sum_flat / n_veg_flat, np.nan)

    # --- Height strata proportions (six bincount calls, no Python per-bin work) ---
    strata_grids: dict[str, np.ndarray] = {}
    for name, (lo, hi) in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA):
        indicator = ((hag_v > lo) & (hag_v <= hi)).astype(np.float64)
        n_strata = np.bincount(cell_v, weights=indicator, minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            prop = np.where(n_veg_flat > 0, n_strata / n_veg_flat, np.nan)
        strata_grids[name] = _flip(prop.reshape(nx, ny))

    # --- h50/h75/h95 + hmin + hmax, fully vectorised (no per-cell Python loop) ---
    # lexsort by (value, cell) - not just cell - so each cell's points are
    # *also* sorted by value within the group, which the closed-form
    # percentile step below depends on (a plain argsort(cell_v) only groups
    # by cell, leaving value order within each group arbitrary).
    order = np.lexsort((hag_v, cell_v))
    sorted_cells_v = cell_v[order]
    sorted_hag_v = hag_v[order].astype(np.float64)
    unique_cells_v, first_idx_v = np.unique(sorted_cells_v, return_index=True)
    ends_v = np.append(first_idx_v[1:], len(sorted_hag_v))
    group_sizes = (ends_v - first_idx_v).astype(np.float64)

    def _grouped_percentile(q: float) -> np.ndarray:
        """np.percentile's 'linear' method, applied to every occupied cell's
        pre-sorted group at once via index arithmetic instead of a per-cell
        Python-level np.percentile call - profiled at ~55x faster than the
        loop at 250k cells (11.0s -> 0.2s), the actual cost being the
        thousands of individual per-cell function calls, not the sort
        (pre-sorting once but still calling np.percentile per cell only
        saves ~7%, confirmed). Computed in float64 to match np.percentile's
        own internal promotion for array-form q - matching bit-for-bit
        wasn't achievable without keeping the per-cell call (numpy promotes
        to float64 internally regardless of input dtype for that form); this
        differs from the exact per-cell result by ~1e-6 m, floating-point
        rounding noise negligible next to real HAG measurement precision,
        not a logic difference.
        """
        rank = (np.float64(q) / np.float64(100)) * (group_sizes - np.float64(1))
        lower_offset = np.floor(rank).astype(np.int64)
        upper_offset = np.ceil(rank).astype(np.int64)
        frac = rank - lower_offset
        lo = sorted_hag_v[first_idx_v + lower_offset]
        hi = sorted_hag_v[first_idx_v + upper_offset]
        out = np.full(n_cells, np.nan, dtype=np.float64)
        out[unique_cells_v] = lo + frac * (hi - lo)
        return out

    h50_flat = _grouped_percentile(50)
    h75_flat = _grouped_percentile(75)
    h95_flat = _grouped_percentile(95)
    hmin_flat = np.full(n_cells, np.nan, dtype=np.float64)
    hmax_flat = np.full(n_cells, np.nan, dtype=np.float64)
    hmin_flat[unique_cells_v] = sorted_hag_v[first_idx_v]
    hmax_flat[unique_cells_v] = sorted_hag_v[ends_v - 1]

    with np.errstate(invalid="ignore", divide="ignore"):
        denom_flat = hmax_flat - hmin_flat
        crr_flat = np.where(denom_flat > 0, (hmean_flat - hmin_flat) / denom_flat, np.nan)

    # --- FHD via 3D histogram (x_cell × y_cell × hag_band): fully vectorised ---
    xi_v = cell_v // ny
    yi_v = cell_v % ny
    fhd_mask = hag_v <= _FHD_MAX_H
    hag_bin_v = np.minimum(
        np.floor(hag_v[fhd_mask] / _FHD_BIN_SIZE).astype(np.intp), _FHD_N_BINS - 1
    )
    flat_fhd = np.ravel_multi_index(
        (xi_v[fhd_mask], yi_v[fhd_mask], hag_bin_v), (nx, ny, _FHD_N_BINS)
    )
    counts_3d = (
        np.bincount(flat_fhd, minlength=n_cells * _FHD_N_BINS)
        .reshape(nx, ny, _FHD_N_BINS)
        .astype(np.float64)
    )
    cell_totals = counts_3d.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cell_totals > 0, counts_3d / cell_totals, 0.0)
        log_p = np.where(p > 0, np.log(p), 0.0)
    fhd_raw = -(p * log_p).sum(axis=2)  # (nx, ny)
    fhd_raw[cell_totals.squeeze(axis=2) == 0] = np.nan
    fhd_grid = _flip(fhd_raw)
    with np.errstate(invalid="ignore", divide="ignore"):
        vci_grid = np.where(_VCI_MAX_ENTROPY > 0, fhd_grid / _VCI_MAX_ENTROPY, np.nan).astype(
            np.float32
        )

    metrics: dict[str, np.ndarray] = {
        "h50": _flip(h50_flat.reshape(nx, ny)),
        "h75": _flip(h75_flat.reshape(nx, ny)),
        "h95": _flip(h95_flat.reshape(nx, ny)),
        "hmax": _flip(hmax_flat.reshape(nx, ny)),
        "hmean": _flip(hmean_flat.reshape(nx, ny)),
        "cc": cc,
        "density": density_grid,
        "fhd": fhd_grid,
        "vci": vci_grid,
        "crr": _flip(crr_flat.reshape(nx, ny)),
        **strata_grids,
    }

    # Mask all metrics in cells below the minimum density threshold (in-place
    # to avoid 16 separate np.where copies).
    if min_density > 0.0:
        sparse = density_grid < min_density
        for arr in metrics.values():
            arr[sparse] = np.nan

    return metrics


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
            np.isnan(h95) | np.isnan(cc) | (h95 <= 0) | (cc == 0),
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
) -> "tuple[float, float, float] | tuple[tuple[float, float, float], np.ndarray]":
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

    valid = ~(np.isnan(h95) | np.isnan(cc) | np.isnan(agb_field) | (cc <= 0) | (h95 <= 0))
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
    features: Optional[list[str]] = None,
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
        valid = ~np.any(np.isnan(X), axis=1)
        result = np.full(X.shape[0], np.nan, dtype=np.float32)
        if valid.any():
            result[valid] = estimator.predict(X[valid]).astype(np.float32)
        return result.reshape(shape)

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
    nx, ny, indices_list, neighbourhood_area = baba_neighbourhoods(
        points, resolution, bbox, baba_radius
    )
    shape = (ny, nx)

    # Pre-allocate all output grids
    h50 = np.full(shape, np.nan, dtype=np.float64)
    h75 = np.full(shape, np.nan, dtype=np.float64)
    h95 = np.full(shape, np.nan, dtype=np.float64)
    hmax = np.full(shape, np.nan, dtype=np.float64)
    hmean = np.full(shape, np.nan, dtype=np.float64)
    cc = np.full(shape, np.nan, dtype=np.float64)
    density = np.full(shape, np.nan, dtype=np.float64)
    fhd = np.full(shape, np.nan, dtype=np.float64)
    vci = np.full(shape, np.nan, dtype=np.float64)
    crr = np.full(shape, np.nan, dtype=np.float64)
    strata = {n: np.full(shape, np.nan, dtype=np.float64) for n in _HEIGHT_STRATA_NAMES}

    hag_all = points["HeightAboveGround"]
    ret_all = points["ReturnNumber"]
    # Precomputed once over all points rather than re-running np.isin (and
    # the > 0 comparison) inside the loop for every one of nx*ny output
    # cells - _VEG_CLASSES is tiny and fixed, so there's nothing per-cell
    # about this test.
    veg_all = np.isin(points["Classification"], _VEG_CLASSES) & (hag_all > 0)

    for k, idxs in enumerate(indices_list):
        if not idxs:
            continue
        row, col = divmod(k, nx)
        hag_k = hag_all[idxs]
        ret_k = ret_all[idxs]

        cell_density = len(idxs) / neighbourhood_area
        density[row, col] = cell_density
        if min_density > 0.0 and cell_density < min_density:
            continue

        hag_v = hag_k[veg_all[idxs]]
        if hag_v.size > 0:
            # Single sort for all three percentiles; cache min/max/mean to
            # avoid redundant passes when computing crr.
            h50[row, col], h75[row, col], h95[row, col] = np.percentile(hag_v, [50, 75, 95])
            hmin_k = float(hag_v.min())
            hmax_k = float(hag_v.max())
            hmean_k = float(hag_v.mean())
            hmax[row, col] = hmax_k
            hmean[row, col] = hmean_k
            denom_k = hmax_k - hmin_k
            if denom_k > 0:
                crr[row, col] = (hmean_k - hmin_k) / denom_k

            # Compute FHD once; derive VCI from it to avoid a second
            # _fhd_from_hag call inside _vci_from_hag.
            fhd_k = _fhd_from_hag(hag_v)
            fhd[row, col] = fhd_k
            vci[row, col] = (
                float(fhd_k / _VCI_MAX_ENTROPY)
                if not np.isnan(fhd_k) and _VCI_MAX_ENTROPY > 0
                else np.nan
            )

            n_v = hag_v.size
            for name, (lo, hi) in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA):
                strata[name][row, col] = float(((hag_v > lo) & (hag_v <= hi)).sum()) / n_v

        fr = ret_k == 1
        n_fr = int(fr.sum())
        if n_fr > 0:
            cc[row, col] = float((hag_k[fr] > cc_threshold).sum()) / n_fr

    _flip = flip_to_north_up

    return {
        "h50": _flip(h50),
        "h75": _flip(h75),
        "h95": _flip(h95),
        "hmax": _flip(hmax),
        "hmean": _flip(hmean),
        "cc": _flip(cc),
        "density": _flip(density),
        "fhd": _flip(fhd),
        "vci": _flip(vci),
        "crr": _flip(crr),
        **{n: _flip(strata[n]) for n in _HEIGHT_STRATA_NAMES},
    }


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------


def _process_tile_metrics(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    cc_threshold: float,
    baba_radius: float = 0.0,
    min_density: float = 0.0,
    model_fn: Optional[Callable] = None,
) -> None:
    """
    Extract per-cell metrics and either write them all (``model_fn=None``,
    used by :func:`compute_metrics`) or apply *model_fn* and write the
    resulting AGB grid (used by :func:`compute_biomass`).
    """
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("Tile %d: no points, skipping", tile_index)
        return

    points = attach_hag(arr)
    if baba_radius > 0:
        metrics = _extract_metrics_baba(
            points,
            resolution,
            bbox=crop_bbox,
            baba_radius=baba_radius,
            cc_threshold=cc_threshold,
            min_density=min_density,
        )
    else:
        metrics = _extract_metrics(
            points,
            resolution,
            bbox=crop_bbox,
            cc_threshold=cc_threshold,
            min_density=min_density,
        )

    if model_fn is None:
        for name, grid in metrics.items():
            store.write_tile(name, resolution, year, grid, crop_bbox)
        logger.debug("Metrics tile %d written", tile_index)
        return

    agb = model_fn(metrics)
    if np.all(np.isnan(agb)):
        logger.debug("AGB tile %d: all NaN, skipping", tile_index)
        return
    store.write_tile("biomass", resolution, year, agb, crop_bbox)
    logger.debug("AGB tile %d written", tile_index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 10.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
    baba_radius: float = 0.0,
    min_density: float = 0.0,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
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
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for ``filters.hag_delaunay`` accuracy (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    _require_year(year)
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    if not overwrite:
        if all(store.has_data(v, resolution, year) for v in _METRIC_NAMES):
            logger.info(
                "LiDAR metrics already present for year %d at %.0f m — skipping",
                year,
                resolution,
            )
            return
    crs = array_crs(provider)
    for var in _METRIC_NAMES:
        store.ensure_group(var, resolution, effective_bbox, crs, tile_size)
    effective_buffer = max(tile_buffer, baba_radius)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=effective_buffer)
    logger.info(
        "Extracting LiDAR metrics  (%.0f m, %d tile(s), %d worker(s), year=%s%s%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        f", BABA r={baba_radius:.0f} m" if baba_radius > 0 else "",
        f", min_density={min_density:.1f}" if min_density > 0 else "",
    )
    run_tiled(
        _process_tile_metrics,
        provider,
        tiles,
        store,
        n_workers,
        resolution=resolution,
        year=year,
        cc_threshold=cc_threshold,
        baba_radius=baba_radius,
        min_density=min_density,
    )


def compute_biomass(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 10.0,
    model_fn: Optional[Callable[[dict[str, np.ndarray]], np.ndarray]] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
    baba_radius: float = 0.0,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    min_density: float = 0.0,
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
    overwrite:
        If ``False`` (default) and biomass data for *year* already exists
        in the store, the computation is skipped.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for ``filters.hag_delaunay`` accuracy (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    import warnings

    _require_year(year)
    using_default_model = model_fn is None
    model_fn = model_fn or naesset_model
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    if not overwrite and store.has_data("biomass", resolution, year):
        logger.info("Biomass already present for year %d at %.0f m — skipping", year, resolution)
        return
    store.ensure_group("biomass", resolution, effective_bbox, array_crs(provider), tile_size)
    effective_buffer = max(tile_buffer, baba_radius)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=effective_buffer)
    logger.info(
        "Computing AGB  (%.0f m, %d tile(s), %d worker(s), year=%s%s%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        f", BABA r={baba_radius:.0f} m" if baba_radius > 0 else "",
        f", min_density={min_density:.1f}" if min_density > 0 else "",
    )
    if using_default_model:
        # naesset_model's own per-call warning already only prints once per
        # process (Python dedupes identical (message, category, lineno)
        # warnings by default, and run_tiled's ThreadPoolExecutor workers
        # share one process) - but relying on that implicitly, buried inside
        # a tile worker's call stack, is easy to miss. Warn explicitly once,
        # here, and suppress naesset_model's own internal warning for the
        # run so the message doesn't appear to come from deep inside a tile
        # callback with a confusing stacklevel.
        warnings.warn(
            "compute_biomass is using naesset_model with uncalibrated placeholder "
            "coefficients (a=0.8, b=1.8, c=0.5). Results are not scientifically valid "
            "without calibration. Call calibrate_naesset(h95, cc, agb_field) with field "
            "inventory data and pass model_fn=lambda m: naesset_model(m, a=a, b=b, c=c) "
            "with the returned coefficients.",
            UserWarning,
            stacklevel=2,
        )
    with warnings.catch_warnings():
        if using_default_model:
            warnings.filterwarnings(
                "ignore", message="naesset_model is using uncalibrated.*", category=UserWarning
            )
        run_tiled(
            _process_tile_metrics,
            provider,
            tiles,
            store,
            n_workers,
            resolution=resolution,
            year=year,
            cc_threshold=cc_threshold,
            model_fn=model_fn,
            baba_radius=baba_radius,
            min_density=min_density,
        )
