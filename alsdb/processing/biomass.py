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

_VEG_CLASSES = (3, 4, 5)
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
# Normalisation uses the total number of bins (not occupied bins) so VCI is
# comparable across cells and scenes regardless of local canopy height range.
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
    """Vegetation Complexity Index — FHD normalised to [0, 1]."""
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
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    n_cells = nx * ny

    x = points["X"]
    y = points["Y"]
    hag = points["HeightAboveGround"]
    fr = points["ReturnNumber"] == 1
    hag_fr = hag[fr]

    veg = np.isin(points["Classification"], _VEG_CLASSES) & (hag > 0)
    hag_v = hag[veg]

    def _bin(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        xi = np.clip(np.digitize(px, x_edges) - 1, 0, nx - 1)
        yi = np.clip(np.digitize(py, y_edges) - 1, 0, ny - 1)
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

    # --- h50/h75/h95 + hmin + hmax in one grouped sort ---
    # Sorting veg points by cell groups them; min/max come free from the same pass.
    order = np.argsort(cell_v, kind="stable")
    sorted_cells_v = cell_v[order]
    sorted_hag_v = hag_v[order]
    unique_cells_v, first_idx_v = np.unique(sorted_cells_v, return_index=True)
    ends_v = np.append(first_idx_v[1:], len(sorted_hag_v))

    h50_flat = np.full(n_cells, np.nan, dtype=np.float64)
    h75_flat = np.full(n_cells, np.nan, dtype=np.float64)
    h95_flat = np.full(n_cells, np.nan, dtype=np.float64)
    hmin_flat = np.full(n_cells, np.nan, dtype=np.float64)
    hmax_flat = np.full(n_cells, np.nan, dtype=np.float64)
    for _i, _cell in enumerate(unique_cells_v):
        _h = sorted_hag_v[first_idx_v[_i] : ends_v[_i]]
        h50_flat[_cell], h75_flat[_cell], h95_flat[_cell] = np.percentile(_h, [50, 75, 95])
        hmin_flat[_cell] = _h.min()
        hmax_flat[_cell] = _h.max()

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
    cls_all = points["Classification"]
    ret_all = points["ReturnNumber"]

    for k, idxs in enumerate(indices_list):
        if not idxs:
            continue
        row, col = divmod(k, nx)
        hag_k = hag_all[idxs]
        cls_k = cls_all[idxs]
        ret_k = ret_all[idxs]

        cell_density = len(idxs) / neighbourhood_area
        density[row, col] = cell_density
        if min_density > 0.0 and cell_density < min_density:
            continue

        veg = np.isin(cls_k, _VEG_CLASSES) & (hag_k > 0)
        hag_v = hag_k[veg]
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
    _require_year(year)
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
