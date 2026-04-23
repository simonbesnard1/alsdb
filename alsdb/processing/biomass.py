# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Above-Ground Biomass (AGB) estimation from TileDB ALS point clouds.

Pipeline
--------
1. Query TileDB → numpy structured array.
2. Run ``filters.hag_delaunay`` via PDAL to attach ``HeightAboveGround``.
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
    array_crs,
    array_data_bbox,
    attach_hag,
    check_bbox_overlap,
    check_year_exists,
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

_METRIC_NAMES = ["h50", "h75", "h95", "hmean", "cc", "density"]


# ---------------------------------------------------------------------------
# Per-cell metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    Compute per-cell LiDAR structural metrics over *bbox*.

    Returns a dict ``{name: (ny, nx) float32 array}`` in north-up
    orientation.  Empty cells are ``np.nan``.
    """
    from scipy.stats import binned_statistic_2d

    x_min, y_min, x_max, y_max = bbox
    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    bins = [x_edges, y_edges]

    x = points["X"]
    y = points["Y"]
    hag = points["HeightAboveGround"]

    veg = np.isin(points["Classification"], _VEG_CLASSES) & (hag > 0)
    x_v, y_v, hag_v = x[veg], y[veg], hag[veg]

    def _pct(p):
        def stat(v):
            return float(np.percentile(v, p)) if len(v) else np.nan

        return stat

    def _flip(g):
        return np.flipud(np.where(np.isnan(g), np.nan, g).T).astype(np.float32)

    cell_area = resolution**2
    n_all = binned_statistic_2d(x, y, hag, statistic="count", bins=bins).statistic

    fr = points["ReturnNumber"] == 1
    x_fr, y_fr, hag_fr = x[fr], y[fr], hag[fr]
    above = (hag_fr > cc_threshold).astype(np.float32)
    n_fr = binned_statistic_2d(
        x_fr, y_fr, np.ones(fr.sum()), statistic="count", bins=bins
    ).statistic
    n_above = binned_statistic_2d(x_fr, y_fr, above, statistic="sum", bins=bins).statistic

    with np.errstate(invalid="ignore", divide="ignore"):
        cc = _flip(np.where(n_fr > 0, n_above / n_fr, np.nan))

    metrics: dict[str, np.ndarray] = {
        "h50": _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(50), bins=bins).statistic),
        "h75": _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(75), bins=bins).statistic),
        "h95": _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(95), bins=bins).statistic),
        "hmean": _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic="mean", bins=bins).statistic),
        "cc": cc,
        "density": _flip(n_all / cell_area),
    }
    return metrics


# ---------------------------------------------------------------------------
# Allometric model
# ---------------------------------------------------------------------------


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
        Model coefficients.  Defaults are approximate generic values —
        **calibrate against field plots** before production use.
    """
    h95 = metrics["h95"]
    cc = metrics["cc"]
    with np.errstate(invalid="ignore"):
        agb = np.where(
            np.isnan(h95) | np.isnan(cc) | (cc == 0),
            np.nan,
            a * np.power(np.where(h95 > 0, h95, 0), b) * np.power(cc, c),
        )
    return agb.astype(np.float32)


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
        Defaults to all six standard metrics:
        ``["h50", "h75", "h95", "hmean", "cc", "density"]``.
        The order must match the feature order used during training.

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
        rf.fit(X_train, y_train)          # X columns = h50, h75, h95, hmean, cc, density

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
    """
    from scipy.spatial import cKDTree

    x_min, y_min, x_max, y_max = bbox
    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))

    # Cell centres; row 0 = min_y (flipped to north-up at the end)
    cx_arr = x_min + (np.arange(nx) + 0.5) * resolution
    cy_arr = y_min + (np.arange(ny) + 0.5) * resolution
    CX, CY = np.meshgrid(cx_arr, cy_arr)  # both (ny, nx)
    centres = np.column_stack([CX.ravel(), CY.ravel()])  # (ny*nx, 2)

    xy = np.column_stack([points["X"].astype(np.float64), points["Y"].astype(np.float64)])
    kd = cKDTree(xy)
    indices_list = kd.query_ball_point(centres, r=baba_radius)

    shape = (ny, nx)
    h50 = np.full(shape, np.nan, dtype=np.float64)
    h75 = np.full(shape, np.nan, dtype=np.float64)
    h95 = np.full(shape, np.nan, dtype=np.float64)
    hmean = np.full(shape, np.nan, dtype=np.float64)
    cc = np.full(shape, np.nan, dtype=np.float64)
    density = np.full(shape, np.nan, dtype=np.float64)

    neighbourhood_area = np.pi * baba_radius**2
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

        veg = np.isin(cls_k, _VEG_CLASSES) & (hag_k > 0)
        hag_v = hag_k[veg]
        if hag_v.size > 0:
            h50[row, col] = np.percentile(hag_v, 50)
            h75[row, col] = np.percentile(hag_v, 75)
            h95[row, col] = np.percentile(hag_v, 95)
            hmean[row, col] = hag_v.mean()

        fr = ret_k == 1
        n_fr = int(fr.sum())
        if n_fr > 0:
            cc[row, col] = float((hag_k[fr] > cc_threshold).sum()) / n_fr

        density[row, col] = len(idxs) / neighbourhood_area

    def _flip(a: np.ndarray) -> np.ndarray:
        return np.flipud(a).astype(np.float32)

    return {
        "h50": _flip(h50),
        "h75": _flip(h75),
        "h95": _flip(h95),
        "hmean": _flip(hmean),
        "cc": _flip(cc),
        "density": _flip(density),
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
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("Metrics tile %d: no points, skipping", tile_index)
        return

    points = attach_hag(arr)
    if baba_radius > 0:
        metrics = _extract_metrics_baba(
            points, resolution, bbox=crop_bbox, baba_radius=baba_radius, cc_threshold=cc_threshold
        )
    else:
        metrics = _extract_metrics(points, resolution, bbox=crop_bbox, cc_threshold=cc_threshold)

    for name, grid in metrics.items():
        if not np.all(np.isnan(grid)):
            store.write_tile(name, resolution, year, grid, crop_bbox)

    logger.debug("Metrics tile %d written", tile_index)


def _process_tile_biomass(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    cc_threshold: float,
    model_fn: Callable,
    baba_radius: float = 0.0,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("AGB tile %d: no points, skipping", tile_index)
        return

    points = attach_hag(arr)
    if baba_radius > 0:
        metrics = _extract_metrics_baba(
            points, resolution, bbox=crop_bbox, baba_radius=baba_radius, cc_threshold=cc_threshold
        )
    else:
        metrics = _extract_metrics(points, resolution, bbox=crop_bbox, cc_threshold=cc_threshold)
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
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
) -> None:
    """
    Compute LiDAR structural metrics and write them into *store*.

    Metrics written: ``h50``, ``h75``, ``h95``, ``hmean``, ``cc``,
    ``density`` — each as a separate variable at *resolution*.

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
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for ``filters.hag_delaunay`` accuracy (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if year is not None and not check_year_exists(year, provider):
        return
    if not overwrite and year is not None:
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
        "Extracting LiDAR metrics  (%.0f m, %d tile(s), %d worker(s), year=%s%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        f", BABA r={baba_radius:.0f} m" if baba_radius > 0 else "",
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
    model_fn = model_fn or naesset_model
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if year is not None and not check_year_exists(year, provider):
        return
    if not overwrite and year is not None and store.has_data("biomass", resolution, year):
        logger.info("Biomass already present for year %d at %.0f m — skipping", year, resolution)
        return
    store.ensure_group("biomass", resolution, effective_bbox, array_crs(provider), tile_size)
    effective_buffer = max(tile_buffer, baba_radius)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=effective_buffer)
    logger.info(
        "Computing AGB  (%.0f m, %d tile(s), %d worker(s), year=%s%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        f", BABA r={baba_radius:.0f} m" if baba_radius > 0 else "",
    )
    run_tiled(
        _process_tile_biomass,
        provider,
        tiles,
        store,
        n_workers,
        resolution=resolution,
        year=year,
        cc_threshold=cc_threshold,
        model_fn=model_fn,
        baba_radius=baba_radius,
    )
