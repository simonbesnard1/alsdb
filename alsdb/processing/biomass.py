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
    from alsdb.processing.biomass import compute_biomass, compute_metrics

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

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pdal

from alsdb.processing._tiling import (
    array_crs, array_data_bbox, check_bbox_overlap, check_year_exists,
    tile_bboxes, query_to_array,
)

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)

_VEG_CLASSES          = (3, 4, 5)
_DEFAULT_CC_THRESHOLD = 2.0   # m — first returns above this count as "canopy"

_METRIC_NAMES = ["h50", "h75", "h95", "hmean", "cc", "density"]


# ---------------------------------------------------------------------------
# HAG attachment
# ---------------------------------------------------------------------------

def _attach_hag(arr: np.ndarray) -> np.ndarray:
    """Run PDAL hag_delaunay on *arr* and return annotated point array."""
    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
    ]
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    result = p.arrays[0]
    logger.debug("HAG attached: %d points", len(result))
    return result


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

    x   = points["X"]
    y   = points["Y"]
    hag = points["HeightAboveGround"]

    veg = np.isin(points["Classification"], _VEG_CLASSES) & (hag > 0)
    x_v, y_v, hag_v = x[veg], y[veg], hag[veg]

    def _pct(p):
        def stat(v):
            return float(np.percentile(v, p)) if len(v) else np.nan
        return stat

    def _flip(g):
        return np.flipud(np.where(np.isnan(g), np.nan, g).T).astype(np.float32)

    cell_area = resolution ** 2
    n_all = binned_statistic_2d(x, y, hag, statistic="count", bins=bins).statistic

    fr = points["ReturnNumber"] == 1
    x_fr, y_fr, hag_fr = x[fr], y[fr], hag[fr]
    above  = (hag_fr > cc_threshold).astype(np.float32)
    n_fr   = binned_statistic_2d(x_fr, y_fr, np.ones(fr.sum()),
                                  statistic="count", bins=bins).statistic
    n_above = binned_statistic_2d(x_fr, y_fr, above,
                                   statistic="sum", bins=bins).statistic

    with np.errstate(invalid="ignore", divide="ignore"):
        cc = _flip(np.where(n_fr > 0, n_above / n_fr, np.nan))

    metrics: dict[str, np.ndarray] = {
        "h50":     _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(50), bins=bins).statistic),
        "h75":     _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(75), bins=bins).statistic),
        "h95":     _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(95), bins=bins).statistic),
        "hmean":   _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic="mean",   bins=bins).statistic),
        "cc":      cc,
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
    cc  = metrics["cc"]
    with np.errstate(invalid="ignore"):
        agb = np.where(
            np.isnan(h95) | np.isnan(cc) | (cc == 0),
            np.nan,
            a * np.power(np.where(h95 > 0, h95, 0), b) * np.power(cc, c),
        )
    return agb.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------

def _process_tile_metrics(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox:  tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    cc_threshold: float,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("Metrics tile %d: no points, skipping", tile_index)
        return

    points = _attach_hag(arr)
    metrics = _extract_metrics(points, resolution, bbox=crop_bbox,
                               cc_threshold=cc_threshold)

    for name, grid in metrics.items():
        if not np.all(np.isnan(grid)):
            store.write_tile(name, resolution, year, grid, crop_bbox)

    logger.debug("Metrics tile %d written", tile_index)


def _process_tile_biomass(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox:  tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    cc_threshold: float,
    model_fn: Callable,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("AGB tile %d: no points, skipping", tile_index)
        return

    points = _attach_hag(arr)
    metrics = _extract_metrics(points, resolution, bbox=crop_bbox,
                               cc_threshold=cc_threshold)
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
    crs = array_crs(provider)
    for var in ["h50", "h75", "h95", "hmean", "cc", "density"]:
        store.ensure_group(var, resolution, effective_bbox, crs, tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    logger.info("Extracting LiDAR metrics  (%.0f m, %d tile(s), %d worker(s), year=%s)",
                resolution, len(tiles), n_workers, year)

    def _work(idx, qb, cb):
        _process_tile_metrics(provider, qb, cb, store, idx,
                              resolution, year, cc_threshold)

    if n_workers == 1:
        for i, (qb, cb) in enumerate(tiles):
            _work(i, qb, cb)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_work, i, qb, cb): i
                       for i, (qb, cb) in enumerate(tiles)}
            for future in as_completed(futures):
                future.result()


def compute_biomass(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 10.0,
    model_fn: Optional[Callable[[dict[str, np.ndarray]], np.ndarray]] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
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
    store.ensure_group("biomass", resolution, effective_bbox,
                       array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    logger.info("Computing AGB  (%.0f m, %d tile(s), %d worker(s), year=%s)",
                resolution, len(tiles), n_workers, year)

    def _work(idx, qb, cb):
        _process_tile_biomass(provider, qb, cb, store, idx,
                              resolution, year, cc_threshold, model_fn)

    if n_workers == 1:
        for i, (qb, cb) in enumerate(tiles):
            _work(i, qb, cb)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_work, i, qb, cb): i
                       for i, (qb, cb) in enumerate(tiles)}
            for future in as_completed(futures):
                future.result()
