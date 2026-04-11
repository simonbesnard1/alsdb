# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Gap fraction and effective LAI from TileDB ALS point clouds.

Theory
------
Gap fraction P_gap is estimated using the MacArthur-Wilson return-count
estimator:

    P_gap = N_gnd / (N_gnd + N_veg)

where *N_gnd* is the number of first returns classified as ground (Class 2)
and *N_veg* is the number of first returns classified as vegetation
(Classes 3–5) within each raster cell.  This is a direct observable —
no assumptions about canopy structure are required.

Effective LAI can be derived optionally via the Beer-Lambert law:

    L_e = -ln(P_gap) / k

where *k* is the extinction coefficient (default 0.5 for a spherical leaf
angle distribution).  The output is "effective LAI" (L_e), not true LAI,
because ALS cannot distinguish leaves from woody material.

Usage::

    from alsdb import ALSProvider
    from alsdb.storage import ALSZarrStore
    from alsdb.processing.gap import compute_gap_fraction

    provider = ALSProvider(storage_type="local", uri="array_")
    store = ALSZarrStore("output/spain.zarr")

    # Gap fraction only
    compute_gap_fraction(provider, store, resolution=10.0, year=2021)

    # Gap fraction + effective LAI
    compute_gap_fraction(provider, store, resolution=10.0, year=2021,
                         lai=True, k=0.5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from alsdb.processing._tiling import (
    array_crs, array_data_bbox, attach_hag, check_bbox_overlap,
    check_year_exists, query_to_array, run_tiled, tile_bboxes,
)

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)

_GROUND_CLASS  = 2
_VEG_CLASSES   = (3, 4, 5)
_LAI_K_DEFAULT = 0.5
_LAI_MAX       = 10.0   # physical ceiling — avoids ln(0) → -inf artefacts


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _compute_gap_grid(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Compute per-cell gap fraction from a HAG-annotated point array.

    Returns a ``(ny, nx)`` float32 north-up array; cells with no first
    returns are ``np.nan``.
    """
    from scipy.stats import binned_statistic_2d

    min_x, min_y, max_x, max_y = bbox
    nx = max(1, int(np.ceil((max_x - min_x) / resolution)))
    ny = max(1, int(np.ceil((max_y - min_y) / resolution)))
    x_edges = np.linspace(min_x, max_x, nx + 1)
    y_edges = np.linspace(min_y, max_y, ny + 1)
    bins = [x_edges, y_edges]

    fr    = points["ReturnNumber"] == 1
    x_fr  = points["X"][fr]
    y_fr  = points["Y"][fr]
    cls_fr = points["Classification"][fr]

    gnd  = (cls_fr == _GROUND_CLASS).astype(np.float32)
    veg  = np.isin(cls_fr, _VEG_CLASSES).astype(np.float32)
    ones = np.ones(fr.sum(), dtype=np.float32)

    n_gnd = binned_statistic_2d(x_fr, y_fr, gnd,  statistic="sum",   bins=bins).statistic
    n_veg = binned_statistic_2d(x_fr, y_fr, veg,  statistic="sum",   bins=bins).statistic
    n_tot = binned_statistic_2d(x_fr, y_fr, ones, statistic="count", bins=bins).statistic

    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(n_tot > 0, n_gnd / (n_gnd + n_veg), np.nan)

    return np.flipud(gap.T).astype(np.float32)


def _gap_to_lai(gap: np.ndarray, k: float) -> np.ndarray:
    """Convert gap fraction to effective LAI via Beer-Lambert."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lai = -np.log(np.where(gap > 0, gap, np.nan)) / k
    return np.clip(lai, 0.0, _LAI_MAX).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------

def _process_tile(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox:  tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    lai: bool,
    k: float,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("Gap tile %d: no points, skipping", tile_index)
        return

    points = attach_hag(arr)

    gap = _compute_gap_grid(points, resolution, crop_bbox)

    if np.all(np.isnan(gap)):
        logger.debug("Gap tile %d: all NaN, skipping", tile_index)
        return

    store.write_tile("gap", resolution, year, gap, crop_bbox)

    if lai:
        lai_grid = _gap_to_lai(gap, k)
        store.write_tile("lai", resolution, year, lai_grid, crop_bbox)

    logger.debug("Gap tile %d written", tile_index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gap_fraction(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 10.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    *,
    lai: bool = False,
    k: float = _LAI_K_DEFAULT,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
) -> None:
    """
    Compute gap fraction (and optionally effective LAI) and write into *store*.

    Gap fraction is the MacArthur-Wilson estimator:

        P_gap = N_gnd_first / (N_gnd_first + N_veg_first)

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.  Must have ``"gap"``
        (and ``"lai"`` if ``lai=True``) pre-allocated at *resolution*.
    resolution:
        Cell size in metres (default 10 m).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.  Written as a time slice in the store.
    lai:
        If ``True``, also compute effective LAI via Beer-Lambert
        ``L_e = -ln(P_gap) / k``.
    k:
        Extinction coefficient for the Beer-Lambert LAI estimate
        (default 0.5, spherical leaf angle distribution).
        Only used when ``lai=True``.
    overwrite:
        If ``False`` (default) and gap (and LAI if requested) already exist
        for *year* in the store, the computation is skipped.
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
        gap_done = store.has_data("gap", resolution, year)
        lai_done = (not lai) or store.has_data("lai", resolution, year)
        if gap_done and lai_done:
            logger.info("Gap fraction already present for year %d at %.0f m — skipping",
                        year, resolution)
            return
    crs = array_crs(provider)
    store.ensure_group("gap", resolution, effective_bbox, crs, tile_size)
    if lai:
        store.ensure_group("lai", resolution, effective_bbox, crs, tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    logger.info(
        "Computing gap fraction  (%.0f m, %d tile(s), %d worker(s), year=%s%s)",
        resolution, len(tiles), n_workers, year,
        ", LAI" if lai else "",
    )

    run_tiled(_process_tile, provider, tiles, store, n_workers,
              resolution=resolution, year=year, lai=lai, k=k)
