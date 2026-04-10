# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Canopy Height Model (CHM), DTM and DSM computation using TileDB + PDAL.

Pattern
-------
Data is queried from TileDB via :class:`~alsdb.core.alsprovider.ALSProvider`,
then injected into a PDAL pipeline as a numpy structured array.  PDAL handles
filtering and height-above-ground computation; rasterisation is done in Python
with ``scipy.stats.binned_statistic_2d`` so results can be written directly
into an :class:`~alsdb.storage.ALSZarrStore` without any intermediate files.

Pipeline (CHM)
--------------
numpy array input
    → ``filters.hag_delaunay``  builds a TIN from Class-2 ground points,
                                 attaches ``HeightAboveGround`` to every point
    → ``filters.range``         keeps vegetation points only (Class 3–5)
    → ``filters.assign``        clamps negative HAG values to 0
    → ``filters.crop``          clips to the non-buffered tile extent
    → rasterise max(HAG) in numpy → written to Zarr

Usage::

    from alsdb import ALSProvider
    from alsdb.storage import ALSZarrStore
    from alsdb.processing.chm import compute_chm, compute_all

    provider = ALSProvider(storage_type="local", uri="array_")
    store = ALSZarrStore("output/spain.zarr")

    compute_chm(provider, store, resolution=1.0, year=2021)
    compute_all(provider, store, year=2021,
                tile_size=500.0, tile_buffer=50.0, n_workers=4)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional

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

_GROUND_CLASS = 2
_VEG_CLASSES  = (3, 4, 5)


# ---------------------------------------------------------------------------
# Numpy rasteriser
# ---------------------------------------------------------------------------

def _rasterise(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    statistic: str = "max",
) -> np.ndarray:
    """
    Bin *values* into a regular grid over *crop_bbox*.

    Returns a ``(ny, nx)`` float32 array in north-up orientation.
    Empty bins are ``np.nan``.
    """
    from scipy.stats import binned_statistic_2d

    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))
    x_edges = np.linspace(cx0, cx1, nx + 1)
    y_edges = np.linspace(cy0, cy1, ny + 1)

    grid = binned_statistic_2d(
        x, y, values, statistic=statistic, bins=[x_edges, y_edges],
    ).statistic                              # shape (nx, ny)

    return np.flipud(grid.T).astype(np.float32)   # → (ny, nx) north-up


# ---------------------------------------------------------------------------
# PDAL helpers
# ---------------------------------------------------------------------------

def _run(stages: list, arr: np.ndarray) -> np.ndarray:
    """Execute a PDAL pipeline and return the output point array."""
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    return p.arrays[0]


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------

def _process_tile_chm(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox:  tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    first_returns_only: bool,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("CHM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox
    # Filter to vegetation classes; optionally restrict to first returns so
    # that only the top-of-canopy surface is modelled (recommended).
    veg_limits = f"Classification[{_VEG_CLASSES[0]}:{_VEG_CLASSES[-1]}]"
    if first_returns_only:
        veg_limits += ",ReturnNumber[1:1]"
    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.range", "limits": veg_limits},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
        {"type": "filters.crop",
         "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
    ]
    try:
        points = _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("CHM tile %d: no vegetation points, skipping", tile_index)
            return
        raise

    if len(points) == 0:
        return

    grid = _rasterise(
        points["X"], points["Y"], points["HeightAboveGround"],
        crop_bbox, resolution, statistic="max",
    )
    store.write_tile("chm", resolution, year, grid, crop_bbox)
    logger.debug("CHM tile %d written", tile_index)


def _process_tile_dtm(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox:  tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("DTM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox
    stages = [
        {"type": "filters.range",
         "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]"},
        {"type": "filters.crop",
         "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
    ]
    try:
        points = _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("DTM tile %d: no ground points, skipping", tile_index)
            return
        raise

    if len(points) == 0:
        return

    grid = _rasterise(
        points["X"], points["Y"], points["Z"],
        crop_bbox, resolution, statistic="max",
    )
    store.write_tile("dtm", resolution, year, grid, crop_bbox)
    logger.debug("DTM tile %d written", tile_index)


def _process_tile_dsm(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox:  tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    first_returns_only: bool,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("DSM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox
    stages: list = []
    if first_returns_only:
        stages.append({"type": "filters.range", "limits": "ReturnNumber[1:1]"})
    stages.append({"type": "filters.crop",
                   "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"})
    try:
        points = _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("DSM tile %d: no points, skipping", tile_index)
            return
        raise

    if len(points) == 0:
        return

    grid = _rasterise(
        points["X"], points["Y"], points["Z"],
        crop_bbox, resolution, statistic="max",
    )
    store.write_tile("dsm", resolution, year, grid, crop_bbox)
    logger.debug("DSM tile %d written", tile_index)


# ---------------------------------------------------------------------------
# Shared tiled executor
# ---------------------------------------------------------------------------

def _run_tiled(worker_fn, provider, tiles, store, n_workers, **kwargs) -> None:
    """Run *worker_fn* over all *tiles*, sequentially or in parallel."""
    if n_workers == 1:
        for idx, (query_bbox, crop_bbox) in enumerate(tiles):
            worker_fn(provider, query_bbox, crop_bbox, store, idx, **kwargs)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    worker_fn, provider, query_bbox, crop_bbox, store, idx, **kwargs
                ): idx
                for idx, (query_bbox, crop_bbox) in enumerate(tiles)
            }
            for future in as_completed(futures):
                future.result()   # re-raise worker exceptions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_chm(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    *,
    first_returns_only: bool = True,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
) -> None:
    """
    Compute a Canopy Height Model and write it into *store*.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Grid cell size in metres (default 1 m).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.  Written as a time slice in the store.
    first_returns_only:
        If ``True`` (default), only first returns are used to build the
        canopy surface.  First returns represent the first surface the
        laser pulse hit — i.e. the top of the canopy — which is the
        physically correct input for a CHM.  Set to ``False`` to include
        all vegetation returns (reproduces the legacy behaviour).
    tile_size:
        Sub-tile width/height in metres (default 500 m).
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
    store.ensure_group("chm", resolution, effective_bbox,
                       array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    logger.info(
        "Computing CHM  (%.1f m, %d tile(s), %d worker(s), year=%s, first_returns=%s)",
        resolution, len(tiles), n_workers, year, first_returns_only,
    )
    _run_tiled(_process_tile_chm, provider, tiles, store, n_workers,
               resolution=resolution, year=year,
               first_returns_only=first_returns_only)


def compute_dtm(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    *,
    tile_size: float = 500.0,
    n_workers: int = 1,
) -> None:
    """
    Rasterize ground points (Class 2) to a DTM and write into *store*.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Grid cell size in metres.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.
    tile_size:
        Sub-tile width/height in metres (default 500 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if year is not None and not check_year_exists(year, provider):
        return
    store.ensure_group("dtm", resolution, effective_bbox,
                       array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=0.0)
    logger.info("Computing DTM  (%.1f m, %d tile(s), %d worker(s), year=%s)",
                resolution, len(tiles), n_workers, year)
    _run_tiled(_process_tile_dtm, provider, tiles, store, n_workers,
               resolution=resolution, year=year)


def compute_dsm(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 1.0,
    first_returns_only: bool = True,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    *,
    tile_size: float = 500.0,
    n_workers: int = 1,
) -> None:
    """
    Rasterize maximum return elevation to a DSM and write into *store*.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Grid cell size in metres.
    first_returns_only:
        Use only first returns for a clean canopy-top signal.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.
    tile_size:
        Sub-tile width/height in metres (default 500 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if year is not None and not check_year_exists(year, provider):
        return
    store.ensure_group("dsm", resolution, effective_bbox,
                       array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=0.0)
    logger.info("Computing DSM  (%.1f m, %d tile(s), %d worker(s), year=%s)",
                resolution, len(tiles), n_workers, year)
    _run_tiled(_process_tile_dsm, provider, tiles, store, n_workers,
               resolution=resolution, year=year,
               first_returns_only=first_returns_only)


def compute_all(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    first_returns_only: bool = True,
) -> None:
    """
    Compute DTM, DSM, and CHM in one call, writing all into *store*.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Grid cell size in metres.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.
    tile_size / n_workers:
        Tiling parameters forwarded to all three products.
    tile_buffer:
        Overlap buffer forwarded to :func:`compute_chm` only.
    first_returns_only:
        Forwarded to :func:`compute_chm`.  See that function for details.
    """
    compute_dtm(provider, store, resolution=resolution, bbox=bbox, year=year,
                tile_size=tile_size, n_workers=n_workers)
    compute_dsm(provider, store, resolution=resolution, bbox=bbox, year=year,
                tile_size=tile_size, n_workers=n_workers)
    compute_chm(provider, store, resolution=resolution, bbox=bbox, year=year,
                first_returns_only=first_returns_only,
                tile_size=tile_size, tile_buffer=tile_buffer, n_workers=n_workers)
