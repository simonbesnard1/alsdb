# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Canopy Height Model (CHM) computation using TileDB + PDAL.

Pattern (inspired by silvimetric)
----------------------------------
Data is queried from TileDB programmatically via
:class:`~alsdb.core.alsprovider.ALSProvider`, then injected into a PDAL
pipeline as a numpy structured array using PDAL's Python API
(``pdal.Pipeline(json, arrays=[arr])``).  This avoids the fragile
``readers.tiledb`` config-file approach entirely — PDAL only handles
filtering and rasterisation; TileDB handles all I/O.

Pipeline (inside PDAL)
-----------------------
numpy array input
    → ``filters.hag_delaunay``  builds a TIN from Class-2 ground points,
                                 attaches ``HeightAboveGround`` to every point
    → ``filters.range``         keeps vegetation points only (Class 3–5)
    → ``filters.assign``        clamps negative HAG values to 0
    → ``filters.crop``          clips to the non-buffered tile extent
    → ``writers.gdal``          rasterizes max(HAG) per cell → GeoTIFF

Tiled processing
----------------
Large areas are processed as a grid of sub-tiles.  For CHM each tile is
queried with an overlap *buffer* so ``filters.hag_delaunay`` has enough
ground points at edges; the buffer is removed by ``filters.crop`` before
writing.  DTM and DSM require no buffer (no TIN).  Sub-tiles are mosaicked
with ``rasterio.merge`` into the final GeoTIFF.

Usage::

    from alsdb import ALSProvider
    from alsdb.processing.chm import compute_chm, compute_dtm, compute_dsm

    provider = ALSProvider(storage_type="local", uri="array_")

    # Full tile CHM
    compute_chm(provider, "output/chm.tif", resolution=1.0)

    # Restrict to a bounding box
    compute_chm(provider, "output/chm.tif", resolution=1.0,
                bbox=(308000, 4688000, 310000, 4690000))

    # Tiled CHM (500 m tiles, 50 m buffer, 4 parallel workers)
    compute_chm(provider, "output/chm.tif", resolution=1.0,
                bbox=(308000, 4688000, 310000, 4690000),
                tile_size=500.0, tile_buffer=50.0, n_workers=4)

    # All three products at once
    compute_all(provider, output_dir="output/", resolution=1.0,
                bbox=(308000, 4688000, 310000, 4690000),
                tile_size=500.0, tile_buffer=50.0, n_workers=4)
"""

from __future__ import annotations

import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pdal

from alsdb.processing._tiling import array_domain_bbox, mosaic_tiles, tile_bboxes
from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)

_GROUND_CLASS = 2
_VEG_CLASSES = (3, 4, 5)

# Canonical PDAL dtype mapping for LAS dimensions
_PDAL_DTYPES: dict[str, type] = {
    "X": np.float64,
    "Y": np.float64,
    **LAS_ATTRIBUTES,
}


# ---------------------------------------------------------------------------
# TileDB → numpy structured array
# ---------------------------------------------------------------------------

def _query_to_array(
    provider: TileDBProvider,
    bbox: Optional[tuple[float, float, float, float]],
    year: Optional[int] = None,
) -> np.ndarray:
    """
    Query the TileDB array and return a PDAL-compatible numpy structured array.

    Parameters
    ----------
    provider:
        TileDB provider (ALSProvider or ALSDatabase).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
        Reads the full array if ``None``.
    year:
        Optional survey year filter.  ``None`` returns all years.

    Returns
    -------
    np.ndarray
        Structured numpy array with X, Y and all LAS attribute fields.
    """
    with provider.open("r") as arr:
        attrs = list(LAS_ATTRIBUTES.keys())
        yr_dim = arr.schema.domain.dim("Year")
        y0 = year if year is not None else int(yr_dim.domain[0])
        y1 = (year + 1) if year is not None else int(yr_dim.domain[1]) + 1
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            data = arr.query(attrs=attrs)[min_x:max_x, min_y:max_y, y0:y1]
        else:
            data = arr.query(attrs=attrs)[:, :, y0:y1]

    n = len(data["X"])
    logger.debug("Queried %d points from TileDB", n)

    dtype = [(name, _PDAL_DTYPES[name]) for name in _PDAL_DTYPES]
    out = np.empty(n, dtype=dtype)
    for name in _PDAL_DTYPES:
        out[name] = data[name].astype(_PDAL_DTYPES[name])
    return out


# ---------------------------------------------------------------------------
# PDAL pipeline builders
# ---------------------------------------------------------------------------

def _gdal_writer(
    output_path: str,
    resolution: float,
    dimension: str = "Z",
    output_type: str = "max",
    nodata: float = -9999.0,
) -> dict:
    return {
        "type": "writers.gdal",
        "filename": output_path,
        "dimension": dimension,
        "resolution": resolution,
        "output_type": output_type,
        "data_type": "float32",
        "nodata": nodata,
        "gdalopts": "COMPRESS=DEFLATE,PREDICTOR=3",
    }


def _run(stages: list, arr: np.ndarray) -> None:
    """Execute a PDAL pipeline with a numpy array as input."""
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    count = p.execute()
    logger.debug("PDAL executed: %d points processed", count)


# ---------------------------------------------------------------------------
# Per-tile worker functions
# ---------------------------------------------------------------------------

def _process_tile_chm(
    provider: TileDBProvider,
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    tmp_dir: str,
    tile_index: int,
    resolution: float,
    nodata: float,
    year: Optional[int],
) -> Optional[Path]:
    """Process a single CHM sub-tile (with HAG buffer)."""
    arr = _query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("CHM tile %d: no points, skipping", tile_index)
        return None

    tmp_path = Path(tmp_dir) / f"chm_tile_{tile_index:04d}.tif"
    cx0, cy0, cx1, cy1 = crop_bbox

    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.range",
         "limits": f"Classification[{_VEG_CLASSES[0]}:{_VEG_CLASSES[-1]}]"},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
        {"type": "filters.crop",
         "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
        _gdal_writer(str(tmp_path), resolution,
                     dimension="HeightAboveGround",
                     output_type="max", nodata=nodata),
    ]
    logger.debug("CHM tile %d: %.0f–%.0f / %.0f–%.0f", tile_index, cx0, cx1, cy0, cy1)
    try:
        _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("CHM tile %d: no vegetation points after filtering, skipping", tile_index)
            return None
        raise
    return tmp_path


def _process_tile_dtm(
    provider: TileDBProvider,
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    tmp_dir: str,
    tile_index: int,
    resolution: float,
    nodata: float,
    year: Optional[int],
) -> Optional[Path]:
    """Process a single DTM sub-tile (ground points only, no buffer needed)."""
    arr = _query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("DTM tile %d: no points, skipping", tile_index)
        return None

    tmp_path = Path(tmp_dir) / f"dtm_tile_{tile_index:04d}.tif"
    cx0, cy0, cx1, cy1 = crop_bbox

    stages = [
        {"type": "filters.range",
         "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]"},
        {"type": "filters.crop",
         "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
        _gdal_writer(str(tmp_path), resolution,
                     dimension="Z", output_type="max", nodata=nodata),
    ]
    logger.debug("DTM tile %d: %.0f–%.0f / %.0f–%.0f", tile_index, cx0, cx1, cy0, cy1)
    try:
        _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("DTM tile %d: no ground points after filtering, skipping", tile_index)
            return None
        raise
    return tmp_path


def _process_tile_dsm(
    provider: TileDBProvider,
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    tmp_dir: str,
    tile_index: int,
    resolution: float,
    nodata: float,
    year: Optional[int],
    first_returns_only: bool,
) -> Optional[Path]:
    """Process a single DSM sub-tile (no buffer needed)."""
    arr = _query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("DSM tile %d: no points, skipping", tile_index)
        return None

    tmp_path = Path(tmp_dir) / f"dsm_tile_{tile_index:04d}.tif"
    cx0, cy0, cx1, cy1 = crop_bbox

    stages: list = []
    if first_returns_only:
        stages.append({"type": "filters.range", "limits": "ReturnNumber[1:1]"})
    stages += [
        {"type": "filters.crop",
         "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
        _gdal_writer(str(tmp_path), resolution,
                     dimension="Z", output_type="max", nodata=nodata),
    ]
    logger.debug("DSM tile %d: %.0f–%.0f / %.0f–%.0f", tile_index, cx0, cx1, cy0, cy1)
    try:
        _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("DSM tile %d: no points after filtering, skipping", tile_index)
            return None
        raise
    return tmp_path


# ---------------------------------------------------------------------------
# Shared tiled executor
# ---------------------------------------------------------------------------

def _run_tiled(
    worker_fn,
    provider: TileDBProvider,
    tiles: list,
    tmp_dir: str,
    n_workers: int,
    **kwargs,
) -> list[Path]:
    """
    Run *worker_fn* over all *tiles*, sequentially or in parallel.

    Returns a sorted list of non-None output paths.
    """
    tile_paths: list[Path] = []

    if n_workers == 1:
        for idx, (query_bbox, crop_bbox) in enumerate(tiles):
            result = worker_fn(provider, query_bbox, crop_bbox,
                               tmp_dir, idx, **kwargs)
            if result is not None:
                tile_paths.append(result)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    worker_fn, provider, query_bbox, crop_bbox,
                    tmp_dir, idx, **kwargs
                ): idx
                for idx, (query_bbox, crop_bbox) in enumerate(tiles)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    tile_paths.append(result)

    tile_paths.sort()
    return tile_paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_chm(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
    *,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    year: Optional[int] = None,
) -> Path:
    """
    Compute a Canopy Height Model from a TileDB array.

    Parameters
    ----------
    provider:
        :class:`~alsdb.providers.tiledb_provider.TileDBProvider` instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres (default 1 m).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for ``filters.hag_delaunay`` accuracy (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    year:
        Optional survey year filter.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_bbox = bbox if bbox is not None else array_domain_bbox(provider)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    n_tiles = len(tiles)
    logger.info("Computing CHM → %s  (%.1f m, %d tile(s), %d worker(s))",
                output_path, resolution, n_tiles, n_workers)

    if n_tiles == 1:
        query_bbox, crop_bbox = tiles[0]
        arr = _query_to_array(provider, query_bbox, year=year)
        cx0, cy0, cx1, cy1 = crop_bbox
        stages = [
            {"type": "filters.hag_delaunay"},
            {"type": "filters.range",
             "limits": f"Classification[{_VEG_CLASSES[0]}:{_VEG_CLASSES[-1]}]"},
            {"type": "filters.assign",
             "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
            {"type": "filters.crop",
             "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
            _gdal_writer(str(output_path), resolution,
                         dimension="HeightAboveGround",
                         output_type="max", nodata=nodata),
        ]
        _run(stages, arr)
        return output_path

    with tempfile.TemporaryDirectory(prefix="alsdb_chm_") as tmp_dir:
        tile_paths = _run_tiled(
            _process_tile_chm, provider, tiles, tmp_dir, n_workers,
            resolution=resolution, nodata=nodata, year=year,
        )
        if not tile_paths:
            logger.warning("No CHM tiles produced (no points in bbox)")
            return output_path
        mosaic_tiles(tile_paths, output_path, nodata=nodata)

    return output_path


def compute_dtm(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
    *,
    tile_size: float = 500.0,
    n_workers: int = 1,
    year: Optional[int] = None,
) -> Path:
    """
    Rasterize ground points (Class 2) to a DTM GeoTIFF.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    year:
        Optional survey year filter.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_bbox = bbox if bbox is not None else array_domain_bbox(provider)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=0.0)
    n_tiles = len(tiles)
    logger.info("Computing DTM → %s  (%.1f m, %d tile(s), %d worker(s))",
                output_path, resolution, n_tiles, n_workers)

    if n_tiles == 1:
        arr = _query_to_array(provider, bbox, year=year)
        stages = [
            {"type": "filters.range",
             "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]"},
            _gdal_writer(str(output_path), resolution,
                         dimension="Z", output_type="max", nodata=nodata),
        ]
        _run(stages, arr)
        return output_path

    with tempfile.TemporaryDirectory(prefix="alsdb_dtm_") as tmp_dir:
        tile_paths = _run_tiled(
            _process_tile_dtm, provider, tiles, tmp_dir, n_workers,
            resolution=resolution, nodata=nodata, year=year,
        )
        if not tile_paths:
            logger.warning("No DTM tiles produced (no points in bbox)")
            return output_path
        mosaic_tiles(tile_paths, output_path, nodata=nodata)

    return output_path


def compute_dsm(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 1.0,
    first_returns_only: bool = True,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
    *,
    tile_size: float = 500.0,
    n_workers: int = 1,
    year: Optional[int] = None,
) -> Path:
    """
    Rasterize maximum return elevation to a DSM GeoTIFF.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres.
    first_returns_only:
        Use only first returns for a clean canopy-top signal.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    year:
        Optional survey year filter.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_bbox = bbox if bbox is not None else array_domain_bbox(provider)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=0.0)
    n_tiles = len(tiles)
    logger.info("Computing DSM → %s  (%.1f m, %d tile(s), %d worker(s))",
                output_path, resolution, n_tiles, n_workers)

    if n_tiles == 1:
        arr = _query_to_array(provider, bbox, year=year)
        stages: list = []
        if first_returns_only:
            stages.append({"type": "filters.range", "limits": "ReturnNumber[1:1]"})
        stages.append(_gdal_writer(str(output_path), resolution,
                                   dimension="Z", output_type="max", nodata=nodata))
        _run(stages, arr)
        return output_path

    with tempfile.TemporaryDirectory(prefix="alsdb_dsm_") as tmp_dir:
        tile_paths = _run_tiled(
            _process_tile_dsm, provider, tiles, tmp_dir, n_workers,
            resolution=resolution, nodata=nodata, year=year,
            first_returns_only=first_returns_only,
        )
        if not tile_paths:
            logger.warning("No DSM tiles produced (no points in bbox)")
            return output_path
        mosaic_tiles(tile_paths, output_path, nodata=nodata)

    return output_path


def compute_all(
    provider: TileDBProvider,
    output_dir: str | Path,
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
    year: Optional[int] = None,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
) -> dict[str, Path]:
    """
    Compute DTM, DSM, and CHM in one call.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_dir:
        Output directory (created if it does not exist).
    resolution:
        Grid cell size in metres.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.
    year:
        Optional survey year filter.
    tile_size / tile_buffer / n_workers:
        Tiling parameters forwarded to all three products.

    Returns
    -------
    dict
        ``{"dtm": Path, "dsm": Path, "chm": Path}``
    """
    output_dir = Path(output_dir)
    return {
        "dtm": compute_dtm(provider, output_dir / "dtm.tif",
                           resolution=resolution, bbox=bbox, nodata=nodata,
                           tile_size=tile_size, n_workers=n_workers, year=year),
        "dsm": compute_dsm(provider, output_dir / "dsm.tif",
                           resolution=resolution, bbox=bbox, nodata=nodata,
                           tile_size=tile_size, n_workers=n_workers, year=year),
        "chm": compute_chm(provider, output_dir / "chm.tif",
                           resolution=resolution, bbox=bbox, nodata=nodata,
                           tile_size=tile_size, tile_buffer=tile_buffer,
                           n_workers=n_workers, year=year),
    }
