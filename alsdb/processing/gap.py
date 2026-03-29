# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Gap fraction and (optionally) effective LAI from TileDB ALS point clouds.

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
angle distribution).  This introduces two assumptions:

1. The canopy behaves as a random turbid medium (violated by clumping).
2. *k = 0.5* (varies by species and canopy structure).

For this reason LAI is opt-in via ``lai=True`` and *k* must be supplied
explicitly so the assumption is visible in the calling code.  The output
is "effective LAI" (L_e), not true LAI, because ALS cannot distinguish
leaves from woody material.

Usage::

    from alsdb import ALSProvider
    from alsdb.processing.gap import compute_gap_fraction

    provider = ALSProvider(storage_type="local", uri="array_")

    # Gap fraction only (no assumptions)
    compute_gap_fraction(provider, "output/gap.tif",
                         bbox=(308000, 4688000, 310000, 4690000),
                         resolution=10.0, year=2021)

    # Gap fraction + effective LAI (Beer-Lambert, k=0.5)
    compute_gap_fraction(provider, "output/gap.tif",
                         resolution=10.0, year=2021,
                         lai=True, k=0.5,
                         lai_path="output/lai.tif")

    # Tiled processing for large areas
    compute_gap_fraction(provider, "output/gap.tif",
                         resolution=10.0, year=2021,
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

from alsdb.processing._tiling import (
    array_domain_bbox, mosaic_tiles, query_to_array, tile_bboxes,
)
from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

_GROUND_CLASS = 2
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

    Parameters
    ----------
    points:
        Structured numpy array with ``X``, ``Y``, ``Classification``,
        ``ReturnNumber`` fields (as returned by PDAL after
        ``filters.hag_delaunay``).
    resolution:
        Cell size in metres.
    bbox:
        ``(min_x, min_y, max_x, max_y)`` — defines the output grid extent.

    Returns
    -------
    np.ndarray
        2-D float32 array (ny, nx) of gap fraction values in [0, 1].
        Cells with no first returns are ``np.nan``.
    """
    from scipy.stats import binned_statistic_2d

    min_x, min_y, max_x, max_y = bbox
    nx = max(1, int(np.ceil((max_x - min_x) / resolution)))
    ny = max(1, int(np.ceil((max_y - min_y) / resolution)))
    x_edges = np.linspace(min_x, max_x, nx + 1)
    y_edges = np.linspace(min_y, max_y, ny + 1)
    bins = [x_edges, y_edges]

    fr = points["ReturnNumber"] == 1
    x_fr = points["X"][fr]
    y_fr = points["Y"][fr]
    cls_fr = points["Classification"][fr]

    gnd = (cls_fr == _GROUND_CLASS).astype(np.float32)
    veg = np.isin(cls_fr, _VEG_CLASSES).astype(np.float32)
    ones = np.ones(fr.sum(), dtype=np.float32)

    n_gnd  = binned_statistic_2d(x_fr, y_fr, gnd,  statistic="sum",   bins=bins).statistic
    n_veg  = binned_statistic_2d(x_fr, y_fr, veg,  statistic="sum",   bins=bins).statistic
    n_tot  = binned_statistic_2d(x_fr, y_fr, ones, statistic="count", bins=bins).statistic

    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(n_tot > 0, n_gnd / (n_gnd + n_veg), np.nan)

    # flip from (nx, ny) → (ny, nx) north-up
    return np.flipud(gap.T).astype(np.float32)


def _gap_to_lai(gap: np.ndarray, k: float) -> np.ndarray:
    """
    Convert a gap fraction grid to effective LAI via Beer-Lambert.

    ``L_e = -ln(P_gap) / k``

    Cells where P_gap == 0 (fully closed canopy) are clamped to
    ``_LAI_MAX`` rather than returning inf.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        lai = -np.log(np.where(gap > 0, gap, np.nan)) / k
    return np.clip(lai, 0.0, _LAI_MAX).astype(np.float32)


# ---------------------------------------------------------------------------
# Rasterio writer
# ---------------------------------------------------------------------------

def _write_raster(
    grid: np.ndarray,
    path: Path,
    bbox: tuple[float, float, float, float],
    nodata: float,
    crs: Optional[str] = None,
) -> Path:
    """Write a single-band float32 GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds

    ny, nx = grid.shape
    min_x, min_y, max_x, max_y = bbox
    transform = from_bounds(min_x, min_y, max_x, max_y, nx, ny)
    out = np.where(np.isnan(grid), nodata, grid).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
        transform=transform, nodata=nodata,
        compress="deflate", predictor=3, tiled=True,
        blockxsize=256, blockysize=256,
    )
    if crs:
        profile["crs"] = crs

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    return path


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------

def _process_tile(
    provider: TileDBProvider,
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    tmp_dir: str,
    tile_index: int,
    resolution: float,
    nodata: float,
    year: Optional[int],
    lai: bool,
    k: float,
) -> Optional[dict[str, Path]]:
    """
    Process a single gap fraction sub-tile.

    Returns ``{"gap": Path}`` or ``{"gap": Path, "lai": Path}``,
    or ``None`` if the tile contains no first returns.
    """
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("Gap tile %d: no points, skipping", tile_index)
        return None

    # Attach HAG so we can filter by classification correctly
    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
    ]
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    points = p.arrays[0]

    gap = _compute_gap_grid(points, resolution, crop_bbox)

    if np.all(np.isnan(gap)):
        logger.debug("Gap tile %d: all NaN, skipping", tile_index)
        return None

    out: dict[str, Path] = {}

    gap_path = Path(tmp_dir) / f"gap_tile_{tile_index:04d}.tif"
    _write_raster(gap, gap_path, crop_bbox, nodata)
    out["gap"] = gap_path

    if lai:
        lai_grid = _gap_to_lai(gap, k)
        lai_path = Path(tmp_dir) / f"lai_tile_{tile_index:04d}.tif"
        _write_raster(lai_grid, lai_path, crop_bbox, nodata)
        out["lai"] = lai_path

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gap_fraction(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 10.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    nodata: float = -9999.0,
    *,
    lai: bool = False,
    k: float = _LAI_K_DEFAULT,
    lai_path: Optional[str | Path] = None,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
) -> dict[str, Path]:
    """
    Compute gap fraction (and optionally effective LAI) from a TileDB array.

    Gap fraction is the MacArthur-Wilson estimator:

        P_gap = N_gnd_first / (N_gnd_first + N_veg_first)

    This is a direct observable — no canopy-structure assumptions required.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_path:
        Output GeoTIFF path for the gap fraction raster.
    resolution:
        Cell size in metres (default 10 m).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Optional survey year filter.
    nodata:
        No-data fill value.
    lai:
        If ``True``, also compute effective LAI via Beer-Lambert
        ``L_e = -ln(P_gap) / k``.  Requires ``lai_path``.
    k:
        Extinction coefficient for the Beer-Lambert LAI estimate
        (default 0.5, spherical leaf angle distribution).
        Only used when ``lai=True``.  **Changing this value changes
        the LAI output** — choose carefully for your vegetation type.
    lai_path:
        Output GeoTIFF path for the LAI raster.  Required when
        ``lai=True``; ignored otherwise.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Overlap buffer for ``filters.hag_delaunay`` accuracy (default 50 m).
    n_workers:
        Parallel workers (default 1 = sequential).

    Returns
    -------
    dict
        ``{"gap": Path}`` or ``{"gap": Path, "lai": Path}``.

    Raises
    ------
    ValueError
        If ``lai=True`` but ``lai_path`` is not provided.
    """
    if lai and not lai_path:
        raise ValueError("lai_path must be provided when lai=True")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_bbox = bbox if bbox is not None else array_domain_bbox(provider)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    n_tiles = len(tiles)
    logger.info(
        "Computing gap fraction → %s  (%.0f m, %d tile(s), %d worker(s))",
        output_path, resolution, n_tiles, n_workers,
    )

    result_paths: dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="alsdb_gap_") as tmp_dir:
        gap_tiles:  list[Path] = []
        lai_tiles:  list[Path] = []

        def _collect(idx, query_bbox, crop_bbox):
            return _process_tile(
                provider, query_bbox, crop_bbox, tmp_dir, idx,
                resolution, nodata, year, lai, k,
            )

        if n_workers == 1:
            results = [
                _collect(i, qb, cb) for i, (qb, cb) in enumerate(tiles)
            ]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_collect, i, qb, cb): i
                    for i, (qb, cb) in enumerate(tiles)
                }
                results = [None] * len(tiles)
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

        for r in results:
            if r is None:
                continue
            if "gap" in r:
                gap_tiles.append(r["gap"])
            if "lai" in r:
                lai_tiles.append(r["lai"])

        if not gap_tiles:
            logger.warning("No gap fraction tiles produced (no points in bbox)")
            return result_paths

        gap_tiles.sort()
        if len(gap_tiles) == 1:
            import shutil
            shutil.copy(gap_tiles[0], output_path)
            gap_tiles[0].unlink()
        else:
            mosaic_tiles(gap_tiles, output_path, nodata=nodata)
        result_paths["gap"] = output_path

        if lai and lai_tiles:
            lai_out = Path(lai_path)
            lai_out.parent.mkdir(parents=True, exist_ok=True)
            lai_tiles.sort()
            if len(lai_tiles) == 1:
                import shutil
                shutil.copy(lai_tiles[0], lai_out)
                lai_tiles[0].unlink()
            else:
                mosaic_tiles(lai_tiles, lai_out, nodata=nodata)
            result_paths["lai"] = lai_out
            logger.info("LAI (k=%.2f) → %s", k, lai_out)

    return result_paths
