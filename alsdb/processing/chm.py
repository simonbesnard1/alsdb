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
    → ``filters.hag_delaunay``  builds a Delaunay TIN from Class-2 ground
                                 points and attaches ``HeightAboveGround``
                                 (falls back to ``filters.hag_nn`` for tiles
                                 with fewer than 3 ground points)
    → ``filters.assign``        clamps negative HAG values to 0
    → ``filters.range``         keeps vegetation points only (Class 3–5)
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
from typing import TYPE_CHECKING, Optional

import numpy as np
import pdal

from alsdb.processing._tiling import (
    _filter_ground_outliers,
    _hag_stage,
    _require_year,
    array_crs,
    array_data_bbox,
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

_GROUND_CLASS = 2
_VEG_CLASSES = (3, 4, 5)


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

    *statistic* may be any scipy built-in string (``"max"``, ``"mean"``, …)
    or a percentile shorthand such as ``"p95"`` (the 95th percentile).
    """
    from scipy.stats import binned_statistic_2d

    # Resolve percentile shorthand → callable
    if isinstance(statistic, str) and statistic.startswith("p") and statistic[1:].isdigit():
        q = int(statistic[1:])
        stat_fn = lambda arr: np.nanpercentile(arr, q) if len(arr) > 0 else np.nan  # noqa: E731
    else:
        stat_fn = statistic  # type: ignore[assignment]

    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))
    x_edges = np.linspace(cx0, cx1, nx + 1)
    y_edges = np.linspace(cy0, cy1, ny + 1)

    grid = binned_statistic_2d(
        x,
        y,
        values,
        statistic=stat_fn,
        bins=[x_edges, y_edges],
    ).statistic  # shape (nx, ny)

    return np.flipud(grid.T).astype(np.float32)  # → (ny, nx) north-up


# ---------------------------------------------------------------------------
# DTM interpolation helpers
# ---------------------------------------------------------------------------


def _nn_fill(
    grid: np.ndarray,
    gnd: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    """
    Fill NaN cells in *grid* using nearest-neighbour interpolation from
    ground points *gnd*.  Used as a fallback for cells outside the TIN
    convex hull (tile edges, data voids).
    """
    from scipy.spatial import cKDTree

    nan_mask = np.isnan(grid)
    if not nan_mask.any() or len(gnd) == 0:
        return grid

    cx0, _, _, cy1 = crop_bbox
    ny, nx = grid.shape
    row_idx, col_idx = np.where(nan_mask)
    # north-up: row 0 = top → actual Y = cy1 - (row + 0.5) * resolution
    qx = cx0 + (col_idx + 0.5) * resolution
    qy = cy1 - (row_idx + 0.5) * resolution

    xy = np.column_stack([gnd["X"].astype(np.float64), gnd["Y"].astype(np.float64)])
    _, idxs = cKDTree(xy).query(np.column_stack([qx, qy]), k=1)

    out = grid.copy()
    out[row_idx, col_idx] = gnd["Z"][idxs].astype(np.float32)
    return out


def _dtm_tin(
    arr: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    """
    TIN interpolation via PDAL ``filters.delaunay`` + ``filters.faceraster``.

    The full buffered point array *arr* is passed to ``filters.delaunay``
    without pre-cropping so that buffer ground points contribute to edge
    triangles, then ``filters.faceraster`` restricts output to *crop_bbox*.
    Any cells that remain NaN after triangulation (outside the convex hull)
    are filled by nearest-neighbour from the ground points.
    """
    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))

    stages = [
        {
            "type": "filters.range",
            "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]",
        },
        # No crop before delaunay — keep buffered points so edge triangles
        # are built from context beyond crop_bbox, then faceraster clips output.
        {"type": "filters.delaunay"},
        {
            "type": "filters.faceraster",
            "resolution": resolution,
            "origin_x": cx0,
            "origin_y": cy0,
            "width": nx,
            "height": ny,
        },
    ]
    raster_pts = _run(stages, arr)

    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    if len(raster_pts) > 0:
        # Cell centres: X = cx0 + (col + 0.5)*res, Y = cy0 + (row + 0.5)*res
        col = np.round((raster_pts["X"] - cx0) / resolution - 0.5).astype(int)
        row = np.round((raster_pts["Y"] - cy0) / resolution - 0.5).astype(int)
        valid = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        grid[row[valid], col[valid]] = raster_pts["Z"][valid].astype(np.float32)
    grid = np.flipud(grid)  # south-up → north-up

    # Fill cells outside the convex hull (tile edges, isolated voids)
    gnd_mask = arr["Classification"] == _GROUND_CLASS
    gnd = arr[gnd_mask]
    if len(gnd) > 0:
        grid = _nn_fill(grid, gnd, crop_bbox, resolution)

    return grid


def _dtm_idw(
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    power: float = 2.0,
    k: int = 8,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """
    IDW interpolation of ground points using scipy kd-tree.

    Equivalent to PDAL ``writers.gdal output_type=idw`` but works on an
    in-memory array without writing a temporary file.  Useful as a fallback
    for sparse tiles where ``filters.delaunay`` cannot build a mesh.

    Parameters
    ----------
    max_distance:
        If set, cells whose nearest ground point is farther than this value
        (metres) are left as NaN instead of being extrapolated.  Prevents
        spurious fill in genuine data voids (water bodies, survey gaps).
        Default ``None`` fills all cells.

    Returns a ``(ny, nx)`` float32 north-up array.
    """
    from scipy.spatial import cKDTree

    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))

    gx = cx0 + (np.arange(nx) + 0.5) * resolution
    gy = cy0 + (np.arange(ny) + 0.5) * resolution
    GX, GY = np.meshgrid(gx, gy)
    query = np.column_stack([GX.ravel(), GY.ravel()])

    xy = np.column_stack([points["X"].astype(np.float64), points["Y"].astype(np.float64)])
    z = points["Z"].astype(np.float64)

    k_actual = min(k, len(xy))
    tree = cKDTree(xy)
    dists, idxs = tree.query(query, k=k_actual)

    if k_actual == 1:
        values = z[idxs].copy()
        nearest_dist = dists
    else:
        weights = 1.0 / np.maximum(dists, 1e-10) ** power
        weights /= weights.sum(axis=1, keepdims=True)
        values = (weights * z[idxs]).sum(axis=1)
        nearest_dist = dists[:, 0]

    if max_distance is not None:
        values[nearest_dist > max_distance] = np.nan

    return np.flipud(values.reshape(ny, nx)).astype(np.float32)


# ---------------------------------------------------------------------------
# CHM post-processing
# ---------------------------------------------------------------------------


def _pit_fill(grid: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Remove spurious pits (NaN holes inside the canopy) and spikes from a CHM.

    Two-pass approach:
    1. Fill NaN pits adjacent to valid canopy with local median.
    2. Detect spikes via Laplacian (sharp local maxima) and replace them with
       the local median.  The Laplacian threshold is adaptive (99th percentile
       of all non-zero Laplacian values over *interior* valid pixels only),
       so mild canopy curvature is kept and boundary artefacts are avoided.

    Parameters
    ----------
    grid:
        Input CHM as ``(ny, nx)`` float32 north-up array.
    window:
        Neighbourhood window size for median filter (default 3 = 3×3 pixels).
    """
    from scipy.ndimage import binary_dilation, binary_erosion, laplace, median_filter

    out = grid.copy()
    nan_mask = np.isnan(out)
    valid_mask = ~nan_mask

    # Pass 1 — fill NaN pits that are directly adjacent to valid canopy
    if nan_mask.any():
        adjacent = nan_mask & binary_dilation(valid_mask, iterations=1)
        if adjacent.any():
            # Fill only the NaN pits with local median of valid neighbours;
            # use reflect mode so the median filter itself doesn't need zero-padding.
            local_med = median_filter(np.where(nan_mask, np.nanmedian(out), out), size=window)
            out = np.where(adjacent, local_med.astype(np.float32), out)
            valid_mask = ~np.isnan(out)

    # Pass 2 — Laplacian spike detection restricted to *interior* valid pixels.
    # Interior = valid pixels that are not adjacent to any NaN region.  Excluding
    # boundary pixels prevents the large Laplacian values caused by the canopy→NaN
    # step from being counted in the threshold or triggering false detections.
    interior_mask = valid_mask & binary_erosion(valid_mask, iterations=1)

    # Compute Laplacian only on valid data; fill NaN with the local median so
    # no artificial discontinuities are introduced at data boundaries.
    filled = np.where(nan_mask, np.nanmedian(out) if valid_mask.any() else 0.0, out)
    lap = np.abs(laplace(filled.astype(np.float64)))

    lap_interior = lap[interior_mask]
    lap_interior_nonzero = lap_interior[lap_interior > 0]
    if lap_interior_nonzero.size > 0:
        threshold = float(np.percentile(lap_interior_nonzero, 99))
        spike_mask = interior_mask & (lap > threshold)
        if spike_mask.any():
            local_med = median_filter(filled, size=window)
            out = np.where(spike_mask, local_med.astype(np.float32), out)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# PDAL helpers
# ---------------------------------------------------------------------------


def _run(stages: list, arr: np.ndarray) -> np.ndarray:
    """Execute a PDAL pipeline and return the output point array."""
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    return p.arrays[0] if p.arrays else arr[:0]


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------


def _process_tile_chm(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    first_returns_only: bool,
    height_statistic: str = "max",
    pit_fill: bool = True,
) -> None:
    arr = _filter_ground_outliers(query_to_array(provider, query_bbox, year=year))
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
        _hag_stage(arr),
        {
            "type": "filters.assign",
            "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
        },
        {"type": "filters.range", "limits": veg_limits},
        {"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
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
        points["X"],
        points["Y"],
        points["HeightAboveGround"],
        crop_bbox,
        resolution,
        statistic=height_statistic,
    )
    if pit_fill:
        grid = _pit_fill(grid)
    store.write_tile("chm", resolution, year, grid, crop_bbox)
    logger.debug("CHM tile %d written", tile_index)


def _process_tile_dtm(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    dtm_method: str = "tin",
) -> None:
    arr = _filter_ground_outliers(query_to_array(provider, query_bbox, year=year))
    if arr.size == 0:
        logger.debug("DTM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox

    try:
        if dtm_method == "tin":
            grid = _dtm_tin(arr, crop_bbox, resolution)
        else:
            # IDW and min both need ground points extracted first
            stages = [
                {
                    "type": "filters.range",
                    "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]",
                },
                {"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
            ]
            points = _run(stages, arr)
            if len(points) == 0:
                logger.debug("DTM tile %d: no ground points, skipping", tile_index)
                return
            if dtm_method == "idw":
                grid = _dtm_idw(points, crop_bbox, resolution)
            else:
                grid = _rasterise(
                    points["X"], points["Y"], points["Z"], crop_bbox, resolution, statistic="min"
                )
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("DTM tile %d: no ground points, skipping", tile_index)
            return
        raise

    if np.all(np.isnan(grid)):
        logger.debug("DTM tile %d: all NaN, skipping", tile_index)
        return

    store.write_tile("dtm", resolution, year, grid, crop_bbox)
    logger.debug("DTM tile %d written", tile_index)


def _process_tile_dsm(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
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
    stages.append({"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"})
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
        points["X"],
        points["Y"],
        points["Z"],
        crop_bbox,
        resolution,
        statistic="max",
    )
    store.write_tile("dsm", resolution, year, grid, crop_bbox)
    logger.debug("DSM tile %d written", tile_index)


# ---------------------------------------------------------------------------
# Combined tile worker (used by compute_all)
# ---------------------------------------------------------------------------


def _process_tile_all(
    provider: "TileDBProvider",
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    store: "ALSZarrStore",
    tile_index: int,
    resolution: float,
    year: Optional[int],
    first_returns_only: bool,
    need_dtm: bool,
    need_dsm: bool,
    need_chm: bool,
    height_statistic: str = "max",
    dtm_method: str = "tin",
    pit_fill: bool = True,
) -> None:
    """
    Single-pass tile worker for :func:`compute_all`.

    Performs one TileDB query and at most one HAG normalisation call
    to produce DTM, DSM, and CHM simultaneously.  Products already present
    in the store for *year* are skipped via the ``need_*`` flags.
    """
    if not (need_dtm or need_dsm or need_chm):
        logger.debug("All tile %d: all products already present, skipping", tile_index)
        return

    arr = _filter_ground_outliers(query_to_array(provider, query_bbox, year=year))
    if arr.size == 0:
        logger.debug("All tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox

    # --- DTM ---------------------------------------------------------------
    if need_dtm:
        try:
            if dtm_method == "tin":
                dtm_grid = _dtm_tin(arr, crop_bbox, resolution)
            else:
                stages = [
                    {
                        "type": "filters.range",
                        "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]",
                    },
                    {"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
                ]
                pts = _run(stages, arr)
                if dtm_method == "idw":
                    dtm_grid = _dtm_idw(pts, crop_bbox, resolution) if len(pts) else None
                else:
                    dtm_grid = (
                        _rasterise(pts["X"], pts["Y"], pts["Z"], crop_bbox, resolution, "min")
                        if len(pts)
                        else None
                    )
            if dtm_grid is not None and not np.all(np.isnan(dtm_grid)):
                store.write_tile("dtm", resolution, year, dtm_grid, crop_bbox)
        except RuntimeError as exc:
            if "no points" not in str(exc).lower():
                raise

    # --- DSM (max Z, optionally first returns only) — no HAG needed -----
    if need_dsm:
        stages = []
        if first_returns_only:
            stages.append({"type": "filters.range", "limits": "ReturnNumber[1:1]"})
        stages.append({"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"})
        try:
            pts = _run(stages, arr)
            if len(pts):
                store.write_tile(
                    "dsm",
                    resolution,
                    year,
                    _rasterise(pts["X"], pts["Y"], pts["Z"], crop_bbox, resolution, "max"),
                    crop_bbox,
                )
        except RuntimeError as exc:
            if "no points" not in str(exc).lower():
                raise

    # --- CHM (hag_delaunay/hag_nn + veg first returns) ------------------
    if need_chm:
        veg_limits = f"Classification[{_VEG_CLASSES[0]}:{_VEG_CLASSES[-1]}]"
        if first_returns_only:
            veg_limits += ",ReturnNumber[1:1]"
        stages = [
            _hag_stage(arr),
            {
                "type": "filters.assign",
                "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
            },
            {"type": "filters.range", "limits": veg_limits},
            {"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"},
        ]
        try:
            pts = _run(stages, arr)
            if len(pts):
                chm_grid = _rasterise(
                    pts["X"],
                    pts["Y"],
                    pts["HeightAboveGround"],
                    crop_bbox,
                    resolution,
                    height_statistic,
                )
                if pit_fill:
                    chm_grid = _pit_fill(chm_grid)
                store.write_tile("chm", resolution, year, chm_grid, crop_bbox)
        except RuntimeError as exc:
            if "no points" not in str(exc).lower():
                raise

    logger.debug(
        "All tile %d written (dtm=%s dsm=%s chm=%s)",
        tile_index,
        need_dtm,
        need_dsm,
        need_chm,
    )


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
    height_statistic: str = "max",
    pit_fill: bool = True,
    overwrite: bool = False,
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
    overwrite:
        If ``False`` (default) and CHM data for *year* already exists in
        the store, the computation is skipped entirely.  Set to ``True``
        to force recomputation.
    tile_size:
        Sub-tile width/height in metres (default 500 m).
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
    if not overwrite and store.has_data("chm", resolution, year):
        logger.info(
            "CHM already present for year %d at %.1f m — skipping "
            "(pass overwrite=True to recompute)",
            year,
            resolution,
        )
        return
    store.ensure_group("chm", resolution, effective_bbox, array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    logger.info(
        "Computing CHM  (%.1f m, %d tile(s), %d worker(s), year=%s, first_returns=%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        first_returns_only,
    )
    run_tiled(
        _process_tile_chm,
        provider,
        tiles,
        store,
        n_workers,
        resolution=resolution,
        year=year,
        first_returns_only=first_returns_only,
        height_statistic=height_statistic,
        pit_fill=pit_fill,
    )


def compute_dtm(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    *,
    dtm_method: str = "tin",
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 10.0,
    n_workers: int = 1,
) -> None:
    """
    Interpolate ground points (Class 2) to a DTM and write into *store*.

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
    dtm_method:
        Interpolation method for terrain surface estimation:

        - ``"tin"`` (default) — PDAL ``filters.delaunay`` +
          ``filters.faceraster``.  Linear interpolation within each
          Delaunay triangle; the gold-standard terrain model.
        - ``"idw"`` — Inverse Distance Weighting via scipy kd-tree (power=2,
          k=8 neighbours).  Smoother than TIN; better behaved on very sparse
          tiles where Delaunay cannot build a mesh.
        - ``"min"`` — Minimum-Z binning.  Fast but leaves NaN gaps wherever
          no ground return falls in a cell.
    overwrite:
        If ``False`` (default) and DTM data for *year* already exists in
        the store, the computation is skipped.
    tile_size:
        Sub-tile width/height in metres (default 500 m).
    tile_buffer:
        Query buffer so ground points near tile edges are available for
        interpolation (default 10 m; increase for very sparse surveys).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    _require_year(year)
    if dtm_method not in ("tin", "idw", "min"):
        raise ValueError(f"dtm_method must be 'tin', 'idw', or 'min'; got {dtm_method!r}")
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    if not overwrite and store.has_data("dtm", resolution, year):
        logger.info("DTM already present for year %d at %.1f m — skipping", year, resolution)
        return
    store.ensure_group("dtm", resolution, effective_bbox, array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    logger.info(
        "Computing DTM  (%.1f m, %d tile(s), %d worker(s), year=%s, method=%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        dtm_method,
    )
    run_tiled(
        _process_tile_dtm,
        provider,
        tiles,
        store,
        n_workers,
        resolution=resolution,
        year=year,
        dtm_method=dtm_method,
    )


def compute_dsm(
    provider: "TileDBProvider",
    store: "ALSZarrStore",
    resolution: float = 1.0,
    first_returns_only: bool = True,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    *,
    overwrite: bool = False,
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
    overwrite:
        If ``False`` (default) and DSM data for *year* already exists in
        the store, the computation is skipped.
    tile_size:
        Sub-tile width/height in metres (default 500 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    _require_year(year)
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    if not overwrite and store.has_data("dsm", resolution, year):
        logger.info("DSM already present for year %d at %.1f m — skipping", year, resolution)
        return
    store.ensure_group("dsm", resolution, effective_bbox, array_crs(provider), tile_size)
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=0.0)
    logger.info(
        "Computing DSM  (%.1f m, %d tile(s), %d worker(s), year=%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
    )
    run_tiled(
        _process_tile_dsm,
        provider,
        tiles,
        store,
        n_workers,
        resolution=resolution,
        year=year,
        first_returns_only=first_returns_only,
    )


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
    height_statistic: str = "max",
    pit_fill: bool = True,
    overwrite: bool = False,
    dtm_method: str = "tin",
) -> None:
    """
    Compute DTM, DSM, and CHM in one call, writing all into *store*.

    Uses a single TileDB query and a single HAG normalisation per
    tile, shared across all three products — avoiding the redundant work
    of calling each function separately.  Products already present in the
    store for *year* are skipped unless ``overwrite=True``.

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
        Tiling parameters.
    tile_buffer:
        Overlap buffer for ``filters.hag_delaunay`` (CHM only).
    first_returns_only:
        Use only first returns for CHM (and DSM).  See :func:`compute_chm`.
    overwrite:
        If ``False`` (default), skip products already present for *year*.
        If ``True``, recompute everything regardless.
    """
    _require_year(year)
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return

    # Determine which products still need computing
    need_dtm = overwrite or year is None or not store.has_data("dtm", resolution, year)
    need_dsm = overwrite or year is None or not store.has_data("dsm", resolution, year)
    need_chm = overwrite or year is None or not store.has_data("chm", resolution, year)

    if not (need_dtm or need_dsm or need_chm):
        logger.info(
            "compute_all: DTM, DSM and CHM already present for year %d at %.1f m "
            "— nothing to do (pass overwrite=True to recompute)",
            year,
            resolution,
        )
        return

    crs = array_crs(provider)
    if need_dtm:
        store.ensure_group("dtm", resolution, effective_bbox, crs, tile_size)
    if need_dsm:
        store.ensure_group("dsm", resolution, effective_bbox, crs, tile_size)
    if need_chm:
        store.ensure_group("chm", resolution, effective_bbox, crs, tile_size)

    # CHM needs the larger buffer for hag_delaunay; use it for all products
    # so a single tile list covers everything.
    buffer = tile_buffer if need_chm else 0.0
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=buffer)
    logger.info(
        "compute_all  (%.1f m, %d tile(s), %d worker(s), year=%s, dtm=%s dsm=%s chm=%s)",
        resolution,
        len(tiles),
        n_workers,
        year,
        need_dtm,
        need_dsm,
        need_chm,
    )
    run_tiled(
        _process_tile_all,
        provider,
        tiles,
        store,
        n_workers,
        resolution=resolution,
        year=year,
        first_returns_only=first_returns_only,
        need_dtm=need_dtm,
        need_dsm=need_dsm,
        need_chm=need_chm,
        height_statistic=height_statistic,
        pit_fill=pit_fill,
        dtm_method=dtm_method,
    )
