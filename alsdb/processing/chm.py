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
    VEG_CLASSES as _VEG_CLASSES,
    _filter_ground_outliers,
    _hag_stage,
    _require_year,
    array_crs,
    array_data_bbox,
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

_GROUND_CLASS = 2
_PITFREE_THRESHOLDS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)


def _classification_ranges(classes: tuple[int, ...]) -> str:
    """
    Build a PDAL ``filters.range`` limits string for a (possibly
    non-contiguous) set of classification codes.

    ``Classification[a:b]`` only expresses one contiguous range - simply
    taking ``min``/``max`` of an arbitrary class tuple like ``(1, 3, 4, 5)``
    would wrongly also admit Class 2 (ground). This groups the sorted classes
    into contiguous runs and joins them, repeating the dimension name per
    group as PDAL requires (bracket-continuation without repeating the name
    fails with "No dimension name"), e.g. ``(1, 3, 4, 5)`` ->
    ``"Classification[1:1],Classification[3:5]"``.
    """
    codes = sorted(set(classes))
    groups: list[list[int]] = []
    for c in codes:
        if groups and c == groups[-1][-1] + 1:
            groups[-1].append(c)
        else:
            groups.append([c])
    parts = [f"Classification[{g[0]}:{g[-1]}]" for g in groups]
    return ",".join(parts)


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

    return flip_to_north_up(grid, transpose=True)  # → (ny, nx) north-up


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


def _delaunay_raster(
    arr: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    value_field: str = "Z",
) -> np.ndarray:
    """
    TIN interpolation via PDAL ``filters.delaunay`` + ``filters.faceraster``.

    Triangulates *arr* and rasterises to *crop_bbox*.  *arr* is passed in
    without pre-cropping so that points beyond *crop_bbox* still contribute
    to edge triangles; ``filters.faceraster`` restricts the output raster to
    *crop_bbox* on its own.

    *value_field* selects which per-point field gets interpolated.
    ``filters.faceraster`` always rasterises whatever is in ``Z`` — so for
    any other field (e.g. ``"HeightAboveGround"``, used by
    :func:`_pitfree_rasterise`), a copy of *arr* has ``Z`` overwritten with
    that field before triangulation.  *arr*'s dtype must already declare a
    ``Z`` field (even if its values are about to be overwritten) since a
    structured array's fields can't be added on the fly.

    ``filters.faceraster``'s interpolated raster is only materialised
    through a downstream raster-writer stage — reading the pipeline's point
    array afterward (what this function used to do) silently returns the
    *input* points completely unchanged: same X/Y/Z, same count, empty
    stage metadata. Confirmed empirically, not documented behaviour any
    docstring here should have assumed. So this writes to GDAL's in-memory
    ``/vsimem/`` filesystem (no real disk I/O) via ``writers.raster`` and
    reads the actual interpolated grid back with rasterio.

    Returns a ``(ny, nx)`` float32 north-up array. All-NaN if *arr* has
    fewer than 3 points, or if the triangulation itself fails (degenerate/
    collinear input) — treated as "this input contributes nothing" rather
    than raised, since callers such as :func:`_pitfree_rasterise` combine
    several such rasters and a single failed layer shouldn't abort the rest.
    """
    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))
    empty = np.full((ny, nx), np.nan, dtype=np.float32)

    if len(arr) < 3:
        return empty

    if value_field != "Z":
        arr = arr.copy()
        arr["Z"] = arr[value_field]

    import uuid

    import rasterio
    import rasterio.shutil as rio_shutil

    vsi_path = f"/vsimem/_delaunay_raster_{uuid.uuid4().hex}.tif"
    stages = [
        {"type": "filters.delaunay"},
        {
            "type": "filters.faceraster",
            "resolution": resolution,
            "origin_x": cx0,
            "origin_y": cy0,
            "width": nx,
            "height": ny,
        },
        {"type": "writers.raster", "gdaldriver": "GTiff", "filename": vsi_path},
    ]
    try:
        _run(stages, arr)
        with rasterio.open(vsi_path) as ds:
            band = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band).astype(np.float32)
        return band
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            return empty
        logger.debug("_delaunay_raster: triangulation failed (%s), returning all-NaN", exc)
        return empty
    finally:
        try:
            rio_shutil.delete(vsi_path)
        except Exception:
            pass


def _dtm_tin(
    arr: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    """
    Ground TIN via :func:`_delaunay_raster`, with nearest-neighbour fallback.

    Ground points (Class 2) are extracted from the full buffered array
    *before* triangulation so that buffer points contribute to edge
    triangles, matching :func:`_delaunay_raster`'s no-precrop contract. Any
    cells that remain NaN after triangulation (outside the convex hull) are
    filled by nearest-neighbour from the same ground points.
    """
    gnd_pts = _run(
        [{"type": "filters.range", "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]"}],
        arr,
    )
    grid = _delaunay_raster(gnd_pts, crop_bbox, resolution, value_field="Z")

    gnd_mask = arr["Classification"] == _GROUND_CLASS
    gnd = arr[gnd_mask]
    if len(gnd) > 0:
        grid = _nn_fill(grid, gnd, crop_bbox, resolution)

    return grid


def _mask_by_point_distance(
    grid: np.ndarray,
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    max_distance: float,
) -> np.ndarray:
    """
    Null out (set NaN) cells in *grid* whose nearest point in *points* is
    farther than *max_distance* (metres).

    Delaunay triangulation interpolates across the full interior of a point
    set's convex hull, including long triangles spanning genuine gaps
    between distant, sparse clusters with nothing supporting the surface in
    between.

    Complements :func:`_gate_by_ground_distance` (which drops vegetation
    points before rasterisation, based on distance to *ground* points) -
    this instead masks the rasterised cells themselves, based on distance to
    the same layer's own points, after triangulation.
    """
    from scipy.spatial import cKDTree

    valid_mask = ~np.isnan(grid)
    if not valid_mask.any() or len(points) == 0:
        return grid

    cx0, _, _, cy1 = crop_bbox
    row_idx, col_idx = np.where(valid_mask)
    # north-up: row 0 = top → actual Y = cy1 - (row + 0.5) * resolution
    qx = cx0 + (col_idx + 0.5) * resolution
    qy = cy1 - (row_idx + 0.5) * resolution

    xy = np.column_stack([points["X"].astype(np.float64), points["Y"].astype(np.float64)])
    dist, _ = cKDTree(xy).query(np.column_stack([qx, qy]), k=1)

    out = grid.copy()
    too_far = dist > max_distance
    out[row_idx[too_far], col_idx[too_far]] = np.nan
    return out


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

    return flip_to_north_up(values.reshape(ny, nx))


def _gate_by_ground_distance(
    points: np.ndarray,
    ground: np.ndarray,
    max_distance: Optional[float],
) -> np.ndarray:
    """
    Drop points farther than *max_distance* from the nearest ground point.

    ``filters.hag_delaunay``/``filters.hag_nn`` will produce a
    ``HeightAboveGround`` value for a point regardless of how far it sits
    from real ground support - extrapolating from a distant or
    poorly-conditioned triangle rather than admitting the estimate is
    unreliable. Dropping the point instead leaves that cell ``NaN`` in the
    output, which is more honest than a confident but unsupported guess.

    Parameters
    ----------
    points:
        Vegetation points with an attached ``HeightAboveGround`` field.
    ground:
        Class-2 ground points from the same (buffered) query - the source
        of "real local support", independent of whichever ground points
        actually fed the HAG interpolation.
    max_distance:
        Maximum distance (metres) to the nearest ground point. ``None``
        disables gating (default - preserves prior behaviour).
    """
    if max_distance is None or len(points) == 0 or len(ground) == 0:
        return points

    from scipy.spatial import cKDTree

    gnd_xy = np.column_stack([ground["X"].astype(np.float64), ground["Y"].astype(np.float64)])
    pts_xy = np.column_stack([points["X"].astype(np.float64), points["Y"].astype(np.float64)])
    dist, _ = cKDTree(gnd_xy).query(pts_xy, k=1)
    return points[dist <= max_distance]


# ---------------------------------------------------------------------------
# Pit-free canopy-top rasterisation
# ---------------------------------------------------------------------------


def _pitfree_rasterise(
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """
    Pit-free CHM rasterisation (Khosravipour et al., 2014), adapted to
    per-point ``HeightAboveGround`` instead of raw elevation.

    Parameters
    ----------
    points:
        HAG-normalised, already classification-filtered (and optionally
        ground-distance-gated) vegetation points — the same input that
        would otherwise go straight into ``_rasterise(..., statistic="max")``.
        Not pre-cropped to *crop_bbox*: each threshold layer's triangulation
        benefits from the same buffered-context handling as
        :func:`_delaunay_raster`/:func:`_dtm_tin`.
    thresholds:
        Increasing sequence of height cutoffs in metres (default
        ``(0, 2, 5, 10, 15, 20)``). Not adaptive to point density — the same
        limitation LAStools' own ``-spike_free`` defaults have; tune for a
        specific survey's point density if needed.
    max_distance:
        Maximum distance (metres) from a cell to that layer's nearest point
        before the cell is masked out instead of trusted. ``None`` (default)
        disables masking; set it to bound extrapolation across genuine
        vegetation gaps (see above).

    Returns
    -------
    A ``(ny, nx)`` float32 north-up array. NaN wherever no threshold layer
    covers a cell (equivalent to a prior "no data" cell).
    """
    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))
    stacked = np.full((len(thresholds), ny, nx), np.nan, dtype=np.float32)

    for i, t in enumerate(thresholds):
        layer_pts = points[points["HeightAboveGround"] >= t]
        if len(layer_pts) < 3:
            continue
        layer_grid = _delaunay_raster(
            layer_pts, crop_bbox, resolution, value_field="HeightAboveGround"
        )
        if max_distance is not None:
            layer_grid = _mask_by_point_distance(
                layer_grid, layer_pts, crop_bbox, resolution, max_distance
            )
        stacked[i] = layer_grid

    all_nan_cols = np.all(np.isnan(stacked), axis=0)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    grid[~all_nan_cols] = np.nanmax(stacked[:, ~all_nan_cols], axis=0)
    return grid


# ---------------------------------------------------------------------------
# CHM post-processing
# ---------------------------------------------------------------------------


def _fill_pits(grid: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Fill NaN pits (small gaps inside the canopy footprint) in a CHM.

    Parameters
    ----------
    grid:
        Input CHM as ``(ny, nx)`` float32 north-up array.
    window:
        Neighbourhood window size for the median filter (default 3 = 3×3 pixels).
    """
    from scipy.ndimage import binary_dilation, median_filter

    out = grid.copy()
    nan_mask = np.isnan(out)
    valid_mask = ~nan_mask

    if nan_mask.any():
        adjacent = nan_mask & binary_dilation(valid_mask, iterations=1)
        if adjacent.any():
            # Fill only the NaN pits with local median of valid neighbours;
            # use reflect mode so the median filter itself doesn't need zero-padding.
            local_med = median_filter(np.where(nan_mask, np.nanmedian(out), out), size=window)
            out = np.where(adjacent, local_med.astype(np.float32), out)

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
    max_ground_distance: Optional[float] = None,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    pitfree: bool = False,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_distance: Optional[float] = None,
) -> None:
    arr = _filter_ground_outliers(query_to_array(provider, query_bbox, year=year))
    if arr.size == 0:
        logger.debug("CHM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox
    # Filter to vegetation classes; optionally restrict to first returns so
    # that only the top-of-canopy surface is modelled (recommended).
    veg_limits = _classification_ranges(veg_classes)
    if first_returns_only:
        veg_limits += ",ReturnNumber[1:1]"
    stages = [
        _hag_stage(arr),
        {
            "type": "filters.assign",
            "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
        },
        {"type": "filters.range", "limits": veg_limits},
    ]
    if not pitfree:
        # Naive max-binning needs points pre-cropped to the tile; pit-free's
        # per-threshold Delaunay layers instead crop via faceraster after
        # triangulating the full buffered extent, so edge triangles have
        # real neighbouring context (same reasoning as _dtm_tin).
        stages.append({"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"})
    try:
        points = _run(stages, arr)
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            logger.debug("CHM tile %d: no vegetation points, skipping", tile_index)
            return
        raise

    if len(points) == 0:
        return

    if max_ground_distance is not None:
        gnd = arr[arr["Classification"] == _GROUND_CLASS]
        points = _gate_by_ground_distance(points, gnd, max_ground_distance)
        if len(points) == 0:
            return

    if pitfree:
        grid = _pitfree_rasterise(
            points,
            crop_bbox,
            resolution,
            thresholds=pitfree_thresholds,
            max_distance=pitfree_max_distance,
        )
    else:
        grid = _rasterise(
            points["X"],
            points["Y"],
            points["HeightAboveGround"],
            crop_bbox,
            resolution,
            statistic=height_statistic,
        )
    if pit_fill:
        grid = _fill_pits(grid)
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
    max_ground_distance: Optional[float] = None,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    pitfree: bool = False,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_distance: Optional[float] = None,
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
        veg_limits = _classification_ranges(veg_classes)
        if first_returns_only:
            veg_limits += ",ReturnNumber[1:1]"
        stages = [
            _hag_stage(arr),
            {
                "type": "filters.assign",
                "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
            },
            {"type": "filters.range", "limits": veg_limits},
        ]
        if not pitfree:
            stages.append({"type": "filters.crop", "bounds": f"([{cx0},{cx1}],[{cy0},{cy1}])"})
        try:
            pts = _run(stages, arr)
            if max_ground_distance is not None:
                gnd = arr[arr["Classification"] == _GROUND_CLASS]
                pts = _gate_by_ground_distance(pts, gnd, max_ground_distance)
            if len(pts):
                if pitfree:
                    chm_grid = _pitfree_rasterise(
                        pts,
                        crop_bbox,
                        resolution,
                        thresholds=pitfree_thresholds,
                        max_distance=pitfree_max_distance,
                    )
                else:
                    chm_grid = _rasterise(
                        pts["X"],
                        pts["Y"],
                        pts["HeightAboveGround"],
                        crop_bbox,
                        resolution,
                        height_statistic,
                    )
                if pit_fill:
                    chm_grid = _fill_pits(chm_grid)
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
    max_ground_distance: Optional[float] = None,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    pitfree: bool = False,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_distance: Optional[float] = None,
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
    max_ground_distance:
        If set, drop vegetation points farther than this distance (metres)
        from the nearest ground point instead of trusting whatever HAG
        value ``hag_delaunay``/``hag_nn`` extrapolated for them - those
        cells are left ``NaN`` rather than filled with an unsupported
        guess.  ``None`` (default) preserves prior behaviour.
    veg_classes:
        LAS classification codes treated as "canopy" for the top surface
        (default ``(3, 4, 5)`` - low/medium/high vegetation). Benchmarking
        against an independent reference found that automatic ground-truth
        classification is least reliable under tall/dense canopy, where a
        genuine canopy-top point can be left ``Class 1`` (unclassified)
        instead of confidently labelled - pass e.g. ``(1, 3, 4, 5)`` to
        include unclassified points alongside vegetation. Does not need to
        be contiguous.
    pitfree:
        If ``True``, replace naive per-cell max-binning with pit-free
        rasterisation (Khosravipour et al., 2014): builds a Delaunay TIN
        per height threshold in ``pitfree_thresholds`` (from per-point
        ``HeightAboveGround`` rather than raw elevation) and takes the
        cell-wise maximum across all layers. A higher threshold excludes
        the spurious low return that would otherwise triangulate into an
        interior pit, while lower thresholds still cover the rest of the
        canopy — so pits are avoided during surface construction instead of
        detected and patched afterward. Adds meaningfully more computation
        (one TIN build per threshold instead of one binning pass).
        ``False`` (default) preserves prior behaviour.
    pitfree_thresholds:
        Increasing height cutoffs (metres) for ``pitfree``'s threshold
        stack (default ``(0, 2, 5, 10, 15, 20)``). Not adaptive to point
        density — tune for a specific survey if the defaults don't fit.
    pitfree_max_distance:
        CHM only, ``pitfree=True`` only. Delaunay triangulation interpolates
        across the full interior of a point set's convex hull, including
        long triangles spanning genuine gaps between distant, sparse
        vegetation clusters - confirmed on real data giving ~100% cell
        coverage where naive per-cell binning only covered ~20-30%. Set this
        (metres) to mask out any cell whose nearest same-layer point is
        farther away, mirroring LAStools' own ``-spike_free`` ``buffer``
        parameter. ``None`` (default) disables masking.
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
        max_ground_distance=max_ground_distance,
        veg_classes=veg_classes,
        pitfree=pitfree,
        pitfree_thresholds=pitfree_thresholds,
        pitfree_max_distance=pitfree_max_distance,
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
    max_ground_distance: Optional[float] = None,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    pitfree: bool = False,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_distance: Optional[float] = None,
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
    max_ground_distance:
        CHM only - see :func:`compute_chm`.
    veg_classes:
        CHM only - see :func:`compute_chm`.
    pitfree / pitfree_thresholds / pitfree_max_distance:
        CHM only - see :func:`compute_chm`.
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
        max_ground_distance=max_ground_distance,
        veg_classes=veg_classes,
        pitfree=pitfree,
        pitfree_thresholds=pitfree_thresholds,
        pitfree_max_distance=pitfree_max_distance,
        dtm_method=dtm_method,
    )
