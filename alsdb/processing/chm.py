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
import warnings
from typing import TYPE_CHECKING, Optional, Union

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

# Module-level, added once at import time - not a per-call
# warnings.catch_warnings() context manager. That would look equivalent but
# isn't safe here: catch_warnings() mutates the *global* filter list on
# __enter__/__exit__, and _pitfree_rasterise runs concurrently across
# run_tiled's worker threads (n_workers up to 50 in production) - one
# thread's __exit__ can restore the filter list while another thread's
# nanmax call is still relying on the "ignore" filter being active,
# letting the warning leak through. Confirmed empirically: 1 leak in 2000
# concurrent calls across 50 threads with the context-manager version.
# A one-time, never-reverted global filter has no such race.
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

_GROUND_CLASS = 2
_PITFREE_THRESHOLDS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)
_NOISE_CLASSES = (7, 18)  # ASPRS LAS: 7 = low noise, 18 = high noise


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


def _exclude_classes_stages(classes: tuple[int, ...]) -> list[dict]:
    """
    Build one ``filters.range`` stage per classification code to exclude.

    A single stage's comma-separated conditions are OR'd together (that's
    what :func:`_classification_ranges` relies on for inclusion), which
    would be wrong here: ``"Classification![7:7],Classification![18:18]"``
    keeps a point if it satisfies *either* negation, i.e. everything except
    points that are simultaneously class 7 *and* class 18 - not what
    "exclude noise" means. Chaining one negated stage per code instead
    ANDs them, which is what exclusion actually requires.
    """
    return [{"type": "filters.range", "limits": f"Classification![{c}:{c}]"} for c in classes]


def _outlier_removal_stages(mean_k: int, multiplier: float) -> list[dict]:
    """
    Statistical outlier detection + exclusion, as a chainable stage list.

    ``filters.outlier`` (``method="statistical"``) reclassifies points whose
    mean distance to their *mean_k* nearest neighbours exceeds *multiplier*
    standard deviations above the point cloud's global mean as class 7 - it
    does not drop them itself (PDAL default: ``class=7``). The trailing
    negation actually removes them; without it they'd still land in the
    veg-class range filter's *complement*, i.e. still be excluded, but only
    because 7 already isn't in ``veg_classes`` - making the exclusion
    explicit here rather than relying on that coincidence.
    """
    return [
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": mean_k,
            "multiplier": multiplier,
        },
        *_exclude_classes_stages((7,)),
    ]


def _validate_grid_alignment(tile_size: float, resolution: float) -> None:
    """
    Require *tile_size* to be a whole multiple of *resolution*.

    ``_rasterise``/``_delaunay_raster`` derive each tile's cell count via
    ``ceil((crop_extent) / resolution)`` then ``linspace`` the *actual*
    extent into that many cells - if ``tile_size`` isn't an exact multiple
    of *resolution*, each tile's true cell width silently drifts away from
    *resolution*, and adjacent tiles' grids no longer align at the shared
    edge. Harmless at whole-multiple combinations (the defaults); silently
    wrong otherwise, so this fails loud rather than producing a warped
    mosaic.
    """
    n = tile_size / resolution
    if not np.isclose(n, round(n), atol=1e-9):
        raise ValueError(
            f"tile_size ({tile_size}) must be a whole multiple of resolution "
            f"({resolution}); got tile_size/resolution = {n}. Otherwise each "
            f"tile's actual cell width drifts away from resolution and "
            f"adjacent tiles won't align at the mosaic seam."
        )


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


def _thin_highest_per_subcell(
    points: np.ndarray,
    subcell_resolution: float,
    value_field: str = "HeightAboveGround",
) -> np.ndarray:
    """
    Keep only the point with the highest *value_field* per subcell of a fine
    grid, dropping every other point in that subcell.

    Approximates spike-free's "use all returns, but effectively only the
    canopy-top ones survive" behaviour cheaply: a faithful spike-free port
    needs an incremental per-point insert-with-freeze loop, which has two
    fatal problems here — no PDAL/scipy primitive supports incremental
    Delaunay insertion with per-point rejection (``scipy.spatial.Delaunay``
    is batch Qhull with no such hook), and a pure-Python loop over real
    survey point counts (hundreds of millions to billions) is a
    non-starter regardless of how clean the geometry code is. Thinning
    first via a vectorised binned-argmax — no Python-level per-point loop —
    then triangulating only the survivors sidesteps both problems.

    Binning is by ``floor(X / subcell_resolution)``, ``floor(Y /
    subcell_resolution)`` against the absolute coordinate grid, not
    *points*' own local extent — so subcells stay aligned across tile
    boundaries instead of drifting per-tile.

    Returns a subset of *points* (all fields preserved, not just
    coordinates) - the survivors feed straight into :func:`_delaunay_raster`.
    """
    if len(points) == 0:
        return points

    col = np.floor(points["X"] / subcell_resolution).astype(np.int64)
    row = np.floor(points["Y"] / subcell_resolution).astype(np.int64)
    # A packed 1-D key (row * col_range + col) identifies each (row, col)
    # subcell just as uniquely as np.unique(stack([row, col]), axis=0) did,
    # without going through numpy's much slower void-type row comparison:
    # consecutive rows' key ranges are contiguous, never overlapping, since
    # the multiplier (col_range) exactly equals the width of one row's own
    # range - collision-free regardless of row/col's absolute magnitude.
    cell_id = row * (col.max() - col.min() + 1) + col

    # Sort by (cell_id, -value) so the first row of each cell_id run is the
    # max-value_field point in that subcell.
    order = np.lexsort((-points[value_field], cell_id))
    sorted_cell_id = cell_id[order]
    first_in_group = np.empty(len(order), dtype=bool)
    first_in_group[0] = True
    if len(order) > 1:
        first_in_group[1:] = sorted_cell_id[1:] != sorted_cell_id[:-1]
    keep_idx = np.sort(order[first_in_group])
    return points[keep_idx]


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

    Cleanup goes through :class:`rasterio.io.MemoryFile` rather than a raw
    ``/vsimem/`` path string + ``rasterio.shutil.delete()``: the latter has
    to open the file to determine its driver before it can delete it, which
    raises ("Invalid dataset") and is silently swallowed whenever the
    pipeline produced no valid raster (degenerate/collinear triangulation,
    or too few surviving points at a given height threshold — routine in
    real data, especially at pitfree's higher threshold layers). Confirmed
    by direct reproduction: that path leaked 100% of the time on degenerate
    input, permanently, since a ``/vsimem/`` file is never touched by
    Python's own garbage collector. ``MemoryFile.close()`` unlinks
    unconditionally, with no driver validation, so it cleans up correctly
    whether or not the pipeline ever wrote anything valid.

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

    from rasterio.io import MemoryFile

    with MemoryFile(filename="_delaunay_raster.tif") as memfile:
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
            {"type": "writers.raster", "gdaldriver": "GTiff", "filename": memfile.name},
        ]
        try:
            _run(stages, arr)
            with memfile.open() as ds:
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

    Selects Class 2 once via a plain numpy mask rather than running a
    second PDAL pipeline just for the same filter: ``filters.range`` is a
    streaming, order-preserving filter, so a PDAL-filtered result and
    ``arr[arr["Classification"] == _GROUND_CLASS]`` are the identical subset
    in the identical order - the PDAL round-trip was pure overhead.
    """
    gnd = arr[arr["Classification"] == _GROUND_CLASS]
    grid = _delaunay_raster(gnd, crop_bbox, resolution, value_field="Z")

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


def _adaptive_max_distance(
    points: np.ndarray,
    percentile: float = 95.0,
    multiplier: float = 2.0,
) -> float:
    """
    Derive a masking distance for :func:`_mask_by_point_distance` from
    *points*' own nearest-neighbour spacing, instead of requiring a fixed
    metres value tuned (or guessed) for one specific survey's point density.

    ``pitfree_max_distance``/``spikefree_max_distance`` previously required a
    single constant - confirmed (this project's own Barcelona benchmark) to
    vary a lot even within one region depending on point density (median
    nearest-neighbour spacing measured at 0.7-1.05 m across three sample
    windows, P95 1.4-1.86 m) - let alone across the very different surveys
    the all-Spain phase will cover. Khosravipour et al.'s own "freeze"
    distance isn't a fixed constant either - it's derived from the point
    cloud's edge-length distribution (99th percentile, per the LAStools
    reference script this project benchmarks against). This is the same
    idea applied to nearest-neighbour spacing rather than triangle edge
    length (the quantity :func:`_mask_by_point_distance` actually works
    with).

    Returns ``multiplier * percentile(self nearest-neighbour distance)``.
    Requires at least 2 points; callers (:func:`_pitfree_rasterise`,
    :func:`_spikefree_rasterise`) already guard smaller point sets before
    reaching this (pitfree's per-threshold-layer ``len(layer_pts) < 3``
    check in particular is what makes computing this *per layer* - not once
    per tile - meaningful, since point density drops sharply as the
    threshold rises).
    """
    from scipy.spatial import cKDTree

    xy = np.column_stack([points["X"].astype(np.float64), points["Y"].astype(np.float64)])
    dist, _ = cKDTree(xy).query(xy, k=2)
    nn_dist = dist[:, 1]  # column 0 is each point matched to itself (distance 0)
    return float(multiplier * np.percentile(nn_dist, percentile))


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
    max_distance: Optional[Union[float, str]] = None,
    max_distance_percentile: float = 95.0,
    max_distance_multiplier: float = 2.0,
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
        vegetation gaps (see above). Applied to every threshold layer
        including the base layer (``thresholds[0]``, all vegetation points
        regardless of height) — deliberately, unlike the ground/DTM layer in
        the original Khosravipour formulation this is adapted from: here the
        base layer's own convex hull already spans the *entire* vegetated
        extent, so exempting it from masking would let it alone reproduce
        the exact over-extrapolation this parameter exists to prevent
        (confirmed empirically: doing so silently restores ~100% coverage
        across genuine gaps, the specific failure mode this masking fixed).
        Pass the string ``"auto"`` instead of a metres value to derive the
        distance from each threshold layer's *own* nearest-neighbour point
        spacing (see :func:`_adaptive_max_distance`) rather than one fixed
        constant for every layer — computed per layer specifically because
        point density drops sharply as the threshold rises, the exact gap
        a single fixed value can't account for.
    max_distance_percentile, max_distance_multiplier:
        Only used when ``max_distance == "auto"`` — see
        :func:`_adaptive_max_distance`.

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
            eff_distance = (
                _adaptive_max_distance(layer_pts, max_distance_percentile, max_distance_multiplier)
                if max_distance == "auto"
                else max_distance
            )
            layer_grid = _mask_by_point_distance(
                layer_grid, layer_pts, crop_bbox, resolution, eff_distance
            )
        stacked[i] = layer_grid

    # np.nanmax over an all-NaN column already returns NaN on its own (with
    # a "All-NaN slice encountered" RuntimeWarning, suppressed at module
    # level - see the filterwarnings call near the top of this file) -
    # equivalent to the previous mask-and-scatter, without the two
    # fancy-indexing copies that approach needed.
    grid = np.nanmax(stacked, axis=0).astype(np.float32)
    return grid


# ---------------------------------------------------------------------------
# Spike-free canopy-top rasterisation
# ---------------------------------------------------------------------------


def _spikefree_rasterise(
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    subcell_resolution: float,
    max_distance: Union[float, str],
    max_distance_percentile: float = 95.0,
    max_distance_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Spike-free-*approximating* CHM rasterisation (cheap proxy for
    Khosravipour et al., 2016), adapted to per-point ``HeightAboveGround``.

    A faithful port needs a genuinely incremental Delaunay triangulation
    with per-point insert-with-freeze rejection - infeasible in this stack
    (see :func:`_thin_highest_per_subcell`'s docstring) at real survey point
    counts. A 2D per-triangle edge-length check on the *final* (batch)
    triangulation was considered and rejected: confirmed empirically that it
    cannot detect an isolated high point embedded in otherwise dense
    coverage, because 2D edge length only reflects X/Y spatial proximity,
    not the Z outlier a spike actually is - the point's neighbouring edges
    look completely normal once the full point set is triangulated. Real
    spike-free's freeze mechanism only works because it's evaluated
    incrementally, tallest point first, against a *partial* triangulation
    that doesn't yet include the point's real (shorter) neighbours; nothing
    static and batch-computed can reproduce that.

    So this only takes the one piece of "path 1" that *is* real and doesn't
    depend on triangulation order: thin to the highest point per fine
    subcell using all returns (not just first, so 2nd/3rd returns that hit
    the true canopy top are available - directly relevant to the
    tall/sparse-canopy underestimation this mode is meant to address), then
    reuse the existing, already-validated :func:`_delaunay_raster` +
    :func:`_mask_by_point_distance` for triangulation and gap protection.
    Protection against an isolated Z-outlier surviving thinning is left to
    ``remove_outliers`` (a 3D k-nearest-neighbour statistical check, which
    - unlike edge length - does incorporate height and so can actually
    catch this case) applied upstream, before this function ever sees the
    points.

    Parameters
    ----------
    points:
        HAG-normalised, classification-filtered vegetation points from
        *all* returns (not just first - the whole point of this mode over
        pitfree/naive is the 2nd/3rd returns a first-returns-only query
        would drop). Not pre-cropped to *crop_bbox*, same reasoning as
        :func:`_pitfree_rasterise`.
    subcell_resolution:
        Fine-grid cell size (metres) for the highest-point thinning step.
        Not adaptive to point density - tune for a specific survey, same
        caveat as ``pitfree_thresholds``.
    max_distance:
        Maximum distance (metres) from a cell to its nearest (thinned)
        point before the cell is masked out by
        :func:`_mask_by_point_distance` instead of trusted. Pass the string
        ``"auto"`` instead of a metres value to derive the distance from the
        thinned points' own nearest-neighbour spacing (see
        :func:`_adaptive_max_distance`) rather than one fixed constant.
    max_distance_percentile, max_distance_multiplier:
        Only used when ``max_distance == "auto"`` — see
        :func:`_adaptive_max_distance`.

    Returns
    -------
    A ``(ny, nx)`` float32 north-up array.
    """
    thinned = _thin_highest_per_subcell(points, subcell_resolution, value_field="HeightAboveGround")
    grid = _delaunay_raster(thinned, crop_bbox, resolution, value_field="HeightAboveGround")
    eff_distance = (
        _adaptive_max_distance(thinned, max_distance_percentile, max_distance_multiplier)
        if max_distance == "auto"
        else max_distance
    )
    return _mask_by_point_distance(grid, thinned, crop_bbox, resolution, eff_distance)


# ---------------------------------------------------------------------------
# CHM post-processing
# ---------------------------------------------------------------------------


def _cap_height(grid: np.ndarray, max_height: Optional[float]) -> np.ndarray:
    """
    Null out (NaN) CHM cells exceeding *max_height* instead of leaving an
    implausible value in place.

    Both the naive and pit-free rasterisers take a per-cell *max* over
    whatever vegetation points land there - a single mislabeled point (e.g.
    a genuine noise return classified as vegetation) becomes the cell value
    outright, with nothing to compete against it. Confirmed on real data: a
    full-region production run turned up ~0.008% of cells above 60 m in a
    region with no canopy anywhere near that tall. NaN (rather than
    clamping to *max_height*) so a later ``pit_fill`` pass can still recover
    a plausible value from real neighbours instead of a false plateau.
    """
    if max_height is None:
        return grid
    out = grid.copy()
    out[out > max_height] = np.nan
    return out


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
    pitfree_max_distance: Optional[Union[float, str]] = None,
    pitfree_max_distance_percentile: float = 95.0,
    pitfree_max_distance_multiplier: float = 2.0,
    max_height: Optional[float] = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 2.0,
    spikefree: bool = False,
    spikefree_subcell_resolution: Optional[float] = None,
    spikefree_max_distance: Optional[Union[float, str]] = None,
    spikefree_max_distance_percentile: float = 95.0,
    spikefree_max_distance_multiplier: float = 2.0,
) -> None:
    arr = _filter_ground_outliers(query_to_array(provider, query_bbox, year=year))
    if arr.size == 0:
        logger.debug("CHM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox
    # Filter to vegetation classes; optionally restrict to first returns so
    # that only the top-of-canopy surface is modelled (recommended).
    veg_limits = _classification_ranges(veg_classes)
    if first_returns_only and not spikefree:
        # spikefree needs all returns by definition - 2nd/3rd returns that
        # hit the true canopy top are exactly what it relies on to thin
        # from, so first_returns_only is silently ignored under spikefree.
        veg_limits += ",ReturnNumber[1:1]"
    stages = [
        _hag_stage(arr),
        {
            "type": "filters.assign",
            "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
        },
        {"type": "filters.range", "limits": veg_limits},
    ]
    if remove_outliers:
        # A max-statistic (either height_statistic="max" or pitfree's
        # cell-wise nanmax) has nothing to compete against a single isolated
        # point - including one left unclassified by veg_classes' own
        # documented (1, 3, 4, 5) recommendation for tall/dense canopy, where
        # a stray high return is exactly what tends to land in Class 1. A
        # height cap (max_height) only catches gross values above a fixed
        # ceiling; this catches a point statistically isolated from its
        # neighbours regardless of its absolute height.
        stages += _outlier_removal_stages(outlier_mean_k, outlier_multiplier)
    if not (pitfree or spikefree):
        # Naive max-binning needs points pre-cropped to the tile; pit-free's
        # per-threshold Delaunay layers and spikefree's thinned single TIN
        # instead crop via faceraster after triangulating the full buffered
        # extent, so edge triangles have real neighbouring context (same
        # reasoning as _dtm_tin).
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
            max_distance_percentile=pitfree_max_distance_percentile,
            max_distance_multiplier=pitfree_max_distance_multiplier,
        )
    elif spikefree:
        grid = _spikefree_rasterise(
            points,
            crop_bbox,
            resolution,
            subcell_resolution=(spikefree_subcell_resolution or resolution / 3.0),
            max_distance=spikefree_max_distance,
            max_distance_percentile=spikefree_max_distance_percentile,
            max_distance_multiplier=spikefree_max_distance_multiplier,
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
    grid = _cap_height(grid, max_height)
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
    exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
) -> None:
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        logger.debug("DSM tile %d: no points, skipping", tile_index)
        return

    cx0, cy0, cx1, cy1 = crop_bbox
    stages: list = list(_exclude_classes_stages(exclude_classes))
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
    pitfree_max_distance: Optional[Union[float, str]] = None,
    pitfree_max_distance_percentile: float = 95.0,
    pitfree_max_distance_multiplier: float = 2.0,
    max_height: Optional[float] = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 2.0,
    spikefree: bool = False,
    spikefree_subcell_resolution: Optional[float] = None,
    spikefree_max_distance: Optional[Union[float, str]] = None,
    spikefree_max_distance_percentile: float = 95.0,
    spikefree_max_distance_multiplier: float = 2.0,
    dsm_exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
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
        stages = list(_exclude_classes_stages(dsm_exclude_classes))
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
        if first_returns_only and not spikefree:
            # spikefree needs all returns by definition - see _process_tile_chm.
            veg_limits += ",ReturnNumber[1:1]"
        stages = [
            _hag_stage(arr),
            {
                "type": "filters.assign",
                "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
            },
            {"type": "filters.range", "limits": veg_limits},
        ]
        if remove_outliers:
            stages += _outlier_removal_stages(outlier_mean_k, outlier_multiplier)
        if not (pitfree or spikefree):
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
                        max_distance_percentile=pitfree_max_distance_percentile,
                        max_distance_multiplier=pitfree_max_distance_multiplier,
                    )
                elif spikefree:
                    chm_grid = _spikefree_rasterise(
                        pts,
                        crop_bbox,
                        resolution,
                        subcell_resolution=(spikefree_subcell_resolution or resolution / 3.0),
                        max_distance=spikefree_max_distance,
                        max_distance_percentile=spikefree_max_distance_percentile,
                        max_distance_multiplier=spikefree_max_distance_multiplier,
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
                chm_grid = _cap_height(chm_grid, max_height)
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
    pitfree_max_distance: Optional[Union[float, str]] = None,
    pitfree_max_distance_percentile: float = 95.0,
    pitfree_max_distance_multiplier: float = 2.0,
    max_height: Optional[float] = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 2.0,
    spikefree: bool = False,
    spikefree_subcell_resolution: Optional[float] = None,
    spikefree_max_distance: Optional[Union[float, str]] = None,
    spikefree_max_distance_percentile: float = 95.0,
    spikefree_max_distance_multiplier: float = 2.0,
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
        parameter. ``None`` (default) disables masking. Pass ``"auto"``
        instead of a metres value to derive the distance from each
        threshold layer's own nearest-neighbour point spacing rather than
        one fixed constant for every layer (density drops sharply as the
        threshold rises, so a single fixed value can't fit every layer
        equally) - see :func:`_adaptive_max_distance`. Neither a fixed
        value nor ``"auto"`` was empirically validated as "correct" for a
        given survey as of this writing - both alsdb's original 3 m guess
        and the LAStools reference's own un-tuned 0.5 m default turned out
        to be arbitrary relative to PNOA's actual point spacing (measured:
        median nearest-neighbour distance 0.7-1.05 m, P95 1.4-1.86 m,
        varying by location) - ``"auto"`` at least adapts to that variation
        automatically rather than requiring a fixed guess to be re-tuned
        per survey.
    pitfree_max_distance_percentile, pitfree_max_distance_multiplier:
        Only used when ``pitfree_max_distance == "auto"``. Default (95th
        percentile, 2x) is a reasonable starting point, not an empirically
        validated one - see :func:`_adaptive_max_distance`.
    max_height:
        If set, any CHM cell exceeding this height (metres) is left ``NaN``
        instead of kept - both ``height_statistic="max"`` and the pit-free
        path take a per-cell maximum, so a single mislabeled vegetation
        point (a genuine noise return, a bird, a mast) becomes the cell
        value outright. Confirmed on real production data: ~0.008% of
        cells above 60 m in a region with no canopy anywhere near that
        tall. ``None`` (default) disables the cap.
    remove_outliers:
        If ``True``, run PDAL's statistical ``filters.outlier`` over the
        classification- and HAG-filtered vegetation points before
        rasterising, dropping any point whose mean distance to its
        ``outlier_mean_k`` nearest neighbours exceeds ``outlier_multiplier``
        standard deviations above the point cloud's global mean. Unlike
        ``max_height``, this catches a point statistically isolated from its
        neighbours regardless of its absolute height - the failure mode
        ``veg_classes``' own documented ``(1, 3, 4, 5)`` recommendation for
        tall/dense canopy opens up (a stray return left ``Class 1`` because
        automatic classification is least reliable there). ``False``
        (default) preserves prior behaviour.
    outlier_mean_k / outlier_multiplier:
        ``remove_outliers=True`` only. PDAL defaults (8, 2.0). Not adaptive
        to point density - tune for a specific survey if needed, same
        caveat as ``pitfree_thresholds``.
    spikefree:
        If ``True``, replace naive max-binning with a cheap approximation of
        spike-free CHM rasterisation (Khosravipour et al., 2016) instead of
        ``pitfree`` (mutually exclusive with it - both are alternative CHM
        strategies for different problems, not stackable; passing both
        raises ``ValueError``). A faithful port needs a genuinely
        incremental Delaunay triangulation with per-point insert-with-freeze
        rejection - infeasible here (no PDAL/scipy primitive supports it,
        and a pure-Python per-point loop is a non-starter at real survey
        point counts). This instead: thins to the highest point per fine
        subcell (``spikefree_subcell_resolution``) using *all* returns, not
        just first (``first_returns_only`` is silently ignored under
        ``spikefree`` - the 2nd/3rd returns are exactly what it needs),
        directly targeting the tall/sparse-canopy underestimation found
        benchmarking against LAStools; then triangulates once and applies
        the same gap-protection masking ``pitfree_max_distance`` uses
        (``spikefree_max_distance``). A per-triangle edge-length check on
        the final mesh was also tried and rejected: confirmed empirically
        it cannot detect an isolated high point embedded in otherwise dense
        coverage (2D edge length is blind to a pure-Z outlier) - real
        spike-free's freeze mechanism only works because it's evaluated
        incrementally against a partial, not-yet-fully-populated
        triangulation, which nothing static and batch-computed can
        reproduce. Protection against that specific case is
        ``remove_outliers``'s job instead (a 3D k-nearest-neighbour check,
        which does incorporate height). ``False`` (default) preserves prior
        behaviour.
    spikefree_subcell_resolution:
        ``spikefree=True`` only. Fine-grid cell size (metres) for the
        highest-point thinning step. ``None`` (default) uses
        ``resolution / 3``. Not adaptive to point density - tune for a
        specific survey, same caveat as ``pitfree_thresholds``.
    spikefree_max_distance:
        ``spikefree=True`` only, required (raises ``ValueError`` if
        ``None``) - same reasoning as ``pitfree_max_distance``: without it,
        triangulation silently extrapolates across arbitrarily large gaps
        with no signal anything is wrong. Also accepts ``"auto"``, same
        meaning as ``pitfree_max_distance``'s (computed once from the
        thinned points, since spikefree has only one layer, not six).
    spikefree_max_distance_percentile, spikefree_max_distance_multiplier:
        Only used when ``spikefree_max_distance == "auto"`` - see
        :func:`_adaptive_max_distance`.
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
    _validate_grid_alignment(tile_size, resolution)
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
    if pitfree and pitfree_max_distance is None:
        raise ValueError(
            "pitfree=True requires an explicit pitfree_max_distance. Without it, "
            "triangulation extrapolates across arbitrarily large gaps between "
            "vegetation clusters (confirmed on real data: ~100% cell coverage vs. "
            "~20-30% for naive per-cell binning) with no signal anything is wrong. "
            "Pick a distance appropriate to the survey's point density."
        )
    if pitfree and spikefree:
        raise ValueError(
            "pitfree and spikefree are alternative, mutually exclusive CHM "
            "rasterisation strategies - pass only one."
        )
    if spikefree and spikefree_max_distance is None:
        raise ValueError(
            "spikefree=True requires an explicit spikefree_max_distance. Without it, "
            "triangulation extrapolates across arbitrarily large gaps between "
            "vegetation clusters with no signal anything is wrong. Pick a distance "
            "appropriate to the survey's point density."
        )
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
        pitfree_max_distance_percentile=pitfree_max_distance_percentile,
        pitfree_max_distance_multiplier=pitfree_max_distance_multiplier,
        max_height=max_height,
        remove_outliers=remove_outliers,
        outlier_mean_k=outlier_mean_k,
        outlier_multiplier=outlier_multiplier,
        spikefree=spikefree,
        spikefree_subcell_resolution=spikefree_subcell_resolution,
        spikefree_max_distance=spikefree_max_distance,
        spikefree_max_distance_percentile=spikefree_max_distance_percentile,
        spikefree_max_distance_multiplier=spikefree_max_distance_multiplier,
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
    _validate_grid_alignment(tile_size, resolution)
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
    exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
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
    exclude_classes:
        LAS classification codes dropped before taking the per-cell max
        (default ``(7, 18)`` — ASPRS low/high noise). A DSM intentionally
        includes buildings and every other real surface, unlike CHM's
        vegetation-only filter — but noise returns aren't a real surface,
        and a max-statistic has nothing to compete against a single stray
        high-noise point. Pass ``()`` to disable. Called ``exclude_classes``
        here (unprefixed) since this function is already DSM-only;
        :func:`compute_all`'s equivalent is ``dsm_exclude_classes``, prefixed
        only because that function multiplexes DTM/DSM/CHM parameters in one
        namespace (same reason it has ``dtm_method`` but not ``chm_pitfree``).
    overwrite:
        If ``False`` (default) and DSM data for *year* already exists in
        the store, the computation is skipped.
    tile_size:
        Sub-tile width/height in metres (default 500 m).
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    _require_year(year)
    _validate_grid_alignment(tile_size, resolution)
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
        exclude_classes=exclude_classes,
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
    pitfree_max_distance: Optional[Union[float, str]] = None,
    pitfree_max_distance_percentile: float = 95.0,
    pitfree_max_distance_multiplier: float = 2.0,
    max_height: Optional[float] = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 2.0,
    spikefree: bool = False,
    spikefree_subcell_resolution: Optional[float] = None,
    spikefree_max_distance: Optional[Union[float, str]] = None,
    spikefree_max_distance_percentile: float = 95.0,
    spikefree_max_distance_multiplier: float = 2.0,
    dsm_exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
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
    pitfree / pitfree_thresholds / pitfree_max_distance /
    pitfree_max_distance_percentile / pitfree_max_distance_multiplier /
    max_height / remove_outliers / outlier_mean_k / outlier_multiplier /
    spikefree / spikefree_subcell_resolution / spikefree_max_distance /
    spikefree_max_distance_percentile / spikefree_max_distance_multiplier:
        CHM only - see :func:`compute_chm`.
    dsm_exclude_classes:
        DSM only - see :func:`compute_dsm`. Prefixed with ``dsm_`` (unlike
        this function's CHM-only parameters above, which stay unprefixed for
        symmetry with :func:`compute_chm`'s own signature) because this is
        the one place DTM/DSM/CHM parameters share a single namespace and
        ``exclude_classes`` alone would be ambiguous - same reason
        ``dtm_method`` is prefixed too.
    overwrite:
        If ``False`` (default), skip products already present for *year*.
        If ``True``, recompute everything regardless.
    """
    _require_year(year)
    _validate_grid_alignment(tile_size, resolution)
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return

    # Determine which products still need computing
    need_dtm = overwrite or year is None or not store.has_data("dtm", resolution, year)
    need_dsm = overwrite or year is None or not store.has_data("dsm", resolution, year)
    need_chm = overwrite or year is None or not store.has_data("chm", resolution, year)

    if need_chm and pitfree and pitfree_max_distance is None:
        raise ValueError(
            "pitfree=True requires an explicit pitfree_max_distance. Without it, "
            "triangulation extrapolates across arbitrarily large gaps between "
            "vegetation clusters (confirmed on real data: ~100% cell coverage vs. "
            "~20-30% for naive per-cell binning) with no signal anything is wrong. "
            "Pick a distance appropriate to the survey's point density."
        )
    if need_chm and pitfree and spikefree:
        raise ValueError(
            "pitfree and spikefree are alternative, mutually exclusive CHM "
            "rasterisation strategies - pass only one."
        )
    if need_chm and spikefree and spikefree_max_distance is None:
        raise ValueError(
            "spikefree=True requires an explicit spikefree_max_distance. Without it, "
            "triangulation extrapolates across arbitrarily large gaps between "
            "vegetation clusters with no signal anything is wrong. Pick a distance "
            "appropriate to the survey's point density."
        )

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
        pitfree_max_distance_percentile=pitfree_max_distance_percentile,
        pitfree_max_distance_multiplier=pitfree_max_distance_multiplier,
        max_height=max_height,
        remove_outliers=remove_outliers,
        outlier_mean_k=outlier_mean_k,
        outlier_multiplier=outlier_multiplier,
        spikefree=spikefree,
        spikefree_subcell_resolution=spikefree_subcell_resolution,
        spikefree_max_distance=spikefree_max_distance,
        spikefree_max_distance_percentile=spikefree_max_distance_percentile,
        spikefree_max_distance_multiplier=spikefree_max_distance_multiplier,
        dsm_exclude_classes=dsm_exclude_classes,
        dtm_method=dtm_method,
    )
