# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""CHM, DTM and DSM processing on a shared grid with support diagnostics.

Public functions query buffered TileDB tiles, use an explicit ground TIN for
normalization, reconstruct surfaces, and write products and provenance to Zarr.
Native spike-free reconstruction requires the optional CGAL build. The historical
highest-subcell approximation remains available for reproducibility.

See ``doc/chm_methods.md`` for algorithms, migration and uncertainty examples.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pdal

from alsdb.processing._tiling import (
    VEG_CLASSES as _VEG_CLASSES,
)
from alsdb.processing._tiling import (
    flip_to_north_up,
)

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)

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
    if not np.isfinite([tile_size, resolution]).all() or min(tile_size, resolution) <= 0:
        raise ValueError("tile_size and resolution must be finite and positive")
    n = tile_size / resolution
    if not np.isclose(n, round(n), rtol=0, atol=1e-9):
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
        stat_fn = lambda arr: np.nanpercentile(arr, q) if len(arr) > 0 else np.nan
    else:
        stat_fn = statistic  # type: ignore[assignment]

    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))
    x_edges = cx0 + np.arange(nx + 1) * resolution
    y_edges = cy1 - np.arange(ny, -1, -1) * resolution

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
    """Keep the highest point per globally aligned subcell; this is a thinning approximation."""
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
    _ny, _nx = grid.shape
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
            if any(
                message in str(exc).lower()
                for message in (
                    "no points",
                    "not enough points",
                    "collinear",
                    "degenerate",
                    "qhull",
                )
            ):
                return empty
            raise


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
    max_distance: float | None = None,
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
    max_distance: float | None,
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
    if max_distance is None or len(points) == 0:
        return points
    if len(ground) == 0:
        return points[:0]

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
    max_distance: float | str | None = None,
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

    # fmax ignores missing layers and preserves all-missing pixels without
    # mutating process-wide warning filters (safe in threaded workers).
    grid = np.fmax.reduce(stacked, axis=0).astype(np.float32)
    return grid


# ---------------------------------------------------------------------------
# Spike-free canopy-top rasterisation
# ---------------------------------------------------------------------------


def _spikefree_rasterise(
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    subcell_resolution: float,
    max_distance: float | str,
    max_distance_percentile: float = 95.0,
    max_distance_multiplier: float = 2.0,
) -> np.ndarray:
    """Legacy highest-subcell TIN approximation. For genuine freezing use spikefree.rasterize_spikefree."""
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


def _cap_height(grid: np.ndarray, max_height: float | None) -> np.ndarray:
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


def _fill_pits(
    grid: np.ndarray, window: int = 3, *, eligible=None, max_cells: int = 4
) -> np.ndarray:
    """Fill small enclosed holes from local observations, never excluded cells.

    Holes touching a raster boundary, exceeding max_cells, or containing an
    ineligible cell remain missing. Values come from the original raster only.
    """
    from scipy.ndimage import binary_dilation, find_objects, label

    if window < 3 or window % 2 != 1 or max_cells < 1:
        raise ValueError("window must be odd and >=3; max_cells must be positive")
    out = np.asarray(grid, dtype=np.float32).copy()
    missing = ~np.isfinite(out)
    allowed = np.ones(out.shape, bool) if eligible is None else np.asarray(eligible, bool)
    if allowed.shape != out.shape:
        raise ValueError("eligible mask must match grid")
    labels, _ = label(missing, structure=np.ones((3, 3)))
    for number, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        rows, cols = bounds
        if (
            rows.start == 0
            or cols.start == 0
            or rows.stop == out.shape[0]
            or cols.stop == out.shape[1]
        ):
            continue
        hole = labels[bounds] == number
        if hole.sum() > max_cells or not allowed[bounds][hole].all():
            continue
        radius = window // 2
        r0, r1 = max(0, rows.start - radius), min(out.shape[0], rows.stop + radius)
        c0, c1 = max(0, cols.start - radius), min(out.shape[1], cols.stop + radius)
        region = (slice(r0, r1), slice(c0, c1))
        local_hole = labels[region] == number
        neighbours = binary_dilation(local_hole, structure=np.ones((window, window)))
        neighbours &= np.isfinite(grid[region]) & allowed[region]
        if neighbours.any():
            out[region][local_hole] = np.median(grid[region][neighbours])
    return out


# ---------------------------------------------------------------------------
# PDAL helpers
# ---------------------------------------------------------------------------


def _run(stages: list, arr: np.ndarray) -> np.ndarray:
    """Execute a PDAL pipeline and return the output point array."""
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    return p.arrays[0] if p.arrays else arr[:0]


def compute_chm(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 1.0,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    *,
    first_returns_only: bool = True,
    height_statistic: str = "max",
    pit_fill: bool = False,
    max_ground_distance: float | None = None,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    pitfree: bool = False,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_distance: float | str | None = None,
    pitfree_max_distance_percentile: float = 95.0,
    pitfree_max_distance_multiplier: float = 2.0,
    max_height: float | None = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 2.0,
    spikefree: bool = False,
    spikefree_subcell_resolution: float | None = None,
    spikefree_max_distance: float | str | None = None,
    spikefree_max_distance_percentile: float = 95.0,
    spikefree_max_distance_multiplier: float = 2.0,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    quality: bool = True,
    ground_outlier_removal: bool = True,
    source_version: str | None = None,
    method: str | None = None,
    freeze_distance: float = 1.5,
    height_buffer: float = 0.5,
    max_triangle_edge: float | None = None,
    lastools_executable: str = "las2dem64",
    lastools_demo: bool = False,
    freeze_interval: float = 0.25,
    ground_extrapolation: bool = False,
    fill_max_cells: int = 4,
) -> None:
    """Compute surface products on one fixed north-up grid and write to Zarr.

    ``method`` (CHM/all): "max", "pitfree", "highest_subcell_tin", or
    "spikefree" (compiled constrained Delaunay; all eligible return numbers).
    Legacy ``spikefree=True`` retains the approximation with a deprecation
    warning. ``pitfree=True`` remains an alias for method="pitfree".

    Terrain is interpolated inside the ground convex hull. Outside it, values
    are missing unless ``ground_extrapolation=True`` explicitly enables nearest
    ground fallback. ``max_ground_distance`` limits accepted terrain support.
    ``pit_fill`` defaults to False; when enabled only bounded local holes are
    filled, with excluded cells preserved. Quality layers are enabled by default.

    The requested upper-left origin is preserved, with the extent extended east
    and south to full pixels. ``tile_size`` must be a multiple of resolution.
    Changed parameters or legacy untracked output require ``overwrite=True``.
    Failed runs restart safely. ``source_version`` should identify the input
    revision when the source array changes in place.

    See doc/chm_methods.md for method parameters, flags, migration and examples.
    """
    from alsdb.processing._products import compute_products

    compute_products(provider, store, ("chm",), locals())


def compute_dtm(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 1.0,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    *,
    dtm_method: str = "tin",
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 10.0,
    n_workers: int = 1,
    quality: bool = True,
    ground_outlier_removal: bool = True,
    source_version: str | None = None,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = False,
) -> None:
    """Compute surface products on one fixed north-up grid and write to Zarr.

    ``method`` (CHM/all): "max", "pitfree", "highest_subcell_tin", or
    "spikefree" (compiled constrained Delaunay; all eligible return numbers).
    Legacy ``spikefree=True`` retains the approximation with a deprecation
    warning. ``pitfree=True`` remains an alias for method="pitfree".

    Terrain is interpolated inside the ground convex hull. Outside it, values
    are missing unless ``ground_extrapolation=True`` explicitly enables nearest
    ground fallback. ``max_ground_distance`` limits accepted terrain support.
    ``pit_fill`` defaults to False; when enabled only bounded local holes are
    filled, with excluded cells preserved. Quality layers are enabled by default.

    The requested upper-left origin is preserved, with the extent extended east
    and south to full pixels. ``tile_size`` must be a multiple of resolution.
    Changed parameters or legacy untracked output require ``overwrite=True``.
    Failed runs restart safely. ``source_version`` should identify the input
    revision when the source array changes in place.

    See doc/chm_methods.md for method parameters, flags, migration and examples.
    """
    from alsdb.processing._products import compute_products

    compute_products(provider, store, ("dtm",), locals())


def compute_dsm(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 1.0,
    first_returns_only: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    *,
    exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
    overwrite: bool = False,
    tile_size: float = 500.0,
    n_workers: int = 1,
    quality: bool = True,
    ground_outlier_removal: bool = True,
    source_version: str | None = None,
) -> None:
    """Compute surface products on one fixed north-up grid and write to Zarr.

    ``method`` (CHM/all): "max", "pitfree", "highest_subcell_tin", or
    "spikefree" (compiled constrained Delaunay; all eligible return numbers).
    Legacy ``spikefree=True`` retains the approximation with a deprecation
    warning. ``pitfree=True`` remains an alias for method="pitfree".

    Terrain is interpolated inside the ground convex hull. Outside it, values
    are missing unless ``ground_extrapolation=True`` explicitly enables nearest
    ground fallback. ``max_ground_distance`` limits accepted terrain support.
    ``pit_fill`` defaults to False; when enabled only bounded local holes are
    filled, with excluded cells preserved. Quality layers are enabled by default.

    The requested upper-left origin is preserved, with the extent extended east
    and south to full pixels. ``tile_size`` must be a multiple of resolution.
    Changed parameters or legacy untracked output require ``overwrite=True``.
    Failed runs restart safely. ``source_version`` should identify the input
    revision when the source array changes in place.

    See doc/chm_methods.md for method parameters, flags, migration and examples.
    """
    from alsdb.processing._products import compute_products

    compute_products(provider, store, ("dsm",), locals())


def compute_all(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 1.0,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    first_returns_only: bool = True,
    height_statistic: str = "max",
    pit_fill: bool = False,
    max_ground_distance: float | None = None,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    pitfree: bool = False,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_distance: float | str | None = None,
    pitfree_max_distance_percentile: float = 95.0,
    pitfree_max_distance_multiplier: float = 2.0,
    max_height: float | None = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 2.0,
    spikefree: bool = False,
    spikefree_subcell_resolution: float | None = None,
    spikefree_max_distance: float | str | None = None,
    spikefree_max_distance_percentile: float = 95.0,
    spikefree_max_distance_multiplier: float = 2.0,
    dsm_exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
    overwrite: bool = False,
    dtm_method: str = "tin",
    quality: bool = True,
    ground_outlier_removal: bool = True,
    source_version: str | None = None,
    method: str | None = None,
    freeze_distance: float = 1.5,
    height_buffer: float = 0.5,
    max_triangle_edge: float | None = None,
    lastools_executable: str = "las2dem64",
    lastools_demo: bool = False,
    freeze_interval: float = 0.25,
    ground_extrapolation: bool = False,
    fill_max_cells: int = 4,
) -> None:
    """Compute surface products on one fixed north-up grid and write to Zarr.

    ``method`` (CHM/all): "max", "pitfree", "highest_subcell_tin", or
    "spikefree" (compiled constrained Delaunay; all eligible return numbers).
    Legacy ``spikefree=True`` retains the approximation with a deprecation
    warning. ``pitfree=True`` remains an alias for method="pitfree".

    Terrain is interpolated inside the ground convex hull. Outside it, values
    are missing unless ``ground_extrapolation=True`` explicitly enables nearest
    ground fallback. ``max_ground_distance`` limits accepted terrain support.
    ``pit_fill`` defaults to False; when enabled only bounded local holes are
    filled, with excluded cells preserved. Quality layers are enabled by default.

    The requested upper-left origin is preserved, with the extent extended east
    and south to full pixels. ``tile_size`` must be a multiple of resolution.
    Changed parameters or legacy untracked output require ``overwrite=True``.
    Failed runs restart safely. ``source_version`` should identify the input
    revision when the source array changes in place.

    See doc/chm_methods.md for method parameters, flags, migration and examples.
    """
    from alsdb.processing._products import compute_products

    compute_products(provider, store, ("dtm", "dsm", "chm"), locals())
