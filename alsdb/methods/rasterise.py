# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
"""Rasterisation of point clouds: binning, TINs, pit-free CHM, gap filling

References
----------
Khosravipour, A., Skidmore, A. K., Isenburg, M., Wang, T., & Hussin, Y. A.
    (2014). Generating pit-free canopy height models from airborne lidar.
    Photogrammetric Engineering & Remote Sensing, 80(9), 863-872.
    https://doi.org/10.14358/PERS.80.9.863
Roussel, J.-R., et al. (2020). lidR: An R package for analysis of Airborne
    Laser Scanning (ALS) data. Remote Sensing of Environment, 251, 112061.
    https://doi.org/10.1016/j.rse.2020.112061
"""

from __future__ import annotations

import json

import numpy as np
import pdal

from alsdb.methods._grid import flip_to_north_up

_GROUND_CLASS = 2

_PITFREE_THRESHOLDS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)

_PITFREE_EDGE_SPACINGS = 3.0  # auto upper-layer edge limit, in pulse spacings

_NOISE_CLASSES = (7, 18)  # ASPRS LAS: 7 = low noise, 18 = high noise


def _classification_ranges(classes: tuple[int, ...]) -> str:
    """Build a PDAL ``filters.range`` string for a set of classification codes

    Codes are grouped into contiguous runs, e.g. ``(1, 3, 4, 5)`` ->
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
    """Build one ``filters.range`` stage per class to exclude

    Conditions within one stage are OR'd, so exclusion needs one stage per code.
    """
    return [{"type": "filters.range", "limits": f"Classification![{c}:{c}]"} for c in classes]


def _outlier_removal_stages(mean_k: int, multiplier: float) -> list[dict]:
    """Return PDAL stages that flag statistical outliers as class 7 and drop them."""
    return [
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": mean_k,
            "multiplier": multiplier,
        },
        *_exclude_classes_stages((7,)),
    ]


def _rasterise(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    statistic: str = "max",
) -> np.ndarray:
    """Bin values into a north-up grid over *crop_bbox*

    *statistic* is a scipy statistic name or a percentile such as ``"p95"``.
    Empty cells are NaN.
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
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Keep the highest point per subcell of a grid anchored at *origin*."""
    if len(points) == 0:
        return points

    col = np.floor((points["X"] - origin[0]) / subcell_resolution).astype(np.int64)
    row = np.floor((points["Y"] - origin[1]) / subcell_resolution).astype(np.int64)
    # Packed (row, col) key, unique per subcell.
    cell_id = row * (col.max() - col.min() + 1) + col

    # Highest point first in each subcell; ties broken by X, Y, not input order.
    order = np.lexsort((points["Y"], points["X"], -points[value_field], cell_id))
    sorted_cell_id = cell_id[order]
    first_in_group = np.empty(len(order), dtype=bool)
    first_in_group[0] = True
    if len(order) > 1:
        first_in_group[1:] = sorted_cell_id[1:] != sorted_cell_id[:-1]
    keep_idx = np.sort(order[first_in_group])
    return points[keep_idx]


def _nn_fill(
    grid: np.ndarray,
    gnd: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    """Fill NaN cells from the nearest ground point (outside the TIN hull)."""
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
    max_edge: float | None = None,
) -> np.ndarray:
    """Rasterise a TIN of *value_field* over *crop_bbox*

    Points outside *crop_bbox* still contribute to edge triangles. Triangles
    with a horizontal edge longer than *max_edge* are dropped. The raster is
    written to GDAL's in-memory file system, because ``filters.faceraster``
    only materialises it through a writer.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` float32 north-up grid; all NaN when fewer than three
        points or a degenerate triangulation.
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
    # Triangulate in a fixed (X, Y, Z) order: on regular scan patterns PDAL's
    # choice among equivalent triangulations depends on input order.
    arr = arr[np.lexsort((arr["Z"], arr["Y"], arr["X"]))]

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
                **({"max_triangle_edge_length": max_edge} if max_edge is not None else {}),
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
    """Ground TIN of class-2 points, filled by nearest neighbour outside the hull."""
    gnd = arr[arr["Classification"] == _GROUND_CLASS]
    grid = _delaunay_raster(gnd, crop_bbox, resolution, value_field="Z")

    if len(gnd) > 0:
        grid = _nn_fill(grid, gnd, crop_bbox, resolution)

    return grid


def _pulse_spacing(points: np.ndarray, cell: float = 5.0) -> float:
    """Mean point spacing, 1 / sqrt(density), over occupied *cell*-sized cells."""
    col = np.floor(points["X"] / cell).astype(np.int64)
    row = np.floor(points["Y"] / cell).astype(np.int64)
    occupied = len(np.unique(row * (col.max() - col.min() + 1) + col))
    return float(np.sqrt(occupied * cell * cell / len(points)))


def _dtm_idw(
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    power: float = 2.0,
    k: int = 8,
    max_distance: float | None = None,
) -> np.ndarray:
    """Inverse-distance-weighted DTM from ground points

    Parameters
    ----------
    max_distance:
        Cells farther than this (m) from any ground point stay NaN; ``None``
        fills every cell.

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
    """Drop points farther than *max_distance* from the nearest ground point

    Parameters
    ----------
    points:
        Vegetation points with ``HeightAboveGround``.
    ground:
        Class-2 ground points of the same buffered query.
    max_distance:
        Maximum distance (m); ``None`` disables the gate.
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


def _pitfree_rasterise(
    points: np.ndarray,
    crop_bbox: tuple[float, float, float, float],
    resolution: float,
    thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    max_edge: float | str = "auto",
    base_max_edge: float | None = None,
) -> np.ndarray:
    """Pit-free CHM (Khosravipour et al. 2014), following lidR's ``pitfree()``

    Points (ground and vegetation, with buffer) are thinned to the highest
    return per cell; each height threshold layer is triangulated and the
    per-cell maximum is kept. Upper layers drop triangles with an edge longer
    than *max_edge* (``"auto"``: three pulse spacings of the thinned points);
    *base_max_edge* trims the base layer.

    Returns a ``(ny, nx)`` float32 north-up array, NaN where no layer covers a cell.
    """
    cx0, cy0, cx1, cy1 = crop_bbox
    nx = max(1, int(np.ceil((cx1 - cx0) / resolution)))
    ny = max(1, int(np.ceil((cy1 - cy0) / resolution)))
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    points = _thin_highest_per_subcell(points, resolution, origin=(cx0, cy0))
    if len(points) < 3:
        return grid
    if max_edge == "auto":
        # Thinned returns are at most one per cell; 5x5-cell spacing bins (5 m
        # at 1 m resolution, as lidR) always hold several of them.
        max_edge = _PITFREE_EDGE_SPACINGS * _pulse_spacing(points, cell=5 * resolution)
    for i, t in enumerate(sorted(thresholds)):
        layer_pts = points[points["HeightAboveGround"] >= t]
        if len(layer_pts) < 3:
            break  # higher layers are subsets
        layer = _delaunay_raster(
            layer_pts,
            crop_bbox,
            resolution,
            value_field="HeightAboveGround",
            max_edge=base_max_edge if i == 0 else max_edge,
        )
        # fmax ignores missing layers and keeps all-missing pixels NaN without
        # touching process-wide warning filters (safe in threaded workers).
        grid = np.fmax(grid, layer)
    return grid


def _cap_height(grid: np.ndarray, max_height: float | None) -> np.ndarray:
    """Set cells above *max_height* to NaN, so pit filling can recover them from neighbours."""
    if max_height is None:
        return grid
    out = grid.copy()
    out[out > max_height] = np.nan
    return out


def _fill_pits(
    grid: np.ndarray, window: int = 3, *, eligible=None, max_cells: int = 4
) -> np.ndarray:
    """Fill small enclosed holes from neighbouring observations

    Holes touching the raster edge, larger than *max_cells* or containing an
    ineligible cell stay missing.
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


def _run(stages: list, arr: np.ndarray) -> np.ndarray:
    """Execute a PDAL pipeline and return the output point array."""
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    return p.arrays[0] if p.arrays else arr[:0]
