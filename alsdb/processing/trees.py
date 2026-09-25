# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Individual tree segmentation from ALS TileDB point clouds.

Uses PDAL's ``filters.litree`` (Li et al. 2012) to segment points. PDAL
``ClusterID`` output is exposed as ``TreeID`` for API compatibility.  Unassigned points
(ground, noise, or sub-threshold clusters) receive ``TreeID = 0``.

After segmentation, per-tree metrics are summarised into a :class:`pandas.DataFrame`:

* ``tree_id``       — unique integer identifier
* ``centroid_x/y`` — crown centroid (UTM metres)
* ``height``        — maximum height above ground (m)
* ``min_point_height`` — minimum HAG of any point assigned to the tree (m);
  lower-bounded by ``min_height / 2`` due to the pre-filter in :func:`_hag_stages`
* ``crown_area``    — 2-D convex hull area (m²)
* ``crown_radius``  — equivalent circular radius = sqrt(area / π) (m)
* ``n_points``      — number of ALS points in the tree

Tiled processing
----------------
For large areas, pass ``tile_size`` to split the bbox into sub-tiles processed
in parallel.  A ``tile_buffer`` overlap ensures trees at tile boundaries are
fully captured; only trees whose apex falls within the non-buffered
``crop_bbox`` are retained; matching apex coordinates are deduplicated::

    points, trees = segment_trees(
        provider,
        bbox=(655000.0, 8901000.0, 658000.0, 8904000.0),
        year=2014,
        tile_size=300.0,
        tile_buffer=30.0,
        n_workers=4,
        voxel_size=0.5,
    )

Reference
---------
Li, W., Guo, Q., Jakubowski, M. K., & Kelly, M. (2012). A new method for
segmenting individual trees from the lidar point cloud. Photogrammetric
Engineering & Remote Sensing, 78(1), 75–84.
"""

from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd
import pdal
from numpy.lib.recfunctions import rename_fields
from scipy.spatial import ConvexHull, QhullError

from alsdb.processing._surface import clean_points
from alsdb.processing._terrain import normalize_points
from alsdb.processing._tiling import (
    _filter_ground_outliers,
    array_data_bbox,
    bounded_map,
    query_to_array,
    tile_bboxes,
)
from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

_MIN_HULL_POINTS = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hag_stages(arr: np.ndarray, min_height: float, voxel_size: float | None) -> list[dict]:
    """Filter normalized heights and optionally sample before litree."""
    stages: list[dict] = [
        {"type": "filters.range", "limits": f"HeightAboveGround[{min_height / 2}:]"},
    ]
    if voxel_size is not None:
        stages.append({"type": "filters.sample", "radius": voxel_size})
    return stages


def _litree_stage(min_height: float, min_points: int, radius: float) -> dict:
    return {
        "type": "filters.litree",
        "min_points": min_points,
        "min_height": min_height,
        "radius": radius,
    }


def _compute_adaptive_radius(points: np.ndarray) -> float:
    """Return litree search radius derived from P75 HAG: wf(h) = 0.07*h + 0.6."""
    hag = points["HeightAboveGround"].astype(np.float64)
    above = hag[hag > 0]
    p75 = float(np.percentile(above, 75)) if above.size > 0 else 5.0
    return 0.07 * p75 + 0.6


def _tree_metrics(points: np.ndarray, crown_fraction: float = 0.5) -> list[dict]:
    """
    Compute per-tree metrics from a point array that already has TreeID.

    Crown area and radius are estimated from the 2-D convex hull of the
    *upper* crown only — points above ``crown_fraction × tree_height``.
    Restricting to the upper crown avoids including wide understory returns
    that inflate the projected area estimate.

    Parameters
    ----------
    points:
        Structured array with ``TreeID`` and ``HeightAboveGround`` fields.
    crown_fraction:
        Fraction of tree height used as lower bound for crown points
        (default 0.5 = upper half of the tree).
    """
    if "TreeID" not in points.dtype.names:
        return []
    selected = points[points["TreeID"] > 0]
    selected = selected[np.argsort(selected["TreeID"], kind="stable")]
    tree_ids, starts, counts = np.unique(selected["TreeID"], return_index=True, return_counts=True)
    records = []
    for tid, start, count in zip(tree_ids, starts, counts):
        pts = selected[start : start + count]
        x = pts["X"].astype(np.float64)
        y = pts["Y"].astype(np.float64)
        hag = pts["HeightAboveGround"].astype(np.float64)

        tree_height = float(hag.max())
        highest = np.flatnonzero(hag == tree_height)
        apex = highest[np.lexsort((y[highest], x[highest]))[0]]

        crown_area = crown_radius = np.nan
        # Use only points in the upper crown to avoid understory inflation.
        crown_mask = hag >= crown_fraction * tree_height
        x_crown = x[crown_mask]
        y_crown = y[crown_mask]
        if len(x_crown) >= _MIN_HULL_POINTS:
            try:
                hull = ConvexHull(np.column_stack([x_crown, y_crown]))
                crown_area = float(hull.volume)  # scipy: volume = area in 2-D
                crown_radius = float(np.sqrt(crown_area / np.pi))
            except QhullError as exc:
                logger.debug(
                    "Crown hull failed for tree %s (%s), leaving crown_area/crown_radius as NaN",
                    tid,
                    exc,
                )

        records.append(
            {
                "tree_id": int(tid),
                "centroid_x": float(x.mean()),
                "centroid_y": float(y.mean()),
                "height": tree_height,
                "min_point_height": float(hag.min()),
                "crown_area": crown_area,
                "crown_radius": crown_radius,
                "n_points": int(count),
                "apex_x": float(x[apex]),
                "apex_y": float(y[apex]),
            }
        )
    return records


def _process_tile(
    provider: TileDBProvider,
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    tile_index: int,
    year: int | None,
    min_points: int,
    min_height: float,
    radius: float,
    voxel_size: float | None,
    adaptive_radius: bool = False,
    crown_fraction: float = 0.5,
    veg_classes: tuple[int, ...] = (3, 4, 5),
    point_attributes: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """
    Segment trees within one sub-tile.

    Normalize and segment the buffered points, compute complete crown metrics,
    and retain crowns whose deterministic apex belongs to the half-open crop.
    """
    arr = clean_points(
        query_to_array(
            provider,
            query_bbox,
            year=year,
            attributes=None
            if point_attributes is None
            else tuple(
                dict.fromkeys(
                    ("Z", "Classification", "ReturnNumber", "Withheld", *point_attributes)
                )
            ),
        )
    )
    arr = _filter_ground_outliers(arr)
    if arr.size == 0:
        return None

    arr, _ = normalize_points(arr)
    arr = arr[np.isfinite(arr["HeightAboveGround"]) & np.isin(arr["Classification"], veg_classes)]
    if not len(arr):
        return None
    try:
        p = pdal.Pipeline(json.dumps(_hag_stages(arr, min_height, voxel_size)), arrays=[arr])
        p.execute()
        hag_points = p.arrays[0] if p.arrays else arr[:0]
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            return None
        raise

    if hag_points.size == 0:
        return None

    r = _compute_adaptive_radius(hag_points) if adaptive_radius else radius
    try:
        p2 = pdal.Pipeline(
            json.dumps(
                [
                    {"type": "filters.sort", "dimension": "HeightAboveGround", "order": "DESC"},
                    _litree_stage(min_height, min_points, r),
                ]
            ),
            arrays=[hag_points],
        )
        p2.execute()
        points = p2.arrays[0] if p2.arrays else hag_points[:0]
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            return None
        raise

    # PDAL > 2.2 emits ClusterID; preserve alsdb's public TreeID name.
    if "ClusterID" in points.dtype.names:
        points = rename_fields(points, {"ClusterID": "TreeID"})
    if len(points) and "TreeID" not in points.dtype.names:
        raise RuntimeError("PDAL litree returned no ClusterID or TreeID dimension")

    # Decide ownership from the complete buffered crown, then retain its points.
    cx0, cy0, cx1, cy1 = crop_bbox
    records = [
        rec
        for rec in _tree_metrics(points, crown_fraction=crown_fraction)
        if cx0 <= rec["apex_x"] < cx1 and cy0 < rec["apex_y"] <= cy1
    ]
    if not records:
        return None
    owned = [rec["tree_id"] for rec in records]
    return points[np.isin(points["TreeID"], owned)].copy(), pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_trees(
    provider: TileDBProvider,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    min_points: int = 10,
    min_height: float = 3.0,
    radius: float = 100.0,
    adaptive_radius: bool = False,
    voxel_size: float | None = None,
    tile_size: float | None = None,
    tile_buffer: float = 30.0,
    n_workers: int = 1,
    crown_fraction: float = 0.5,
    *,
    veg_classes: tuple[int, ...] = (3, 4, 5),
    point_attributes: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Segment individual trees using ``filters.litree``.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    bbox:
        Spatial filter ``(min_x, min_y, max_x, max_y)``.  Uses the full
        array domain when ``None``.
    year:
        Optional survey year filter.
    min_points:
        Minimum ALS points for a cluster to be labelled as a tree.
    min_height:
        Minimum tree height in metres.
    radius:
        PDAL non-tree seed radius (m), default 100. This controls the competing
        non-tree seed, not a crown search radius. Used when adaptive_radius=False.
    adaptive_radius:
        Deprecated legacy height-based radius heuristic. Overrides radius,
        but has not been calibrated for PDAL non-tree seed insertion.
    voxel_size:
        Poisson disk sampling radius (m) applied before ``filters.litree``
        to speed up the graph build.  ``0.5`` m is a good starting point for
        dense ALS (> 5 pts/m²).  ``None`` disables downsampling.
    tile_size:
        Sub-tile side length (m).  When provided the area is partitioned into
        tiles and processed in parallel (recommended for areas > 0.5 km²).
        ``None`` processes the full bbox in one pass.
    tile_buffer:
        Overlap buffer (m) added to each tile query so trees at edges are
        fully captured.  Default 30 m.
    n_workers:
        Thread pool size.  Effective only when ``tile_size`` is set.
    crown_fraction:
        Fraction of tree height used as the lower bound for crown-area
        computation (default 0.5 = upper half of each tree).  Only points
        at or above ``crown_fraction × height`` contribute to the convex
        hull used for ``crown_area`` and ``crown_radius``.  Reducing this
        value includes more understory returns and will increase the area
        estimate; increasing it focuses on the uppermost crown.

    Returns
    -------
    points : np.ndarray
        Structured array with ``HeightAboveGround`` and ``TreeID`` fields.
        ``TreeID = 0`` means unassigned.  Points from all tiles are
        concatenated; TreeIDs are re-numbered 1 … N globally.
    trees : pd.DataFrame
        One row per tree, sorted by descending height.  Empty if none found.
    """
    parts = list(
        iter_segment_trees(
            provider,
            bbox,
            year,
            min_points,
            min_height,
            radius,
            adaptive_radius,
            voxel_size,
            tile_size,
            tile_buffer,
            n_workers,
            crown_fraction,
            veg_classes=veg_classes,
            point_attributes=point_attributes,
        )
    )
    if not parts:
        return np.array([]), pd.DataFrame()
    points = np.concatenate([part[0] for part in parts])
    trees = pd.concat([part[1] for part in parts], ignore_index=True)
    return points, trees.sort_values("height", ascending=False).reset_index(drop=True)


def iter_segment_trees(
    provider,
    bbox=None,
    year=None,
    min_points=10,
    min_height=3.0,
    radius=100.0,
    adaptive_radius=False,
    voxel_size=None,
    tile_size=None,
    tile_buffer=30.0,
    n_workers=1,
    crown_fraction=0.5,
    *,
    veg_classes=(3, 4, 5),
    point_attributes=None,
):
    """Yield (points, trees) per tile with compact global IDs and bounded memory.

    Ownership uses the full crown's highest point (XY breaks height ties).
    Crowns crossing the output boundary retain their buffered points. Finite
    buffers can still change segmentation; choose a buffer wider than crowns.
    Optional Poisson sampling changes the points used for crown metrics.
    """
    if (
        not np.isfinite([min_height, radius, tile_buffer, crown_fraction]).all()
        or min_height <= 0
        or radius <= 0
        or min_points < 1
        or tile_buffer < 0
        or not 0 <= crown_fraction <= 1
    ):
        raise ValueError("Invalid tree segmentation parameters")
    if adaptive_radius:
        warnings.warn(
            "adaptive_radius uses a legacy height-based heuristic for PDAL's non-tree seed radius; set radius explicitly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if voxel_size is not None and (not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("voxel_size must be finite and positive")
    if tile_size is not None and (not np.isfinite(tile_size) or tile_size <= 0):
        raise ValueError("tile_size must be finite and positive")
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)
    from alsdb.processing._grid import GridSpec

    GridSpec.from_bbox(effective_bbox, 1.0)  # validate extent without changing it
    if tile_size is None:
        x0, y0, x1, y1 = effective_bbox
        tiles = [
            (
                (x0 - tile_buffer, y0 - tile_buffer, x1 + tile_buffer, y1 + tile_buffer),
                effective_bbox,
            )
        ]
    else:
        tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)

    def worker(item):
        index, (query, crop) = item
        return _process_tile(
            provider,
            query,
            crop,
            index,
            year,
            min_points,
            min_height,
            radius,
            voxel_size,
            adaptive_radius,
            crown_fraction,
            veg_classes,
            point_attributes,
        )

    offset = 0
    seen = set()
    for result in bounded_map(worker, enumerate(tiles), n_workers):
        if result is None:
            continue
        points, trees = result
        keep = []
        for rec in trees.itertuples():
            key = (rec.apex_x, rec.apex_y)
            keep.append(key not in seen)
            seen.add(key)
        trees = trees.loc[keep].copy().sort_values("tree_id")
        if trees.empty:
            continue
        ids = trees.tree_id.to_numpy()
        points = points[np.isin(points["TreeID"], ids)]
        dtype = [
            (name, np.uint64 if name == "TreeID" else points.dtype[name])
            for name in points.dtype.names
        ]
        output = np.empty(len(points), dtype=dtype)
        for name in points.dtype.names:
            output[name] = points[name]
        output["TreeID"] = np.searchsorted(ids, points["TreeID"]) + offset + 1
        trees["tree_id"] = np.arange(offset + 1, offset + 1 + len(ids), dtype=np.uint64)
        offset += len(ids)
        yield output, trees.reset_index(drop=True)
