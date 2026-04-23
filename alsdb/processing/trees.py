# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Individual tree segmentation from ALS TileDB point clouds.

Uses PDAL's ``filters.litree`` (Li et al. 2012 — graph-based minimum spanning
tree algorithm) to assign a unique ``TreeID`` to each point.  Unassigned points
(ground, noise, or sub-threshold clusters) receive ``TreeID = 0``.

After segmentation, per-tree metrics are summarised into a :class:`pandas.DataFrame`:

* ``tree_id``       — unique integer identifier
* ``centroid_x/y`` — crown centroid (UTM metres)
* ``height``        — maximum height above ground (m)
* ``base_height``   — minimum height above ground within the crown (m)
* ``crown_area``    — 2-D convex hull area (m²)
* ``crown_radius``  — equivalent circular radius = sqrt(area / π) (m)
* ``n_points``      — number of ALS points in the tree

Tiled processing
----------------
For large areas, pass ``tile_size`` to split the bbox into sub-tiles processed
in parallel.  A ``tile_buffer`` overlap ensures trees at tile boundaries are
fully captured; only trees whose centroid falls within the non-buffered
``crop_bbox`` are retained so each tree appears exactly once::

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import pdal
from scipy.spatial import ConvexHull

from alsdb.processing._tiling import array_data_bbox, query_to_array, tile_bboxes
from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

_MIN_HULL_POINTS = 3
# Each tile's TreeIDs are offset by tile_index × _TREE_ID_STRIDE so they are
# globally unique before the final re-numbering step.
_TREE_ID_STRIDE = 100_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hag_stages(min_height: float, voxel_size: Optional[float]) -> list[dict]:
    """HAG + pre-filter stages shared by all code paths (run before litree)."""
    stages: list[dict] = [
        {"type": "filters.hag_delaunay"},
        {
            "type": "filters.assign",
            "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
        },
        # Drop ground / low points before the graph build — single biggest
        # speedup for filters.litree on dense ALS data.
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


def _tree_metrics(points: np.ndarray) -> list[dict]:
    """Compute per-tree metrics from a point array that already has TreeID."""
    if "TreeID" not in points.dtype.names:
        return []
    tree_ids = np.unique(points["TreeID"])
    tree_ids = tree_ids[tree_ids > 0]
    records = []
    for tid in tree_ids:
        mask = points["TreeID"] == tid
        pts = points[mask]
        x = pts["X"].astype(np.float64)
        y = pts["Y"].astype(np.float64)
        hag = pts["HeightAboveGround"].astype(np.float64)

        crown_area = crown_radius = np.nan
        if len(pts) >= _MIN_HULL_POINTS:
            try:
                hull = ConvexHull(np.column_stack([x, y]))
                crown_area = float(hull.volume)
                crown_radius = float(np.sqrt(crown_area / np.pi))
            except Exception:
                pass

        records.append(
            {
                "tree_id": int(tid),
                "centroid_x": float(x.mean()),
                "centroid_y": float(y.mean()),
                "height": float(hag.max()),
                "base_height": float(hag.min()),
                "crown_area": crown_area,
                "crown_radius": crown_radius,
                "n_points": int(mask.sum()),
            }
        )
    return records


def _process_tile(
    provider: TileDBProvider,
    query_bbox: tuple[float, float, float, float],
    crop_bbox: tuple[float, float, float, float],
    tile_index: int,
    year: Optional[int],
    min_points: int,
    min_height: float,
    radius: float,
    voxel_size: Optional[float],
    adaptive_radius: bool = False,
) -> Optional[tuple[np.ndarray, pd.DataFrame]]:
    """
    Segment trees within one sub-tile.

    Queries *query_bbox* (buffered) for HAG accuracy, then:
    * crops the point array to *crop_bbox* (no buffer — avoids duplicate trees),
    * offsets TreeIDs by ``tile_index × _TREE_ID_STRIDE`` for global uniqueness,
    * discards trees whose centroid falls outside *crop_bbox*.
    """
    arr = query_to_array(provider, query_bbox, year=year)
    if arr.size == 0:
        return None

    try:
        p = pdal.Pipeline(json.dumps(_hag_stages(min_height, voxel_size)), arrays=[arr])
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
            json.dumps([_litree_stage(min_height, min_points, r)]), arrays=[hag_points]
        )
        p2.execute()
        points = p2.arrays[0] if p2.arrays else hag_points[:0]
    except RuntimeError as exc:
        if "no points" in str(exc).lower():
            return None
        raise

    # Crop to non-buffered extent
    cx0, cy0, cx1, cy1 = crop_bbox
    in_crop = (
        (points["X"] >= cx0) & (points["X"] <= cx1) & (points["Y"] >= cy0) & (points["Y"] <= cy1)
    )
    points = points[in_crop].copy()

    if points.size == 0:
        return None

    # Offset TreeIDs so they are unique across tiles
    valid = points["TreeID"] > 0
    points["TreeID"][valid] += tile_index * _TREE_ID_STRIDE

    # Compute metrics and filter to trees whose centroid is inside crop_bbox
    records = []
    for rec in _tree_metrics(points):
        if cx0 <= rec["centroid_x"] <= cx1 and cy0 <= rec["centroid_y"] <= cy1:
            records.append(rec)
        else:
            # Centroid outside crop_bbox → this tree belongs to a neighbour tile
            points["TreeID"][points["TreeID"] == rec["tree_id"]] = 0

    if not records:
        return None

    return points, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_trees(
    provider: TileDBProvider,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    min_points: int = 10,
    min_height: float = 3.0,
    radius: float = 2.0,
    adaptive_radius: bool = False,
    voxel_size: Optional[float] = None,
    tile_size: Optional[float] = None,
    tile_buffer: float = 30.0,
    n_workers: int = 1,
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
        ``filters.litree`` search radius (m).  Used when *adaptive_radius*
        is ``False``.  Increase for sparser clouds or wider-crowned trees.
    adaptive_radius:
        If ``True``, compute the search radius per tile from P75 HAG using
        ``wf(h) = 0.07 * h + 0.6`` (Murphy et al. lidar-forestry approach).
        Overrides *radius*.
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

    Returns
    -------
    points : np.ndarray
        Structured array with ``HeightAboveGround`` and ``TreeID`` fields.
        ``TreeID = 0`` means unassigned.  Points from all tiles are
        concatenated; TreeIDs are re-numbered 1 … N globally.
    trees : pd.DataFrame
        One row per tree, sorted by descending height.  Empty if none found.
    """
    effective_bbox = bbox if bbox is not None else array_data_bbox(provider)

    # ------------------------------------------------------------------ #
    # Single-tile (no tiling) fast path                                   #
    # ------------------------------------------------------------------ #
    if tile_size is None:
        logger.info(
            "Segmenting trees  bbox=%s  year=%s  min_height=%.1f m  "
            "radius=%s  voxel_size=%s",
            effective_bbox,
            year,
            min_height,
            "adaptive" if adaptive_radius else f"{radius:.1f} m",
            f"{voxel_size} m" if voxel_size else "none",
        )
        arr = query_to_array(provider, effective_bbox, year=year)
        if arr.size == 0:
            logger.warning("segment_trees: no points in bbox %s", effective_bbox)
            return arr, pd.DataFrame()

        logger.info("  %d points queried — running filters.litree…", arr.size)
        p = pdal.Pipeline(json.dumps(_hag_stages(min_height, voxel_size)), arrays=[arr])
        p.execute()
        hag_points = p.arrays[0] if p.arrays else arr[:0]
        r = _compute_adaptive_radius(hag_points) if adaptive_radius else radius
        p2 = pdal.Pipeline(
            json.dumps([_litree_stage(min_height, min_points, r)]), arrays=[hag_points]
        )
        p2.execute()
        points = p2.arrays[0] if p2.arrays else hag_points[:0]

        records = _tree_metrics(points)
        if not records:
            logger.warning("segment_trees: no trees found (try lowering min_points or min_height)")
            return points, pd.DataFrame()

        trees = pd.DataFrame(records).sort_values("height", ascending=False).reset_index(drop=True)
        logger.info(
            "  Done: %d trees  |  tallest %.1f m  |  mean crown %.0f m²",
            len(trees),
            trees["height"].max(),
            trees["crown_area"].mean(),
        )
        return points, trees

    # ------------------------------------------------------------------ #
    # Tiled path                                                           #
    # ------------------------------------------------------------------ #
    tiles = tile_bboxes(effective_bbox, tile_size=tile_size, buffer=tile_buffer)
    n_tiles = len(tiles)
    logger.info(
        "Segmenting trees  bbox=%s  year=%s  "
        "%d tile(s)  %.0f m tiles / %.0f m buffer  %d worker(s)  voxel_size=%s",
        effective_bbox,
        year,
        n_tiles,
        tile_size,
        tile_buffer,
        n_workers,
        f"{voxel_size} m" if voxel_size else "none",
    )

    tile_results: list[Optional[tuple[np.ndarray, pd.DataFrame]]] = [None] * n_tiles

    def _worker(idx: int, qb, cb):
        result = _process_tile(
            provider, qb, cb, idx, year, min_points, min_height, radius, voxel_size,
            adaptive_radius=adaptive_radius,
        )
        n = len(result[1]) if result is not None else 0
        logger.debug("  tile %d/%d: %d trees", idx + 1, n_tiles, n)
        return idx, result

    if n_workers == 1:
        for i, (qb, cb) in enumerate(tiles):
            _, result = _worker(i, qb, cb)
            tile_results[i] = result
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, i, qb, cb): i for i, (qb, cb) in enumerate(tiles)}
            for future in as_completed(futures):
                idx, result = future.result()
                tile_results[idx] = result

    valid = [r for r in tile_results if r is not None]
    if not valid:
        logger.warning("segment_trees: no trees found in any tile")
        return np.array([]), pd.DataFrame()

    all_points = np.concatenate([r[0] for r in valid])
    all_trees = pd.concat([r[1] for r in valid], ignore_index=True)

    # Re-number TreeIDs 1…N globally (tile offsets served their purpose)
    id_map = {old: new for new, old in enumerate(all_trees["tree_id"].values, start=1)}
    all_trees["tree_id"] = all_trees["tree_id"].map(id_map)
    for old_id, new_id in id_map.items():
        all_points["TreeID"][all_points["TreeID"] == old_id] = new_id

    all_trees = all_trees.sort_values("height", ascending=False).reset_index(drop=True)
    logger.info(
        "  Done: %d trees from %d/%d tiles  |  tallest %.1f m  |  mean crown %.0f m²",
        len(all_trees),
        len(valid),
        n_tiles,
        all_trees["height"].max(),
        all_trees["crown_area"].mean(),
    )
    return all_points, all_trees
