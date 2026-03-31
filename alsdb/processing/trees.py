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

Usage::

    from alsdb import ALSProvider
    from alsdb.processing.trees import segment_trees

    provider = ALSProvider(storage_type="local", uri="array_")
    points, trees = segment_trees(
        provider,
        bbox=(655000.0, 8901000.0, 656000.0, 8902000.0),
        year=2014,
        min_height=3.0,
        min_points=10,
    )
    print(trees.head())
    print(f"{len(trees)} trees detected")

Reference
---------
Li, W., Guo, Q., Jakubowski, M. K., & Kelly, M. (2012). A new method for
segmenting individual trees from the lidar point cloud. Photogrammetric
Engineering & Remote Sensing, 78(1), 75–84.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
import pdal
from scipy.spatial import ConvexHull

from alsdb.processing._tiling import query_to_array
from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

_MIN_HULL_POINTS = 3   # need ≥ 3 points for a 2-D convex hull


def segment_trees(
    provider: TileDBProvider,
    bbox: tuple[float, float, float, float],
    year: Optional[int] = None,
    min_points: int = 10,
    min_height: float = 3.0,
    radius: float = 2.0,
    voxel_size: Optional[float] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Segment individual trees within *bbox* using ``filters.litree``.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    bbox:
        Spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Optional survey year filter.
    min_points:
        Minimum number of points for a cluster to be labelled as a tree.
        Smaller clusters receive ``TreeID = 0``.
    min_height:
        Minimum tree height in metres.  Points below this threshold are
        excluded before segmentation (they are returned in the output array
        with ``TreeID = 0``).
    radius:
        Search radius (m) used to connect points in the minimum spanning tree
        graph.  Increase for sparser point clouds or wider-crowned trees.
    voxel_size:
        If provided, downsample the vegetation point cloud using Poisson disk
        sampling (``filters.sample``) with this minimum inter-point distance (m)
        before running ``filters.litree``.
        Dramatically speeds up segmentation on dense ALS data (e.g. 0.5 m).
        Does not affect the returned *points* array — all original points are
        returned with their ``TreeID`` assigned by nearest-voxel label.

    Returns
    -------
    points : np.ndarray
        Structured numpy array (all input points) with two fields added:

        * ``HeightAboveGround`` — HAG in metres (from ``filters.hag_delaunay``)
        * ``TreeID``            — integer tree label (0 = unassigned)

    trees : pd.DataFrame
        One row per detected tree, sorted by descending height.
        Returns an empty DataFrame if no trees are found.
    """
    arr = query_to_array(provider, bbox, year=year)
    if arr.size == 0:
        logger.warning("segment_trees: no points in bbox %s", bbox)
        return arr, pd.DataFrame()

    # Pre-filter to vegetation points only — ground returns are irrelevant for
    # tree segmentation and are the main source of slowness in filters.litree.
    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
        {"type": "filters.range",
         "limits": f"HeightAboveGround[{min_height / 2}:]"},
    ]
    if voxel_size is not None:
        # filters.sample = Poisson disk sampling: keeps at most one point per
        # sphere of radius voxel_size — equivalent to voxel downsampling but
        # available in all PDAL builds (no plugin required).
        stages.append({
            "type": "filters.sample",
            "radius": voxel_size,
        })
    stages.append({
        "type": "filters.litree",
        "min_points": min_points,
        "min_height": min_height,
        "radius": radius,
    })

    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    points = p.arrays[0]

    tree_ids = np.unique(points["TreeID"])
    tree_ids = tree_ids[tree_ids > 0]

    if len(tree_ids) == 0:
        logger.warning(
            "segment_trees: no trees found in bbox %s "
            "(try lowering min_points or min_height)", bbox,
        )
        return points, pd.DataFrame()

    records = []
    for tid in tree_ids:
        mask = points["TreeID"] == tid
        pts  = points[mask]
        x    = pts["X"].astype(np.float64)
        y    = pts["Y"].astype(np.float64)
        hag  = pts["HeightAboveGround"].astype(np.float64)

        crown_area   = np.nan
        crown_radius = np.nan
        if len(pts) >= _MIN_HULL_POINTS:
            try:
                hull = ConvexHull(np.column_stack([x, y]))
                crown_area   = float(hull.volume)          # area in 2-D
                crown_radius = float(np.sqrt(crown_area / np.pi))
            except Exception:
                pass

        records.append({
            "tree_id":      int(tid),
            "centroid_x":   float(x.mean()),
            "centroid_y":   float(y.mean()),
            "height":       float(hag.max()),
            "base_height":  float(hag.min()),
            "crown_area":   crown_area,
            "crown_radius": crown_radius,
            "n_points":     int(mask.sum()),
        })

    trees = (
        pd.DataFrame(records)
        .sort_values("height", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "segment_trees: %d trees detected (bbox=%s, year=%s)",
        len(trees), bbox, year,
    )
    return points, trees
