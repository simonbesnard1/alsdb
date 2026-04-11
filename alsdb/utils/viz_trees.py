# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Visualisation utilities for individual tree segmentation results.

Two main plots:

* :func:`plot_trees`    — 2-D plan view: crown circles or convex hulls coloured
                          by tree height, with centroids marked.
* :func:`plot_trees_3d` — 3-D scatter of the raw point cloud, each tree
                          coloured by a distinct hue; ground / unassigned
                          points shown in light grey.

Usage::

    from alsdb.processing.trees import segment_trees
    from alsdb.utils.viz_trees import plot_trees, plot_trees_3d

    points, trees = segment_trees(provider, bbox=(...), year=2014)

    plot_trees(trees, output_path="trees_2d.png")
    plot_trees_3d(points, trees, output_path="trees_3d.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hull_patch(x: np.ndarray, y: np.ndarray, **kwargs) -> mpatches.Polygon:
    """Return a matplotlib Polygon patch for the convex hull of (x, y)."""
    hull = ConvexHull(np.column_stack([x, y]))
    verts = np.column_stack([x, y])[hull.vertices]
    return mpatches.Polygon(verts, closed=True, **kwargs)


def _discrete_colors(n: int, cmap: str = "tab20") -> np.ndarray:
    """Return (n, 4) RGBA array with distinct colours cycling through *cmap*."""
    cm = plt.get_cmap(cmap)
    return np.array([cm(i % cm.N / cm.N) for i in range(n)])


# ---------------------------------------------------------------------------
# 2-D crown map
# ---------------------------------------------------------------------------


def plot_trees(
    trees: pd.DataFrame,
    points: Optional[np.ndarray] = None,
    cmap: str = "YlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_crowns: bool = True,
    output_path: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    2-D plan-view map of detected trees.

    Each tree is drawn as:

    * a filled crown circle (radius = ``crown_radius``) coloured by height, OR
    * a convex hull polygon when *points* is provided (more accurate shape).

    Centroids are marked with a small cross.

    Parameters
    ----------
    trees:
        DataFrame returned by :func:`~alsdb.processing.trees.segment_trees`.
    points:
        Optional raw point array (with ``TreeID`` and ``X``, ``Y`` fields).
        When supplied, actual crown convex hulls are drawn instead of circles.
    cmap:
        Matplotlib colormap for tree height.
    vmin, vmax:
        Color scale limits (m).  Defaults to data min/max.
    show_crowns:
        Draw crown outlines (circles or hulls).  Disable for very dense stands.
    output_path:
        Save figure to this path when provided.
    ax:
        Existing axes to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if trees.empty:
        raise ValueError("trees DataFrame is empty — nothing to plot.")

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=(10, 9))

    heights = trees["height"].values
    _vmin = vmin if vmin is not None else float(np.nanmin(heights))
    _vmax = vmax if vmax is not None else float(np.nanmax(heights))
    norm = Normalize(vmin=_vmin, vmax=_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cm = plt.get_cmap(cmap)

    if show_crowns:
        patches = []
        colors = []

        for _, row in trees.iterrows():
            color = cm(norm(row["height"]))
            if (
                points is not None
                and "TreeID" in points.dtype.names
                and not np.isnan(row.get("crown_area", np.nan))
            ):
                mask = points["TreeID"] == row["tree_id"]
                x_t = points["X"][mask].astype(float)
                y_t = points["Y"][mask].astype(float)
                if len(x_t) >= 3:
                    try:
                        patches.append(_hull_patch(x_t, y_t))
                        colors.append(color)
                        continue
                    except Exception:
                        pass
            # Fallback: circle with crown_radius
            r = row.get("crown_radius", 5.0)
            r = r if (r and not np.isnan(r)) else 5.0
            patches.append(mpatches.Circle((row["centroid_x"], row["centroid_y"]), r))
            colors.append(color)

        col = PatchCollection(
            patches,
            facecolors=colors,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.75,
            zorder=2,
        )
        ax.add_collection(col)

    # Centroids
    ax.scatter(
        trees["centroid_x"],
        trees["centroid_y"],
        c=heights,
        cmap=cmap,
        norm=norm,
        s=15,
        zorder=3,
        linewidths=0,
    )

    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Tree height (m)", rotation=270, labelpad=15)

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Individual tree segmentation — {len(trees)} trees")
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3-D coloured point cloud
# ---------------------------------------------------------------------------


def plot_trees_3d(
    points: np.ndarray,
    trees: Optional[pd.DataFrame] = None,
    max_trees: int = 30,
    point_size: float = 0.5,
    alpha: float = 0.6,
    output_path: Optional[str | Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    3-D scatter of the segmented point cloud.

    Each tree receives a distinct colour; unassigned points (``TreeID == 0``)
    are shown in light grey.

    Parameters
    ----------
    points:
        Structured array with ``X``, ``Y``, ``HeightAboveGround``, ``TreeID``.
    trees:
        Optional trees DataFrame.  When provided, up to *max_trees* tallest
        trees are labelled with their height.
    max_trees:
        Maximum number of trees to colour distinctly (the rest are grey).
    point_size:
        Marker size for ``ax.scatter``.
    alpha:
        Point transparency.
    output_path:
        Save figure when provided.
    ax:
        Existing 3-D axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "TreeID" not in points.dtype.names:
        raise ValueError("points array must have a 'TreeID' field — run segment_trees first.")
    if "HeightAboveGround" not in points.dtype.names:
        raise ValueError("points array must have a 'HeightAboveGround' field.")

    fig = ax.get_figure() if ax is not None else plt.figure(figsize=(12, 8))
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")

    x = points["X"].astype(float)
    y = points["Y"].astype(float)
    hag = points["HeightAboveGround"].astype(float)
    tid = points["TreeID"].astype(int)

    # Ground / unassigned in grey
    gnd = tid == 0
    ax.scatter(
        x[gnd],
        y[gnd],
        hag[gnd],
        c="lightgrey",
        s=point_size * 0.5,
        alpha=alpha * 0.5,
        linewidths=0,
        zorder=1,
    )

    # Tree points
    unique_ids = np.unique(tid[~gnd])
    if len(unique_ids) > max_trees:
        unique_ids = unique_ids[:max_trees]

    colors = _discrete_colors(len(unique_ids))
    for i, t in enumerate(unique_ids):
        mask = tid == t
        ax.scatter(
            x[mask],
            y[mask],
            hag[mask],
            c=[colors[i]],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            zorder=2,
        )

    # Label tallest trees
    if trees is not None and not trees.empty:
        for _, row in trees.head(min(10, len(trees))).iterrows():
            if int(row["tree_id"]) in unique_ids:
                ax.text(
                    row["centroid_x"],
                    row["centroid_y"],
                    row["height"] + 0.5,
                    f"{row['height']:.1f} m",
                    fontsize=6,
                    ha="center",
                    color="black",
                    zorder=5,
                )

    ax.set_xlabel("Easting (m)", labelpad=8)
    ax.set_ylabel("Northing (m)", labelpad=8)
    ax.set_zlabel("Height above ground (m)", labelpad=8)
    ax.set_title(
        f"3-D tree segmentation — {len(unique_ids)} trees shown "
        f"({'all' if len(unique_ids) == len(np.unique(tid[~gnd])) else f'top {max_trees}'})"
    )
    ax.ticklabel_format(style="plain", useOffset=False, axis="both")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
