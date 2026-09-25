"""Native incremental constrained-Delaunay spike-free canopy reconstruction."""

import numpy as np

from alsdb.processing._grid import GridSpec


def require_backend():
    try:
        from alsdb.processing import _spikefree_native
    except ImportError as exc:
        raise ImportError(
            "Native spike-free backend is not built. Run "
            "`pixi run -e spikefree build-spikefree` and use the spikefree "
            "environment. For the approximation select "
            "method='highest_subcell_tin'."
        ) from exc
    return _spikefree_native


def rasterize_spikefree(
    points,
    bbox,
    resolution,
    *,
    freeze_distance=1.5,
    height_buffer=0.5,
    max_triangle_edge=None,
    value_field="HeightAboveGround",
):
    """Rasterize all eligible returns, sorted by value_field (HAG by default).

    freeze_distance is a horizontal edge threshold; height_buffer delays
    freezing vertically. max_triangle_edge optionally trims unsupported long
    triangles in the final mesh. There is no highest-per-subcell thinning.
    This implements the published mechanism, not binary LAStools equivalence.
    """
    if (
        not np.isfinite([freeze_distance, height_buffer]).all()
        or freeze_distance <= 0
        or height_buffer < 0
    ):
        raise ValueError("freeze_distance must be positive and height_buffer nonnegative")
    if max_triangle_edge is not None and (
        not np.isfinite(max_triangle_edge) or max_triangle_edge <= 0
    ):
        raise ValueError("max_triangle_edge must be positive")
    grid = GridSpec.from_bbox(bbox, resolution)
    xyz = np.column_stack((points["X"], points["Y"], points[value_field])).astype(np.float64)
    return require_backend().rasterize(
        xyz,
        grid.x0,
        grid.y1,
        resolution,
        grid.nx,
        grid.ny,
        freeze_distance,
        height_buffer,
        max_triangle_edge or 0.0,
    )
