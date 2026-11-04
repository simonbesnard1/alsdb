"""Spike-free canopy reconstruction (Khosravipour et al. 2016)

Points are inserted from highest to lowest into a constrained Delaunay
triangulation (CGAL); a point is skipped where the triangle above it is
frozen (all edges shorter than the freeze distance and at least the height
buffer above it). ``freeze_distance="local"`` sets the freeze distance per
5 m cell from the pulse density, as in the locally adaptive spike-free of
Fischer et al. (2024; lidR ``lspikefree()``).

References
----------
Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. (2024). Robust
    characterisation of forest structure from airborne laser scanning - A
    systematic assessment and sample workflow for ecologists. Methods in
    Ecology and Evolution, 15, 1873-1888.
    https://doi.org/10.1111/2041-210X.14416
Khosravipour, A., Skidmore, A. K., & Isenburg, M. (2016). Generating
    spike-free digital surface models using LiDAR raw point clouds: A new
    approach for forestry applications. International Journal of Applied
    Earth Observation and Geoinformation, 52, 104-114.
    https://doi.org/10.1016/j.jag.2016.06.005
The CGAL Project. CGAL User and Reference Manual, 2D Triangulations.
    https://doc.cgal.org/latest/Triangulation_2/
"""

from dataclasses import dataclass

import numpy as np

from alsdb.methods._grid import GridSpec

LOCAL_FREEZE_CELL = 5.0  # m, cells for the local pulse density (as PULSE_DENSITY_CELL)
# Fischer et al. (2024): freeze = m / ln(s * pd + o) + 1, pd in pulses/m2,
# m = 3.1, s = 1.75, o = 2.1. The paper prints m / (ln(s * pd + o) + 1); the
# authors' R workflow computes the form above, which lidR's lspikefree()
# approximates as 1.42 + 1.96 / sqrt(pd).
LOCAL_FREEZE_PARAMETERS = (3.1, 1.75, 2.1)
# Densities are clamped to this range, as in the authors' workflow.
LOCAL_DENSITY_RANGE = (0.5, 100.0)


def local_freeze_distance(density):
    """Freeze distance (m) for pulse densities (pulses/m2), Fischer et al. (2024)."""
    m, s, o = LOCAL_FREEZE_PARAMETERS
    pd = np.clip(np.asarray(density, dtype=float), *LOCAL_DENSITY_RANGE)
    return m / np.log(s * pd + o) + 1.0


@dataclass(frozen=True)
class FreezeGrid:
    """North-up freeze distances (m) with upper-left corner (x0, y1)."""

    distance: np.ndarray
    x0: float
    y1: float
    cell: float


def local_freeze_grid(pulses, bbox, cell=LOCAL_FREEZE_CELL):
    """Freeze distances from the density of *pulses* (one return per pulse) per cell.

    Cells lie on multiples of *cell*, so neighbouring tiles share them; cells
    without pulses count as the lowest density of :data:`LOCAL_DENSITY_RANGE`.
    """
    from alsdb.methods._surface import pulse_density_grid

    density, x0, y1 = pulse_density_grid(pulses, bbox, cell)
    return FreezeGrid(local_freeze_distance(density), x0, y1, float(cell))


def require_backend():
    try:
        from alsdb.methods import _spikefree_native
    except ImportError as exc:
        raise ImportError(
            "Native spike-free backend is not built. Run "
            "`pixi run -e spikefree build-spikefree` and use the spikefree "
            "environment."
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

    freeze_distance is a horizontal edge threshold, one value or a
    :class:`FreezeGrid` (locally adaptive); height_buffer delays freezing
    vertically. max_triangle_edge optionally trims unsupported long
    triangles in the final mesh. There is no highest-per-subcell thinning.
    This implements the published mechanism, not binary LAStools equivalence.
    """
    local = freeze_distance if isinstance(freeze_distance, FreezeGrid) else None
    scalar = local.distance.max() if local is not None else freeze_distance
    if not np.isfinite([scalar, height_buffer]).all() or scalar <= 0 or height_buffer < 0:
        raise ValueError("freeze_distance must be positive and height_buffer nonnegative")
    if local is not None and not (np.isfinite(local.distance).all() and local.distance.min() > 0):
        raise ValueError("local freeze distances must be finite and positive")
    if max_triangle_edge is not None and (
        not np.isfinite(max_triangle_edge) or max_triangle_edge <= 0
    ):
        raise ValueError("max_triangle_edge must be positive")
    grid = GridSpec.from_bbox(bbox, resolution)
    xyz = np.column_stack((points["X"], points["Y"], points[value_field])).astype(np.float64)
    backend = require_backend()
    extra = {}
    if local is not None:
        if not getattr(backend, "freeze_grid", False):
            raise ImportError(
                "The native spike-free backend predates local freeze distances; "
                "rebuild it with `pixi run -e spikefree build-spikefree`."
            )
        extra = {
            "freeze_grid": local.distance,
            "fx0": local.x0,
            "fy1": local.y1,
            "fcell": local.cell,
        }
    return backend.rasterize(
        xyz,
        grid.x0,
        grid.y1,
        resolution,
        grid.nx,
        grid.ny,
        float(scalar),
        height_buffer,
        max_triangle_edge or 0.0,
        **extra,
    )
