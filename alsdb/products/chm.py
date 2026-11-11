# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""CHM, DTM and DSM processing.

Query buffered tiles from the database, normalise heights with a ground TIN,
build surfaces and write products and provenance to Zarr.

References
----------
Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. (2024). Robust
    characterisation of forest structure from airborne laser scanning - A
    systematic assessment and sample workflow for ecologists. Methods in
    Ecology and Evolution, 15, 1873-1888.
    https://doi.org/10.1111/2041-210X.14416 (``freeze_distance="local"``)
Khosravipour, A., Skidmore, A. K., Isenburg, M., Wang, T., & Hussin, Y. A.
    (2014). Generating pit-free canopy height models from airborne lidar.
    Photogrammetric Engineering & Remote Sensing, 80(9), 863-872.
    https://doi.org/10.14358/PERS.80.9.863 (``method="pitfree"``)
Khosravipour, A., Skidmore, A. K., & Isenburg, M. (2016). Generating
    spike-free digital surface models using LiDAR raw point clouds: A new
    approach for forestry applications. International Journal of Applied
    Earth Observation and Geoinformation, 52, 104-114.
    https://doi.org/10.1016/j.jag.2016.06.005 (``method="spikefree"``)
Roussel, J.-R., Auty, D., Coops, N. C., Tompalski, P., Goodbody, T. R. H.,
    Sánchez Meador, A., Bourdon, J.-F., de Boissieu, F., & Achim, A. (2020).
    lidR: An R package for analysis of Airborne Laser Scanning (ALS) data.
    Remote Sensing of Environment, 251, 112061.
    https://doi.org/10.1016/j.rse.2020.112061 (pitfree and spike-free
    conventions, three-pulse-spacing freeze distance)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from alsdb.methods._terrain import GROUND_CLASSES
from alsdb.methods._terrain import VEG_CLASSES as _VEG_CLASSES
from alsdb.methods.rasterise import _NOISE_CLASSES, _PITFREE_THRESHOLDS

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)


def _validate_grid_alignment(tile_size: float, resolution: float) -> None:
    """Raise ValueError unless *tile_size* is a whole multiple of *resolution*."""
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


# ---------------------------------------------------------------------------
# DTM interpolation helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pit-free canopy-top rasterisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spike-free canopy-top rasterisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CHM post-processing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PDAL helpers
# ---------------------------------------------------------------------------


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
    ground_classes: tuple[int, ...] = GROUND_CLASSES,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_edge: float | str = "auto",
    max_height: float | None = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 6.0,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    quality: bool = True,
    ground_outlier_removal: bool = True,
    source_version: str | None = None,
    method: str = "max",
    freeze_distance: float | str = "auto",
    spikefree_first_returns: bool = False,
    height_buffer: float = 0.5,
    min_pulse_density: float | None = 2.0,
    max_scan_angle: float | None = 20.0,
    max_triangle_edge: float | None = None,
    lastools_executable: str = "las2dem64",
    lastools_demo: bool = False,
    freeze_interval: float = 0.25,
    ground_extrapolation: bool = False,
    fill_max_cells: int = 4,
    parallel: str = "threads",
) -> None:
    """Compute a canopy height model and write it to Zarr.

    ``method``: "max", "pitfree", "tin" (first-return TIN), "spikefree" or
    "lastools". ``freeze_distance`` for spike-free: a
    number (m), "auto" (three pulse spacings per tile) or "local" (per 5 m
    cell, Fischer et al. 2024). Quality flags mark cells below
    ``min_pulse_density`` (pulses/m2) or above ``max_scan_angle`` (degrees);
    ``None`` disables a flag. Terrain is missing outside the ground hull
    unless ``ground_extrapolation=True``.

    ``parallel="processes"`` runs the tiles of ``n_workers > 1`` in worker
    processes instead of threads: faster on many cores (threads level off at
    two to three times one worker), with one tile in memory per process;
    scripts then need an ``if __name__ == "__main__":`` guard.

    Products share one north-up grid that keeps the requested upper-left
    origin. ``tile_size`` must be a multiple of ``resolution``. Changing
    parameters requires ``overwrite=True``; interrupted runs resume safely.
    See ``doc/chm_methods.md`` for methods, flags and examples.
    """
    from alsdb.products._surfaces import compute_products

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
    ground_classes: tuple[int, ...] = GROUND_CLASSES,
    parallel: str = "threads",
) -> None:
    """Compute a digital terrain model and write it to Zarr.

    Terrain is missing outside the ground hull unless
    ``ground_extrapolation=True``; ``max_ground_distance`` limits support.

    ``parallel="processes"`` runs the tiles of ``n_workers > 1`` in worker
    processes instead of threads: faster on many cores (threads level off at
    two to three times one worker), with one tile in memory per process;
    scripts then need an ``if __name__ == "__main__":`` guard.

    Products share one north-up grid that keeps the requested upper-left
    origin. ``tile_size`` must be a multiple of ``resolution``. Changing
    parameters requires ``overwrite=True``; interrupted runs resume safely.
    See ``doc/chm_methods.md`` for methods, flags and examples.
    """
    from alsdb.products._surfaces import compute_products

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
    parallel: str = "threads",
) -> None:
    """Compute a digital surface model and write it to Zarr.

    ``exclude_classes`` are left out of the surface.

    ``parallel="processes"`` runs the tiles of ``n_workers > 1`` in worker
    processes instead of threads: faster on many cores (threads level off at
    two to three times one worker), with one tile in memory per process;
    scripts then need an ``if __name__ == "__main__":`` guard.

    Products share one north-up grid that keeps the requested upper-left
    origin. ``tile_size`` must be a multiple of ``resolution``. Changing
    parameters requires ``overwrite=True``; interrupted runs resume safely.
    See ``doc/chm_methods.md`` for methods, flags and examples.
    """
    from alsdb.products._surfaces import compute_products

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
    ground_classes: tuple[int, ...] = GROUND_CLASSES,
    pitfree_thresholds: tuple[float, ...] = _PITFREE_THRESHOLDS,
    pitfree_max_edge: float | str = "auto",
    max_height: float | None = None,
    remove_outliers: bool = False,
    outlier_mean_k: int = 8,
    outlier_multiplier: float = 6.0,
    dsm_exclude_classes: tuple[int, ...] = _NOISE_CLASSES,
    overwrite: bool = False,
    dtm_method: str = "tin",
    quality: bool = True,
    ground_outlier_removal: bool = True,
    source_version: str | None = None,
    method: str = "max",
    freeze_distance: float | str = "auto",
    spikefree_first_returns: bool = False,
    height_buffer: float = 0.5,
    min_pulse_density: float | None = 2.0,
    max_scan_angle: float | None = 20.0,
    max_triangle_edge: float | None = None,
    lastools_executable: str = "las2dem64",
    lastools_demo: bool = False,
    freeze_interval: float = 0.25,
    ground_extrapolation: bool = False,
    fill_max_cells: int = 4,
    parallel: str = "threads",
) -> None:
    """Compute DTM, DSM and CHM on one grid and write them to Zarr.

    Accepts the options of :func:`compute_chm`, :func:`compute_dtm` and
    :func:`compute_dsm`.

    ``parallel="processes"`` runs the tiles of ``n_workers > 1`` in worker
    processes instead of threads: faster on many cores (threads level off at
    two to three times one worker), with one tile in memory per process;
    scripts then need an ``if __name__ == "__main__":`` guard.

    Products share one north-up grid that keeps the requested upper-left
    origin. ``tile_size`` must be a multiple of ``resolution``. Changing
    parameters requires ``overwrite=True``; interrupted runs resume safely.
    See ``doc/chm_methods.md`` for methods, flags and examples.
    """
    from alsdb.products._surfaces import compute_products

    compute_products(provider, store, ("dtm", "dsm", "chm"), locals())
