# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Gap fraction and effective LAI from TileDB ALS point clouds.

Theory
------
Gap fraction P_gap is estimated using the MacArthur-Wilson return-count
estimator:

    P_gap = N_gnd / (N_gnd + N_veg)

where *N_gnd* is the number of first returns classified as ground (Class 2)
and *N_veg* is the number of first returns classified as vegetation
(Classes 3–5) within each raster cell.  This is a direct observable —
no assumptions about canopy structure are required.

Effective LAI can be derived optionally via the Beer-Lambert law:

    L_e = -ln(P_gap) / k

where *k* is the extinction coefficient (default 0.5 for a spherical leaf
angle distribution).  The output is "effective LAI" (L_e), not true LAI,
because ALS cannot distinguish leaves from woody material.

Usage::

    from alsdb import ALSProvider
    from alsdb.storage import ALSZarrStore
    from alsdb.processing.gap import compute_gap_fraction

    provider = ALSProvider(storage_type="local", uri="array_")
    store = ALSZarrStore("output/spain.zarr")

    # Gap fraction only
    compute_gap_fraction(provider, store, resolution=10.0, year=2021)

    # Gap fraction + effective LAI
    compute_gap_fraction(provider, store, resolution=10.0, year=2021,
                         lai=True, k=0.5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from alsdb.processing._tiling import (
    VEG_CLASSES as _VEG_CLASSES,
)

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)

_GROUND_CLASS = 2
_LAI_K_DEFAULT = 0.5
_LAI_MAX = 15.0  # physical ceiling — raised from 10 to cover dense tropical canopies

# Extinction coefficient presets for common leaf angle distributions.
# Pass the appropriate value as `k` to compute_gap_fraction(lai=True, k=...).
LAI_K_PRESETS: dict[str, float] = {
    "spherical": 0.5,  # random leaf angles — standard default
    "planophile": 0.8,  # predominantly horizontal (tropical broadleaf, crops)
    "erectophile": 0.35,  # predominantly vertical (grasses, some conifers)
    "conifer": 0.45,  # needle-leaf average across species
}


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def _compute_gap_grid(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
) -> np.ndarray:
    """
    Compute per-cell gap fraction directly from classified first returns.

    Returns a ``(ny, nx)`` float32 north-up array; cells with no first
    returns are ``np.nan``.

    veg_classes:
        LAS classification codes treated as vegetation (default ``(3, 4,
        5)``, standard ASPRS). Not universal across surveys -- see
        ``pavd.py``'s module docstring for a real dataset (DE-Hai) that
        uses a different code entirely; verify with
        ``np.unique(points["Classification"], return_counts=True)`` before
        trusting the default on a new array.
    """
    return gap_statistics(points, resolution, bbox, min_density, veg_classes)["gap"]


def gap_statistics(
    points, resolution, bbox, min_density=0.0, veg_classes=_VEG_CLASSES, baba_radius=0.0
):
    """Gap estimate, first-return support counts and zero-gap saturation flag."""
    from scipy.spatial import cKDTree

    from alsdb.processing._grid import GridSpec

    grid = GridSpec.from_bbox(bbox, resolution)
    if baba_radius > 0:
        first = (points["ReturnNumber"] == 1) & np.isfinite(points["X"]) & np.isfinite(points["Y"])
        classification = points["Classification"][first]
        ground = classification == _GROUND_CLASS
        classified = ground | np.isin(classification, veg_classes)
        x, y = grid.centers()
        centers = np.column_stack((x.ravel(), y.ravel()))
        xy = np.column_stack((points["X"][first], points["Y"][first]))

        def count(mask):
            return cKDTree(xy[mask]).query_ball_point(centers, baba_radius, return_length=True)

        ng, nc, nf = (count(mask) for mask in (ground, classified, np.ones(len(xy), bool)))
        area = np.pi * baba_radius**2
    else:
        inside, cells = grid.point_bins(points)
        first = points["ReturnNumber"][inside] == 1
        cells = cells[first]
        classification = points["Classification"][inside][first]
        ground = classification == _GROUND_CLASS
        classified = ground | np.isin(classification, veg_classes)
        n = grid.nx * grid.ny
        ng = np.bincount(cells, weights=ground, minlength=n)
        nc = np.bincount(cells, weights=classified, minlength=n)
        nf = np.bincount(cells, minlength=n)
        area = resolution**2
    supported = (nc > 0) & (nf / area >= min_density)
    gap = np.divide(ng, nc, out=np.full(len(nc), np.nan), where=supported)
    saturated = np.where(supported, (ng == 0).astype(float), np.nan)
    return {
        name: values.reshape(grid.shape).astype(np.float32)
        for name, values in (
            ("gap", gap),
            ("gap_n_ground", ng),
            ("gap_n_classified", nc),
            ("gap_n_first", nf),
            ("gap_saturated", saturated),
        )
    }


def _gap_to_lai(gap: np.ndarray, k: float, clumping_index: float = 1.0) -> np.ndarray:
    """
    Convert gap fraction to effective LAI via Beer-Lambert with optional
    clumping correction (Jonckheere et al. 2004 / Chen & Black 1992).

    ``L_true = -ln(P_gap) / (k × Ω)``

    where Ω is the element clumping index (0 < Ω ≤ 1).  Random foliage → Ω = 1
    (no correction).  Clumped canopies have Ω < 1, so L_true > L_e.
    """
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    if not np.isfinite(clumping_index) or clumping_index <= 0 or clumping_index > 1:
        raise ValueError(f"clumping_index must be in (0, 1], got {clumping_index}")
    with np.errstate(invalid="ignore", divide="ignore"):
        lai = -np.log(np.where(gap > 0, gap, np.nan)) / (k * clumping_index)
    return np.clip(lai, 0.0, _LAI_MAX).astype(np.float32)


# ---------------------------------------------------------------------------
# BABA gap fraction
# ---------------------------------------------------------------------------


def _compute_gap_grid_baba(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    baba_radius: float,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
) -> np.ndarray:
    """
    Compute per-cell gap fraction using a circular neighbourhood of radius
    *baba_radius* around each cell centre (Buffered Area-Based Approach).

    The denominator is ``N_gnd + N_veg`` (classified first returns only),
    consistent with the standard grid estimator.  Unclassified, noise, and
    building returns are excluded so they do not dilute the gap estimate.

    veg_classes:
        LAS classification codes treated as vegetation (default ``(3, 4,
        5)``, standard ASPRS) -- see :func:`_compute_gap_grid`'s docstring;
        not universal across surveys.
    """
    return gap_statistics(
        points, resolution, bbox, min_density, veg_classes, baba_radius=baba_radius
    )["gap"]


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------


def compute_gap_fraction(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 10.0,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    *,
    lai: bool = False,
    k: float = _LAI_K_DEFAULT,
    clumping_index: float = 1.0,
    baba_radius: float = 0.0,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    ground_outlier_removal: bool = False,
    quality: bool = True,
) -> None:
    """
    Compute gap fraction (and optionally effective LAI) and write into *store*.

    Gap fraction is the MacArthur-Wilson estimator:

        P_gap = N_gnd_first / (N_gnd_first + N_veg_first)

    When ``lai=True``, effective LAI is derived via the Beer-Lambert law with
    an optional clumping correction (Jonckheere et al. 2004):

        L_true = -ln(P_gap) / (k × Ω)

    where *k* is the extinction coefficient and *Ω* is the element clumping
    index.  For biome-appropriate *k* values see :data:`LAI_K_PRESETS`.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.  Must have ``"gap"``
        (and ``"lai"`` if ``lai=True``) pre-allocated at *resolution*.
    resolution:
        Cell size in metres (default 10 m).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year filter.  Written as a time slice in the store.
    lai:
        If ``True``, also compute effective LAI via Beer-Lambert.
    k:
        Extinction coefficient (default 0.5, spherical leaf angle
        distribution).  See :data:`LAI_K_PRESETS` for biome presets.
        Only used when ``lai=True``.
    clumping_index:
        Element clumping index Ω ∈ (0, 1] (default 1.0 = no correction).
        Values < 1 correct for foliage clumping — typical ranges:
        0.5–0.7 for conifers, 0.7–0.9 for broadleaf forests.
        Only used when ``lai=True``.
    min_density:
        Minimum first-return density (returns m⁻²) for a cell to receive
        a gap fraction estimate.  Cells below this threshold are set to
        ``np.nan``.  Default ``0.0`` disables the guard.
    veg_classes:
        LAS classification codes treated as vegetation (default ``(3, 4,
        5)``, standard ASPRS). Not universal across surveys -- see
        ``pavd.py``'s module docstring for a real dataset that uses a
        different code entirely; verify with
        ``np.unique(classification, return_counts=True)`` before trusting
        the default on a new array.
    overwrite:
        If ``False`` (default) and gap (and LAI if requested) already exist
        for *year* in the store, the computation is skipped.
    tile_size:
        Sub-tile width and height in metres (default 500 m).
    tile_buffer:
        Query overlap buffer (default 50 m), expanded to cover baba_radius.
    n_workers:
        Parallel workers (default 1 = sequential).
    """
    from alsdb.processing.forest import compute_forest_products

    compute_forest_products(
        provider,
        store,
        resolution,
        bbox,
        year,
        metrics=(),
        gap=True,
        lai=lai,
        k=k,
        clumping_index=clumping_index,
        baba_radius=baba_radius,
        min_density=min_density,
        veg_classes=veg_classes,
        overwrite=overwrite,
        tile_size=tile_size,
        tile_buffer=tile_buffer,
        n_workers=n_workers,
        ground_outlier_removal=ground_outlier_removal,
        quality=quality,
    )
