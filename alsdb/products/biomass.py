# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Above-ground biomass (AGB) from ALS point clouds.

Points are normalised with a buffered terrain TIN, per-cell metrics are
computed (height percentiles, canopy cover, density, foliage height
diversity, height-stratum proportions) and an allometric model turns them
into AGB (Mg/ha), written to an :class:`~alsdb.storage.ALSZarrStore`.

Height metrics use all vegetation returns above ground; canopy cover
(``cc``) is the share of classified (ground and vegetation) first returns
that are vegetation above ``cc_threshold``, so buildings, wires and
unclassified points neither add cover nor dilute it (as the gap fraction
of :mod:`alsdb.products.gap`).

The models are in :mod:`alsdb.products.allometry`: the default Næsset-style
power law (placeholder coefficients: calibrate it before using the results),
linear models in the functional forms of GEDI L4A with prediction intervals,
and wrappers for scikit-learn and PyTorch models.

Example::

    compute_metrics(provider, store, resolution=10.0, year=2021)
    compute_biomass(provider, store, resolution=10.0, year=2021)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from alsdb.methods._terrain import VEG_CLASSES as _VEG_CLASSES
from alsdb.products._tiling import PointGrid

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)

_DEFAULT_CC_THRESHOLD = 2.0  # m — first returns above this count as "canopy"

_METRIC_NAMES = [
    # Height percentiles
    "h50",
    "h75",
    "h95",
    "hmax",
    "hmean",
    # Canopy structure
    "cc",
    "density",
    "fhd",
    "vci",
    "crr",
    # Height stratum proportions (fraction of veg returns per layer)
    "pv_0_2",
    "pv_2_5",
    "pv_5_10",
    "pv_10_20",
    "pv_20_40",
    "pv_above40",
]

# Height strata for pv_* metrics: (lower, upper) bounds in metres.
# The last stratum is open-ended (upper = ∞).
_HEIGHT_STRATA: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, np.inf),
)
_HEIGHT_STRATA_NAMES: tuple[str, ...] = (
    "pv_0_2",
    "pv_2_5",
    "pv_5_10",
    "pv_10_20",
    "pv_20_40",
    "pv_above40",
)

# FHD/VCI vertical binning: 1 m bands up to _FHD_MAX_H. VCI divides by
# log(_FHD_N_BINS), the total bin count (Schneider et al. 2017), so it grows
# with canopy height as well as evenness.
_FHD_BIN_SIZE: float = 1.0
_FHD_MAX_H: float = 80.0  # raised from 60 m to cover tall tropical/boreal forests
_FHD_BINS = np.arange(0.0, _FHD_MAX_H + _FHD_BIN_SIZE, _FHD_BIN_SIZE)
_FHD_N_BINS: int = len(_FHD_BINS) - 1  # number of 1 m bands
_VCI_MAX_ENTROPY: float = np.log(_FHD_N_BINS)


def _fhd_from_hag(hag_vals: np.ndarray) -> float:
    """Return the foliage height diversity (Shannon entropy of the height distribution)."""
    if len(hag_vals) == 0:
        return np.nan
    counts, _ = np.histogram(hag_vals, bins=_FHD_BINS)
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        return np.nan
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def _vci_from_hag(hag_vals: np.ndarray) -> float:
    """Return the vegetation complexity index (FHD normalised by its maximum); it grows with canopy height."""
    fhd = _fhd_from_hag(hag_vals)
    if np.isnan(fhd) or _VCI_MAX_ENTROPY == 0:
        return np.nan
    return float(fhd / _VCI_MAX_ENTROPY)


# ---------------------------------------------------------------------------
# Per-cell metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-cell structural metrics over *bbox*.

    Parameters
    ----------
    min_density:
        Cells with fewer points per m2 get NaN; 0 disables the check.
    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``); check the
        classes present in your survey.

    Returns a dict ``{name: (ny, nx) float32 array}``, north-up, NaN where empty.
    """
    from alsdb.methods._grid import GridSpec

    requested = tuple(_METRIC_NAMES if metrics is None else dict.fromkeys(metrics))
    unknown = set(requested) - set(_METRIC_NAMES)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    grid = GridSpec.from_bbox(bbox, resolution)
    inside, cells = grid.point_bins(points)
    n = grid.nx * grid.ny
    density = np.bincount(cells, minlength=n) / resolution**2
    hag = points["HeightAboveGround"][inside]
    valid = np.isfinite(hag)
    vegetation = valid & np.isin(points["Classification"][inside], veg_classes) & (hag > 0)
    cv, hv = cells[vegetation], hag[vegetation].astype(np.float64)
    nv = np.bincount(cv, minlength=n)
    out = {}
    if "density" in requested:
        out["density"] = density
    if "cc" in requested:
        cls = points["Classification"][inside]
        canopy_class = np.isin(cls, veg_classes)
        first = valid & (points["ReturnNumber"][inside] == 1) & (canopy_class | (cls == 2))
        denominator = np.bincount(cells[first], minlength=n)
        canopy = canopy_class[first] & (hag[first] > cc_threshold)
        numerator = np.bincount(cells[first], weights=canopy, minlength=n)
        out["cc"] = np.divide(numerator, denominator, out=np.full(n, np.nan), where=denominator > 0)
    if {"hmean", "crr"} & set(requested):
        mean = np.divide(
            np.bincount(cv, weights=hv, minlength=n), nv, out=np.full(n, np.nan), where=nv > 0
        )
        out["hmean"] = mean
    for name, (lo, hi) in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA):
        if name in requested:
            counts = np.bincount(cv, weights=(hv > lo) & (hv <= hi), minlength=n)
            out[name] = np.divide(counts, nv, out=np.full(n, np.nan), where=nv > 0)
    if {"h50", "h75", "h95", "hmax", "crr"} & set(requested):
        order = np.lexsort((hv, cv))
        heights = hv[order]
        occupied, starts, counts = np.unique(cv[order], return_index=True, return_counts=True)
        for name, q in (("h50", 0.5), ("h75", 0.75), ("h95", 0.95), ("hmax", 1.0)):
            if name in requested or (name == "hmax" and "crr" in requested):
                rank = q * (counts - 1)
                low = np.floor(rank).astype(int)
                high = np.ceil(rank).astype(int)
                values = np.full(n, np.nan)
                values[occupied] = heights[starts + low] + (rank - low) * (
                    heights[starts + high] - heights[starts + low]
                )
                out[name] = values
        if "crr" in requested:
            minimum = np.full(n, np.nan)
            minimum[occupied] = heights[starts]
            spread = out["hmax"] - minimum
            out["crr"] = np.divide(mean - minimum, spread, out=np.full(n, np.nan), where=spread > 0)
    if {"fhd", "vci"} & set(requested):
        # Sparse occupied (cell, height-band) counts avoid a cells × bands cube.
        keep = hv <= _FHD_MAX_H
        bands = np.minimum((hv[keep] / _FHD_BIN_SIZE).astype(int), _FHD_N_BINS - 1)
        keys, counts = np.unique(cv[keep] * _FHD_N_BINS + bands, return_counts=True)
        cell = keys // _FHD_N_BINS
        totals = np.bincount(cell, weights=counts, minlength=n)
        probabilities = counts / totals[cell]
        entropy = np.bincount(
            cell, weights=-probabilities * np.log(probabilities), minlength=n
        ).astype(float)
        entropy[totals == 0] = np.nan
        out["fhd"] = entropy
        out["vci"] = entropy / _VCI_MAX_ENTROPY
    result = {}
    for name in requested:
        values = out[name].astype(np.float32).reshape(grid.shape)
        if min_density > 0:
            values[density.reshape(grid.shape) < min_density] = np.nan
        result[name] = values
    return result


# ---------------------------------------------------------------------------
# BABA (Buffered Area-Based Approach) metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics_baba(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    baba_radius: float,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-cell metrics from all points within *baba_radius* of each cell centre.

    Buffered area-based approach: robust at fine resolutions. The points must
    extend at least *baba_radius* beyond *bbox*. Metrics match
    :func:`_extract_metrics` applied to each disc.

    Parameters
    ----------
    min_density:
        Neighbourhoods with fewer points per m2 get NaN.
    """
    from alsdb.methods._grid import GridSpec

    requested = tuple(_METRIC_NAMES if metrics is None else dict.fromkeys(metrics))
    if set(requested) - set(_METRIC_NAMES):
        raise ValueError("Unknown structural metric")
    requested_set = set(requested)
    grid = GridSpec.from_bbox(bbox, resolution)
    centre_x, centre_y = (c.ravel() for c in grid.centers())  # north-up order
    area = np.pi * baba_radius**2
    n = len(centre_x)
    out = {name: np.full(n, np.nan, np.float32) for name in requested}
    if not len(points):
        return {name: values.reshape(grid.shape) for name, values in out.items()}
    quantiles = tuple(
        (name, q)
        for name, q in (("h50", 0.5), ("h75", 0.75), ("h95", 0.95))
        if name in requested_set
    )
    strata = tuple(
        (name, bounds)
        for name, bounds in zip(_HEIGHT_STRATA_NAMES, _HEIGHT_STRATA)
        if name in requested_set
    )
    needs_entropy = bool({"fhd", "vci"} & requested_set)
    needs_heights = bool(requested_set - {"density", "cc"})
    hag = points["HeightAboveGround"].astype(np.float64)
    finite = np.isfinite(hag)
    cls = points["Classification"]
    canopy_class = np.isin(cls, veg_classes)
    veg = finite & (hag > 0) & canopy_class
    first = finite & (points["ReturnNumber"] == 1) & (canopy_class | (cls == 2))
    canopy = first & canopy_class & (hag > cc_threshold)
    x, y = points["X"].astype(np.float64), points["Y"].astype(np.float64)
    search = PointGrid(x, y, (x.min(), y.min()), baba_radius)
    for cell in range(n):
        idx = search.disc(centre_x[cell], centre_y[cell], baba_radius)
        density = len(idx) / area
        if density < min_density:
            continue
        if "density" in out:
            out["density"][cell] = density
        if not len(idx):
            continue
        if "cc" in out:
            n_first = np.count_nonzero(first[idx])
            if n_first:
                out["cc"][cell] = np.count_nonzero(canopy[idx]) / n_first
        if not needs_heights:
            continue
        hv = hag[idx[veg[idx]]]
        m = len(hv)
        if not m:
            continue
        # One sort serves percentiles, extremes, strata and the height histogram.
        hs = np.sort(hv)
        for name, q in quantiles:
            rank = q * (m - 1)
            low = int(np.floor(rank))
            t = rank - low
            a, b = hs[low], hs[min(low + 1, m - 1)]
            # numpy's linear interpolation, so values equal np.percentile's
            out[name][cell] = b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t
        if {"hmean", "crr"} & requested_set:
            height_mean = hv.mean()
            if "hmean" in out:
                out["hmean"][cell] = height_mean
        if "hmax" in out:
            out["hmax"][cell] = hs[-1]
        if "crr" in out and hs[-1] > hs[0]:
            out["crr"][cell] = (height_mean - hs[0]) / (hs[-1] - hs[0])
        if needs_entropy:
            # np.histogram's bins: half-open, the last one closed
            edges = np.searchsorted(hs, _FHD_BINS, side="left")
            edges[-1] = np.searchsorted(hs, _FHD_BINS[-1], side="right")
            counts = np.diff(edges)
            c = counts[counts > 0].astype(np.float64)
            share = c / c.sum()
            entropy = float(-np.sum(share * np.log(share))) if c.size else np.nan
            if "fhd" in out:
                out["fhd"][cell] = entropy
            if "vci" in out:
                out["vci"][cell] = entropy / _VCI_MAX_ENTROPY
        for name, (lo, hi) in strata:
            count = np.searchsorted(hs, hi, side="right") - np.searchsorted(hs, lo, side="right")
            out[name][cell] = count / m
    return {name: values.reshape(grid.shape) for name, values in out.items()}


# ---------------------------------------------------------------------------
# Per-tile workers
# ---------------------------------------------------------------------------


def compute_metrics(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 10.0,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
    baba_radius: float = 0.0,
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    parallel: str = "threads",
    metrics: tuple[str, ...] | None = None,
    ground_outlier_removal: bool = True,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = True,
) -> None:
    """Compute structural metrics and write them to *store*.

    Writes h50, h75, h95, hmax, hmean, cc, density, fhd, vci, crr and the
    height-stratum proportions pv_0_2 ... pv_above40.

    Parameters
    ----------
    provider:
        Point cloud provider.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Cell size (m).
    bbox:
        Optional ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year.
    cc_threshold:
        Height (m) defining canopy for cover.
    min_density:
        Cells with fewer points per m2 get NaN; 0 disables the check.
    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``); check the
        classes present in your survey.
    tile_size:
        Processing tile size (m).
    tile_buffer:
        Buffer for terrain interpolation (m).
    n_workers:
        Parallel workers.
    parallel:
        ``"threads"`` (default) or ``"processes"``: worker processes use one
        core each (threads level off at two to three times one worker) and
        hold one tile in memory each; scripts need an
        ``if __name__ == "__main__":`` guard.
    ground_extrapolation:
        Give returns outside the ground TIN the height above the nearest
        ground (default, as :mod:`alsdb.products.profiles`); off, they get
        no height and are left out of every metric.
    """
    from alsdb.products._area import compute_area_products

    compute_area_products(
        provider,
        store,
        resolution,
        bbox,
        year,
        cc_threshold=cc_threshold,
        baba_radius=baba_radius,
        min_density=min_density,
        veg_classes=veg_classes,
        overwrite=overwrite,
        tile_size=tile_size,
        tile_buffer=tile_buffer,
        n_workers=n_workers,
        parallel=parallel,
        ground_outlier_removal=ground_outlier_removal,
        max_ground_distance=max_ground_distance,
        ground_extrapolation=ground_extrapolation,
        metrics=metrics,
    )


def compute_biomass(
    provider: TileDBProvider,
    store: ALSZarrStore,
    resolution: float = 10.0,
    model_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    year: int | None = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    *,
    baba_radius: float = 0.0,
    overwrite: bool = False,
    tile_size: float = 500.0,
    tile_buffer: float = 50.0,
    n_workers: int = 1,
    parallel: str = "threads",
    min_density: float = 0.0,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    model_features: tuple[str, ...] | None = None,
    model_id: str | None = None,
    ground_outlier_removal: bool = True,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = True,
) -> None:
    """Estimate above-ground biomass and write it to *store*.

    Parameters
    ----------
    provider:
        Point cloud provider.
    store:
        :class:`~alsdb.storage.ALSZarrStore` target.
    resolution:
        Cell size (m).
    model_fn:
        ``model_fn(metrics) -> AGB grid`` (Mg/ha); default
        :func:`~alsdb.products.allometry.naesset_model`.
    bbox:
        Optional ``(min_x, min_y, max_x, max_y)``.
    year:
        Survey year.
    cc_threshold:
        Height (m) defining canopy for cover.
    min_density:
        Cells with fewer points per m2 get NaN.
    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``); check the
        classes present in your survey.
    overwrite:
        Recompute when biomass for *year* exists.
    tile_size:
        Processing tile size (m).
    tile_buffer:
        Buffer for terrain interpolation (m).
    n_workers:
        Parallel workers.
    parallel:
        ``"threads"`` (default) or ``"processes"``: worker processes use one
        core each (threads level off at two to three times one worker) and
        hold one tile in memory each; scripts need an
        ``if __name__ == "__main__":`` guard.
    ground_extrapolation:
        As :func:`compute_metrics`.
    """
    from alsdb.products._area import compute_area_products

    compute_area_products(
        provider,
        store,
        resolution,
        bbox,
        year,
        cc_threshold=cc_threshold,
        baba_radius=baba_radius,
        min_density=min_density,
        veg_classes=veg_classes,
        overwrite=overwrite,
        tile_size=tile_size,
        tile_buffer=tile_buffer,
        n_workers=n_workers,
        parallel=parallel,
        ground_outlier_removal=ground_outlier_removal,
        max_ground_distance=max_ground_distance,
        ground_extrapolation=ground_extrapolation,
        metrics=(),
        biomass=True,
        model_fn=model_fn,
        model_features=model_features,
        model_id=model_id,
    )
