# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Above-Ground Biomass (AGB) estimation from TileDB ALS point clouds.

Pipeline
--------
1. Query TileDB → numpy structured array (same as CHM).
2. Run ``filters.hag_delaunay`` via PDAL to attach ``HeightAboveGround`` to
   every point.  All points (ground + vegetation) are needed at this stage so
   the ground TIN is complete.
3. Compute per-cell LiDAR metrics in Python/scipy:

   ========  ===============================================================
   h50       50th-percentile HAG of vegetation points (m)
   h75       75th-percentile HAG of vegetation points (m)
   h95       95th-percentile HAG of vegetation points (m)
   hmean     Mean HAG of vegetation points (m)
   cc        Canopy cover — fraction of first returns with HAG > threshold
   density   Total point density (points m⁻²)
   ========  ===============================================================

4. Apply an allometric model ``AGB = f(metrics)`` → Mg ha⁻¹.
5. Write GeoTIFF via rasterio.

Default model
-------------
A Næsset-style power law::

    AGB = a × h95^b × cc^c

with default coefficients ``a=0.8, b=1.8, c=0.5``.  These are approximate
generic values — **calibrate against field inventory plots** for your region
and species composition before using the output scientifically.

Usage::

    from alsdb import ALSProvider
    from alsdb.processing.biomass import compute_biomass, compute_metrics

    provider = ALSProvider(storage_type="local", uri="array_")

    # All metrics as separate GeoTIFFs
    compute_metrics(provider, "output/metrics/", resolution=10.0)

    # AGB with default model
    compute_biomass(provider, "output/agb.tif", resolution=10.0)

    # AGB with a custom model
    def my_model(metrics):
        return 1.2 * metrics["h95"] ** 2.1 * metrics["cc"] ** 0.6

    compute_biomass(provider, "output/agb.tif", resolution=10.0,
                    model_fn=my_model,
                    bbox=(308000, 4688000, 310000, 4690000))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pdal

from alsdb.processing.chm import _query_to_array
from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

_VEG_CLASSES = (3, 4, 5)
_DEFAULT_CC_THRESHOLD = 2.0   # m — first returns above this count as "canopy"
_CRS = "EPSG:25830"


# ---------------------------------------------------------------------------
# Step 1 — get HAG-annotated point array from PDAL
# ---------------------------------------------------------------------------

def _attach_hag(provider: TileDBProvider,
                bbox: Optional[tuple[float, float, float, float]],
                year: Optional[int] = None) -> np.ndarray:
    """
    Query TileDB and return all points with ``HeightAboveGround`` attached.

    Ground points (Class 2) are kept so the Delaunay TIN is complete.
    Negative HAG values are clamped to 0.
    """
    arr = _query_to_array(provider, bbox, year=year)

    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
    ]
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    result = p.arrays[0]
    logger.debug("HAG attached: %d points", len(result))
    return result


# ---------------------------------------------------------------------------
# Step 2 — per-cell metric extraction
# ---------------------------------------------------------------------------

def _bin_edges(
    points: np.ndarray,
    resolution: float,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Return (x_edges, y_edges, x_min, x_max, y_min, y_max)."""
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
    else:
        x_min, x_max = float(points["X"].min()), float(points["X"].max())
        y_min, y_max = float(points["Y"].min()), float(points["Y"].max())

    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    return x_edges, y_edges, x_min, x_max, y_min, y_max


def _extract_metrics(
    points: np.ndarray,
    resolution: float,
    bbox: Optional[tuple[float, float, float, float]] = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
) -> tuple[dict[str, np.ndarray], list[float]]:
    """
    Compute per-cell LiDAR structural metrics.

    Returns
    -------
    metrics : dict
        ``{name: 2-D array (ny, nx)}`` — nodata cells are ``np.nan``.
    extent : [x_min, x_max, y_min, y_max]
        Spatial extent for rasterio / imshow.
    """
    from scipy.stats import binned_statistic_2d

    x_edges, y_edges, x_min, x_max, y_min, y_max = _bin_edges(points, resolution, bbox)
    bins = [x_edges, y_edges]

    x = points["X"]
    y = points["Y"]
    hag = points["HeightAboveGround"]

    # Vegetation mask
    veg = np.isin(points["Classification"], _VEG_CLASSES) & (hag > 0)
    x_v, y_v, hag_v = x[veg], y[veg], hag[veg]

    def _pct(p):
        def stat(v):
            return float(np.percentile(v, p)) if len(v) else np.nan
        return stat

    def _flip(g):
        """binned_statistic_2d → (nx, ny); convert to (ny, nx) north-up."""
        return np.flipud(np.where(np.isnan(g), np.nan, g).T)

    # Height percentiles and mean (vegetation only)
    h50 = _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(50), bins=bins).statistic)
    h75 = _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(75), bins=bins).statistic)
    h95 = _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic=_pct(95), bins=bins).statistic)
    hmean = _flip(binned_statistic_2d(x_v, y_v, hag_v, statistic="mean", bins=bins).statistic)

    # Point density — all points per m²
    cell_area = resolution ** 2
    n_all = binned_statistic_2d(x, y, hag, statistic="count", bins=bins).statistic
    density = _flip(n_all / cell_area)

    # Canopy cover — first returns above cc_threshold
    fr = points["ReturnNumber"] == 1
    x_fr, y_fr, hag_fr = x[fr], y[fr], hag[fr]
    above = (hag_fr > cc_threshold).astype(np.float32)
    n_fr = binned_statistic_2d(x_fr, y_fr, np.ones(fr.sum()), statistic="count", bins=bins).statistic
    n_above = binned_statistic_2d(x_fr, y_fr, above, statistic="sum", bins=bins).statistic
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = _flip(np.where(n_fr > 0, n_above / n_fr, np.nan))

    logger.debug(
        "Metrics extracted: grid %d×%d, h95 range %.1f–%.1f m, cc range %.2f–%.2f",
        h95.shape[1], h95.shape[0],
        float(np.nanmin(h95)), float(np.nanmax(h95)),
        float(np.nanmin(cc)), float(np.nanmax(cc)),
    )

    metrics = {
        "h50": h50, "h75": h75, "h95": h95,
        "hmean": hmean, "cc": cc, "density": density,
    }
    return metrics, [x_min, x_max, y_min, y_max]


# ---------------------------------------------------------------------------
# Step 3 — allometric model
# ---------------------------------------------------------------------------

def naesset_model(
    metrics: dict[str, np.ndarray],
    a: float = 0.8,
    b: float = 1.8,
    c: float = 0.5,
) -> np.ndarray:
    """
    Næsset-style power-law AGB model (Mg ha⁻¹).

    ``AGB = a × h95^b × cc^c``

    Parameters
    ----------
    metrics:
        Dict as returned by :func:`_extract_metrics`.
    a, b, c:
        Model coefficients.  Defaults are approximate generic temperate-forest
        values — **calibrate against field plots** for production use.

    Returns
    -------
    np.ndarray
        AGB grid in Mg ha⁻¹; cells with no canopy (``cc == 0``) → 0.
    """
    h95 = metrics["h95"]
    cc = metrics["cc"]
    with np.errstate(invalid="ignore"):
        agb = np.where(
            np.isnan(h95) | np.isnan(cc) | (cc == 0),
            np.nan,
            a * np.power(np.where(h95 > 0, h95, 0), b) * np.power(cc, c),
        )
    return agb.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 — rasterio writer
# ---------------------------------------------------------------------------

def _write_raster(
    grid: np.ndarray,
    path: Path,
    extent: list[float],
    nodata: float,
    crs: str = _CRS,
) -> Path:
    """Write a single-band float32 GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds

    ny, nx = grid.shape
    x_min, x_max, y_min, y_max = extent
    transform = from_bounds(x_min, y_min, x_max, y_max, nx, ny)

    out = np.where(np.isnan(grid), nodata, grid).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=ny, width=nx,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(out, 1)

    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    provider: TileDBProvider,
    output_dir: str | Path,
    resolution: float = 10.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    nodata: float = -9999.0,
) -> dict[str, Path]:
    """
    Compute and save all LiDAR structural metrics as individual GeoTIFFs.

    Useful for model calibration, visual inspection, and downstream analysis.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_dir:
        Output directory.
    resolution:
        Grid cell size in metres (10–25 m typical for biomass).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    cc_threshold:
        HAG threshold (m) used to define "canopy" for the cover metric.
    nodata:
        No-data fill value.

    Returns
    -------
    dict
        ``{"h50": Path, "h75": Path, "h95": Path, "hmean": Path,
           "cc": Path, "density": Path}``
    """
    output_dir = Path(output_dir)
    logger.info("Extracting LiDAR metrics → %s  (%.0f m resolution)", output_dir, resolution)

    points = _attach_hag(provider, bbox, year=year)
    metrics, extent = _extract_metrics(points, resolution, bbox=bbox,
                                       cc_threshold=cc_threshold)

    paths = {}
    for name, grid in metrics.items():
        p = _write_raster(grid, output_dir / f"{name}.tif", extent, nodata)
        paths[name] = p
        logger.info("  wrote %s → %s", name, p)

    return paths


def compute_biomass(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 10.0,
    model_fn: Optional[Callable[[dict[str, np.ndarray]], np.ndarray]] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    year: Optional[int] = None,
    cc_threshold: float = _DEFAULT_CC_THRESHOLD,
    nodata: float = -9999.0,
) -> Path:
    """
    Estimate Above-Ground Biomass (AGB) and write a GeoTIFF.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres.
    model_fn:
        Callable ``model_fn(metrics) → np.ndarray`` mapping the metric dict
        to an AGB grid (Mg ha⁻¹).  Defaults to :func:`naesset_model` with
        generic coefficients — calibrate for your site before use.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    cc_threshold:
        HAG threshold (m) for the canopy cover metric.
    nodata:
        No-data fill value.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    model_fn = model_fn or naesset_model

    logger.info("Computing AGB → %s  (%.0f m resolution)", output_path, resolution)

    points = _attach_hag(provider, bbox, year=year)
    metrics, extent = _extract_metrics(points, resolution, bbox=bbox,
                                       cc_threshold=cc_threshold)
    agb = model_fn(metrics)

    logger.info(
        "AGB range: %.1f – %.1f Mg/ha  (mean %.1f)",
        float(np.nanmin(agb)), float(np.nanmax(agb)), float(np.nanmean(agb)),
    )

    _write_raster(agb, output_path, extent, nodata)
    return output_path
