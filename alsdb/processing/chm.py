# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Canopy Height Model (CHM) computation using TileDB + PDAL.

Pattern (inspired by silvimetric)
----------------------------------
Data is queried from TileDB programmatically via
:class:`~alsdb.core.alsprovider.ALSProvider`, then injected into a PDAL
pipeline as a numpy structured array using PDAL's Python API
(``pdal.Pipeline(json, arrays=[arr])``).  This avoids the fragile
``readers.tiledb`` config-file approach entirely — PDAL only needs to
apply filters and rasterize; TileDB handles all I/O.

Pipeline (inside PDAL)
-----------------------
numpy array input
    → ``filters.hag_delaunay``  builds a TIN from Class-2 ground points,
                                 attaches ``HeightAboveGround`` to every point
    → ``filters.range``         keeps vegetation points only (Class 3–5)
    → ``filters.assign``        clamps negative HAG values to 0
    → ``writers.gdal``          rasterizes max(HAG) per cell → GeoTIFF

Usage::

    from alsdb import ALSProvider
    from alsdb.processing.chm import compute_chm, compute_dtm, compute_dsm

    provider = ALSProvider(
        storage_type="s3",
        uri="s3://new-bucket-2f37f541/test",
        url="https://s3.gfz-potsdam.de",
        region="eu-central-1",
        credentials=credentials,
    )

    # Full tile CHM
    compute_chm(provider, "output/chm.tif", resolution=1.0)

    # Restrict to a bounding box
    compute_chm(provider, "output/chm_tile.tif", resolution=1.0,
                bbox=(308000, 4688000, 310000, 4690000))

    # All three products at once
    compute_all(provider, output_dir="output/", resolution=1.0,
                bbox=(308000, 4688000, 310000, 4690000))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pdal

from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)

_GROUND_CLASS = 2
_VEG_CLASSES = (3, 4, 5)

# Canonical PDAL dtype mapping for LAS dimensions
_PDAL_DTYPES: dict[str, type] = {
    "X": np.float64,
    "Y": np.float64,
    **LAS_ATTRIBUTES,
}


# ---------------------------------------------------------------------------
# TileDB → numpy structured array
# ---------------------------------------------------------------------------

def _query_to_array(
    provider: TileDBProvider,
    bbox: Optional[tuple[float, float, float, float]],
    year: Optional[int] = None,
) -> np.ndarray:
    """
    Query the TileDB array and return a PDAL-compatible numpy structured array.

    Parameters
    ----------
    provider:
        TileDB provider (ALSProvider or ALSDatabase).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
        Reads the full array if ``None``.
    year:
        Optional survey year filter.  ``None`` returns all years.

    Returns
    -------
    np.ndarray
        Structured numpy array with X, Y and all LAS attribute fields.
    """
    with provider.open("r") as arr:
        attrs = list(LAS_ATTRIBUTES.keys())
        yr_dim = arr.schema.domain.dim("Year")
        y0 = year if year is not None else int(yr_dim.domain[0])
        # +1: TileDB-Py int-dimension slices are exclusive-end (like Python slices)
        y1 = (year + 1) if year is not None else int(yr_dim.domain[1]) + 1
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            data = arr.query(attrs=attrs)[min_x:max_x, min_y:max_y, y0:y1]
        else:
            data = arr.query(attrs=attrs)[:, :, y0:y1]

    n = len(data["X"])
    logger.debug("Queried %d points from TileDB", n)

    dtype = [(name, _PDAL_DTYPES[name]) for name in _PDAL_DTYPES]
    out = np.empty(n, dtype=dtype)
    for name in _PDAL_DTYPES:
        out[name] = data[name].astype(_PDAL_DTYPES[name])
    return out


# ---------------------------------------------------------------------------
# PDAL pipeline builders
# ---------------------------------------------------------------------------

def _gdal_writer(
    output_path: str,
    resolution: float,
    dimension: str = "Z",
    output_type: str = "max",
    nodata: float = -9999.0,
) -> dict:
    return {
        "type": "writers.gdal",
        "filename": output_path,
        "dimension": dimension,
        "resolution": resolution,
        "output_type": output_type,
        "data_type": "float32",
        "nodata": nodata,
        "gdalopts": "COMPRESS=DEFLATE,PREDICTOR=3",
    }


def _run(stages: list, arr: np.ndarray) -> None:
    """Execute a PDAL pipeline with a numpy array as input."""
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    count = p.execute()
    logger.debug("PDAL executed: %d points processed", count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_chm(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
) -> Path:
    """
    Compute a Canopy Height Model from a TileDB array.

    Queries TileDB for all points (ground + vegetation), injects them into
    PDAL as a numpy array, applies ``filters.hag_delaunay`` to compute
    height above ground, then rasterizes the maximum HAG of vegetation
    points to a GeoTIFF.

    Parameters
    ----------
    provider:
        :class:`~alsdb.providers.tiledb_provider.TileDBProvider` instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres (default 1 m).
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)`` in the
        array's native CRS.  Reads the full array if ``None``.
    nodata:
        No-data fill value.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _query_to_array(provider, bbox)

    stages = [
        {"type": "filters.hag_delaunay"},
        {"type": "filters.range",
         "limits": f"Classification[{_VEG_CLASSES[0]}:{_VEG_CLASSES[-1]}]"},
        {"type": "filters.assign",
         "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
        _gdal_writer(str(output_path), resolution,
                     dimension="HeightAboveGround",
                     output_type="max", nodata=nodata),
    ]

    logger.info("Computing CHM → %s  (%.1f m resolution)", output_path, resolution)
    _run(stages, arr)
    return output_path


def compute_dtm(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
) -> Path:
    """
    Rasterize ground points (Class 2) to a DTM GeoTIFF.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _query_to_array(provider, bbox)

    stages = [
        {"type": "filters.range",
         "limits": f"Classification[{_GROUND_CLASS}:{_GROUND_CLASS}]"},
        _gdal_writer(str(output_path), resolution,
                     dimension="Z", output_type="max", nodata=nodata),
    ]

    logger.info("Computing DTM → %s  (%.1f m resolution)", output_path, resolution)
    _run(stages, arr)
    return output_path


def compute_dsm(
    provider: TileDBProvider,
    output_path: str | Path,
    resolution: float = 1.0,
    first_returns_only: bool = True,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
) -> Path:
    """
    Rasterize maximum return elevation to a DSM GeoTIFF.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_path:
        Output GeoTIFF path.
    resolution:
        Grid cell size in metres.
    first_returns_only:
        Use only first returns for a clean canopy-top signal.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _query_to_array(provider, bbox)

    stages: list = []
    if first_returns_only:
        stages.append({"type": "filters.range", "limits": "ReturnNumber[1:1]"})
    stages.append(
        _gdal_writer(str(output_path), resolution,
                     dimension="Z", output_type="max", nodata=nodata)
    )

    logger.info("Computing DSM → %s  (%.1f m resolution)", output_path, resolution)
    _run(stages, arr)
    return output_path


def compute_all(
    provider: TileDBProvider,
    output_dir: str | Path,
    resolution: float = 1.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    nodata: float = -9999.0,
) -> dict[str, Path]:
    """
    Compute DTM, DSM, and CHM in one call.

    The TileDB array is queried once per product.

    Parameters
    ----------
    provider:
        TileDB provider instance.
    output_dir:
        Output directory (created if it does not exist).
    resolution:
        Grid cell size in metres.
    bbox:
        Optional spatial filter ``(min_x, min_y, max_x, max_y)``.
    nodata:
        No-data fill value.

    Returns
    -------
    dict
        ``{"dtm": Path, "dsm": Path, "chm": Path}``
    """
    output_dir = Path(output_dir)
    return {
        "dtm": compute_dtm(provider, output_dir / "dtm.tif",
                           resolution=resolution, bbox=bbox, nodata=nodata),
        "dsm": compute_dsm(provider, output_dir / "dsm.tif",
                           resolution=resolution, bbox=bbox, nodata=nodata),
        "chm": compute_chm(provider, output_dir / "chm.tif",
                           resolution=resolution, bbox=bbox, nodata=nodata),
    }
