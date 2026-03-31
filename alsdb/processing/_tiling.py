# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Shared tiling utilities for large-area raster processing.

Used by :mod:`alsdb.processing.chm` and :mod:`alsdb.processing.biomass`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from alsdb.utils.schema import LAS_ATTRIBUTES

if TYPE_CHECKING:
    from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

# Canonical PDAL dtype mapping shared by chm.py and biomass.py
PDAL_DTYPES: dict[str, type] = {
    "X": np.float64,
    "Y": np.float64,
    **LAS_ATTRIBUTES,
}


def query_to_array(
    provider: "TileDBProvider",
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
        y1 = (year + 1) if year is not None else int(yr_dim.domain[1]) + 1
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            data = arr.query(attrs=attrs)[min_x:max_x, min_y:max_y, y0:y1]
        else:
            data = arr.query(attrs=attrs)[:, :, y0:y1]

    n = len(data["X"])
    logger.debug("Queried %d points from TileDB", n)

    dtype = [(name, PDAL_DTYPES[name]) for name in PDAL_DTYPES]
    out = np.empty(n, dtype=dtype)
    for name in PDAL_DTYPES:
        out[name] = data[name].astype(PDAL_DTYPES[name])
    return out


def array_domain_bbox(provider) -> tuple[float, float, float, float]:
    """
    Derive a bounding box from the X/Y dimension domains of the TileDB array.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, min_y, max_x, max_y)``
    """
    dom = provider.schema.domain
    x_dim = dom.dim("X")
    y_dim = dom.dim("Y")
    return (
        float(x_dim.domain[0]),
        float(y_dim.domain[0]),
        float(x_dim.domain[1]),
        float(y_dim.domain[1]),
    )


def tile_bboxes(
    bbox: tuple[float, float, float, float],
    tile_size: float,
    buffer: float,
) -> list[tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]]:
    """
    Partition *bbox* into a grid of sub-tiles.

    Returns a list of ``(query_bbox, crop_bbox)`` pairs:

    * ``query_bbox`` — inflated by *buffer* on all sides; used for the
      TileDB query so ``filters.hag_delaunay`` has enough ground points at
      tile edges.
    * ``crop_bbox``  — the actual non-overlapping tile extent; used to
      restrict output to avoid duplicate pixels in the mosaic.

    Parameters
    ----------
    bbox:
        Full area ``(min_x, min_y, max_x, max_y)``.
    tile_size:
        Sub-tile width and height in metres.
    buffer:
        Overlap buffer in metres added to the query bbox on each side.
        Set to ``0.0`` when no TIN-based filter is used (DTM, DSM).
    """
    min_x, min_y, max_x, max_y = bbox
    cols = math.ceil((max_x - min_x) / tile_size)
    rows = math.ceil((max_y - min_y) / tile_size)

    tiles = []
    for row in range(rows):
        for col in range(cols):
            cx0 = min_x + col * tile_size
            cy0 = min_y + row * tile_size
            cx1 = min_x + (col + 1) * tile_size if col < cols - 1 else max_x
            cy1 = min_y + (row + 1) * tile_size if row < rows - 1 else max_y

            crop_bbox  = (cx0, cy0, cx1, cy1)
            query_bbox = (cx0 - buffer, cy0 - buffer, cx1 + buffer, cy1 + buffer)
            tiles.append((query_bbox, crop_bbox))

    return tiles


def mosaic_tiles(
    tile_paths: list[Path],
    output_path: Path,
    nodata: float,
    bounds: Optional[tuple[float, float, float, float]] = None,
) -> None:
    """
    Merge *tile_paths* into *output_path* using ``rasterio.merge``.

    Parameters
    ----------
    tile_paths:
        List of GeoTIFF paths to merge.
    output_path:
        Destination GeoTIFF path.
    nodata:
        No-data value.
    bounds:
        Optional ``(min_x, min_y, max_x, max_y)`` to clip the merged output.
        When provided, the output is exactly this extent (padded with nodata if
        needed).  Pass *effective_bbox* here to avoid the merged raster
        extending beyond the requested area due to floating-point tile alignment.

    Source files are deleted after a successful write.
    """
    import rasterio
    from rasterio.merge import merge as rasterio_merge

    merge_kwargs: dict = {"nodata": nodata}
    if bounds is not None:
        merge_kwargs["bounds"] = bounds

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rasterio_merge(sources, **merge_kwargs)
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=nodata,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in sources:
            src.close()
        for p in tile_paths:
            try:
                p.unlink()
            except OSError:
                logger.warning("Could not delete temp tile %s", p)
