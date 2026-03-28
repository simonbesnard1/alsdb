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
from typing import Optional

logger = logging.getLogger(__name__)


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
) -> None:
    """
    Merge *tile_paths* into *output_path* using ``rasterio.merge``.

    Source files are deleted after a successful write.
    """
    import rasterio
    from rasterio.merge import merge as rasterio_merge

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rasterio_merge(sources, nodata=nodata)
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
