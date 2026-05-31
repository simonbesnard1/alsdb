# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Shared tiling utilities for large-area raster processing.

Used by :mod:`alsdb.processing.chm`, :mod:`alsdb.processing.gap`,
and :mod:`alsdb.processing.biomass`.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pdal

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
        if name in data:
            out[name] = data[name].astype(PDAL_DTYPES[name])
        else:
            out[name] = np.zeros(n, dtype=PDAL_DTYPES[name])
    return out


def array_domain_bbox(provider) -> tuple[float, float, float, float]:
    """
    Return the **declared** X/Y domain of the TileDB schema.

    This is the full extent the array *could* cover, not the extent of the
    data that has actually been written.  For processing use
    :func:`array_data_bbox` instead so you don't tile over empty space.

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


def array_crs(provider) -> str:
    """
    Return the CRS string stored in the TileDB array metadata.

    Returns an empty string if no CRS has been written (e.g. the array was
    created without one).
    """
    with provider.open("r") as arr:
        return arr.meta.get("crs", "")


def array_data_bbox(provider) -> tuple[float, float, float, float]:
    """
    Return the bounding box of **actually stored** data via ``nonempty_domain``.

    Unlike :func:`array_domain_bbox`, this reflects the real extent of points
    in the array, not the declared schema domain.  Use this as the default
    extent for processing so tiles are only generated where data exists.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, min_y, max_x, max_y)``

    Raises
    ------
    RuntimeError
        If the array is empty (no points ingested yet).
    """
    with provider.open("r") as arr:
        ned = arr.nonempty_domain()
    if ned is None:
        raise RuntimeError("TileDB array is empty — no points have been ingested yet.")
    # nonempty_domain() returns a tuple of (min, max) pairs in dimension order:
    # ned[0] = (min_x, max_x), ned[1] = (min_y, max_y), ned[2] = (min_year, max_year)
    return (
        float(ned[0][0]),
        float(ned[1][0]),
        float(ned[0][1]),
        float(ned[1][1]),
    )


def check_year_exists(year: int, provider) -> bool:
    """
    Return ``True`` if *year* falls within the ingested year range.

    Logs a warning and returns ``False`` when the year is outside the stored
    range, so callers can bail out early.
    """
    with provider.open("r") as arr:
        ned = arr.nonempty_domain()
    if ned is None:
        logger.warning("TileDB array is empty — nothing to process.")
        return False
    # ned[2] = (min_year, max_year) for the Year dimension
    y_min, y_max = int(ned[2][0]), int(ned[2][1])
    if year < y_min or year > y_max:
        logger.warning(
            "Requested year %d is outside the stored year range [%d, %d] "
            "— nothing will be computed.",
            year,
            y_min,
            y_max,
        )
        return False
    return True


def check_bbox_overlap(
    requested: tuple[float, float, float, float],
    provider,
) -> bool:
    """
    Return ``True`` if *requested* overlaps the stored data extent.

    Logs a warning and returns ``False`` when there is no overlap, so the
    caller can bail out early rather than processing tiles that will all be
    empty.

    Parameters
    ----------
    requested:
        ``(min_x, min_y, max_x, max_y)`` passed by the user.
    provider:
        TileDB provider; used to read ``nonempty_domain``.
    """
    try:
        data_bbox = array_data_bbox(provider)
    except RuntimeError:
        logger.warning("TileDB array is empty — nothing to process.")
        return False

    rx0, ry0, rx1, ry1 = requested
    dx0, dy0, dx1, dy1 = data_bbox

    if rx1 <= dx0 or rx0 >= dx1 or ry1 <= dy0 or ry0 >= dy1:
        logger.warning(
            "Requested bbox (%.0f, %.0f, %.0f, %.0f) does not overlap "
            "the stored data extent (%.0f, %.0f, %.0f, %.0f) — "
            "nothing will be computed.",
            rx0,
            ry0,
            rx1,
            ry1,
            dx0,
            dy0,
            dx1,
            dy1,
        )
        return False

    return True


def attach_hag(arr: np.ndarray) -> np.ndarray:
    """
    Run ``filters.hag_nn`` on *arr* and return the HAG-annotated array.

    Uses k-nearest ground points (kd-tree, non-recursive) to interpolate
    height above ground for every point.  Negative HAG values (artefacts
    at tile edges) are clamped to zero.

    Shared by :mod:`alsdb.processing.chm`, :mod:`alsdb.processing.gap`,
    and :mod:`alsdb.processing.biomass`.
    """
    stages = [
        {"type": "filters.hag_nn", "count": 10, "allow_extrapolation": True},
        {
            "type": "filters.assign",
            "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
        },
    ]
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    result = p.arrays[0]
    logger.debug("HAG attached: %d points", len(result))
    return result


def run_tiled(worker_fn: Callable, provider, tiles, store, n_workers: int, **kwargs) -> None:
    """
    Run *worker_fn* over all *tiles*, sequentially or in a thread pool.

    The worker signature must be::

        worker_fn(provider, query_bbox, crop_bbox, store, tile_index, **kwargs)

    Shared by :mod:`alsdb.processing.chm`, :mod:`alsdb.processing.gap`,
    and :mod:`alsdb.processing.biomass`.
    """
    if n_workers == 1:
        for idx, (query_bbox, crop_bbox) in enumerate(tiles):
            worker_fn(provider, query_bbox, crop_bbox, store, idx, **kwargs)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    worker_fn, provider, query_bbox, crop_bbox, store, idx, **kwargs
                ): idx
                for idx, (query_bbox, crop_bbox) in enumerate(tiles)
            }
            for future in as_completed(futures):
                future.result()  # re-raise worker exceptions


def tile_bboxes(
    bbox: tuple[float, float, float, float],
    tile_size: float,
    buffer: float,
) -> list[
    tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
]:
    """
    Partition *bbox* into a grid of sub-tiles.

    Returns a list of ``(query_bbox, crop_bbox)`` pairs:

    * ``query_bbox`` — inflated by *buffer* on all sides; used for the
      TileDB query so ``filters.hag_nn`` has enough ground points at
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

            crop_bbox = (cx0, cy0, cx1, cy1)
            query_bbox = (cx0 - buffer, cy0 - buffer, cx1 + buffer, cy1 + buffer)
            tiles.append((query_bbox, crop_bbox))

    return tiles
