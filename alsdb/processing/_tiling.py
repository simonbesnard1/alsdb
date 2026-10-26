# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Shared tiling utilities for large-area raster processing.

Used by :mod:`alsdb.processing.chm`, :mod:`alsdb.processing.gap`,
and :mod:`alsdb.processing.biomass`.
"""

from __future__ import annotations

import gc
import json
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

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

# LAS classification codes treated as "vegetation".
VEG_CLASSES = (3, 4, 5)


def query_to_array(
    provider: TileDBProvider,
    bbox: tuple[float, float, float, float] | None,
    year: int | None = None,
    attributes: tuple[str, ...] | None = None,
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
        Structured numpy array with X, Y and the selected LAS attribute fields.
        ``attributes=None`` preserves the default of reading all attributes.
    """
    with provider.open("r") as arr:
        attrs = list(LAS_ATTRIBUTES if attributes is None else dict.fromkeys(attributes))
        unknown = set(attrs) - LAS_ATTRIBUTES.keys()
        if unknown:
            raise ValueError(f"Unknown LAS attributes: {sorted(unknown)}")
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

    fields = ["X", "Y", *attrs]
    dtype = [(name, PDAL_DTYPES[name]) for name in fields]
    out = np.empty(n, dtype=dtype)
    for name in fields:
        np_dtype = PDAL_DTYPES[name]
        if name in data:
            out[name] = data[name].astype(np_dtype)
        else:
            out[name] = np.zeros(n, dtype=np_dtype)
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


def _require_year(year) -> None:
    """Raise a clear ValueError when *year* is None in a processing function."""
    if year is None:
        raise ValueError(
            "year must be an integer survey year (e.g. year=2021), not None. "
            "Call provider.available_years() to see what years are stored."
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


_HAG_DELAUNAY_MIN_GND: int = 3  # minimum ground points to build a Delaunay TIN


def _filter_ground_outliers(arr: np.ndarray) -> np.ndarray:
    """
    Reclassify low ground outliers to Class 7 with PDAL ``filters.elm``.

    Applied **only to Class-2 (ground) points**. ELM (Extended Local Minimum,
    10 m cells, 1 m threshold) flags the lowest returns of a cell while they
    sit >= 1 m below the next one: multipath/water returns and other
    below-ground noise, which pull the TIN down and inflate heights above it.
    Within-cell spacing also grows with slope, so on clean synthetic ground
    at 0.25 pts/m2 it flags 0.08% of points at 35 degrees and 0.4% at 45.

    There used to be a second, ``filters.outlier`` statistical stage. It flags
    *isolated* points (3-D mean neighbour distance), not points off the
    surface: on Barcelona windows it removed 2-5% of ground, on flat land 93%
    of those within 0.5 m of a local plane, and on steep slopes it moved the
    DTM by > 0.5 m on 9-23% of 1 m cells. Like lidR and LAStools, high points
    in Class 2 are now trusted to the classification.

    Points are reclassified in place in a copy (order preserved), so callers'
    index correspondence holds. Returns *arr* unchanged when nothing is
    flagged or the PDAL pipeline fails (logged at DEBUG level).
    """
    gnd_mask = arr["Classification"] == 2
    n_gnd = int(gnd_mask.sum())
    if n_gnd < 4:
        return arr

    gnd_arr = arr[gnd_mask].copy()
    stages = [{"type": "filters.elm"}]

    try:
        p = pdal.Pipeline(json.dumps(stages), arrays=[gnd_arr])
        p.execute()
        filtered_gnd = p.arrays[0] if p.arrays else gnd_arr
    except RuntimeError as exc:
        logger.debug(
            "_filter_ground_outliers: PDAL pipeline failed (%s) — skipping outlier removal", exc
        )
        return arr

    # Points reclassified away from Class 2 (→ Class 7 noise) are outliers.
    # PDAL preserves point order for in-place filters, so index correspondence holds.
    outlier_in_gnd = filtered_gnd["Classification"] != 2
    n_out = int(outlier_in_gnd.sum())

    if n_out == 0:
        return arr

    out = arr.copy()
    gnd_indices = np.where(gnd_mask)[0]
    out["Classification"][gnd_indices[outlier_in_gnd]] = 7  # noise, never canopy
    logger.debug(
        "_filter_ground_outliers: reclassified %d/%d low ground outliers (ELM)",
        n_out,
        n_gnd,
    )
    return out


def _hag_stage(arr: np.ndarray) -> dict:
    """
    Return the most accurate available PDAL HAG stage for *arr*.

    Legacy PDAL-based workers prefer ``filters.hag_delaunay`` when the
    tile contains at least
    ``_HAG_DELAUNAY_MIN_GND`` ground-classified points.  Falls back to
    ``filters.hag_nn`` when the ground point count is too low to build a
    valid triangulation (sparse surveys, dense-canopy void tiles).

    Parameters
    ----------
    arr:
        Point array as returned by :func:`query_to_array`.  Must contain
        a ``Classification`` field.

    Returns
    -------
    dict
        A single PDAL stage descriptor ready for inclusion in a pipeline list.
    """
    n_gnd = int((arr["Classification"] == 2).sum())
    if n_gnd >= _HAG_DELAUNAY_MIN_GND:
        return {"type": "filters.hag_delaunay"}
    logger.debug("_hag_stage: only %d ground points — falling back to filters.hag_nn", n_gnd)
    return {"type": "filters.hag_nn", "count": max(1, n_gnd), "allow_extrapolation": True}


def attach_hag(arr: np.ndarray) -> np.ndarray:
    """Normalize against a ground TIN; unsupported heights remain NaN.

    Point-normalized CHM statistics do not in general equal DSM minus DTM.
    Additional fields record ground distance, extrapolation and negative HAG.
    """
    from alsdb.processing._terrain import normalize_points

    return normalize_points(_filter_ground_outliers(arr))[0]


def run_tiled(
    worker_fn: Callable,
    provider,
    tiles,
    store,
    n_workers: int,
    progress_every: int = 500,
    gc_every: int = 100,
    skip: set[int] | frozenset[int] = frozenset(),
    **kwargs,
) -> None:
    """
    Run *worker_fn* over all *tiles*, sequentially or in a thread pool.

    The worker signature must be::

        worker_fn(provider, query_bbox, crop_bbox, store, tile_index, **kwargs)

    Logs "N/total tiles completed" every *progress_every* completions (and
    once at the end) - a crash that kills the process before it can log
    anything else still leaves behind how far tile completion actually got,
    which memory usage alone can't tell you (e.g. distinguishing "tiles are
    completing steadily and it's genuinely just a lot of data" from "nothing
    has finished in the last N minutes").

    Also forces a ``gc.collect()`` every *gc_every* completions. PDAL/GDAL
    point/array objects form reference cycles that Python's generational GC
    only reclaims once its own allocation-count thresholds trigger - on a
    long, allocation-heavy run those thresholds fire far too rarely, letting
    cycles pile up for a long time before eventually being collected. This
    reads identically to a real memory leak from `top`/RSS alone, but a
    direct A/B reproduction (same pipeline, same real tiles) showed RSS
    staying flat for thousands of tiles with periodic forced collection,
    versus climbing into the hundreds of GB without it - so this isn't
    optional cleanup, it's the actual fix for that failure mode.

    Tile indices in *skip* (already written by an interrupted run being
    resumed) are not processed but keep their original index, so a worker's
    ``tile_index`` always identifies the same tile across attempts.

    Tiles are consumed as they finish, not in submission order: with ordered
    consumption one slow (dense) tile idles every other worker once the
    bounded window behind it fills - measured 3x slower on a synthetic mix
    with 1 in 25 tiles slow.

    Shared by :mod:`alsdb.processing.chm`, :mod:`alsdb.processing.gap`,
    and :mod:`alsdb.processing.biomass`.
    """
    total = len(tiles)
    completed = 0
    if skip:
        completed = sum(1 for i in range(total) if i in skip)
        logger.info("run_tiled: resuming, %d/%d tiles already complete", completed, total)

    def _note_progress() -> None:
        nonlocal completed
        completed += 1
        if completed % progress_every == 0 or completed == total:
            logger.info("run_tiled: %d/%d tiles completed", completed, total)
        if completed % gc_every == 0:
            gc.collect()

    def process(item):
        idx, (query_bbox, crop_bbox) = item
        return worker_fn(provider, query_bbox, crop_bbox, store, idx, **kwargs)

    todo = (item for item in enumerate(tiles) if item[0] not in skip)
    for _ in bounded_map_unordered(process, todo, n_workers):
        _note_progress()


def bounded_map_unordered(fn, items, n_workers=1):
    """Like :func:`bounded_map` but yields results as soon as each finishes.

    Keeps at most ``2 * n_workers`` items in flight and refills from whichever
    task completes first, so a slow item never blocks the rest of the pool.
    Use when result order doesn't matter.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if n_workers == 1:
        yield from map(fn, items)
        return
    items = iter(items)
    sentinel = object()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending = set()
        for item in items:
            pending.add(executor.submit(fn, item))
            if len(pending) >= 2 * n_workers:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()  # re-raise worker exceptions
                item = next(items, sentinel)
                if item is not sentinel:
                    pending.add(executor.submit(fn, item))
                yield result


def bounded_map(fn, items, n_workers=1):
    """Ordered results with at most two pending items per worker."""
    from collections import deque

    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if n_workers == 1:
        yield from map(fn, items)
        return
    items = iter(items)
    sentinel = object()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending = deque()
        for _ in range(2 * n_workers):
            item = next(items, sentinel)
            if item is sentinel:
                break
            pending.append(executor.submit(fn, item))
        while pending:
            yield pending.popleft().result()
            item = next(items, sentinel)
            if item is not sentinel:
                pending.append(executor.submit(fn, item))


def flip_to_north_up(grid: np.ndarray, transpose: bool = False) -> np.ndarray:
    """
    Convert a south-up raster to north-up float32 orientation.

    Set *transpose* when *grid* is shaped ``(nx, ny)`` (e.g. straight out of
    ``reshape(nx, ny)`` or ``binned_statistic_2d``); leave it ``False`` when
    *grid* is already ``(ny, nx)``.

    Shared by :mod:`alsdb.processing.chm`, :mod:`alsdb.processing.gap`,
    and :mod:`alsdb.processing.biomass`.
    """
    if transpose:
        grid = grid.T
    return np.flipud(grid).astype(np.float32)


def baba_neighbourhoods(
    points: np.ndarray,
    resolution: float,
    bbox: tuple[float, float, float, float],
    baba_radius: float,
) -> tuple[int, int, list, float]:
    """
    Build per-cell circular-neighbourhood point indices for the Buffered
    Area-Based Approach (BABA).

    Returns ``(nx, ny, indices_list, neighbourhood_area)`` where
    ``indices_list[row * nx + col]`` holds the indices into *points* that
    fall within *baba_radius* of that cell's centre.

    Shared by :mod:`alsdb.processing.gap` and :mod:`alsdb.processing.biomass`.
    """
    from scipy.spatial import cKDTree

    from alsdb.processing._grid import GridSpec

    grid = GridSpec.from_bbox(bbox, resolution)
    nx, ny = grid.nx, grid.ny
    cx, cy = grid.centers()
    # Preserve this helper's historical south-up index order for its callers.
    centres = np.column_stack((np.flipud(cx).ravel(), np.flipud(cy).ravel()))

    xy = np.column_stack([points["X"].astype(np.float64), points["Y"].astype(np.float64)])
    indices_list = cKDTree(xy).query_ball_point(centres, r=baba_radius)

    return nx, ny, indices_list, np.pi * baba_radius**2


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
      TileDB query so ``filters.hag_delaunay`` has enough ground points
      at tile edges to form complete edge triangles.
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
