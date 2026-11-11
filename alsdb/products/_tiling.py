# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tiling, querying and parallel execution shared by the products."""

from __future__ import annotations

import gc
import json
import logging
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
        if name in data:
            out[name] = data[name]  # assignment casts to the field type; no astype copy
        else:
            out[name] = 0  # attribute absent from the array: zeros
    return out


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

    This is the real extent of the points in the array, not the declared
    schema domain.  Use it as the default extent for processing so tiles are
    only generated where data exists.

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
    """Return True if *requested* overlaps the stored data extent.

    Logs a warning and returns False otherwise.

    Parameters
    ----------
    requested:
        ``(min_x, min_y, max_x, max_y)``.
    provider:
        TileDB provider.
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


def _filter_ground_outliers(arr: np.ndarray) -> np.ndarray:
    """Reclassify low ground outliers as class 7 with PDAL ``filters.elm``.

    Only ground (class 2) is checked: ELM (10 m cells, 1 m threshold) flags the
    lowest returns of a cell lying at least 1 m below the next ones, such as
    multipath returns. High ground points are trusted to the classification,
    as in lidR and LAStools. Returns a reclassified copy (order preserved), or
    *arr* unchanged when nothing is flagged or PDAL fails.
    """
    gnd_mask = arr["Classification"] == 2
    n_gnd = int(gnd_mask.sum())
    if n_gnd < 4:
        return arr

    # ELM reads coordinates and writes classes: the other fields stay behind.
    gnd_arr = np.empty(
        n_gnd, dtype=[(name, arr.dtype[name]) for name in ("X", "Y", "Z", "Classification")]
    )
    for name in gnd_arr.dtype.names:
        gnd_arr[name] = arr[name][gnd_mask]
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


_TILE_WORKER: dict = {}  # per-process state of run_tiled worker processes


def _available_cpus() -> int:
    """CPUs this process may use (its affinity, e.g. a Slurm allocation)."""
    import os

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def _init_tile_worker(worker_fn, provider, store, kwargs, threads):
    """Set up one tile worker process: its own TileDB context and store handle."""
    # Several processes share the machine: fewer TileDB threads and a small
    # tile cache each, instead of every process claiming all cores and 1 GiB.
    n = str(threads)
    provider = provider.with_config(**{
        "sm.num_reader_threads": n, "sm.num_tiledb_threads": n,
        "sm.compute_concurrency_level": n, "sm.io_concurrency_level": n,
        "sm.tile_cache_size": str(128 * 1024**2),
    })  # fmt: skip
    _TILE_WORKER.update(fn=worker_fn, provider=provider, store=store, kwargs=kwargs, done=0)


def _run_tile(item):
    """Process one tile in a worker process; return the data years it wrote."""
    w = _TILE_WORKER
    idx, (query_bbox, crop_bbox) = item
    w["fn"](w["provider"], query_bbox, crop_bbox, w["store"], idx, **w["kwargs"])
    w["done"] += 1
    if w["done"] % 20 == 0:
        gc.collect()  # reference cycles of PDAL/GDAL objects
    return w["store"].take_pending_years()


def _map_in_processes(items, n_workers, initargs):
    """Yield ``_run_tile(item)`` from spawned worker processes as tiles finish.

    Workers start with ``spawn`` (a forked child would inherit TileDB's
    threads in whatever state they were); at most two tiles per worker are
    in flight.
    """
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    items = iter(items)
    sentinel = object()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        n_workers, mp_context=context, initializer=_init_tile_worker, initargs=initargs
    ) as pool:
        pending = set()
        for item in items:
            pending.add(pool.submit(_run_tile, item))
            if len(pending) >= 2 * n_workers:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()  # re-raise worker exceptions
                item = next(items, sentinel)
                if item is not sentinel:
                    pending.add(pool.submit(_run_tile, item))
                yield result


def run_tiled(
    worker_fn: Callable,
    provider,
    tiles,
    store,
    n_workers: int,
    progress_every: int = 500,
    gc_every: int = 100,
    skip: set[int] | frozenset[int] = frozenset(),
    parallel: str = "threads",
    **kwargs,
) -> None:
    """Run *worker_fn* over all *tiles*, sequentially, in threads or in processes.

    The worker signature is
    ``worker_fn(provider, query_bbox, crop_bbox, store, tile_index, **kwargs)``.
    Progress is logged every *progress_every* tiles, and ``gc.collect()`` runs
    every *gc_every* tiles to free reference cycles of PDAL/GDAL objects.
    Tiles in *skip* keep their index but are not processed. Results are
    consumed as tiles finish, so one slow tile does not idle the others.

    ``parallel="processes"`` (with ``n_workers > 1``) runs the tiles in
    worker processes instead of threads. Most of a tile's work holds Python's
    global lock, so threads level off at two to three times one worker, while
    processes use one core each. It needs a module-level *worker_fn*,
    picklable *kwargs*, a store prepared with
    :meth:`~alsdb.storage.ALSZarrStore.prepare_tile_writes`, and, in scripts,
    an ``if __name__ == "__main__":`` guard (processes start with ``spawn``).
    Each process holds its own tile in memory. Arguments that cannot be
    pickled (e.g. a lambda model) fall back to threads, with a warning.
    """
    if parallel not in ("threads", "processes"):
        raise ValueError("parallel must be 'threads' or 'processes'")
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
    if parallel == "processes" and n_workers > 1:
        import pickle

        try:
            pickle.dumps((worker_fn, kwargs))
        except Exception as exc:  # noqa: BLE001 - pickling failures vary (lambdas, closures)
            logger.warning(
                "run_tiled: worker arguments cannot be sent to processes (%s); using threads",
                type(exc).__name__,
            )
        else:
            threads = max(1, _available_cpus() // n_workers)
            written = set()
            try:
                for years in _map_in_processes(
                    todo, n_workers, (worker_fn, provider, store, kwargs, threads)
                ):
                    written |= years
                    _note_progress()
            finally:
                store.record_data_years(written)
            return
    for _ in bounded_map_unordered(process, todo, n_workers):
        _note_progress()


def bounded_map_unordered(fn, items, n_workers=1):
    """Map *fn* over *items* in a thread pool, yielding results as each finishes.

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


class PointGrid:
    """Points sorted into square cells of the search radius, for disc queries.

    A disc of that radius touches at most the 3 x 3 cells around its centre,
    each a contiguous run of the sorted points, so a query returns an index
    array without building per-point Python lists (as cKDTree's
    ``query_ball_point`` does: millions of ints for dense footprints).
    """

    def __init__(self, x, y, origin, cell):
        self.x, self.y, self.origin, self.cell = x, y, origin, cell
        cx = np.floor((x - origin[0]) / cell).astype(np.int64)
        cy = np.floor((y - origin[1]) / cell).astype(np.int64)
        self.nx, self.ny = int(cx.max()) + 1, int(cy.max()) + 1
        key = cy * self.nx + cx
        self.order = np.argsort(key, kind="stable")
        self.bounds = np.searchsorted(key[self.order], np.arange(self.nx * self.ny + 1))

    def disc(self, x0, y0, radius):
        """Return the indices of the points within *radius* of (x0, y0), ascending."""
        fx = int(np.floor((x0 - self.origin[0]) / self.cell))
        fy = int(np.floor((y0 - self.origin[1]) / self.cell))
        runs = []
        for gy in range(max(fy - 1, 0), min(fy + 2, self.ny)):
            lo, hi = max(fx - 1, 0), min(fx + 2, self.nx)
            if lo < hi:  # the cells of one row are consecutive in the sorted order
                runs.append(
                    self.order[self.bounds[gy * self.nx + lo] : self.bounds[gy * self.nx + hi]]
                )
        if not runs:
            return np.empty(0, np.int64)
        idx = np.concatenate(runs)
        inside = (self.x[idx] - x0) ** 2 + (self.y[idx] - y0) ** 2 <= radius**2
        return np.sort(idx[inside])
