# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Zarr-backed storage for ALS gridded products.

Layout
------
Each :class:`ALSZarrStore` is a Zarr directory store with one group per
resolution:

    {root}/
    ├── .zattrs              → bbox, crs_wkt
    ├── 1m/
    │   ├── .zattrs          → resolution, x_origin, y_origin, nx, ny, crs_wkt
    │   ├── time             (T,)         int32   [survey years]
    │   ├── y                (ny,)        float64 [cell centres, north-up]
    │   ├── x                (nx,)        float64 [cell centres]
    │   ├── chm              (T, ny, nx)  float32
    │   ├── dtm              (T, ny, nx)  float32
    │   └── dsm              (T, ny, nx)  float32
    └── 10m/
        ├── ...              (same coordinates)
        ├── gap              (T, ny, nx)  float32
        ├── lai              (T, ny, nx)  float32
        ├── biomass          (T, ny, nx)  float32
        └── h50 … density    (T, ny, nx)  float32

The CRS is stored as a string in group attributes (``crs_wkt`` key) rather
than as a separate Zarr array, which avoids object-dtype compatibility issues
across Zarr versions.  :meth:`to_dataset` writes it via rioxarray so the
returned :class:`xarray.Dataset` is CRS-aware.

Concurrent tile writes
----------------------
Multiple threads may call :meth:`write_tile` simultaneously for different
spatial regions (which the tiling guarantees are non-overlapping).  Zarr
writes each chunk to a separate file atomically, so concurrent writes to
different chunks are safe. A per-resolution threading lock currently protects
the entire tile write, including time-axis resize and index lookup, to prevent
partial-chunk and metadata races between tiles.

Usage::

    from alsdb.storage import ALSZarrStore

    # Processing functions auto-initialize groups as needed:
    store = ALSZarrStore("output/spain.zarr")
    compute_chm(provider, store, resolution=1.0, year=2021)

    # Or create explicitly upfront:
    store = ALSZarrStore.create(
        "output/spain.zarr",
        bbox=(308000, 4688000, 780000, 4900000),
        crs_wkt="EPSG:25830",
        variables={
            "1m":  ["chm", "dtm", "dsm"],
            "10m": ["gap", "lai", "biomass",
                    "h50", "h75", "h95", "hmean", "cc", "density"],
        },
        tile_size=500.0,
    )

    # Read as xarray Dataset
    ds = store.to_dataset(resolution=1.0)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from retry import retry

logger = logging.getLogger(__name__)

_COORD_ARRAYS = {"x", "y", "time"}

_S3_DEFAULT_RETRIES = 10
# botocore's own default (10) is far smaller than realistic n_workers for
# compute_all/compute_biomass (production configs use 50) - every thread
# past the pool size blocks waiting for a connection rather than doing real
# work, which reads as "process alive, barely any CPU used, no visible
# progress" - confirmed as the actual cause of exactly that symptom on a
# real all-Spain run. Comfortably larger than any realistic n_workers.
_S3_DEFAULT_MAX_POOL_CONNECTIONS = 100


def _apply_s3_client_defaults(storage_options: dict) -> dict:
    """Inject sensible defaults for a busy, multi-threaded S3 backend.

    Configures botocore adaptive retry mode (covers throttling and
    transient AWS/S3-compatible endpoint errors) and a connection pool large
    enough for realistic worker counts (see _S3_DEFAULT_MAX_POOL_CONNECTIONS).
    User-supplied values in *storage_options* are never overwritten.

    Note: this must go through s3fs's ``config_kwargs`` (a plain dict s3fs
    merges into its own ``AioConfig(...)``), not ``client_kwargs["config"]``
    — s3fs already passes its own ``config=`` positional into
    ``create_client(..., **client_kwargs)``, so a ``config`` key inside
    ``client_kwargs`` collides with it and raises "multiple values for
    keyword argument 'config'".
    """
    opts = storage_options.copy()
    config_kwargs = opts.setdefault("config_kwargs", {})
    config_kwargs.setdefault("retries", {"max_attempts": _S3_DEFAULT_RETRIES, "mode": "adaptive"})
    config_kwargs.setdefault("max_pool_connections", _S3_DEFAULT_MAX_POOL_CONNECTIONS)
    return opts


@retry(
    (OSError, ConnectionError),
    tries=10,
    delay=5,
    backoff=3,
    logger=logger,
)
def _open_group_retrying(path: str, mode: str, storage_options: dict | None = None):
    """Open a Zarr group, retrying on transient S3 failures.

    ``ALSZarrStore.__init__`` calls this exactly once per store. Unlike
    ``ensure_group``/``write_tile``/``has_data``, this used to be a bare
    ``zarr.open_group(...)`` call with no retry protection - a single
    transient GetObject failure while reading the root's zarr.json would
    crash the whole process before any tile work started, with no
    top-level except around it in the calling workflow script.
    """
    import zarr

    if storage_options is not None:
        return zarr.open_group(path, mode=mode, storage_options=storage_options)
    return zarr.open_group(path, mode=mode)


def _res_str(resolution: float) -> str:
    """Format resolution as a group name, e.g. 1.0 → '1m', 0.5 → '0.5m'."""
    if resolution == int(resolution):
        return f"{int(resolution)}m"
    return f"{resolution}m"


class ALSZarrStore:
    """
    Zarr-backed store for ALS gridded products.

    Parameters
    ----------
    path:
        Directory path for the Zarr store.  Local path or an ``s3://`` URI.
    mode:
        ``"a"`` (default) opens existing store or creates a new empty one.
        ``"r"`` opens read-only.
    storage_options:
        Keyword arguments forwarded to the fsspec/s3fs backend when *path*
        is an S3 URI.  Typical keys: ``key``, ``secret``, ``endpoint_url``,
        ``client_kwargs``.  Ignored for local paths.
    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "a",
        storage_options: dict | None = None,
    ) -> None:
        self.path = path  # keep as-is so S3 URIs survive repr
        self._storage_options = storage_options or {}
        _path_str = str(path)
        if self._storage_options and _path_str.startswith("s3://"):
            s3_opts = _apply_s3_client_defaults(self._storage_options)
            self._root = _open_group_retrying(_path_str, mode, s3_opts)
        else:
            self._root = _open_group_retrying(_path_str, mode)
        # Per-resolution-group locks so different variables can be written
        # concurrently (e.g. CHM and DTM in parallel) without blocking each other.
        self._group_locks: dict[str, threading.Lock] = {}
        self._group_locks_create = threading.Lock()  # protects the dict itself
        # Per-instance caches for write_tile. On S3 every group/array open is a
        # zarr.json GET; uncached, one write_tile cost ~25 store round trips
        # (23 of them metadata reads), all inside the group lock. Grid attrs are
        # immutable after _init_group; the rest are invalidated by begin_run.
        self._grid_attrs: dict[str, tuple[float, float, int, int]] = {}
        self._time_index: dict[tuple[str, int], int] = {}
        self._arrays: dict[tuple[str, str, int], object] = {}  # (res, var, t) -> sized handle
        self._years_recorded: set[tuple[str, str, int]] = set()
        self._var_locks: dict[tuple[str, str], threading.Lock] = {}
        self._tile_group_ready = False

    # ------------------------------------------------------------------
    # Internal lock management
    # ------------------------------------------------------------------

    def _group_lock(self, res_key: str) -> threading.Lock:
        """Return the lock for resolution group *res_key*, creating it if needed."""
        with self._group_locks_create:
            if res_key not in self._group_locks:
                self._group_locks[res_key] = threading.Lock()
            return self._group_locks[res_key]

    def _var_lock(self, res_key: str, variable: str) -> threading.Lock:
        """Lock for chunk read-modify-write on one array (unaligned windows only)."""
        with self._group_locks_create:
            return self._var_locks.setdefault((res_key, variable), threading.Lock())

    def _invalidate_caches(self, res_key: str) -> None:
        """Drop cached handles/indices for *res_key* (caller holds its group lock)."""
        self._time_index = {k: v for k, v in self._time_index.items() if k[0] != res_key}
        self._arrays = {k: v for k, v in self._arrays.items() if k[0] != res_key}
        self._years_recorded = {k for k in self._years_recorded if k[0] != res_key}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        bbox: tuple[float, float, float, float],
        crs_wkt: str,
        variables: dict[str, list[str]],
        tile_size: float = 500.0,
        storage_options: dict | None = None,
    ) -> ALSZarrStore:
        """
        Create a new Zarr store pre-allocated for *bbox* and *variables*.

        Parameters
        ----------
        path:
            Store directory.  Existing content is overwritten.
        bbox:
            Full spatial extent ``(min_x, min_y, max_x, max_y)`` in metres.
        crs_wkt:
            CRS as an EPSG string (e.g. ``"EPSG:25830"``) or WKT.
        variables:
            Mapping of resolution string → list of variable names, e.g.
            ``{"1m": ["chm", "dtm", "dsm"], "10m": ["gap", "biomass"]}``.
        tile_size:
            Chunk size in CRS units.  Should match the ``tile_size`` used
            during processing so chunk boundaries align with tile boundaries.
        """

        store = cls(path, mode="w", storage_options=storage_options)
        store._root.attrs.update({"bbox": list(bbox), "crs_wkt": crs_wkt})

        for res_key, var_names in variables.items():
            store._init_group(res_key, bbox, crs_wkt, var_names, tile_size, root=store._root)

        return cls(path, mode="a", storage_options=storage_options)

    # ------------------------------------------------------------------
    # Group initialisation (internal)
    # ------------------------------------------------------------------

    def _init_group(
        self,
        res_key: str,
        bbox: tuple[float, float, float, float],
        crs_wkt: str,
        var_names: list[str],
        tile_size: float,
        root=None,
    ) -> None:
        """Create a resolution group with coordinate arrays and data variables."""
        if root is None:
            root = self._root

        min_x, _min_y, _max_x, max_y = bbox
        res = float(res_key.rstrip("m"))
        from alsdb.processing._grid import GridSpec

        grid = GridSpec.from_bbox(bbox, res)
        ny, nx = grid.shape
        chunk_ny = min(ny, max(1, int(tile_size / res)))
        chunk_nx = min(nx, max(1, int(tile_size / res)))

        grp = root.require_group(res_key)
        grp.attrs.update(
            {
                "resolution": res,
                "x_origin": min_x,
                "y_origin": max_y,  # top-left corner, north-up
                "nx": nx,
                "ny": ny,
                "crs_wkt": crs_wkt,
            }
        )

        # Coordinate arrays (cell centres, written once)
        if "x" not in grp:
            x_coords = min_x + (np.arange(nx, dtype=np.float64) + 0.5) * res
            x_arr = grp.create_array("x", shape=(nx,), dtype=np.float64, chunks=(nx,))
            x_arr[:] = x_coords
        if "y" not in grp:
            y_coords = max_y - (np.arange(ny, dtype=np.float64) + 0.5) * res
            y_arr = grp.create_array("y", shape=(ny,), dtype=np.float64, chunks=(ny,))
            y_arr[:] = y_coords
        if "time" not in grp:
            grp.create_array("time", shape=(0,), chunks=(1,), dtype=np.int32)

        # Data variables (time axis starts at 0, grows on demand)
        for var in var_names:
            if var not in grp:
                arr = grp.create_array(
                    var,
                    shape=(0, ny, nx),
                    chunks=(1, chunk_ny, chunk_nx),
                    dtype=np.float32,
                    fill_value=np.nan,
                )
                arr.attrs["_FillValue"] = "NaN"
                arr.attrs["data_years"] = []

        logger.info(
            "ALSZarrStore: initialised group '%s'  (%d×%d px, chunk %d×%d, vars: %s)",
            res_key,
            ny,
            nx,
            chunk_ny,
            chunk_nx,
            ", ".join(var_names),
        )

    @retry(
        (OSError, ConnectionError),
        tries=10,
        delay=5,
        backoff=3,
        logger=logger,
    )
    def ensure_group(
        self,
        variable: str,
        resolution: float,
        bbox: tuple[float, float, float, float],
        crs_wkt: str,
        tile_size: float = 500.0,
    ) -> None:
        """
        Ensure the resolution group and *variable* array exist.

        Called automatically by processing functions before the first tile
        write.  Safe to call from multiple threads — uses the store lock.

        Parameters
        ----------
        variable:
            Variable name to ensure, e.g. ``"chm"``.
        resolution:
            Cell size in metres.
        bbox:
            Spatial extent ``(min_x, min_y, max_x, max_y)`` for the global
            grid.  Typically ``array_data_bbox(provider)``.
        crs_wkt:
            CRS string stored in group attributes.
        tile_size:
            Chunk size in CRS units.
        """
        res_key = _res_str(resolution)
        with self._group_lock(res_key):
            if res_key in self._root:
                from alsdb.processing._grid import GridSpec

                expected = GridSpec.from_bbox(bbox, resolution)
                attrs = self._root[res_key].attrs
                if (
                    int(attrs["nx"]) != expected.nx
                    or int(attrs["ny"]) != expected.ny
                    or not np.allclose(
                        [attrs["x_origin"], attrs["y_origin"]],
                        [expected.x0, expected.y1],
                        rtol=0,
                        atol=resolution * 1e-8,
                    )
                    or attrs["crs_wkt"] != crs_wkt
                ):
                    raise ValueError("Requested grid/CRS differs from the existing store grid")
            if res_key not in self._root:
                self._init_group(res_key, bbox, crs_wkt, [variable], tile_size)
            elif variable not in self._root[res_key]:
                grp = self._root[res_key]
                ny = int(grp.attrs["ny"])
                nx = int(grp.attrs["nx"])
                chunk_ny = min(ny, max(1, int(tile_size / resolution)))
                chunk_nx = min(nx, max(1, int(tile_size / resolution)))
                # Always start at 0; _upsert_year will resize to match the
                # current time axis before the first write.
                arr = grp.create_array(
                    variable,
                    shape=(0, ny, nx),
                    chunks=(1, chunk_ny, chunk_nx),
                    dtype=np.float32,
                    fill_value=np.nan,
                )
                arr.attrs["_FillValue"] = "NaN"
                arr.attrs["data_years"] = []
                logger.info("ALSZarrStore: added variable '%s' to group '%s'", variable, res_key)

    def check_run(self, variable, resolution, year, configuration, *, overwrite=False):
        """Return True only for a completed run with identical provenance.

        Existing untracked data cannot be silently reused. Failed/running runs
        with the same configuration are restarted and their old slice cleared.
        """
        import hashlib
        import json

        encoded = json.dumps(configuration, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        key = _res_str(resolution)
        if key not in self._root or variable not in self._root[key]:
            return False
        records = dict(self._root[key][variable].attrs.get("processing_runs", {}))
        record = records.get(str(year))
        if overwrite:
            return False
        if record is None:
            if self.has_data(variable, resolution, year):
                raise ValueError(
                    f"{variable}/{year} has no processing provenance; use overwrite=True"
                )
            return False
        if record["fingerprint"] != fingerprint:
            raise ValueError(
                f"{variable}/{year} has a different processing configuration; use overwrite=True"
            )
        return record["status"] == "complete"

    def begin_run(self, variable, resolution, year, configuration, variables, *, resume=False):
        """Mark an attempt running; clear stale data unless resuming the same run.

        With ``resume=True`` and an existing record carrying the same
        configuration fingerprint (an interrupted or failed attempt), written
        tiles and their completion markers are kept so the caller can skip
        them via :meth:`completed_tiles`. Otherwise the year slice of every
        name in *variables* and the run's tile markers are cleared.

        Returns ``True`` if the run is resumed, ``False`` if started fresh.
        """
        import hashlib
        import json

        key = _res_str(resolution)
        configuration = json.loads(json.dumps(configuration, sort_keys=True, allow_nan=False))
        fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        with self._group_lock(key):
            self._invalidate_caches(key)
            group = self._root[key]
            arr = group[variable]
            records = dict(arr.attrs.get("processing_runs", {}))
            previous = records.get(str(year))
            resumed = bool(resume and previous and previous["fingerprint"] == fingerprint)
            records[str(year)] = {
                "status": "running",
                "fingerprint": fingerprint,
                "configuration": configuration,
            }
            arr.attrs["processing_runs"] = records
            if resumed:
                return True
            self._delete_prefix(self._tiles_prefix(key, variable, year))
            years = np.asarray(group["time"][:])
            found = np.flatnonzero(years == year)
            if len(found):
                t = int(found[0])
                for name in variables:
                    if name in group and group[name].shape[0] > t:
                        self._clear_time_slice(group[name], t)
                        group[name].attrs["data_years"] = [
                            y for y in group[name].attrs.get("data_years", []) if y != year
                        ]
            return False

    # ------------------------------------------------------------------
    # Raw store helpers (chunk sweep, tile completion markers)
    # ------------------------------------------------------------------

    def _store_key(self, *parts) -> str:
        base = self._root.path
        return "/".join(([base] if base else []) + [str(p) for p in parts])

    def _list_prefix(self, prefix: str) -> list[str]:
        from zarr.core.sync import sync

        async def collect():
            return [k async for k in self._root.store.list_prefix(prefix)]

        return sync(collect())

    def _delete_prefix(self, prefix: str) -> None:
        from zarr.core.sync import sync

        for k in self._list_prefix(prefix):
            sync(self._root.store.delete(k))

    def _clear_time_slice(self, arr, t: int) -> None:
        """Remove the stored chunks of time slice *t* (reads back as NaN fill).

        Writing NaN over the whole slice made zarr visit every chunk of the
        grid - one S3 DELETE per possible chunk (~58k per variable for zone 31
        at 1 m), whether or not it held data. Listing c/<t>/ touches only the
        chunks that exist (one LIST per 1000 keys).
        """
        from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

        encoding = arr.metadata.chunk_key_encoding
        if not isinstance(encoding, DefaultChunkKeyEncoding) or encoding.separator != "/":
            arr[t, :, :] = np.nan  # unknown key layout: safe, slow fallback
            return
        self._delete_prefix(self._store_key(arr.path, "c", t) + "/")

    def _tiles_prefix(self, res_key: str, variable: str, year: int) -> str:
        return self._store_key("_alsdb_tiles", res_key, variable, int(year)) + "/"

    def mark_tile_complete(self, variable, resolution, year, tile_index) -> None:
        """Record that *tile_index* of the current run of *variable*/*year* is written.

        One empty object per tile (independent keys, so concurrent workers
        never contend). Stored under a root-level ``_alsdb_tiles`` zarr group
        so hierarchy listings don't warn about unrecognised objects.
        """
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.sync import sync

        self._ensure_tile_group()
        key = self._tiles_prefix(_res_str(resolution), variable, year) + str(int(tile_index))
        sync(self._root.store.set(key, default_buffer_prototype().buffer.from_bytes(b"")))

    def _ensure_tile_group(self) -> None:
        if not self._tile_group_ready:
            with self._group_locks_create:
                if not self._tile_group_ready:
                    self._root.require_group("_alsdb_tiles")
                    self._tile_group_ready = True

    def completed_tiles(self, variable, resolution, year) -> set[int]:
        """Tile indices marked complete for the current run of *variable*/*year*."""
        prefix = self._tiles_prefix(_res_str(resolution), variable, year)
        return {int(k.rsplit("/", 1)[1]) for k in self._list_prefix(prefix)}

    def finish_run(self, variable, resolution, year, *, failed=False):
        key = _res_str(resolution)
        with self._group_lock(key):
            arr = self._root[key][variable]
            records = dict(arr.attrs["processing_runs"])
            records[str(year)]["status"] = "failed" if failed else "complete"
            arr.attrs["processing_runs"] = records

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @retry(
        (OSError, ConnectionError),
        tries=10,
        delay=5,
        backoff=3,
        logger=logger,
    )
    def write_tile(
        self,
        variable: str,
        resolution: float,
        year: int,
        data: np.ndarray,
        crop_bbox: tuple[float, float, float, float],
    ) -> None:
        """
        Write a processed tile into the store.

        Parameters
        ----------
        variable:
            Variable name, e.g. ``"chm"``.
        resolution:
            Cell size in metres, e.g. ``1.0``.
        year:
            Survey year (integer).  A new time slice is appended if *year* is
            not already present; otherwise the existing slice is overwritten.
        data:
            2-D float32 array ``(ny_tile, nx_tile)`` in north-up orientation.
            NaN encodes missing values.
        crop_bbox:
            ``(min_x, min_y, max_x, max_y)`` of the tile's non-buffered
            extent.  Used to compute pixel offsets into the global grid.
        """
        if year is None:
            raise ValueError(
                "write_tile() requires an integer survey year (e.g. year=2021), not None. "
                "All processing functions that write to a store must receive an explicit year."
            )
        res_key = _res_str(resolution)
        if res_key not in self._grid_attrs:
            attrs = dict(self._root[res_key].attrs)
            self._grid_attrs[res_key] = (
                float(attrs["x_origin"]),
                float(attrs["y_origin"]),
                int(attrs["ny"]),
                int(attrs["nx"]),
            )
        x_origin, y_origin, ny_store, nx_store = self._grid_attrs[res_key]

        cx0, cy0, cx1, cy1 = crop_bbox

        # Pixel offsets (north-up: row 0 = top of grid)
        col0 = round((cx0 - x_origin) / resolution)
        col1 = round((cx1 - x_origin) / resolution)
        row0 = round((y_origin - cy1) / resolution)
        row1 = round((y_origin - cy0) / resolution)

        # Clamp to store bounds (last tile may be smaller than tile_size)
        col1 = min(col1, nx_store)
        row1 = min(row1, ny_store)

        if row0 >= row1 or col0 >= col1:
            logger.debug("write_tile: empty slice for %s, skipping", variable)
            return

        tile_ny = row1 - row0
        tile_nx = col1 - col0
        if row0 < 0 or col0 < 0 or data.shape[0] < tile_ny or data.shape[1] < tile_nx:
            raise ValueError("Tile data/window does not fit the store grid")
        year = int(year)
        window = data[:tile_ny, :tile_nx].astype(np.float32, copy=False)
        arr = self._sized_array(res_key, variable, year)
        t_idx = self._time_index[(res_key, year)]
        # Unaligned windows require a lock because they share Zarr chunks.
        cy, cx = arr.chunks[1], arr.chunks[2]
        aligned = (
            row0 % cy == 0
            and col0 % cx == 0
            and (row1 % cy == 0 or row1 == ny_store)
            and (col1 % cx == 0 or col1 == nx_store)
        )
        if aligned:
            arr[t_idx, row0:row1, col0:col1] = window
        else:
            with self._var_lock(res_key, variable):
                arr[t_idx, row0:row1, col0:col1] = window
        if (res_key, variable, year) not in self._years_recorded and np.isfinite(window).any():
            with self._group_lock(res_key):
                years = list(arr.attrs.get("data_years", []))
                if year not in years:
                    arr.attrs["data_years"] = sorted(years + [year])
                self._years_recorded.add((res_key, variable, year))

    def _sized_array(self, res_key: str, variable: str, year: int):
        """Array handle whose time axis already covers *year* (cached per instance)."""
        cached = self._arrays.get((res_key, variable, year))
        if cached is not None:
            return cached
        with self._group_lock(res_key):
            cached = self._arrays.get((res_key, variable, year))
            if cached is None:
                grp = self._root[res_key]
                self._time_index[(res_key, year)] = self._upsert_year(grp, year)
                # Re-open after _upsert_year: a handle opened before a resize
                # keeps the old shape and rejects writes to the new slice.
                cached = grp[variable]
                self._arrays[(res_key, variable, year)] = cached
            return cached

    def _upsert_year(self, grp, year: int) -> int:
        """Return the time index for *year*, appending a new slice if needed.

        Caller must hold the per-group lock for *grp* before calling this method.

        Also ensures every data variable in the group is at least (t+1) deep —
        this handles variables added after the time axis was already populated.
        """
        time_arr = grp["time"]
        existing = time_arr[:] if time_arr.shape[0] > 0 else np.array([], dtype=np.int32)
        match = np.where(existing == year)[0]

        if len(match):
            t = int(match[0])
        else:
            # New year: grow time coordinate and all existing data arrays
            t = int(time_arr.shape[0])
            time_arr.resize((t + 1,))
            time_arr[t] = year
            logger.debug("ALSZarrStore: appended year %d at t=%d", year, t)

        # Resize any variable whose time axis is too short (covers both
        # newly appended years and variables added after the time axis grew)
        ny = int(grp.attrs["ny"])
        nx = int(grp.attrs["nx"])
        for name in grp.array_keys():
            if name in _COORD_ARRAYS:
                continue
            arr = grp[name]
            if arr.shape[0] <= t:
                arr.resize((t + 1, ny, nx))

        return t

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def to_dataset(self, resolution: float):
        """
        Return an :class:`xarray.Dataset` for *resolution*.

        Parameters
        ----------
        resolution:
            Resolution group to open, e.g. ``1.0`` or ``10.0``.
        """
        import xarray as xr

        res_key = _res_str(resolution)
        grp = self._root[res_key]
        attrs = dict(grp.attrs)

        time_vals = grp["time"][:]
        x_vals = grp["x"][:]
        y_vals = grp["y"][:]
        crs_wkt = attrs.get("crs_wkt", "")

        coords = {
            "time": ("time", time_vals),
            "y": ("y", y_vals),
            "x": ("x", x_vals),
        }

        n_time = len(time_vals)
        n_y = len(y_vals)
        n_x = len(x_vals)

        data_vars: dict = {}
        for name in grp.array_keys():
            if name in _COORD_ARRAYS:
                continue
            arr = grp[name]
            if arr.shape[0] == 0:
                # Variable was initialised but no data written yet — fill with NaN
                data = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)
            else:
                data = arr[:]
            da = xr.DataArray(
                data,
                dims=["time", "y", "x"],
                coords=coords,
                attrs=dict(arr.attrs),
            )
            data_vars[name] = da

        ds = xr.Dataset(data_vars, attrs=attrs)

        if crs_wkt:
            try:
                import rioxarray  # noqa: F401

                ds = ds.rio.write_crs(crs_wkt)
            except ImportError:
                pass

        return ds

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @retry(
        (OSError, ConnectionError),
        tries=10,
        delay=5,
        backoff=3,
        logger=logger,
    )
    def has_data(self, variable: str, resolution: float, year: int) -> bool:
        """
        Return True if finite values have been written for this variable/year.

        Legacy arrays without per-variable write metadata fall back to the time
        axis. This does not prove completion; surface workflows use check_run
        and processing provenance to distinguish complete and interrupted runs.

        Parameters
        ----------
        variable:
            Variable name, e.g. ``"chm"``.
        resolution:
            Resolution group in metres.
        year:
            Survey year to check.
        """
        res_key = _res_str(resolution)
        if res_key not in self._root:
            return False
        grp = self._root[res_key]
        if variable not in grp:
            return False
        time_arr = grp["time"]
        if time_arr.shape[0] == 0:
            return False
        written = grp[variable].attrs.get("data_years")
        return bool(year in written) if written is not None else bool(year in time_arr[:])

    @property
    def resolutions(self) -> list[float]:
        """Resolution values present in the store."""
        result = []
        for key in self._root.group_keys():
            try:
                result.append(float(key.rstrip("m")))
            except ValueError:
                pass
        return sorted(result)

    def variables(self, resolution: float) -> list[str]:
        """Variable names available at *resolution*."""
        res_key = _res_str(resolution)
        grp = self._root[res_key]
        return [n for n in grp.array_keys() if n not in _COORD_ARRAYS]

    def __repr__(self) -> str:
        parts = [f"  {_res_str(r)}: {self.variables(r)}" for r in self.resolutions]
        return f"ALSZarrStore({self.path})\n" + "\n".join(parts)
