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
different chunks are safe.  A threading lock protects the time-axis
``resize`` + index look-up sequence, and group initialisation.

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

logger = logging.getLogger(__name__)

_COORD_ARRAYS = {"x", "y", "time"}


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
        import zarr

        self.path = path  # keep as-is so S3 URIs survive repr
        self._storage_options = storage_options or {}
        self._root = zarr.open_group(str(path), mode=mode, storage_options=self._storage_options)
        self._lock = threading.Lock()  # protects time-axis resize + group init

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
    ) -> "ALSZarrStore":
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

        min_x, min_y, max_x, max_y = bbox
        res = float(res_key.rstrip("m"))
        ny = int(np.ceil((max_y - min_y) / res))
        nx = int(np.ceil((max_x - min_x) / res))
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

        logger.info(
            "ALSZarrStore: initialised group '%s'  (%d×%d px, chunk %d×%d, vars: %s)",
            res_key,
            ny,
            nx,
            chunk_ny,
            chunk_nx,
            ", ".join(var_names),
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
        with self._lock:
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
                logger.info("ALSZarrStore: added variable '%s' to group '%s'", variable, res_key)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

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
        res_key = _res_str(resolution)
        grp = self._root[res_key]
        attrs = dict(grp.attrs)
        x_origin = float(attrs["x_origin"])
        y_origin = float(attrs["y_origin"])
        ny_store = int(attrs["ny"])
        nx_store = int(attrs["nx"])

        cx0, cy0, cx1, cy1 = crop_bbox

        # Pixel offsets (north-up: row 0 = top of grid)
        col0 = int(round((cx0 - x_origin) / resolution))
        col1 = int(round((cx1 - x_origin) / resolution))
        row0 = int(round((y_origin - cy1) / resolution))
        row1 = int(round((y_origin - cy0) / resolution))

        # Clamp to store bounds (last tile may be smaller than tile_size)
        col1 = min(col1, nx_store)
        row1 = min(row1, ny_store)

        if row0 >= row1 or col0 >= col1:
            logger.debug("write_tile: empty slice for %s, skipping", variable)
            return

        t_idx = self._upsert_year(grp, year)

        tile_ny = row1 - row0
        tile_nx = col1 - col0
        grp[variable][t_idx, row0:row1, col0:col1] = data[:tile_ny, :tile_nx].astype(np.float32)

    def _upsert_year(self, grp, year: int) -> int:
        """Return the time index for *year*, appending a new slice if needed.

        Also ensures every data variable in the group is at least (t+1) deep —
        this handles variables added after the time axis was already populated.
        """
        with self._lock:
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

    def has_data(self, variable: str, resolution: float, year: int) -> bool:
        """
        Return ``True`` if *variable* already has a time slice for *year*.

        Used by compute functions to skip work that is already done.
        The check is at the year level — it does not verify that every
        spatial tile was written (e.g. after a partial run).

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
        return bool(year in time_arr[:])

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
