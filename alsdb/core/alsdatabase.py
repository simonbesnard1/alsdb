# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""High-level management of an ALS TileDB database."""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tiledb
from retry import retry

from alsdb.core.alstile import ALSTile
from alsdb.methods.classification import ClassificationConfig
from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.tile.tile import Tile
from alsdb.utils.schema import TileDBSchemaConfig, create_schema

logger = logging.getLogger(__name__)

_MANIFEST_KEY = "ingestion_manifest"  # arrays ingested before per-file entries: one JSON
_MANIFEST_PREFIX = "ingestion_manifest/"  # one metadata key per ingested file
_CRS_KEY = "crs"

_EPSG_RE = re.compile(r'AUTHORITY\["EPSG","(\d+)"\]\s*\]?\s*$')
_NAME_RE = re.compile(r'(?:PROJCS|GEOGCS)\["([^"]+)"')


_NO_CRS = "EPSG:0"  # what tile-name parsing reports for headers without a CRS


def _classification_neighbors(paths: list[Path], buffer: float) -> dict[Path, tuple[Path, ...]]:
    """Files whose native bounds intersect each file's buffered bounds."""
    if buffer <= 0:
        return {path: () for path in paths}
    tiles = [Tile(path) for path in paths]
    bounds = np.asarray([tile.native_bbox for tile in tiles], dtype=float)
    years = np.asarray([tile.name.year for tile in tiles])
    crs = np.asarray([tile.name.crs for tile in tiles], dtype=object)
    result = {}
    for index, path in enumerate(paths):
        x0, y0, x1, y1 = bounds[index]
        intersects = (
            (bounds[:, 2] >= x0 - buffer)
            & (bounds[:, 0] <= x1 + buffer)
            & (bounds[:, 3] >= y0 - buffer)
            & (bounds[:, 1] <= y1 + buffer)
            & (years == years[index])
            & (crs == crs[index])
        )
        intersects[index] = False
        result[path] = tuple(paths[i] for i in np.flatnonzero(intersects))
    return result


def _keep_crs(tile_crs: str, stored_crs: str | None) -> str:
    """CRS for a recreated array: the tile's, unless it has none and the array had one."""
    if tile_crs == _NO_CRS and stored_crs and stored_crs != _NO_CRS:
        return stored_crs
    return tile_crs


def _short_crs(crs_str: str) -> str:
    """Return a compact CRS label for logging (e.g. EPSG:25830)."""
    m = _EPSG_RE.search(crs_str.strip())
    if m:
        return f"EPSG:{m.group(1)}"
    m2 = _NAME_RE.match(crs_str)
    if m2:
        name = m2.group(1)
        return name if len(name) <= 40 else name[:37] + "..."
    return crs_str[:40]


def _available_cpus() -> int:
    """CPUs this process may use (its affinity, e.g. a Slurm allocation)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


class ALSDatabase(TileDBProvider):
    """
    Write interface for the ALS TileDB array.

    Inherits storage / context management from
    :class:`~alsdb.providers.tiledb_provider.TileDBProvider`.  Creates the
    TileDB array on first write if it does not exist, then appends subsequent
    tiles as new fragments.

    The array uses three dimensions — X, Y (UTM metres) and Year (int16) —
    so repeated surveys of the same tile in different years are stored and
    queryable independently.

    An ingestion manifest is kept in the array's metadata, one key per file
    (``"ingestion_manifest/<filename>"``), so recording a file costs the same
    however many are already ingested.  Each entry records the acquisition
    year, point count, status (``"ok"`` / ``"failed"``), and UTC timestamp.
    Re-ingesting a file that is already marked ``"ok"`` is a no-op unless
    ``overwrite=True`` is passed.

    Parameters
    ----------
    storage_type:
        ``"local"`` or ``"s3"``.
    uri:
        Array URI (filesystem path or ``s3://`` URI).
    schema_cfg:
        Domain and tile-size configuration.  Uses sensible UTM Zone 30N
        defaults when not provided.
    url:
        S3 endpoint URL (required for S3 storage).
    region:
        S3 region.
    credentials:
        S3 credentials dict (keys: ``"AccessKeyId"``, ``"SecretAccessKey"``,
        ``"SessionToken"``).
    s3_config_overrides:
        Raw TileDB ``vfs.s3.*`` overrides.
    max_reader_threads:
        Threads per query inside TileDB; see
        :class:`~alsdb.providers.tiledb_provider.TileDBProvider`.
    """

    def __init__(
        self,
        storage_type: str = "local",
        uri: str | None = None,
        schema_cfg: TileDBSchemaConfig | None = None,
        url: str | None = None,
        region: str = "eu-central-1",
        credentials: dict[str, str] | None = None,
        s3_config_overrides: dict[str, str] | None = None,
        max_reader_threads: int | None = None,
    ) -> None:
        super().__init__(
            storage_type=storage_type,
            uri=uri,
            url=url,
            region=region,
            credentials=credentials,
            s3_config_overrides=s3_config_overrides,
            max_reader_threads=max_reader_threads,
        )
        # None means "derive from first tile's CRS at ingest time"
        self._schema_cfg = schema_cfg

    # ------------------------------------------------------------------
    # Array lifecycle
    # ------------------------------------------------------------------

    def stored_crs(self) -> str | None:
        """Return the CRS stored in array metadata, or ``None`` if not set."""
        if not self.array_exists():
            return None
        with self.open("r") as arr:
            return arr.meta.get(_CRS_KEY)

    def create(self, overwrite: bool = False, crs: str | None = None) -> None:
        """
        Explicitly create the TileDB array.

        Parameters
        ----------
        overwrite:
            If ``True`` and the array already exists, it is deleted first.
        crs:
            CRS string (e.g. ``"EPSG:25830"``) used to select domain bounds
            when no ``schema_cfg`` was provided at construction time.
            Stored in array metadata so subsequent ingestions can validate CRS
            consistency.
        """
        if self.array_exists():
            if not overwrite:
                logger.debug("Array already exists at %s — skipping creation.", self.array_uri)
                return
            logger.info("Removing existing array at %s", self.array_uri)
            tiledb.remove(self.array_uri, ctx=self.ctx)
            self._schema_cache = None

        cfg = self._schema_cfg
        if cfg is None:
            cfg = TileDBSchemaConfig.for_crs(crs) if crs else TileDBSchemaConfig()

        schema = create_schema(cfg)
        tiledb.Array.create(self.array_uri, schema, ctx=self.ctx)
        logger.info(
            "Created array at %s (CRS=%s)", self.array_uri, _short_crs(crs) if crs else "unknown"
        )

        if crs:
            with self.open("w") as arr:
                arr.meta[_CRS_KEY] = crs

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def load_manifest(self) -> dict[str, dict]:
        """
        Return the ingestion manifest stored in array metadata.

        Returns an empty dict if the array does not exist yet or no manifest
        has been written.
        """
        if not self.array_exists():
            return {}
        with self.open("r") as arr:
            meta = dict(arr.meta.items())
        # Arrays written before per-file entries hold the whole manifest in one key.
        manifest = json.loads(meta.get(_MANIFEST_KEY, "{}"))
        for key, raw in meta.items():
            if key.startswith(_MANIFEST_PREFIX):
                manifest[key[len(_MANIFEST_PREFIX) :]] = json.loads(raw)
        return manifest

    def _save_manifest_entry(self, filename: str, entry: dict) -> None:
        """Record one file: a single small metadata write, whatever the manifest size."""
        if not self.array_exists():
            return
        self._write_metadata(_MANIFEST_PREFIX + filename, json.dumps(entry))

    def list_ingested(self) -> list[dict]:
        """
        Return a list of manifest entries, sorted by timestamp.

        Each entry is a dict with keys: ``filename``, ``year``, ``region``,
        ``tile_x_km``, ``tile_y_km``, ``n_points``, ``status``, ``ts``,
        and optionally ``error``.
        """
        manifest = self.load_manifest()
        entries = [{"filename": k, **v} for k, v in manifest.items()]
        return sorted(entries, key=lambda e: e.get("ts", ""))

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def fragment_count(self) -> int:
        """Return the current number of fragments in the array."""
        return len(tiledb.array_fragments(self.array_uri, ctx=self.ctx).uri)

    def consolidate(
        self,
        mode: str = "fragments",
        fragment_size: int = 100_000_000,
        memory_budget: int = 8_000_000_000,
        step_max_frags: int = 50,
        step_size_ratio: float = 0.5,
    ) -> None:
        """
        Consolidate and vacuum the TileDB array.

        Parameters
        ----------
        mode:
            TileDB consolidation mode.  Use ``"commits"`` or
            ``"fragment_meta"`` for lightweight metadata-only passes (fast,
            negligible memory).  Use ``"fragments"`` for full data compaction.
        fragment_size:
            Per-fragment read buffer (bytes).  Keep small to avoid OOM on
            large arrays — 100 MB is safe even with thousands of fragments.
        memory_budget:
            Total memory across all buffers in one consolidation pass.
        step_max_frags:
            Maximum fragments merged in a single pass.  Lower = less memory
            per pass, more passes needed.
        step_size_ratio:
            Size ratio window (0.0 = merge all regardless of size;
            0.5 = merge fragments within 50 % of each other in size).
        """
        if mode != "fragments":
            cfg = tiledb.Config({"sm.consolidation.mode": mode})
            tiledb.consolidate(self.array_uri, config=cfg, ctx=self.ctx)
            vac_cfg = tiledb.Config({"sm.vacuum.mode": mode})
            tiledb.vacuum(self.array_uri, config=vac_cfg, ctx=self.ctx)
            return

        # Large fragment sets require the available RAM for tile-offset reads.
        cfg = tiledb.Config(
            {
                "sm.consolidation.mode": "fragments",
                "sm.consolidation.buffer_size": str(fragment_size),
                "sm.consolidation.total_buffer_size": str(memory_budget),
                "sm.consolidation.step_min_frags": "2",
                "sm.consolidation.step_max_frags": str(step_max_frags),
                "sm.consolidation.step_size_ratio": str(step_size_ratio),
                "sm.consolidation.amplification": "1.5",
                "sm.memory_budget": str(memory_budget),
                "sm.memory_budget_var": str(memory_budget // 2),
            }
        )
        vac_cfg = tiledb.Config({"sm.vacuum.mode": "fragments"})
        n_frags = self.fragment_count()
        logger.info("Consolidating %d fragments (step_max=%d)…", n_frags, step_max_frags)
        tiledb.consolidate(self.array_uri, config=cfg, ctx=self.ctx)
        tiledb.vacuum(self.array_uri, config=vac_cfg, ctx=self.ctx)
        logger.info("Consolidation done.")

    # ------------------------------------------------------------------
    # Retry-protected TileDB I/O
    # ------------------------------------------------------------------

    @retry(
        (tiledb.TileDBError, ConnectionError),
        tries=10,
        delay=5,
        backoff=3,
        logger=logger,
    )
    def _write_to_tiledb(self, coords: tuple, data: dict[str, np.ndarray]) -> None:
        with tiledb.open(self.array_uri, mode="w", ctx=self.ctx) as array:
            array[coords] = data

    @retry(
        (tiledb.TileDBError, ConnectionError),
        tries=10,
        delay=5,
        backoff=3,
        logger=logger,
    )
    def _write_metadata(self, key: str, value: str) -> None:
        with tiledb.open(self.array_uri, mode="w", ctx=self.ctx) as array:
            array.meta[key] = value

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        x: np.ndarray,
        y: np.ndarray,
        year: int,
        attrs: dict[str, np.ndarray],
        crs: str | None = None,
    ) -> None:
        """
        Append a batch of points to the TileDB array.

        The array is created (with default schema) if it does not exist yet.

        Parameters
        ----------
        x:
            1-D ``float64`` array of UTM easting values.
        y:
            1-D ``float64`` array of UTM northing values.
        year:
            Acquisition year (scalar int, broadcast to all points).
        attrs:
            Dictionary mapping LAS attribute names to typed 1-D numpy arrays.
        crs:
            CRS string passed to :meth:`create` when the array does not yet
            exist (ignored if the array already exists).
        """
        if not self.array_exists():
            self.create(crs=crs)
        year_arr = np.full(len(x), year, dtype=np.int16)
        # Fill any schema attributes missing from this tile (e.g. RGB absent in
        # intensity-only datasets) with zeros so TileDB always gets a complete row.
        n = len(x)
        with self.open("r") as _arr:
            for i in range(_arr.schema.nattr):
                a = _arr.schema.attr(i)
                if a.name not in attrs:
                    attrs[a.name] = np.zeros(n, dtype=a.dtype)
        self._write_to_tiledb((x, y, year_arr), attrs)
        logger.debug("Wrote %d points (year=%d) to %s", len(x), year, self.array_uri)

    def _ingest_tile(
        self,
        laz_path: Path,
        chunk_size: int,
        classification_filter: list[int] | None,
        stored_crs: str | None,
        _tile: ALSTile | None = None,
        reproject_to: str | None = None,
        classification_config: ClassificationConfig | None = None,
        classification_context: tuple[Path, ...] = (),
        classification_workers: int = 1,
    ) -> tuple[int, dict]:
        """
        Read a LAZ file and write its points to the array.

        Does **not** touch the manifest — that is the caller's responsibility.
        Returns ``(n_points, manifest_entry)``.

        Raises ``ValueError`` on CRS mismatch.
        """
        filename = laz_path.name
        classification_config = classification_config or ClassificationConfig(enabled=False)
        tile = _tile or ALSTile(
            laz_path,
            classification_filter=classification_filter,
            reproject_to=reproject_to,
            classification_config=classification_config,
            classification_context=classification_context,
            classification_workers=classification_workers,
        )
        tile_name = tile.name  # metadata cached on first access
        year = tile_name.year
        crs = tile.target_crs  # use post-reprojection CRS for storage
        if crs == _NO_CRS and stored_crs and stored_crs != _NO_CRS:
            # Header without a CRS (e.g. Sustainable Landscapes Brazil, DE-Hai):
            # trust the CRS the array was created with instead of rejecting it.
            logger.info(
                "%s has no CRS in its header; using the array's %s",
                filename,
                _short_crs(stored_crs),
            )
            crs = stored_crs

        if stored_crs and stored_crs != crs:
            raise ValueError(
                f"CRS mismatch: array was created with {stored_crs!r}, "
                f"but {filename!r} reports {crs!r}. "
                "Use a separate array or reproject the tile."
            )

        # Accumulate all chunks in memory and write once → exactly 1 fragment per file.
        # Multiple writes within a single open("w") context still create separate
        # S3 objects in TileDB's sparse array model; only a single write avoids this.
        if not self.array_exists():
            self.create(crs=crs)

        xs, ys, attr_chunks = [], [], []
        for x, y, attrs in tile.iter_chunks(chunk_size=chunk_size):
            xs.append(x)
            ys.append(y)
            attr_chunks.append(attrs)

        if not xs:
            entry = {
                "year": year,
                "crs": crs,
                "bbox": list(tile.points_bbox),
                "n_points": 0,
                "status": "ok",
                "ts": datetime.now(UTC).isoformat(),
            }
            if tile.classification_config.active:
                entry["classification"] = tile.classification_config.to_dict()
                entry["classification_hash"] = tile.classification_config.fingerprint
            return 0, entry

        def joined(parts):  # a whole tile usually arrives as one chunk: no copy then
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

        x_all = joined(xs)
        y_all = joined(ys)
        total = len(x_all)
        year_arr = np.full(total, year, dtype=np.int16)

        # Merge per-chunk attribute dicts; attributes a file lacks are zeros.
        # Zero arrays are made only for attributes the file lacks.
        schema = self.schema
        attrs_all: dict[str, np.ndarray] = {}
        for attr in (schema.attr(i) for i in range(schema.nattr)):
            attrs_all[attr.name] = joined(
                [
                    ch[attr.name] if attr.name in ch else np.zeros(len(xs[i]), dtype=attr.dtype)
                    for i, ch in enumerate(attr_chunks)
                ]
            )

        self._write_to_tiledb((x_all, y_all, year_arr), attrs_all)

        logger.debug(
            "Wrote %d pts from %s (year=%d, crs=%s) → %s",
            total,
            filename,
            year,
            _short_crs(crs),
            self.array_uri,
        )

        entry = {
            "year": year,
            "crs": crs,
            "bbox": list(tile.points_bbox),
            "n_points": total,
            "status": "ok",
            "ts": datetime.now(UTC).isoformat(),
        }
        if tile.classification_config.active:
            entry["classification"] = tile.classification_config.to_dict()
            entry["classification_hash"] = tile.classification_config.fingerprint
        return total, entry

    def ingest(
        self,
        laz_path: str | Path,
        chunk_size: int | None = None,
        classification_filter: list[int] | None = None,
        overwrite: bool = False,
        reproject_to: str | None = None,
        classification_config: ClassificationConfig | None = None,
        classification_workers: int | None = None,
    ) -> None:
        """
        Ingest a single LAZ tile into the TileDB array.

        If the file is already recorded as successfully ingested in the
        manifest, the call is a no-op.  Pass ``overwrite=True`` to force
        re-ingestion regardless of manifest state.

        Points are stored under their survey year (read from the LAZ file
        header), so the same spatial tile can be ingested multiple times from
        different survey years without collision.

        Parameters
        ----------
        laz_path:
            Path to the ``.laz`` file.
        chunk_size:
            Points per write batch.  Defaults to ``TileDBSchemaConfig.chunk_size``.
        classification_filter:
            Optional list of LAS classification codes to retain.
        overwrite:
            Re-ingest even if the file is already in the manifest as ``"ok"``.
        reproject_to:
            Target CRS for the stored points.  ``"auto"`` detects feet-based
            CRS and reprojects to the appropriate UTM zone; an explicit
            ``"EPSG:XXXX"`` string reprojects to that CRS; ``None`` (default)
            keeps the native CRS.
        classification_workers:
            Threads for the runs of the PTD tilt ensemble (default: the CPUs
            available to this process). The labels do not depend on it.
        """
        laz_path = Path(laz_path)
        filename = laz_path.name
        if classification_workers is None:
            classification_workers = _available_cpus()
        classification_config = classification_config or ClassificationConfig(enabled=False)
        if classification_config.active and classification_config.buffer > 0:
            logger.warning(
                "Classification buffer %.1f m has no neighboring files in ingest(); "
                "use ingest_many() for buffered classification",
                classification_config.buffer,
            )
        chunk_size = chunk_size or (self._schema_cfg.chunk_size if self._schema_cfg else 1_000_000)

        manifest = self.load_manifest()
        existing = manifest.get(filename, {})
        if (
            not overwrite
            and existing.get("status") == "ok"
            and classification_config.active
            and existing.get("classification_hash") != classification_config.fingerprint
        ):
            raise ValueError(
                f"{filename} was ingested with a different classification configuration; "
                "pass overwrite=True to rebuild the array"
            )
        if not overwrite and existing.get("status") == "ok":
            logger.info(
                "Already ingested %s — skipping (pass overwrite=True to force)",
                filename,
            )
            return

        # Read tile metadata once up front so the CRS is available before the
        # array is (re)created and is reused in _ingest_tile without a second
        # PDAL pass.
        tile = ALSTile(
            laz_path,
            classification_filter=classification_filter,
            reproject_to=reproject_to,
            classification_config=classification_config,
            classification_workers=classification_workers,
        )
        tile_crs = tile.target_crs

        if overwrite and self.array_exists():
            self.create(overwrite=True, crs=_keep_crs(tile_crs, self.stored_crs()))

        stored = self.stored_crs()

        try:
            total, entry = self._ingest_tile(
                laz_path,
                chunk_size,
                classification_filter,
                stored,
                _tile=tile,
                reproject_to=reproject_to,
                classification_config=classification_config,
            )
        except Exception as exc:
            self._save_manifest_entry(
                filename,
                {"status": "failed", "error": str(exc), "ts": datetime.now(UTC).isoformat()},
            )
            raise

        self._save_manifest_entry(filename, entry)
        logger.info(
            "Ingested %s — %s pts  |  year=%d  |  crs=%s  |  bbox=(%.0f, %.0f, %.0f, %.0f)",
            filename,
            f"{total:,}",
            entry["year"],
            _short_crs(entry["crs"]),
            *entry["bbox"],
        )

    def ingest_many(
        self,
        laz_paths: list[str | Path],
        chunk_size: int | None = None,
        classification_filter: list[int] | None = None,
        consolidate_every: int = 50,
        n_workers: int = 1,
        overwrite: bool = False,
        reproject_to: str | None = None,
        classification_config: ClassificationConfig | None = None,
        optimize: bool = True,
        classification_workers: int | None = None,
    ) -> dict[str, int]:
        """
        Ingest a list of LAZ files, skipping already-ingested ones.

        Automatically consolidates fragments every *consolidate_every* newly
        written tiles to keep read performance healthy.

        Parameters
        ----------
        laz_paths:
            Ordered list of ``.laz`` file paths.
        chunk_size:
            Points per write batch.
        classification_filter:
            Optional LAS classification filter applied to every tile.
        consolidate_every:
            Consolidate and vacuum the array's metadata (commits, fragment
            metadata, manifest entries) after this many files.
        n_workers:
            Number of parallel worker threads.  ``1`` (default) runs
            sequentially.  Values > 1 use a :class:`ThreadPoolExecutor` —
            each worker reads a different LAZ file and appends its own TileDB
            fragment concurrently.  The manifest is written by the main thread
            only, so there is no race condition.
        overwrite:
            If ``True``, wipe the array and re-ingest all files from scratch,
            ignoring the manifest.  If ``False`` (default), files already
            marked ``"ok"`` in the manifest are skipped.
        optimize:
            Merge the per-file fragments at the end (:meth:`optimize`);
            ``False`` leaves it to a separate call (it is the memory peak of
            an ingest).
        classification_workers:
            Threads per file for the runs of the PTD tilt ensemble (default:
            the CPUs available to this process divided by *n_workers*, so
            files and runs together do not oversubscribe the machine). The
            labels do not depend on it.

        Returns
        -------
        dict
            ``{filename: n_points_written}`` for every path in *laz_paths*.
            Skipped files have value 0.
        """
        if classification_workers is None:
            classification_workers = max(1, _available_cpus() // max(1, n_workers))
        chunk_size = chunk_size or (self._schema_cfg.chunk_size if self._schema_cfg else 1_000_000)
        classification_config = classification_config or ClassificationConfig(enabled=False)
        laz_paths = [Path(p) for p in laz_paths]
        neighbors = _classification_neighbors(laz_paths, classification_config.buffer)

        # Pre-load manifest; filter already-ingested files unless overwrite
        manifest = self.load_manifest()
        pending: list[Path] = []
        results: dict[str, int] = {}
        for p in laz_paths:
            existing = manifest.get(p.name, {})
            if (
                not overwrite
                and existing.get("status") == "ok"
                and classification_config.active
                and existing.get("classification_hash") != classification_config.fingerprint
            ):
                raise ValueError(
                    f"{p.name} was ingested with a different classification configuration; "
                    "pass overwrite=True to rebuild the array"
                )
            if not overwrite and existing.get("status") == "ok":
                logger.debug("Already ingested %s — skipping", p.name)
                results[p.name] = 0
            else:
                pending.append(p)

        n_skip = len(laz_paths) - len(pending)
        if n_skip:
            logger.info("Skipping %d already-ingested file(s)", n_skip)

        if not pending:
            logger.info("Nothing to ingest — all files already present in manifest")
            return results

        # Ensure the array exists (or recreate it) before dispatching workers
        # so they never race on array creation.
        first_tile = ALSTile(
            pending[0],
            classification_filter=classification_filter,
            reproject_to=reproject_to,
            classification_config=classification_config,
        )
        first_crs = first_tile.target_crs
        if overwrite and self.array_exists():
            self.create(overwrite=True, crs=_keep_crs(first_crs, self.stored_crs()))
        elif not self.array_exists():
            self.create(crs=first_crs)

        stored = self.stored_crs()
        newly_written = 0
        n_pending = len(pending)
        # ~20 progress lines regardless of file count; at least every 50 files
        progress_every = max(1, min(50, n_pending // 20))

        logger.info(
            "Ingesting %d file(s) → %s  [workers=%d, consolidate_every=%d]",
            n_pending,
            self.array_uri,
            n_workers,
            consolidate_every,
        )

        # Progress counters, updated by the main thread only
        _completed = 0
        _total_pts = 0
        _t0 = time.monotonic()

        def _worker(path: Path) -> tuple[str, int, dict]:
            total, entry = self._ingest_tile(
                path,
                chunk_size,
                classification_filter,
                stored,
                reproject_to=reproject_to,
                classification_config=classification_config,
                classification_context=neighbors[path],
                classification_workers=classification_workers,
            )
            return path.name, total, entry

        # Process in batches so consolidation only runs after all workers in a
        # batch have finished — avoids the race where consolidate() tries to
        # access .wrt commit files that concurrent writers are still using.
        for batch_start in range(0, n_pending, consolidate_every):
            batch = pending[batch_start : batch_start + consolidate_every]

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_path = {executor.submit(_worker, p): p for p in batch}
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        filename, n, entry = future.result()
                        manifest[filename] = entry
                        results[filename] = n
                        if n > 0:
                            newly_written += 1
                    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole batch
                        filename = path.name
                        logger.error("Failed to ingest %s: %s", filename, exc)
                        manifest[filename] = {
                            "status": "failed",
                            "error": str(exc),
                            "ts": datetime.now(UTC).isoformat(),
                        }
                        results[filename] = 0

                    _completed += 1
                    _total_pts += results[filename]

                    if _completed % progress_every == 0 or _completed == n_pending:
                        elapsed = time.monotonic() - _t0
                        rate = _total_pts / elapsed if elapsed > 0 else 0
                        eta = (
                            (n_pending - _completed) * elapsed / _completed if _completed > 0 else 0
                        )
                        logger.info(
                            "  [%d/%d] %5.1f%%  |  %.2f B pts  |  %5.1f M pts/s  |  ETA ~%.0f min",
                            _completed,
                            n_pending,
                            100.0 * _completed / n_pending,
                            _total_pts / 1e9,
                            rate / 1e6,
                            eta / 60,
                        )

                    # Record every tile as it finishes so progress survives crashes
                    self._save_manifest_entry(filename, manifest[filename])

            # All workers in this batch are done.
            # Consolidate only metadata — fast, near-zero memory. "array_meta"
            # merges the per-file manifest entries, so opening the array stays
            # quick. Fragment data consolidation is deferred to the end.
            if newly_written > 0:
                for meta_mode in ("commits", "fragment_meta", "array_meta"):
                    try:
                        self.consolidate(mode=meta_mode)
                    except Exception as exc:  # noqa: BLE001 - best-effort cleanup, never fatal
                        logger.debug("Metadata consolidation (%s) skipped: %s", meta_mode, exc)

        elapsed_total = time.monotonic() - _t0
        n_failed = sum(1 for v in manifest.values() if v.get("status") == "failed")
        logger.info(
            "Ingestion done: %d/%d files  |  %.2f B pts  |  avg %.1f M pts/s  |  %.1f min%s",
            newly_written,
            n_pending,
            _total_pts / 1e9,
            (_total_pts / elapsed_total / 1e6) if elapsed_total > 0 else 0,
            elapsed_total / 60,
            f"  |  {n_failed} FAILED" if n_failed else "",
        )

        if newly_written > 0 and optimize:
            self.optimize()
        return results

    def optimize(self, memory_budget: int = 8_000_000_000, max_passes: int = 200) -> int:
        """Merge fragments for faster reads; return the number left.

        Iterative compaction: each pass merges at most 50 fragments of similar
        size, and passes repeat until the count stops falling, so memory stays
        bounded (*memory_budget* per pass) even with thousands of fragments.
        Run after ingesting; reading works without it, only slower.
        """
        logger.info("Starting iterative fragment compaction…")
        prev = None
        for pass_num in range(1, max_passes + 1):
            n_frags = self.fragment_count()
            if n_frags <= 1 or (prev is not None and n_frags >= prev):
                break
            prev = n_frags
            logger.info("  Compaction pass %d: %d fragments remaining", pass_num, n_frags)
            self.consolidate(
                mode="fragments",
                fragment_size=100_000_000,
                memory_budget=memory_budget,
                step_max_frags=50,
                step_size_ratio=0.5,
            )
        final_frags = self.fragment_count()
        logger.info("Compaction complete: %d fragment(s) remaining", final_frags)
        return final_frags
