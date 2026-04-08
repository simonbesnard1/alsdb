# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tiledb

from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.utils.schema import TileDBSchemaConfig, create_schema
from alsdb.core.alstile import ALSTile

logger = logging.getLogger(__name__)

_MANIFEST_KEY = "ingestion_manifest"
_CRS_KEY = "crs"


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

    An ingestion manifest is kept in the array's metadata (key
    ``"ingestion_manifest"``).  Each entry records the filename, acquisition
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
    """

    def __init__(
        self,
        storage_type: str = "local",
        uri: Optional[str] = None,
        schema_cfg: Optional[TileDBSchemaConfig] = None,
        url: Optional[str] = None,
        region: str = "eu-central-1",
        credentials: Optional[Dict[str, str]] = None,
        s3_config_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            storage_type=storage_type,
            uri=uri,
            url=url,
            region=region,
            credentials=credentials,
            s3_config_overrides=s3_config_overrides,
        )
        # None means "derive from first tile's CRS at ingest time"
        self._schema_cfg = schema_cfg

    # ------------------------------------------------------------------
    # Array lifecycle
    # ------------------------------------------------------------------

    def stored_crs(self) -> Optional[str]:
        """Return the CRS stored in array metadata, or ``None`` if not set."""
        if not self.array_exists():
            return None
        with self.open("r") as arr:
            return arr.meta.get(_CRS_KEY)

    def create(self, overwrite: bool = False, crs: Optional[str] = None) -> None:
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
        logger.info("Created array at %s (CRS=%s)", self.array_uri, crs or "unknown")

        if crs:
            with self.open("w") as arr:
                arr.meta[_CRS_KEY] = crs

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def load_manifest(self) -> Dict[str, dict]:
        """
        Return the ingestion manifest stored in array metadata.

        Returns an empty dict if the array does not exist yet or no manifest
        has been written.
        """
        if not self.array_exists():
            return {}
        with self.open("r") as arr:
            raw = arr.meta.get(_MANIFEST_KEY, "{}")
        return json.loads(raw)

    def _save_manifest(self, manifest: Dict[str, dict]) -> None:
        with self.open("w") as arr:
            arr.meta[_MANIFEST_KEY] = json.dumps(manifest)

    def list_ingested(self) -> List[dict]:
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

    def consolidate(
        self,
        fragment_size: int = 2_000_000_000,
        memory_budget: int = 4_000_000_000,
        step_size_ratio: float = 0.0,
    ) -> None:
        """
        Consolidate and vacuum the TileDB array.

        TileDB's consolidation is size-tiered: it merges fragments whose sizes
        fall within *step_size_ratio* of each other, up to *fragment_size*.
        Setting ``step_size_ratio=0.0`` ignores size differences and merges
        all fragments regardless of size — effectively a full compaction.

        Parameters
        ----------
        fragment_size:
            Target consolidated fragment size in bytes (default 2 GB).
            Fragments smaller than this are candidates for merging.
        memory_budget:
            Total memory budget for the consolidation pass (bytes).
            Should be set to available RAM for large arrays.
        step_size_ratio:
            Size ratio window for fragment eligibility (0.0 = merge all,
            1.0 = only merge identically-sized fragments).
        """
        cfg = tiledb.Config({
            "sm.consolidation.mode":            "fragments",
            "sm.consolidation.buffer_size":     str(fragment_size),
            "sm.consolidation.total_buffer_size": str(memory_budget),
            "sm.consolidation.step_min_frags":  "2",
            "sm.consolidation.step_max_frags":  "200",
            "sm.consolidation.step_size_ratio": str(step_size_ratio),
            "sm.consolidation.amplification":   "1.0",
        })
        vac_cfg = tiledb.Config({"sm.vacuum.mode": "fragments"})
        n_frags = len(tiledb.array_fragments(self.array_uri).uri)
        logger.info(
            "Consolidating %d fragments (fragment_size=%.0f MB, "
            "memory_budget=%.0f MB)…",
            n_frags, fragment_size / 1e6, memory_budget / 1e6,
        )
        tiledb.consolidate(self.array_uri, config=cfg, ctx=self.ctx)
        tiledb.vacuum(self.array_uri, config=vac_cfg, ctx=self.ctx)
        logger.info("Consolidation done.")

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        x: np.ndarray,
        y: np.ndarray,
        year: int,
        attrs: Dict[str, np.ndarray],
        crs: Optional[str] = None,
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
        with tiledb.open(self.array_uri, mode="w", ctx=self.ctx) as arr:
            arr[x, y, year_arr] = attrs
        logger.debug("Wrote %d points (year=%d) to %s", len(x), year, self.array_uri)

    def _ingest_tile(
        self,
        laz_path: Path,
        chunk_size: int,
        classification_filter: Optional[list[int]],
        stored_crs: Optional[str],
        _tile: Optional[ALSTile] = None,
    ) -> Tuple[int, dict]:
        """
        Read a LAZ file and write its points to the array.

        Does **not** touch the manifest — that is the caller's responsibility.
        Returns ``(n_points, manifest_entry)``.

        Raises ``ValueError`` on CRS mismatch.
        """
        filename = laz_path.name
        tile = _tile or ALSTile(laz_path, classification_filter=classification_filter)
        tile_name = tile.name  # metadata cached on first access
        year = tile_name.year
        crs = tile_name.crs

        if stored_crs and stored_crs != crs:
            raise ValueError(
                f"CRS mismatch: array was created with {stored_crs!r}, "
                f"but {filename!r} reports {crs!r}. "
                "Use a separate array or reproject the tile."
            )

        total = 0
        for x, y, attrs in tile.iter_chunks(chunk_size=chunk_size):
            self.write(x, y, year, attrs, crs=crs)
            total += len(x)
            logger.debug(
                "Ingested %d points from %s (year=%d, crs=%s) → %s  (running total: %d)",
                len(x), filename, year, crs, self.array_uri, total,
            )

        logger.info(
            "Done: %d points from %s (year=%d, crs=%s) → %s",
            total, filename, year, crs, self.array_uri,
        )
        entry = {
            "year": year,
            "crs": crs,
            "bbox": list(tile_name.bbox_native),
            "n_points": total,
            "status": "ok",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        return total, entry

    def ingest(
        self,
        laz_path: str | Path,
        chunk_size: Optional[int] = None,
        classification_filter: Optional[list[int]] = None,
        overwrite: bool = False,
    ) -> int:
        """
        Ingest a single LAZ tile into the TileDB array.

        If the file is already recorded as successfully ingested in the
        manifest, the call is a no-op and returns 0.  Pass ``overwrite=True``
        to force re-ingestion regardless of manifest state.

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

        Returns
        -------
        int
            Total number of points written (0 if skipped).
        """
        laz_path = Path(laz_path)
        filename = laz_path.name
        chunk_size = chunk_size or (self._schema_cfg.chunk_size if self._schema_cfg else 1_000_000)

        manifest = self.load_manifest()
        if not overwrite and manifest.get(filename, {}).get("status") == "ok":
            logger.info("Already ingested %s — skipping (pass overwrite=True to force)", filename)
            return 0

        # Read tile metadata once up front so the CRS is available before the
        # array is (re)created and is reused in _ingest_tile without a second
        # PDAL pass.
        tile = ALSTile(laz_path, classification_filter=classification_filter)
        tile_crs = tile.name.crs

        if overwrite and self.array_exists():
            self.create(overwrite=True, crs=tile_crs)

        stored = self.stored_crs()

        try:
            total, entry = self._ingest_tile(laz_path, chunk_size, classification_filter, stored, _tile=tile)
            manifest[filename] = entry
        except Exception as exc:
            manifest[filename] = {
                "status": "failed",
                "error": str(exc),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            self._save_manifest(manifest)
            raise

        self._save_manifest(manifest)
        return total

    def ingest_many(
        self,
        laz_paths: List[str | Path],
        chunk_size: Optional[int] = None,
        classification_filter: Optional[list[int]] = None,
        consolidate_every: int = 50,
        max_workers: int = 1,
        overwrite: bool = False,
    ) -> Dict[str, int]:
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
            Consolidate + vacuum after this many newly ingested tiles.
        max_workers:
            Number of parallel worker threads.  ``1`` (default) runs
            sequentially.  Values > 1 use a :class:`ThreadPoolExecutor` —
            each worker reads a different LAZ file and appends its own TileDB
            fragment concurrently.  The manifest is written by the main thread
            only, so there is no race condition.
        overwrite:
            If ``True``, wipe the array and re-ingest all files from scratch,
            ignoring the manifest.  If ``False`` (default), files already
            marked ``"ok"`` in the manifest are skipped.

        Returns
        -------
        dict
            ``{filename: n_points_written}`` for every path in *laz_paths*.
            Skipped files have value 0.
        """
        chunk_size = chunk_size or (self._schema_cfg.chunk_size if self._schema_cfg else 1_000_000)
        laz_paths = [Path(p) for p in laz_paths]

        # Pre-load manifest; filter already-ingested files unless overwrite
        manifest = self.load_manifest()
        pending: List[Path] = []
        results: Dict[str, int] = {}
        for p in laz_paths:
            if not overwrite and manifest.get(p.name, {}).get("status") == "ok":
                logger.info("Already ingested %s — skipping", p.name)
                results[p.name] = 0
            else:
                pending.append(p)

        if not pending:
            return results

        # Ensure the array exists (or recreate it) before dispatching workers
        # so they never race on array creation.
        first_tile = ALSTile(pending[0], classification_filter=classification_filter)
        first_crs = first_tile.name.crs
        if overwrite and self.array_exists():
            self.create(overwrite=True, crs=first_crs)
        elif not self.array_exists():
            self.create(crs=first_crs)

        stored = self.stored_crs()
        newly_written = 0

        def _worker(path: Path) -> Tuple[str, int, dict]:
            total, entry = self._ingest_tile(path, chunk_size, classification_filter, stored)
            return path.name, total, entry

        # Process in batches so consolidation only runs after all workers in a
        # batch have finished — avoids the race where consolidate() tries to
        # access .wrt commit files that concurrent writers are still using.
        for batch_start in range(0, len(pending), consolidate_every):
            batch = pending[batch_start : batch_start + consolidate_every]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(_worker, p): p for p in batch}
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        filename, n, entry = future.result()
                        manifest[filename] = entry
                        results[filename] = n
                        if n > 0:
                            newly_written += 1
                    except Exception as exc:
                        filename = path.name
                        logger.error("Failed to ingest %s: %s", filename, exc)
                        manifest[filename] = {
                            "status": "failed",
                            "error": str(exc),
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        results[filename] = 0

                    # Save manifest after every tile so progress survives crashes
                    self._save_manifest(manifest)

            # All workers in this batch are done — safe to consolidate.
            # Size-tiered config keeps fragment count manageable during ingest
            # without over-merging (small fragments won't be merged into the
            # large consolidated ones from previous batches until the final pass).
            if newly_written > 0:
                self.consolidate()

        # Final full compaction: step_size_ratio=0.0 ignores size differences
        # and merges ALL remaining fragments into one regardless of size.
        n_batches = max(1, len(pending) // consolidate_every)
        if newly_written > 0 and n_batches > 1:
            logger.info("Final compaction — merging all batch fragments…")
            self.consolidate(step_size_ratio=0.0)

        return results
