# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
import re
import threading
import time
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

_EPSG_RE = re.compile(r'AUTHORITY\["EPSG","(\d+)"\]\s*\]?\s*$')
_NAME_RE = re.compile(r'(?:PROJCS|GEOGCS)\["([^"]+)"')


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
        logger.info(
            "Created array at %s (CRS=%s)", self.array_uri, _short_crs(crs) if crs else "unknown"
        )

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
        if not self.array_exists():
            return
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
        mode: str = "fragments",
        fragment_size: int = 100_000_000,
        memory_budget: int = 256_000_000,
        step_max_frags: int = 10,
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

        cfg = tiledb.Config(
            {
                "sm.consolidation.mode": "fragments",
                "sm.consolidation.buffer_size": str(fragment_size),
                "sm.consolidation.total_buffer_size": str(memory_budget),
                "sm.consolidation.step_min_frags": "2",
                "sm.consolidation.step_max_frags": str(step_max_frags),
                "sm.consolidation.step_size_ratio": str(step_size_ratio),
                "sm.consolidation.amplification": "1.5",
            }
        )
        vac_cfg = tiledb.Config({"sm.vacuum.mode": "fragments"})
        n_frags = len(tiledb.array_fragments(self.array_uri, ctx=self.ctx).uri)
        logger.info("Consolidating %d fragments (step_max=%d)…", n_frags, step_max_frags)
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
        reclassify: bool = False,
        ground_classifier: str = "csf",
        denoise: bool = False,
        reproject_to: Optional[str] = None,
    ) -> Tuple[int, dict]:
        """
        Read a LAZ file and write its points to the array.

        Does **not** touch the manifest — that is the caller's responsibility.
        Returns ``(n_points, manifest_entry)``.

        Raises ``ValueError`` on CRS mismatch.
        """
        filename = laz_path.name
        tile = _tile or ALSTile(
            laz_path,
            classification_filter=classification_filter,
            reclassify=reclassify,
            ground_classifier=ground_classifier,
            denoise=denoise,
            reproject_to=reproject_to,
        )
        tile_name = tile.name  # metadata cached on first access
        year = tile_name.year
        crs = tile.target_crs  # use post-reprojection CRS for storage

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
            return 0, {
                "year": year,
                "crs": crs,
                "bbox": list(tile_name.bbox_native),
                "n_points": 0,
                "status": "ok",
                "ts": datetime.now(timezone.utc).isoformat(),
            }

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        total = len(x_all)
        year_arr = np.full(total, year, dtype=np.int16)

        # Merge per-chunk attribute dicts and fill any missing attributes with zeros.
        with tiledb.open(self.array_uri, mode="r", ctx=self.ctx) as rdr:
            schema_attrs = {
                rdr.schema.attr(i).name: rdr.schema.attr(i) for i in range(rdr.schema.nattr)
            }
        attrs_all: Dict[str, np.ndarray] = {}
        for name, a in schema_attrs.items():
            parts = [
                ch.get(name, np.zeros(len(xs[i]), dtype=a.dtype))
                for i, ch in enumerate(attr_chunks)
            ]
            attrs_all[name] = np.concatenate(parts)

        with tiledb.open(self.array_uri, mode="w", ctx=self.ctx) as tdb_arr:
            tdb_arr[x_all, y_all, year_arr] = attrs_all

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
        reclassify: bool = False,
        ground_classifier: str = "csf",
        denoise: bool = False,
        reproject_to: Optional[str] = None,
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
        reclassify:
            If ``True``, run SMRF + HAG during ingestion to assign ground (2)
            and vegetation (3/4/5) classes to unclassified (class 1) points.
            Use this for surveys delivered with minimal classification (e.g.
            USGS LPC files where non-ground returns are all class 1).
        reproject_to:
            Target CRS for the stored points.  ``"auto"`` detects feet-based
            CRS and reprojects to the appropriate UTM zone; an explicit
            ``"EPSG:XXXX"`` string reprojects to that CRS; ``None`` (default)
            keeps the native CRS.

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
            logger.info(
                "Already ingested %s — skipping (pass overwrite=True to force)",
                filename,
            )
            return 0

        # Read tile metadata once up front so the CRS is available before the
        # array is (re)created and is reused in _ingest_tile without a second
        # PDAL pass.
        tile = ALSTile(
            laz_path,
            classification_filter=classification_filter,
            reclassify=reclassify,
            ground_classifier=ground_classifier,
            denoise=denoise,
            reproject_to=reproject_to,
        )
        tile_crs = tile.target_crs

        if overwrite and self.array_exists():
            self.create(overwrite=True, crs=tile_crs)

        stored = self.stored_crs()

        try:
            total, entry = self._ingest_tile(
                laz_path,
                chunk_size,
                classification_filter,
                stored,
                _tile=tile,
                reclassify=reclassify,
                ground_classifier=ground_classifier,
                denoise=denoise,
                reproject_to=reproject_to,
            )
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
        reclassify: bool = False,
        ground_classifier: str = "csf",
        denoise: bool = False,
        reproject_to: Optional[str] = None,
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
            ground_classifier=ground_classifier,
            denoise=denoise,
        )
        first_crs = first_tile.target_crs
        if overwrite and self.array_exists():
            self.create(overwrite=True, crs=first_crs)
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
            max_workers,
            consolidate_every,
        )

        # Progress counters — updated under _progress_lock by the main thread
        _progress_lock = threading.Lock()
        _completed = 0
        _total_pts = 0
        _n_failed = 0
        _t0 = time.monotonic()

        def _worker(path: Path) -> Tuple[str, int, dict]:
            total, entry = self._ingest_tile(
                path,
                chunk_size,
                classification_filter,
                stored,
                reclassify=reclassify,
                ground_classifier=ground_classifier,
                denoise=denoise,
                reproject_to=reproject_to,
            )
            return path.name, total, entry

        # Process in batches so consolidation only runs after all workers in a
        # batch have finished — avoids the race where consolidate() tries to
        # access .wrt commit files that concurrent writers are still using.
        for batch_start in range(0, n_pending, consolidate_every):
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
                        n = 0

                    # Update progress counters (main thread only — no race)
                    nonlocal_n = results.get(path.name, 0)
                    _completed += 1
                    _total_pts += nonlocal_n
                    if results.get(path.name, -1) == 0 and path.name not in manifest:
                        _n_failed += 1

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

                    # Save manifest after every tile so progress survives crashes
                    self._save_manifest(manifest)

            # All workers in this batch are done.
            # Consolidate only metadata — fast, near-zero memory.
            # Fragment data consolidation is deferred to the end.
            if newly_written > 0:
                for meta_mode in ("commits", "fragment_meta"):
                    try:
                        self.consolidate(mode=meta_mode)
                    except Exception as exc:
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

        # Iterative fragment compaction: merge in small steps until stable.
        # Each pass merges at most step_max_frags fragments; multiple passes
        # converge the fragment count without exhausting memory.
        if newly_written > 0:
            logger.info("Starting iterative fragment compaction…")
            prev = None
            for pass_num in range(1, 201):
                n_frags = len(tiledb.array_fragments(self.array_uri, ctx=self.ctx).uri)
                if n_frags <= 1 or (prev is not None and n_frags >= prev):
                    break
                prev = n_frags
                logger.info("  Compaction pass %d: %d fragments remaining", pass_num, n_frags)
                self.consolidate(
                    mode="fragments",
                    fragment_size=100_000_000,
                    memory_budget=1_000_000_000,
                    step_max_frags=50,
                    step_size_ratio=0.5,
                )
            final_frags = len(tiledb.array_fragments(self.array_uri, ctx=self.ctx).uri)
            logger.info("Compaction complete: %d fragment(s) remaining", final_frags)

        return results
