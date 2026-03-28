# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tiledb

from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.utils.schema import TileDBSchemaConfig, create_schema
from alsdb.core.alstile import ALSTile

logger = logging.getLogger(__name__)

_MANIFEST_KEY = "ingestion_manifest"


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
        self._schema_cfg = schema_cfg or TileDBSchemaConfig()

    # ------------------------------------------------------------------
    # Array lifecycle
    # ------------------------------------------------------------------

    def create(self, overwrite: bool = False) -> None:
        """
        Explicitly create the TileDB array.

        Parameters
        ----------
        overwrite:
            If ``True`` and the array already exists, it is deleted first.
        """
        if self.array_exists():
            if not overwrite:
                logger.debug("Array already exists at %s — skipping creation.", self.array_uri)
                return
            logger.info("Removing existing array at %s", self.array_uri)
            tiledb.remove(self.array_uri, ctx=self.ctx)
            self._schema_cache = None

        schema = create_schema(self._schema_cfg)
        tiledb.Array.create(self.array_uri, schema, ctx=self.ctx)
        logger.info("Created array at %s", self.array_uri)

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
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        x: np.ndarray,
        y: np.ndarray,
        year: int,
        attrs: Dict[str, np.ndarray],
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
        """
        if not self.array_exists():
            self.create()
        year_arr = np.full(len(x), year, dtype=np.int16)
        with tiledb.open(self.array_uri, mode="w", ctx=self.ctx) as arr:
            arr[x, y, year_arr] = attrs
        logger.debug("Wrote %d points (year=%d) to %s", len(x), year, self.array_uri)

    def ingest(
        self,
        laz_path: str | Path,
        chunk_size: Optional[int] = None,
        classification_filter: Optional[list[int]] = None,
        overwrite: bool = False,
    ) -> int:
        """
        Ingest a single PNOA LAZ tile into the TileDB array.

        If the file is already recorded as successfully ingested in the
        manifest, the call is a no-op and returns 0.  Pass ``overwrite=True``
        to force re-ingestion regardless of manifest state.

        If the array already exists the new points are appended as a new
        fragment.  Points are stored under their survey year (parsed from the
        PNOA filename), so the same spatial tile can be ingested multiple times
        from different survey years without collision.

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
        chunk_size = chunk_size or self._schema_cfg.chunk_size

        # --- manifest check ---
        manifest = self.load_manifest()
        if not overwrite and manifest.get(filename, {}).get("status") == "ok":
            logger.info("Already ingested %s — skipping (pass overwrite=True to force)", filename)
            return 0

        if overwrite and self.array_exists():
            self.create(overwrite=True)

        tile = ALSTile(laz_path, classification_filter=classification_filter)
        year = tile.name.year
        total = 0

        try:
            for x, y, attrs in tile.iter_chunks(chunk_size=chunk_size):
                self.write(x, y, year, attrs)
                total += len(x)
                logger.info(
                    "Ingested %d points from %s (year=%d) → %s  (running total: %d)",
                    len(x), filename, year, self.array_uri, total,
                )

            manifest[filename] = {
                "year": year,
                "region": tile.name.region,
                "tile_x_km": tile.name.tile_x_km,
                "tile_y_km": tile.name.tile_y_km,
                "n_points": total,
                "status": "ok",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            logger.info("Done: %d points from %s (year=%d) → %s", total, filename, year, self.array_uri)

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

        Returns
        -------
        dict
            ``{filename: n_points_written}`` for every path in *laz_paths*.
            Skipped files have value 0.
        """
        results: Dict[str, int] = {}
        newly_written = 0

        for path in laz_paths:
            filename = Path(path).name
            n = self.ingest(path, chunk_size=chunk_size,
                            classification_filter=classification_filter)
            results[filename] = n
            if n > 0:
                newly_written += 1

            if newly_written > 0 and newly_written % consolidate_every == 0:
                logger.info("Consolidating after %d tiles…", newly_written)
                tiledb.consolidate(self.array_uri, ctx=self.ctx)
                tiledb.vacuum(self.array_uri, ctx=self.ctx)

        return results
