# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tiledb

from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.utils.schema import TileDBSchemaConfig, create_schema
from alsdb.core.alstile import ALSTile

logger = logging.getLogger(__name__)


class ALSDatabase(TileDBProvider):
    """
    Write interface for the ALS TileDB array.

    Inherits storage / context management from :class:`~alsdb.providers.tiledb_provider.TileDBProvider`.
    Creates the TileDB array on first write if it does not exist, then appends
    subsequent tiles as new fragments.

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
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        x: np.ndarray,
        y: np.ndarray,
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
        attrs:
            Dictionary mapping LAS attribute names to typed 1-D numpy arrays.
        """
        if not self.array_exists():
            self.create()
        with tiledb.open(self.array_uri, mode="w", ctx=self.ctx) as arr:
            arr[x, y] = attrs
        logger.debug("Wrote %d points to %s", len(x), self.array_uri)

    def ingest(
        self,
        laz_path: str | Path,
        chunk_size: Optional[int] = None,
        classification_filter: Optional[list[int]] = None,
        overwrite: bool = False,
    ) -> int:
        """
        Ingest a single PNOA LAZ tile into the TileDB array.

        If the array already exists the new points are appended as a new
        fragment.  Pass ``overwrite=True`` to delete and recreate the array
        before writing.

        Parameters
        ----------
        laz_path:
            Path to the ``.laz`` file.
        chunk_size:
            Points per write batch.  Defaults to ``TileDBSchemaConfig.chunk_size``.
        classification_filter:
            Optional list of LAS classification codes to retain.
        overwrite:
            Delete and recreate the array before ingesting.

        Returns
        -------
        int
            Total number of points written.
        """
        chunk_size = chunk_size or self._schema_cfg.chunk_size

        if overwrite:
            self.create(overwrite=True)

        tile = ALSTile(laz_path, classification_filter=classification_filter)
        total = 0

        for x, y, attrs in tile.iter_chunks(chunk_size=chunk_size):
            self.write(x, y, attrs)
            total += len(x)
            logger.info(
                "Ingested %d points from %s → %s  (running total: %d)",
                len(x),
                Path(laz_path).name,
                self.array_uri,
                total,
            )

        logger.info(
            "Done: %d points from %s → %s",
            total,
            Path(laz_path).name,
            self.array_uri,
        )
        return total
