# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging
import os
from typing import Dict, Optional

import tiledb

logger = logging.getLogger(__name__)


class TileDBProvider:
    """
    Base class for managing low-level interactions with TileDB arrays for ALS data.

    Supports both local filesystem arrays and S3-backed arrays (including
    S3-compatible stores such as MinIO).

    Parameters
    ----------
    storage_type:
        Either ``"local"`` or ``"s3"``.
    uri:
        Full URI for the TileDB array.

        - *local*: absolute or relative filesystem path (e.g. ``"/data/als_array"``).
        - *s3*: ``s3://`` URI (e.g. ``"s3://my-bucket/als/pnoa_2021"``).
    url:
        S3 endpoint URL (required when ``storage_type="s3"``).
        For AWS use ``"s3.amazonaws.com"``; for MinIO use the host:port of your instance.
    region:
        S3 region (default ``"eu-central-1"``).
    credentials:
        Dictionary with S3 credentials keys: ``"AccessKeyId"``,
        ``"SecretAccessKey"``, and optionally ``"SessionToken"``.
        If None, unsigned (public) access is attempted.
    s3_config_overrides:
        Optional dictionary of raw TileDB ``vfs.s3.*`` config keys to
        override after the defaults are applied (useful for experiments).
    """

    def __init__(
        self,
        storage_type: str = "local",
        uri: Optional[str] = None,
        url: Optional[str] = None,
        region: str = "eu-central-1",
        credentials: Optional[Dict[str, str]] = None,
        s3_config_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        if not storage_type or not isinstance(storage_type, str):
            raise ValueError("'storage_type' must be a non-empty string.")

        self.storage_type = storage_type.lower()
        self.s3_config_overrides = s3_config_overrides or {}

        if self.storage_type == "s3":
            if not uri:
                raise ValueError("'uri' must be provided when storage_type='s3'.")
            if not url:
                raise ValueError(
                    "'url' (S3 endpoint) must be provided when storage_type='s3'."
                )
            self.array_uri = uri
            self._raw_cfg, self.ctx = self._initialize_s3_context(
                credentials, url, region
            )

        elif self.storage_type == "local":
            if not uri:
                raise ValueError("'uri' must be provided when storage_type='local'.")
            self.array_uri = uri
            self._raw_cfg, self.ctx = self._initialize_local_context()

        else:
            raise ValueError(
                f"Invalid storage_type {storage_type!r}. Must be 'local' or 's3'."
            )

        self._schema_cache: Optional[tiledb.ArraySchema] = None

    # ------------------------------------------------------------------
    # Context initialisation
    # ------------------------------------------------------------------

    def _initialize_s3_context(
        self,
        credentials: Optional[Dict[str, str]],
        url: str,
        region: str,
    ) -> tuple[Dict[str, str], tiledb.Ctx]:
        cores = os.cpu_count() or 8
        min(cores * 4, 64)
        min(cores * 8, 256)

        # endpoint_override must be hostname[:port] only — strip scheme if present
        endpoint = url.removeprefix("https://").removeprefix("http://").rstrip("/")

        cfg: Dict[str, str] = {
            # Endpoint (Ceph/MinIO — path-style, no virtual addressing)
            "vfs.s3.endpoint_override": endpoint,
            "vfs.s3.region": region,
            "vfs.s3.scheme": "https",
            "vfs.s3.use_virtual_addressing": "false",
            # Multipart upload — required for large LAZ files
            "vfs.s3.use_multipart_upload": "true",
            "vfs.s3.multipart_part_size": "52428800",  # 50 MB
            "vfs.s3.multipart_threshold": "52428800",
            "vfs.s3.max_parallel_ops": "8",
            # Timeouts and retries
            "vfs.s3.connect_timeout_ms": "60000",
            "vfs.s3.request_timeout_ms": "600000",
            "vfs.s3.backoff_scale": "2.0",
            "vfs.s3.backoff_max_ms": "120000",
        }

        if credentials:
            cfg["vfs.s3.aws_access_key_id"] = credentials.get("AccessKeyId", "")
            cfg["vfs.s3.aws_secret_access_key"] = credentials.get("SecretAccessKey", "")
            # Only set session token when actually present — empty string confuses TileDB
            if credentials.get("SessionToken"):
                cfg["vfs.s3.aws_session_token"] = credentials["SessionToken"]
            cfg["vfs.s3.no_sign_request"] = "false"
        else:
            cfg["vfs.s3.no_sign_request"] = "true"

        cfg.update(self.s3_config_overrides)
        return cfg, tiledb.Ctx(tiledb.Config(cfg))

    def _initialize_local_context(self) -> tuple[Dict[str, str], tiledb.Ctx]:
        cfg = {
            "py.init_buffer_bytes": str(4 * 1024**3),  # 4 GiB
            "sm.tile_cache_size": str(4 * 1024**3),  # 4 GiB
            "sm.num_reader_threads": "32",
            "sm.num_tiledb_threads": "32",
            "sm.compute_concurrency_level": "32",
            "sm.io_concurrency_level": "32",
        }
        return cfg, tiledb.Ctx(cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def array_exists(self) -> bool:
        """Return True if the TileDB array already exists."""
        return tiledb.array_exists(self.array_uri, ctx=self.ctx)

    def open(self, mode: str = "r") -> tiledb.Array:
        """Open the TileDB array in the given mode (``"r"`` or ``"w"``)."""
        return tiledb.open(self.array_uri, mode=mode, ctx=self.ctx)

    @property
    def schema(self) -> tiledb.ArraySchema:
        """Return (and cache) the array schema."""
        if self._schema_cache is None:
            with self.open("r") as arr:
                self._schema_cache = arr.schema
        return self._schema_cache
