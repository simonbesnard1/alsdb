# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging
import os

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
    max_reader_threads:
        Per-query internal thread count for TileDB (``sm.num_reader_threads``,
        ``sm.num_tiledb_threads``, ``sm.compute_concurrency_level``,
        ``sm.io_concurrency_level``). Each query on this context can fan out
        internally to this many threads *inside libtiledb* — if the caller
        also runs several queries concurrently at the Python level (e.g. a
        ``ThreadPoolExecutor`` with ``n_workers`` workers), the two multiply,
        so this should be sized relative to that: e.g.
        ``max(1, os.cpu_count() // n_workers)``. Defaults to a conservative
        ``min(cpu_count, 8)`` for local storage (CPU-bound decompression) and
        ``min(cpu_count * 4, 64)`` for S3 (network-bound, tolerates more
        concurrent threads) when left unset.
    """

    def __init__(
        self,
        storage_type: str = "local",
        uri: str | None = None,
        url: str | None = None,
        region: str = "eu-central-1",
        credentials: dict[str, str] | None = None,
        s3_config_overrides: dict[str, str] | None = None,
        max_reader_threads: int | None = None,
    ) -> None:
        if not storage_type or not isinstance(storage_type, str):
            raise ValueError("'storage_type' must be a non-empty string.")

        self.storage_type = storage_type.lower()
        self.s3_config_overrides = s3_config_overrides or {}
        self.max_reader_threads = max_reader_threads

        if self.storage_type == "s3":
            if not uri:
                raise ValueError("'uri' must be provided when storage_type='s3'.")
            if not url:
                raise ValueError("'url' (S3 endpoint) must be provided when storage_type='s3'.")
            self.array_uri = uri
            self._raw_cfg, self.ctx = self._initialize_s3_context(credentials, url, region)

        elif self.storage_type == "local":
            if not uri:
                raise ValueError("'uri' must be provided when storage_type='local'.")
            self.array_uri = uri
            self._raw_cfg, self.ctx = self._initialize_local_context()

        else:
            raise ValueError(f"Invalid storage_type {storage_type!r}. Must be 'local' or 's3'.")

        self._schema_cache: tiledb.ArraySchema | None = None

    # ------------------------------------------------------------------
    # Context initialisation
    # ------------------------------------------------------------------

    def _initialize_s3_context(
        self,
        credentials: dict[str, str] | None,
        url: str,
        region: str,
    ) -> tuple[dict[str, str], tiledb.Ctx]:
        cores = os.cpu_count() or 8
        # S3 reads are network-bound (threads mostly wait on I/O), so a higher
        # default than the local/CPU-bound case is fine when unset.
        reader_threads = str(self.max_reader_threads or min(cores * 4, 64))

        # endpoint_override must be hostname[:port] only — strip scheme if present
        endpoint = url.removeprefix("https://").removeprefix("http://").rstrip("/")

        cfg: dict[str, str] = {
            # Endpoint (Ceph/MinIO — path-style, no virtual addressing)
            "vfs.s3.endpoint_override": endpoint,
            "vfs.s3.region": region,
            "vfs.s3.scheme": "https",
            "vfs.s3.use_virtual_addressing": "false",
            "sm.num_reader_threads": reader_threads,
            "sm.num_tiledb_threads": reader_threads,
            "sm.compute_concurrency_level": reader_threads,
            "sm.io_concurrency_level": reader_threads,
            # Ceph compatibility: disable chunked payload signing (XAmzContentSHA256Mismatch)
            "vfs.s3.aws_payload_signing": "false",
            # Multipart upload — required for large LAZ files
            "vfs.s3.use_multipart_upload": "true",
            "vfs.s3.multipart_part_size": "52428800",  # 50 MB
            "vfs.s3.multipart_threshold": "52428800",
            "vfs.s3.max_parallel_ops": "8",
            # Timeouts and retries
            "vfs.s3.connect_timeout_ms": "60000",
            "vfs.s3.request_timeout_ms": "600000",
            "vfs.s3.max_retries": "10",
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

    def _initialize_local_context(self) -> tuple[dict[str, str], tiledb.Ctx]:
        # Reader/compute/IO thread counts are per-query, inside libtiledb — and
        # callers commonly run several TileDB queries concurrently themselves
        # (e.g. run_tiled()'s ThreadPoolExecutor). A hardcoded 32 here means
        # N concurrent Python workers each fan out to 32 more internal threads,
        # oversubscribing the machine badly. Scale by core count instead, or
        # use max_reader_threads if the caller specified one explicitly.
        cores = os.cpu_count() or 8
        reader_threads = str(self.max_reader_threads or min(cores, 8))
        cfg = {
            # Initial per-query read buffer. 4 GiB was requested unconditionally
            # on every query regardless of actual tile size (typically tens of
            # MB) - with several concurrent worker threads this caused genuine
            # MemoryErrors. TileDB transparently grows this via incomplete-query
            # retries if a tile genuinely needs more.
            "py.init_buffer_bytes": str(256 * 1024**2),  # 256 MiB
            "sm.tile_cache_size": str(1 * 1024**3),  # 1 GiB
            "sm.num_reader_threads": reader_threads,
            "sm.num_tiledb_threads": reader_threads,
            "sm.compute_concurrency_level": reader_threads,
            "sm.io_concurrency_level": reader_threads,
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
