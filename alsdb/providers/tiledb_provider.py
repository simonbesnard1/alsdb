# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Low-level access to local and S3-backed TileDB arrays."""

import logging
import os

import tiledb

logger = logging.getLogger(__name__)


def local_uri(uri: str) -> str:
    """Absolute form of a local array path (URIs with a scheme are returned as given).

    TileDB resolves relative paths against the working directory it saw when
    first used in the process, not the current one: after ``os.chdir`` (e.g.
    ``%cd`` in Jupyter) a relative path would silently point to the old
    folder, and worker processes, which start from the current one, would not
    find the array. Resolving once, when the provider is made, avoids both.
    """
    uri = str(uri)
    return uri if "://" in uri else os.path.abspath(uri)


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
        Threads per query inside TileDB. Size it against concurrent Python
        workers, e.g. ``max(1, os.cpu_count() // n_workers)``; default
        ``min(cpus, 8)`` locally and ``min(cpus * 4, 64)`` on S3.
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
            self.array_uri = local_uri(uri)
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
            # Multipart upload — required for large LAZ files
            "vfs.s3.use_multipart_upload": "true",
            "vfs.s3.multipart_part_size": "52428800",  # 50 MB
            "vfs.s3.max_parallel_ops": "8",
            # Timeouts and retries. request_timeout_ms is an idle timeout (no byte
            # received for that long), not a limit on the whole transfer: 10 min
            # let a read on a connection that died in an S3 outage wait ~50 min
            # (x connect_max_tries) before failing, and cluster jobs sat for hours.
            # 60 s without a byte means the request is dead; a slow transfer is not
            # cut off. connect_max_tries and connect_scale_factor (ms, doubled per
            # try: 0.5, 1, 2, 4 s) are TileDB's retry settings; the max_retries /
            # backoff_* keys used before were not TileDB options and were ignored
            # (as were aws_payload_signing and multipart_threshold, now removed).
            "vfs.s3.connect_timeout_ms": "60000",
            "vfs.s3.request_timeout_ms": "60000",
            "vfs.s3.connect_max_tries": "5",
            "vfs.s3.connect_scale_factor": "500",
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
        # Thread counts are per query; callers often run queries concurrently,
        # so scale by core count unless max_reader_threads is given.
        cores = os.cpu_count() or 8
        reader_threads = str(self.max_reader_threads or min(cores, 8))
        cfg = {
            # Initial per-query read buffer; TileDB grows it when needed.
            "py.init_buffer_bytes": str(256 * 1024**2),  # 256 MiB
            "sm.tile_cache_size": str(1 * 1024**3),  # 1 GiB
            "sm.num_reader_threads": reader_threads,
            "sm.num_tiledb_threads": reader_threads,
            "sm.compute_concurrency_level": reader_threads,
            "sm.io_concurrency_level": reader_threads,
        }
        return cfg, tiledb.Ctx(cfg)

    # ------------------------------------------------------------------
    # Pickling (worker processes): the TileDB context is rebuilt from its config
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return the picklable state: everything but the TileDB context."""
        state = self.__dict__.copy()
        state.pop("ctx", None)
        state.pop("_exists", None)
        state["_schema_cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state and rebuild the TileDB context from its config."""
        self.__dict__.update(state)
        self.ctx = tiledb.Ctx(tiledb.Config(self._raw_cfg))

    def with_config(self, **overrides: str):
        """Return a copy whose TileDB context uses *overrides* on top of this config.

        Parameters
        ----------
        **overrides : str
            TileDB config keys and values, e.g. ``{"sm.tile_cache_size": "0"}``
            passed as a dict with ``**``.

        Returns
        -------
        TileDBProvider
            Same class and array, new context.
        """
        import copy

        clone = copy.copy(self)
        clone._raw_cfg = {**self._raw_cfg, **{k: str(v) for k, v in overrides.items()}}
        clone.ctx = tiledb.Ctx(tiledb.Config(clone._raw_cfg))
        clone._schema_cache = None
        clone._exists = False
        return clone

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def array_exists(self) -> bool:
        """Return True if the TileDB array already exists.

        One listing of the array's ``__schema`` folder, not an open:
        ``tiledb.array_exists`` opens the array, which reads the metadata of
        every fragment not yet consolidated. Ingest asks once per file, and on
        S3 that became millions of requests per batch (HTTP 429 from Ceph).
        An array once seen is remembered (``create(overwrite=True)`` forgets).
        """
        if getattr(self, "_exists", False):
            return True
        if tiledb.object_type(self.array_uri, ctx=self.ctx) != "array":
            return False
        schema_dir = self.array_uri.rstrip("/") + "/__schema"
        try:  # object_type also says "array" after a delete (TileDB SC-27854)
            exists = bool(tiledb.VFS(ctx=self.ctx).ls(schema_dir))
        except tiledb.TileDBError:
            exists = False
        self._exists = exists
        return exists

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
