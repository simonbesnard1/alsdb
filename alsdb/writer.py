from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import tiledb

from .config import S3Config, TileDBConfig
from .schema import LAS_ATTRIBUTES, create_schema

logger = logging.getLogger(__name__)


def _tiledb_ctx(s3: Optional[S3Config]) -> tiledb.Ctx:
    cfg = tiledb.Config()
    if s3 is not None:
        cfg["vfs.s3.region"] = s3.region
        if s3.endpoint_url:
            scheme = "https" if s3.endpoint_url.startswith("https") else "http"
            cfg["vfs.s3.scheme"] = scheme
            cfg["vfs.s3.endpoint_override"] = (
                s3.endpoint_url
                .removeprefix("https://")
                .removeprefix("http://")
            )
            cfg["vfs.s3.use_virtual_addressing"] = "false"
        if s3.access_key_id:
            cfg["vfs.s3.aws_access_key_id"] = s3.access_key_id
        if s3.secret_access_key:
            cfg["vfs.s3.aws_secret_access_key"] = s3.secret_access_key
    return tiledb.Ctx(cfg)


def build_uri(array_name: str, s3: Optional[S3Config]) -> str:
    """Construct a local path or s3:// URI for the TileDB array."""
    if s3 is None:
        return array_name
    prefix = s3.prefix.strip("/")
    if prefix:
        return f"s3://{s3.bucket}/{prefix}/{array_name}"
    return f"s3://{s3.bucket}/{array_name}"


def ensure_array(
    uri: str,
    tdb_cfg: TileDBConfig,
    s3: Optional[S3Config] = None,
    overwrite: bool = False,
) -> None:
    """Create the TileDB array if it does not exist (or overwrite it)."""
    ctx = _tiledb_ctx(s3)
    if tiledb.array_exists(uri, ctx=ctx):
        if overwrite:
            logger.info("Removing existing array %s", uri)
            tiledb.remove(uri, ctx=ctx)
        else:
            logger.debug("Array %s already exists", uri)
            return
    logger.info("Creating array %s", uri)
    schema = create_schema(tdb_cfg)
    tiledb.Array.create(uri, schema, ctx=ctx)


def write_points(
    uri: str,
    data: np.ndarray,
    s3: Optional[S3Config] = None,
) -> None:
    """Append a structured numpy array of LAS points to a TileDB array."""
    ctx = _tiledb_ctx(s3)
    attrs = {
        name: data[name].astype(dtype)
        for name, dtype in LAS_ATTRIBUTES.items()
        if name in data.dtype.names
    }
    with tiledb.open(uri, mode="w", ctx=ctx) as arr:
        arr[data["X"], data["Y"]] = attrs
    logger.debug("Wrote %d points to %s", len(data), uri)
