from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from .config import S3Config, TileDBConfig
from .reader import read_laz
from .writer import build_uri, ensure_array, write_points

logger = logging.getLogger(__name__)


def ingest(
    laz_path: str | Path,
    array_name: str,
    tdb_cfg: TileDBConfig | None = None,
    s3: Optional[S3Config] = None,
    overwrite: bool = False,
) -> str:
    """
    Ingest a LAZ/LAS file into a TileDB array.

    Parameters
    ----------
    laz_path:
        Source .laz / .las file.
    array_name:
        Local path or array name within the S3 bucket.
    tdb_cfg:
        TileDB schema / chunking configuration. Uses defaults if None.
    s3:
        S3 storage configuration. Writes locally if None.
    overwrite:
        Delete and recreate the array if it already exists.

    Returns
    -------
    str
        The URI of the created TileDB array.
    """
    if tdb_cfg is None:
        tdb_cfg = TileDBConfig()

    uri = build_uri(array_name, s3)
    ensure_array(uri, tdb_cfg, s3, overwrite=overwrite)

    total = 0
    for chunk in read_laz(laz_path, chunk_size=tdb_cfg.chunk_size):
        write_points(uri, chunk, s3)
        total += len(chunk)
        logger.info("Ingested %d points → %s (running total)", len(chunk), uri)

    logger.info("Finished: %d points total → %s", total, uri)
    return uri
