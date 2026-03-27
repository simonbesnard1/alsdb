from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
import pdal

logger = logging.getLogger(__name__)


def read_laz(
    path: str | Path,
    chunk_size: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Read a LAZ/LAS file via PDAL, yielding numpy structured arrays.

    Parameters
    ----------
    path:
        Path to the .laz or .las file.
    chunk_size:
        If given, yield arrays of at most this many points.
        If None, yield the whole file as a single array.
    """
    path = Path(path)
    logger.debug("Reading %s", path)

    pipeline = pdal.Pipeline(json.dumps([
        {"type": "readers.las", "filename": str(path)}
    ]))
    pipeline.execute()

    arrays = pipeline.arrays
    if not arrays:
        logger.warning("PDAL returned no arrays for %s", path)
        return

    data: np.ndarray = arrays[0]
    n = len(data)
    logger.info("Read %d points from %s", n, path)

    if chunk_size is None:
        yield data
        return

    for start in range(0, n, chunk_size):
        yield data[start : start + chunk_size]


def read_metadata(path: str | Path) -> dict:
    """Return PDAL pipeline metadata (stats, CRS, etc.) without loading all points."""
    path = Path(path)
    pipeline = pdal.Pipeline(json.dumps([
        {"type": "readers.las", "filename": str(path)},
        {"type": "filters.stats"},
    ]))
    pipeline.execute()
    return json.loads(pipeline.metadata)


def get_native_bbox(path: str | Path) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) in the file's native CRS."""
    meta = read_metadata(path)
    stages = meta.get("metadata", {})
    for key, val in stages.items():
        if "filters.stats" in key:
            bbox = val.get("bbox", {}).get("native", {}).get("bbox", {})
            if bbox:
                return (bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"])
    raise ValueError(f"Could not extract bbox from {path}")
