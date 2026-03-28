# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pdal

from alsdb.tile.tile_name import GenericTileName, TileNameBase

logger = logging.getLogger(__name__)


class Tile:
    """
    Wrapper around any LAZ/LAS tile file.

    Provides lazy metadata access and a chunked point iterator backed by PDAL.
    Year, bounding box and CRS are read directly from the file header — no
    provider-specific filename convention is assumed.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._metadata: Optional[dict] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> TileNameBase:
        """Tile metadata (year, bbox, CRS) read from the LAZ file header."""
        return GenericTileName.from_pdal_metadata(self._path, self.metadata)

    @property
    def metadata(self) -> dict:
        """Lazily load and cache full PDAL pipeline metadata (runs ``filters.stats``)."""
        if self._metadata is None:
            pipeline = pdal.Pipeline(
                json.dumps([
                    {"type": "readers.las", "filename": str(self._path)},
                    {"type": "filters.stats"},
                ])
            )
            pipeline.execute()
            meta = pipeline.metadata
            self._metadata = meta if isinstance(meta, dict) else json.loads(meta)
        return self._metadata

    @property
    def n_points(self) -> int:
        """Total number of points in the file (from PDAL metadata)."""
        for key, val in self.metadata.get("metadata", {}).items():
            if "readers.las" in key:
                return int(val.get("count", 0))
        return 0

    @property
    def native_bbox(self) -> tuple[float, float, float, float]:
        """
        Return ``(min_x, min_y, max_x, max_y)`` in the file's native CRS,
        as reported by ``filters.stats``.
        """
        for key, val in self.metadata.get("metadata", {}).items():
            if "filters.stats" in key:
                b = val.get("bbox", {}).get("native", {}).get("bbox", {})
                if b:
                    return (b["minx"], b["miny"], b["maxx"], b["maxy"])
        raise ValueError(f"Could not extract bounding box from {self._path}")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(
        self, chunk_size: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield structured numpy arrays of LAS points read via PDAL.

        Parameters
        ----------
        chunk_size:
            Maximum number of points per yielded array.
            If None, the entire file is returned as a single array.
        """
        pipeline = pdal.Pipeline(
            json.dumps([{"type": "readers.las", "filename": str(self._path)}])
        )
        pipeline.execute()

        arrays = pipeline.arrays
        if not arrays:
            logger.warning("PDAL returned no arrays for %s", self._path.name)
            return

        data: np.ndarray = arrays[0]
        logger.debug("Loaded %d points from %s", len(data), self._path.name)

        if chunk_size is None:
            yield data
            return

        for start in range(0, len(data), chunk_size):
            yield data[start : start + chunk_size]
