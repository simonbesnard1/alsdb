# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
from collections.abc import Generator
from functools import cached_property
from pathlib import Path

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
        self._metadata: dict | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @cached_property
    def name(self) -> TileNameBase:
        """Tile metadata (year, bbox, CRS) read from the LAZ file header."""
        return GenericTileName.from_pdal_metadata(self._path, self.metadata)

    @property
    def metadata(self) -> dict:
        """PDAL metadata of the LAS header (year, CRS, bounds, count), cached.

        Reads only the header (``count: 0``): the previous full read with
        ``filters.stats`` decompressed every point just for the name, and the
        points were then read again for ingestion (about 13 of 32 s per PNOA
        tile on the cluster). :meth:`read` adds the ``filters.stats`` block from
        its own pass, so :attr:`native_bbox` reports the points' extent once
        the tile has been read.
        """
        if self._metadata is None:
            pipeline = pdal.Pipeline(
                json.dumps([{"type": "readers.las", "filename": str(self._path), "count": 0}])
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
        Return ``(min_x, min_y, max_x, max_y)`` in the file's native CRS:
        the points' extent from ``filters.stats`` once :meth:`read` has run,
        otherwise the header bounds.
        """
        blocks = self.metadata.get("metadata", {})
        for key, val in blocks.items():
            if "filters.stats" in key:
                b = val.get("bbox", {}).get("native", {}).get("bbox", {})
                if b:
                    return (b["minx"], b["miny"], b["maxx"], b["maxy"])
        for key, val in blocks.items():
            if "readers.las" in key and "minx" in val:
                return (val["minx"], val["miny"], val["maxx"], val["maxy"])
        raise ValueError(f"Could not extract bounding box from {self._path}")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(
        self,
        chunk_size: int | None = None,
        out_crs: str | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield structured numpy arrays of LAS points read via PDAL.

        Parameters
        ----------
        chunk_size:
            Maximum number of points per yielded array.
            If None, the entire file is returned as a single array.
        out_crs:
            If provided, reproject points to this CRS (e.g. ``"EPSG:32616"``)
            using ``filters.reprojection`` before returning.
        """
        # Stats before any reprojection: the native extent of the points,
        # computed in this single pass rather than in a separate full read.
        stages: list = [
            {"type": "readers.las", "filename": str(self._path)},
            {"type": "filters.stats"},
        ]
        if out_crs:
            stages.append({"type": "filters.reprojection", "out_srs": out_crs})
        pipeline = pdal.Pipeline(json.dumps(stages))
        pipeline.execute()
        meta = pipeline.metadata
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        stats = {k: v for k, v in meta.get("metadata", {}).items() if "filters.stats" in k}
        if stats:
            self.metadata.setdefault("metadata", {}).update(stats)

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
