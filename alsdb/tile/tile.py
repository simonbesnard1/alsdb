# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Read point data and metadata from one LAS or LAZ file."""

import json
import logging
import math
import os
import struct
from collections.abc import Generator
from functools import cached_property
from pathlib import Path

import numpy as np
import pdal

from alsdb.tile.tile_name import GenericTileName, TileNameBase

logger = logging.getLogger(__name__)


class CorruptLASError(ValueError):
    """A LAS/LAZ file whose structure is damaged (see :func:`check_las_file`)."""


def check_las_file(path: str | Path) -> None:
    """Check that a LAS/LAZ file is complete, reading only its header and end.

    A LAZ file whose end is missing or zero-filled (an incomplete copy) can
    make the LAZ decoder loop forever instead of raising an error; a 74 MB PNOA
    tile whose last 6.6 MB were zeros kept PDAL busy indefinitely. Reading only
    a few KB, this checks the LAS signature and point data offset and, for an
    uncompressed file, its size; for LAZ, that the chunk table pointer lies
    inside the file, that the table lists at least ``ceil(points / chunk
    size)`` chunks (fixed-size chunks only) and that the last 64 KB are not
    all zero bytes. The header layout follows the ASPRS LAS 1.4 specification
    (R15) and the chunk table the LASzip format (Isenburg 2013, "LASzip:
    lossless compression of LiDAR data", PE&RS 79(2), 209-217).

    Parameters
    ----------
    path : str or Path
        LAS or LAZ file.

    Raises
    ------
    CorruptLASError
        Describing the first problem found.
    """
    path = Path(path)
    size = os.path.getsize(path)

    def bad(reason):
        raise CorruptLASError(f"corrupt LAS/LAZ file {path.name}: {reason}")

    with path.open("rb") as f:
        head = f.read(375)
        if len(head) < 227 or head[:4] != b"LASF":
            bad("not a LAS file (no LASF signature)")
        header_size, data_offset, n_vlr = struct.unpack_from("<HII", head, 94)
        point_format, record_length = head[104], struct.unpack_from("<H", head, 105)[0]
        points = struct.unpack_from("<I", head, 107)[0]  # legacy count; 0 in LAS 1.4 beyond 2^32
        if header_size >= 375 and not points:
            points = struct.unpack_from("<Q", head, 247)[0]
        if data_offset > size:
            bad(f"point data offset {data_offset} beyond the file size {size}")
        if not point_format & 0x80:  # bit 7 set: LASzip-compressed point format
            need = data_offset + points * record_length
            if size < need:
                bad(f"truncated: {size} bytes, {need} needed for {points} points")
            return
        f.seek(header_size)
        vlrs = f.read(max(0, data_offset - header_size))
        chunk_size, pos = None, 0
        for _ in range(n_vlr):
            if pos + 54 > len(vlrs):
                break
            length = struct.unpack_from("<H", vlrs, pos + 20)[0]
            if vlrs[pos + 2 : pos + 18].rstrip(b"\0") == b"laszip encoded" and length >= 16:
                chunk_size = struct.unpack_from("<I", vlrs, pos + 54 + 12)[0]
            pos += 54 + length
        if chunk_size is None:
            bad("compressed point format but no LASzip VLR")
        f.seek(data_offset)
        table = struct.unpack("<q", f.read(8))[0]
        if table == -1:  # the writer could not seek back: pointer stored at the end of the file
            f.seek(size - 8)
            table = struct.unpack("<q", f.read(8))[0]
        if not data_offset + 8 <= table <= size - 8:
            bad(f"LAZ chunk table pointer {table} outside the file ({size} bytes)")
        f.seek(table)
        _, n_chunks = struct.unpack("<II", f.read(8))
        if chunk_size not in (0, 0xFFFFFFFF) and points:  # 0xFFFFFFFF: variable-size chunks
            expected = math.ceil(points / chunk_size)
            if n_chunks < expected:
                bad(
                    f"LAZ chunk table lists {n_chunks} chunks, {expected} expected for {points} points"
                )
        f.seek(max(data_offset, size - 65536))
        if not f.read().strip(b"\0"):
            bad("the last 64 KB of the file are all zero bytes (incomplete copy)")


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
        """Return the LAS or LAZ path."""
        return self._path

    @cached_property
    def name(self) -> TileNameBase:
        """Tile metadata (year, bbox, CRS) read from the LAZ file header."""
        return GenericTileName.from_pdal_metadata(self._path, self.metadata)

    @property
    def metadata(self) -> dict:
        """PDAL metadata of the LAS header (year, CRS, bounds, count), cached.

        Only the header is read; :meth:`read` adds point statistics once the tile
        has been read.
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
        """Return ``(min_x, min_y, max_x, max_y)`` in the native CRS.

        The result is the point extent after :meth:`read` has run and the
        header bounds otherwise.
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
        bbox: tuple[float, float, float, float] | None = None,
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
        bbox:
            Optional output-CRS bounds used to crop points after reprojection.

        Raises
        ------
        CorruptLASError
            If :func:`check_las_file` finds the file damaged; checked before
            PDAL reads any point.
        """
        check_las_file(self._path)  # a damaged LAZ can hang PDAL instead of raising
        # Stats before any reprojection: the native extent of the points,
        # computed in this single pass rather than in a separate full read.
        # Only the coordinates: statistics of every dimension cost a second
        # more per 10 M points and nothing reads them.
        stages: list = [
            {"type": "readers.las", "filename": str(self._path)},
            {"type": "filters.stats", "dimensions": "X,Y,Z"},
        ]
        if out_crs:
            stages.append({"type": "filters.reprojection", "out_srs": out_crs})
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            stages.append({"type": "filters.crop", "bounds": f"([{x0},{x1}],[{y0},{y1}])"})
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
