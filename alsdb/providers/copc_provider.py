# SPDX-License-Identifier: EUPL-1.2
"""Spatial reads from COPC files for the ALS product pipeline."""

from __future__ import annotations

import json
import logging
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pdal

from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)

PDAL_DTYPES: dict[str, type] = {
    "X": np.float64,
    "Y": np.float64,
    **LAS_ATTRIBUTES,
}

BBox = tuple[float, float, float, float]


class COPCProvider:
    """Read spatial subsets from COPC files.

    The survey year is taken from the name of the directory containing each
    COPC file. File XY bounds are read once from the LAS header and cached.

    Parameters
    ----------
    files:
        Local COPC file paths. The immediate parent directory of each file
        must be named with its survey year (for example, ``2021``).
    """

    def __init__(self, files: Sequence[str | Path]) -> None:
        if not files:
            raise ValueError("At least one COPC file must be provided.")

        self.files = tuple(self._normalise_path(path) for path in files)

        print()

        # Cached index: (file path, year from parent directory, XY bounds)
        self._file_index: tuple[tuple[str, int, BBox], ...] = tuple(
            (path, self._read_file_year(path), self._read_file_bounds(path))
            for path in self.files
        )

    @staticmethod
    def _normalise_path(path: str | Path) -> str:
        value = str(path)
        if "://" in value:
            raise ValueError(
                "COPCProvider currently needs local files to read LAS headers."
            )
        return str(Path(value).expanduser().resolve())

    @staticmethod
    def _read_file_year(path: str) -> int:

        print(path)

        """Read the survey year from the COPC file's parent directory name."""
        directory_name = Path(path).parent.name
        try:
            return int(directory_name)
        except ValueError as exc:
            raise ValueError(
                f"Expected the parent directory to be a year, "
                f"but got {directory_name!r} for {path}"
            ) from exc

    @staticmethod
    def _read_file_bounds(path: str) -> BBox:
        """Read XY extents from the LAS public header."""
        with Path(path).open("rb") as stream:
            header = stream.read(227)

        if len(header) < 227 or header[:4] != b"LASF":
            raise ValueError(f"Not a valid LAS/LAZ/COPC header: {path}")

        # LAS header offsets 179..226: maxX, minX, maxY, minY, maxZ, minZ.
        max_x, min_x, max_y, min_y, _max_z, _min_z = struct.unpack_from(
            "<6d", header, 179
        )
        bounds = (min_x, min_y, max_x, max_y)

        if (
            not np.isfinite(bounds).all()
            or min_x > max_x
            or min_y > max_y
        ):
            raise ValueError(f"Invalid XY bounds in LAS header: {path}")

        return bounds

    @staticmethod
    def _intersects(a: BBox, b: BBox) -> bool:
        """Return whether two XY bounding boxes intersect."""
        return not (
            a[2] < b[0]
            or a[0] > b[2]
            or a[3] < b[1]
            or a[1] > b[3]
        )

    @staticmethod
    def _empty_array(attributes: Sequence[str]) -> np.ndarray:
        dtype = [(name, PDAL_DTYPES[name]) for name in ("X", "Y", *attributes)]
        return np.empty(0, dtype=dtype)

    def data_bbox(self) -> BBox:
        """Return the bounding box covering all configured COPC files."""

        if not self._file_index:
            raise RuntimeError("COPC provider contains no files.")

        bounds = [file_bbox for _, _, file_bbox in self._file_index]

        return (
            min(bbox[0] for bbox in bounds),  # min_x
            min(bbox[1] for bbox in bounds),  # min_y
            max(bbox[2] for bbox in bounds),  # max_x
            max(bbox[3] for bbox in bounds),  # max_y
        )

    def query_to_array(
        self,
        bbox: BBox | None,
        year: int | None = None,
        attributes: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Query COPC points and return a TileDB-compatible structured array.

        Parameters
        ----------
        bbox:
            Spatial filter ``(min_x, min_y, max_x, max_y)``. ``None`` reads
            all points from the selected file(s).
        year:
            Survey year taken from the COPC file's parent directory name.
            ``None`` reads all configured years.
        attributes:
            LAS attributes to return. ``None`` returns all known attributes.
        """
        attrs = list(
            LAS_ATTRIBUTES if attributes is None else dict.fromkeys(attributes)
        )
        unknown = set(attrs) - LAS_ATTRIBUTES.keys()
        if unknown:
            raise ValueError(f"Unknown LAS attributes: {sorted(unknown)}")

        query_bbox = None
        pdal_bounds = None
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox: {bbox!r}")

            query_bbox = tuple(map(float, bbox))
            if not (
                np.isfinite(query_bbox).all()
                and query_bbox[0] < query_bbox[2]
                and query_bbox[1] < query_bbox[3]
            ):
                raise ValueError(f"Invalid bbox: {bbox!r}")

            min_x, min_y, max_x, max_y = query_bbox
            pdal_bounds = f"([{min_x}, {max_x}], [{min_y}, {max_y}])"

        selected_year = None if year is None else int(year)
        files = [
            path
            for path, file_year, file_bbox in self._file_index
            if (selected_year is None or file_year == selected_year)
            and (query_bbox is None or self._intersects(file_bbox, query_bbox))
        ]

        if not files:
            return self._empty_array(attrs)

        fields = ("X", "Y", *attrs)
        dtype = [(name, PDAL_DTYPES[name]) for name in fields]
        arrays: list[np.ndarray] = []

        for filename in files:
            reader = {"type": "readers.copc", "filename": filename}
            if pdal_bounds is not None:
                reader["bounds"] = pdal_bounds

            pipeline = pdal.Pipeline(json.dumps([reader]))
            pipeline.execute()
            if not pipeline.arrays:
                continue

            data = pipeline.arrays[0]
            output = np.zeros(len(data), dtype=dtype)
            available_fields = set(data.dtype.names or ())
            for field in fields:
                if field in available_fields:
                    output[field] = data[field]

            arrays.append(output)
            logger.debug("Read %d points from %s", len(data), filename)

        if not arrays:
            return self._empty_array(attrs)
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays)
