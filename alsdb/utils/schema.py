# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

from dataclasses import dataclass

import numpy as np
import tiledb

# ---------------------------------------------------------------------------
# LAS dimension registry
# X and Y are the TileDB *dimensions* (spatial index); all others are
# *attributes* stored alongside each point.
# ---------------------------------------------------------------------------
LAS_ATTRIBUTES: dict[str, type] = {
    "Z": np.float64,
    "Intensity": np.uint16,
    "ReturnNumber": np.uint8,
    "NumberOfReturns": np.uint8,
    "ScanDirectionFlag": np.uint8,
    "EdgeOfFlightLine": np.uint8,
    "Classification": np.uint8,
    "ScanAngleRank": np.int8,
    "UserData": np.uint8,
    "PointSourceId": np.uint16,
    "Red": np.uint16,
    "Green": np.uint16,
    "Blue": np.uint16,
    "GpsTime": np.float64,
    "Synthetic": np.uint8,
    "KeyPoint": np.uint8,
    "Withheld": np.uint8,
    "Overlap": np.uint8,
}

_ZSTD9 = tiledb.FilterList([tiledb.ZstdFilter(level=9)])


@dataclass
class TileDBSchemaConfig:
    """
    Schema and domain configuration for the ALS TileDB array.

    The default domain covers ETRS89 / UTM Zone 30N (EPSG:25830) at a scale
    suitable for Iberian Peninsula PNOA surveys.
    """

    tile_extent_x: float = 500.0
    tile_extent_y: float = 500.0
    domain_min_x: float = 100_000.0
    domain_max_x: float = 900_000.0
    domain_min_y: float = 3_000_000.0
    domain_max_y: float = 9_999_900.0
    chunk_size: int = 1_000_000


def create_schema(cfg: TileDBSchemaConfig) -> tiledb.ArraySchema:
    """
    Build a sparse TileDB array schema for LAS point-cloud data.

    Parameters
    ----------
    cfg:
        Domain and tiling configuration.

    Returns
    -------
    tiledb.ArraySchema
        A sparse 2-D schema with X/Y spatial dimensions and one attribute
        per LAS dimension.  All attributes and coordinates use ZSTD-9 compression.
        ``allows_duplicates=True`` accommodates multiple returns at the same XY.
    """
    domain = tiledb.Domain(
        tiledb.Dim(
            name="X",
            domain=(cfg.domain_min_x, cfg.domain_max_x),
            tile=cfg.tile_extent_x,
            dtype=np.float64,
        ),
        tiledb.Dim(
            name="Y",
            domain=(cfg.domain_min_y, cfg.domain_max_y),
            tile=cfg.tile_extent_y,
            dtype=np.float64,
        ),
    )
    attrs = [
        tiledb.Attr(name=name, dtype=dtype, filters=_ZSTD9)
        for name, dtype in LAS_ATTRIBUTES.items()
    ]
    return tiledb.ArraySchema(
        domain=domain,
        attrs=attrs,
        sparse=True,
        allows_duplicates=True,
        coords_filters=_ZSTD9,
    )
