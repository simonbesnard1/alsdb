from __future__ import annotations
import numpy as np
import tiledb
from .config import TileDBConfig

# LAS dimension name -> (numpy dtype)
# X and Y are TileDB dimensions; everything else is an attribute.
LAS_ATTRIBUTES: dict[str, np.dtype] = {
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


def create_schema(cfg: TileDBConfig) -> tiledb.ArraySchema:
    """Build a sparse TileDB schema for LAS point-cloud data."""
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
        allows_duplicates=True,   # multiple returns share the same XY
        coords_filters=_ZSTD9,
    )
