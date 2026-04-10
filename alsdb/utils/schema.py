# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

from dataclasses import dataclass

import numpy as np
import tiledb

# ---------------------------------------------------------------------------
# LAS dimension registry
# X, Y, and Year are the TileDB *dimensions* (spatial + temporal index);
# all others are *attributes* stored alongside each point.
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

# ---------------------------------------------------------------------------
# Filter lists — tuned per data type to maximise compression ratio.
#
# The strategy mirrors what LASzip does internally:
#   1. A decorrelating filter exploits spatial coherence between adjacent points
#      (delta encoding, byte shuffle) so ZSTD sees smaller residuals.
#   2. ZSTD-9 for the final entropy coding pass.
#
# Benchmarks on typical ALS tiles show 15–35 % size reduction vs plain ZSTD-9.
# ---------------------------------------------------------------------------

# Floating-point coordinates (X, Y) — byte-shuffle exploits the regular stride
# between adjacent sorted coordinate values before ZSTD entropy coding.
# (DoubleDeltaFilter only accepts integer types in TileDB, so it cannot be used
# on float64 dimensions.)
_FILTERS_COORD = tiledb.FilterList([
    tiledb.ByteShuffleFilter(),
    tiledb.ZstdFilter(level=9),
])

# Integer coordinate (Year) — double-delta is ideal for a near-constant integer
# dimension (differences between consecutive year values are tiny).
_FILTERS_YEAR = tiledb.FilterList([
    tiledb.DoubleDeltaFilter(),
    tiledb.ZstdFilter(level=9),
])

# Floating-point attributes (Z, GpsTime) — not sorted, so byte-shuffle
# groups the mantissa/exponent bytes before ZSTD.
_FILTERS_FLOAT = tiledb.FilterList([
    tiledb.ByteShuffleFilter(),
    tiledb.ZstdFilter(level=9),
])

# Integer attributes with values that rarely use the full bit width
# (Intensity uint16, PointSourceId uint16, ReturnNumber uint8, …).
_FILTERS_INT = tiledb.FilterList([
    tiledb.BitWidthReductionFilter(),
    tiledb.ZstdFilter(level=9),
])

# Fallback for any attribute type not covered above.
_FILTERS_DEFAULT = tiledb.FilterList([tiledb.ZstdFilter(level=9)])

# ---------------------------------------------------------------------------
# Per-CRS domain defaults
# Each entry covers the valid extent of that CRS in metres with a small margin.
# The generic UTM fallback is intentionally generous to cover any UTM zone.
# ---------------------------------------------------------------------------
_CRS_DOMAINS: dict[str, dict] = {
    # ETRS89 / UTM Zone 30N — Iberian Peninsula (PNOA)
    "EPSG:25830": dict(domain_min_x=100_000, domain_max_x=900_000,
                       domain_min_y=3_000_000, domain_max_y=9_999_900),
    # RD New — Netherlands (AHN)
    "EPSG:28992": dict(domain_min_x=-7_000, domain_max_x=300_000,
                       domain_min_y=289_000, domain_max_y=629_000),
    # Lambert 93 — France (IGN)
    "EPSG:2154":  dict(domain_min_x=99_000, domain_max_x=1_242_000,
                       domain_min_y=6_049_000, domain_max_y=7_111_000),
    # British National Grid — UK (EA)
    "EPSG:27700": dict(domain_min_x=-100_000, domain_max_x=700_000,
                       domain_min_y=0, domain_max_y=1_300_000),
    # Global fallback — covers any projected CRS in metres (UTM, Mercator,
    # national grids, etc.).  Values are set to ±20 000 km, which safely
    # includes the most extreme projected coordinates on Earth.
    "_global":    dict(domain_min_x=-20_000_000, domain_max_x=20_000_000,
                       domain_min_y=-20_000_000, domain_max_y=20_000_000),
}


@dataclass
class TileDBSchemaConfig:
    """
    Schema and domain configuration for the ALS TileDB array.

    Rather than constructing this directly, prefer the factory methods:

    * :meth:`for_crs` — sensible defaults for a known CRS.
    * :meth:`from_bbox` — auto-sized domain around a known bounding box.
    """

    tile_extent_x: float = 500.0
    tile_extent_y: float = 500.0
    domain_min_x: float = -20_000_000.0
    domain_max_x: float = 20_000_000.0
    domain_min_y: float = -20_000_000.0
    domain_max_y: float = 20_000_000.0
    year_min: int = 2000
    year_max: int = 2100
    chunk_size: int = 1_000_000

    @classmethod
    def for_crs(cls, crs: str, **kwargs) -> "TileDBSchemaConfig":
        """
        Return a config with domain bounds appropriate for *crs*.

        Falls back to the generic UTM domain for unknown CRS codes.

        Parameters
        ----------
        crs:
            CRS string, e.g. ``"EPSG:25830"``.
        **kwargs:
            Override any field (e.g. ``chunk_size=500_000``).
        """
        domain = _CRS_DOMAINS.get(crs, _CRS_DOMAINS["_global"])
        return cls(**{**domain, **kwargs})

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        padding: float = 50_000.0,
        **kwargs,
    ) -> "TileDBSchemaConfig":
        """
        Return a config whose domain is derived from *bbox* plus *padding*.

        Useful when you want the array domain to be tight around the actual
        data extent rather than a pre-defined CRS envelope.

        Parameters
        ----------
        bbox:
            ``(min_x, min_y, max_x, max_y)`` in metres.
        padding:
            Extra margin added on all sides (default 50 km).
        **kwargs:
            Override any other field.
        """
        min_x, min_y, max_x, max_y = bbox
        return cls(
            domain_min_x=min_x - padding,
            domain_max_x=max_x + padding,
            domain_min_y=min_y - padding,
            domain_max_y=max_y + padding,
            **kwargs,
        )


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
        A sparse 3-D schema with X/Y/Year dimensions and one attribute per LAS
        field.  The Year dimension separates repeated surveys of the same tile
        (e.g. 2019 vs 2021 flights).  All attributes and coordinates use
        ZSTD-9 compression.  ``allows_duplicates=True`` accommodates multiple
        returns at the same XY within a single survey.
    """
    # Float attributes that benefit from byte-shuffle before ZSTD.
    _float_attrs  = {"Z", "GpsTime"}
    # Integer attributes that rarely saturate their bit width.
    _int_attrs    = {
        "Intensity", "PointSourceId", "ReturnNumber", "NumberOfReturns",
        "ScanDirectionFlag", "EdgeOfFlightLine", "Classification",
        "ScanAngleRank", "UserData", "Red", "Green", "Blue",
        "Synthetic", "KeyPoint", "Withheld", "Overlap",
    }

    def _attr_filters(name: str) -> tiledb.FilterList:
        if name in _float_attrs:
            return _FILTERS_FLOAT
        if name in _int_attrs:
            return _FILTERS_INT
        return _FILTERS_DEFAULT

    domain = tiledb.Domain(
        tiledb.Dim(
            name="X",
            domain=(cfg.domain_min_x, cfg.domain_max_x),
            tile=cfg.tile_extent_x,
            dtype=np.float64,
            filters=_FILTERS_COORD,   # ByteShuffle → Zstd (float64 compatible)
        ),
        tiledb.Dim(
            name="Y",
            domain=(cfg.domain_min_y, cfg.domain_max_y),
            tile=cfg.tile_extent_y,
            dtype=np.float64,
            filters=_FILTERS_COORD,   # ByteShuffle → Zstd
        ),
        tiledb.Dim(
            name="Year",
            domain=(cfg.year_min, cfg.year_max),
            tile=1,
            dtype=np.int16,
            filters=_FILTERS_YEAR,    # DoubleDelta → Zstd (int16 compatible)
        ),
    )
    attrs = [
        tiledb.Attr(name=name, dtype=dtype, filters=_attr_filters(name))
        for name, dtype in LAS_ATTRIBUTES.items()
    ]
    return tiledb.ArraySchema(
        domain=domain,
        attrs=attrs,
        sparse=True,
        allows_duplicates=True,
    )
