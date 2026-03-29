# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from alsdb.utils.constants import PNOA_TILE_SIZE_M


# ---------------------------------------------------------------------------
# Protocol — the interface every tile-name implementation must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class TileNameBase(Protocol):
    """
    Structural interface for tile-name objects.

    Any class that exposes ``year``, ``bbox_native``, and ``crs`` satisfies
    this protocol — no explicit inheritance required.
    """

    @property
    def year(self) -> int:
        """Acquisition year (e.g. 2021)."""
        ...

    @property
    def bbox_native(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` in the file's native CRS (metres)."""
        ...

    @property
    def crs(self) -> str:
        """CRS as an EPSG string, e.g. ``"EPSG:25830"``."""
        ...


# ---------------------------------------------------------------------------
# PNOA Spain
# ---------------------------------------------------------------------------

PNOA_FILENAME_PATTERN = re.compile(
    r"PNOA"
    r"_(?P<year>\d{4})"
    r"_(?P<region>[A-Za-z0-9]+-[A-Za-z0-9]+)"
    r"_(?P<tile_x_km>\d+)-(?P<tile_y_km>\d+)"
    r"_(?P<product>.+?)"
    r"\.laz$",
    re.IGNORECASE,
)

_PNOA_CRS = "EPSG:25830"


@dataclass(frozen=True)
class PNOATileName:
    """
    Metadata derived from a PNOA LAZ tile filename.

    Satisfies :class:`TileNameBase` via structural (Protocol) subtyping.
    """

    year: int
    region: str
    tile_x_km: int
    tile_y_km: int
    product: str

    @property
    def bbox_native(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` in ETRS89 / UTM Zone 30N (m)."""
        min_x = float(self.tile_x_km * 1000)
        max_x = min_x + PNOA_TILE_SIZE_M
        max_y = float(self.tile_y_km * 1000)
        min_y = max_y - PNOA_TILE_SIZE_M
        return (min_x, min_y, max_x, max_y)

    @property
    def crs(self) -> str:
        return _PNOA_CRS


def parse_tile_filename(filename: str | Path) -> PNOATileName:
    """
    Parse a PNOA LAZ filename and return a :class:`PNOATileName`.

    Raises
    ------
    ValueError
        If the filename does not match the PNOA naming convention.
    """
    stem = Path(filename).name
    match = PNOA_FILENAME_PATTERN.match(stem)
    if match is None:
        raise ValueError(
            f"Filename {stem!r} does not match the expected PNOA pattern."
        )
    return PNOATileName(
        year=int(match.group("year")),
        region=match.group("region"),
        tile_x_km=int(match.group("tile_x_km")),
        tile_y_km=int(match.group("tile_y_km")),
        product=match.group("product"),
    )


# ---------------------------------------------------------------------------
# Generic — reads year, bbox, and CRS from LAZ header via PDAL metadata
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


@dataclass(frozen=True)
class GenericTileName:
    """
    Tile name derived from LAZ file header metadata via PDAL.

    Used as a fallback for any dataset whose filename does not match a known
    provider pattern (USGS 3DEP, AHN Netherlands, IGN France, etc.).

    Satisfies :class:`TileNameBase` via structural (Protocol) subtyping.
    """

    year: int
    _bbox: tuple[float, float, float, float]
    _crs: str
    filename: str

    @property
    def bbox_native(self) -> tuple[float, float, float, float]:
        return self._bbox

    @property
    def crs(self) -> str:
        return self._crs

    def __repr__(self) -> str:
        return (
            f"GenericTileName(filename={self.filename!r}, year={self.year}, "
            f"crs={self._crs!r}, bbox={self._bbox})"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_pdal_metadata(
        path: str | Path,
        metadata: dict,
    ) -> "GenericTileName":
        """
        Build a :class:`GenericTileName` from a PDAL metadata dict.

        Parameters
        ----------
        path:
            Path to the LAZ file (used for filename-based year fallback).
        metadata:
            Dict as returned by ``json.loads(pdal.Pipeline(...).metadata)``.
        """
        meta = metadata.get("metadata", {})

        # Find the readers.las metadata block
        reader_key = next(
            (k for k in meta if "readers.las" in k),
            None,
        )
        reader_meta: dict = meta.get(reader_key, {}) if reader_key else {}

        # --- Year ---
        # Filename year (e.g. "PNOA_2021_...", "AHN_2020_...") is the survey
        # year and takes priority.  The LAZ header creation_year is often the
        # file processing/delivery year, which can differ by 1–2 years.
        m = _YEAR_RE.search(Path(path).name)
        if m:
            year = int(m.group())
        else:
            year = int(reader_meta.get("creation_year", 0) or 0)
            if year < 1980 or year > 2100:
                year = 0

        # --- BBox ---
        minx = reader_meta.get("minx") or reader_meta.get("maxx", 0)
        miny = reader_meta.get("miny") or reader_meta.get("maxy", 0)
        maxx = reader_meta.get("maxx", 0)
        maxy = reader_meta.get("maxy", 0)

        # Prefer filters.stats bbox if present (more precise)
        stats_key = next(
            (k for k in meta if "filters.stats" in k),
            None,
        )
        if stats_key:
            b = meta[stats_key].get("bbox", {}).get("native", {}).get("bbox", {})
            if b:
                minx = b.get("minx", minx)
                miny = b.get("miny", miny)
                maxx = b.get("maxx", maxx)
                maxy = b.get("maxy", maxy)

        bbox = (float(minx), float(miny), float(maxx), float(maxy))

        # --- CRS ---
        crs_str = _parse_crs(reader_meta.get("srs", {}))

        return GenericTileName(
            year=year,
            _bbox=bbox,
            _crs=crs_str,
            filename=Path(path).name,
        )


def _parse_crs(srs: dict) -> str:
    """
    Extract a CRS string from a PDAL SRS metadata dict.

    Returns an ``"EPSG:XXXX"`` string when possible, falling back to the
    WKT string or ``"EPSG:0"`` if nothing can be determined.
    """
    if not srs:
        return "EPSG:0"

    wkt = srs.get("wkt") or srs.get("compoundwkt", "")
    if not wkt:
        return "EPSG:0"

    try:
        from pyproj import CRS as ProjCRS
        crs_obj = ProjCRS.from_wkt(wkt)
        epsg = crs_obj.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        # Fallback: authority code from the CRS object
        auth = crs_obj.to_authority()
        if auth:
            return f"{auth[0]}:{auth[1]}"
    except (ImportError, AttributeError, ValueError):
        pass

    # Last resort: first 120 chars of WKT name
    return wkt[:120]
