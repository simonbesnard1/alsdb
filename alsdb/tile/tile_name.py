# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import re
from dataclasses import dataclass
from pathlib import Path

from alsdb.utils.constants import PNOA_TILE_SIZE_M

# ---------------------------------------------------------------------------
# Filename pattern
# Example: PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz
#          PNOA_<year>_<region>_<tile_x_km>-<tile_y_km>_<product>.laz
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


@dataclass(frozen=True)
class PNOATileName:
    """
    Metadata derived from a PNOA LAZ tile filename.

    Attributes
    ----------
    year:
        Acquisition year (e.g. 2021).
    region:
        Survey region code (e.g. ``"CYL-NW"``).
    tile_x_km:
        UTM easting of the tile's *left* edge in km (e.g. 308 → 308 000 m).
    tile_y_km:
        UTM northing of the tile's *top* edge in km (e.g. 4690 → 4 690 000 m).
    product:
        Product descriptor string (e.g. ``"ORT-CLA-RGB"``).
    """

    year: int
    region: str
    tile_x_km: int
    tile_y_km: int
    product: str

    @property
    def bbox_native(self) -> tuple[float, float, float, float]:
        """
        Return ``(min_x, min_y, max_x, max_y)`` in UTM metres.

        The PNOA grid uses 2 km × 2 km tiles.  The filename encodes the
        left (west) easting and the top (north) northing in km, so:

        - ``min_x = tile_x_km * 1000``
        - ``max_x = min_x + 2000``
        - ``max_y = tile_y_km * 1000``
        - ``min_y = max_y - 2000``
        """
        min_x = float(self.tile_x_km * 1000)
        max_x = min_x + PNOA_TILE_SIZE_M
        max_y = float(self.tile_y_km * 1000)
        min_y = max_y - PNOA_TILE_SIZE_M
        return (min_x, min_y, max_x, max_y)


def parse_tile_filename(filename: str | Path) -> PNOATileName:
    """
    Parse a PNOA LAZ filename and return a :class:`PNOATileName`.

    Parameters
    ----------
    filename:
        File path or bare filename.  Only the basename is examined.

    Returns
    -------
    PNOATileName

    Raises
    ------
    ValueError
        If the filename does not match the expected PNOA naming convention.
    """
    stem = Path(filename).name
    match = PNOA_FILENAME_PATTERN.match(stem)
    if match is None:
        raise ValueError(
            f"Filename {stem!r} does not match the expected PNOA pattern: "
            f"{PNOA_FILENAME_PATTERN.pattern}"
        )
    return PNOATileName(
        year=int(match.group("year")),
        region=match.group("region"),
        tile_x_km=int(match.group("tile_x_km")),
        tile_y_km=int(match.group("tile_y_km")),
        product=match.group("product"),
    )
