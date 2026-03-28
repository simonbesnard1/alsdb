# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure alsdb logging. Call once at the top of your script or notebook."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("alsdb").setLevel(level)


from alsdb.core.alsdatabase import ALSDatabase
from alsdb.core.alsprovider import ALSProvider
from alsdb.core.alstile import ALSTile
from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.tile.Tile import Tile
from alsdb.tile.tile_name import PNOATileName, parse_tile_filename
from alsdb.utils.constants import ALSProduct, PNOA_TILE_SIZE_M, UTM30N, WGS84
from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig, create_schema

__all__ = [
    # Core
    "ALSDatabase",
    "ALSProvider",
    "ALSTile",
    # Provider base
    "TileDBProvider",
    # Tile
    "Tile",
    "PNOATileName",
    "parse_tile_filename",
    # Utils
    "ALSProduct",
    "PNOA_TILE_SIZE_M",
    "UTM30N",
    "WGS84",
    "LAS_ATTRIBUTES",
    "TileDBSchemaConfig",
    "create_schema",
]
