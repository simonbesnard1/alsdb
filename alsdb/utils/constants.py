# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

from enum import Enum


class ALSProduct(Enum):
    """
    Enum representing PNOA ALS data product types.

    Attributes:
        ORT_CLA_RGB: Orthometric, classified, RGB-colourised point cloud.
        ORT_CLA:     Orthometric, classified point cloud (no colour).
    """

    ORT_CLA_RGB = "ORT-CLA-RGB"
    ORT_CLA = "ORT-CLA"

    @classmethod
    def list_products(cls) -> list[str]:
        return [p.value for p in cls]


# PNOA tile side length in metres (2 km × 2 km grid)
PNOA_TILE_SIZE_M: float = 2000.0

# Coordinate reference systems
UTM30N = "EPSG:25830"  # ETRS89 / UTM Zone 30N — native CRS of the PNOA data
WGS84 = "EPSG:4326"
