# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alsdb.providers.tiledb_provider import TileDBProvider
from alsdb.utils.constants import PNOA_TILE_SIZE_M
from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)

_ALL_ATTRS = list(LAS_ATTRIBUTES.keys())


class ALSProvider(TileDBProvider):
    """
    Read interface for the ALS TileDB array.

    Inherits storage and context management from
    :class:`~alsdb.providers.tiledb_provider.TileDBProvider` and exposes
    spatial queries that return :class:`pandas.DataFrame` or
    :class:`xarray.Dataset`.

    Parameters
    ----------
    storage_type:
        ``"local"`` or ``"s3"``.
    uri:
        Array URI (filesystem path or ``s3://`` URI).
    url:
        S3 endpoint URL (required for S3 storage).
    region:
        S3 region.
    credentials:
        S3 credentials dict (keys: ``"AccessKeyId"``, ``"SecretAccessKey"``,
        ``"SessionToken"``).
    s3_config_overrides:
        Raw TileDB ``vfs.s3.*`` overrides.
    """

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_bbox(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        attributes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Query all points within a bounding box.

        Parameters
        ----------
        min_x, min_y, max_x, max_y:
            Bounding box in the array's native CRS (UTM metres).
        attributes:
            Subset of LAS attribute names to return.  Returns all if ``None``.

        Returns
        -------
        pandas.DataFrame
            One row per point; columns are ``X``, ``Y``, and the requested attributes.
        """
        attrs = attributes or _ALL_ATTRS
        with self.open("r") as arr:
            data = arr.query(attrs=attrs)[min_x:max_x, min_y:max_y]
        df = pd.DataFrame(data)
        logger.debug(
            "query_bbox [%.0f–%.0f, %.0f–%.0f]: %d points",
            min_x, max_x, min_y, max_y, len(df),
        )
        return df

    def query_tile(
        self,
        tile_x_km: int,
        tile_y_km: int,
        attributes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Query all points within a PNOA tile identified by its km-grid coordinates.

        Parameters
        ----------
        tile_x_km:
            Tile easting in km (e.g. ``308`` covers 308 000 – 310 000 m).
        tile_y_km:
            Tile northing in km (e.g. ``4690`` covers 4 688 000 – 4 690 000 m).
        attributes:
            Subset of LAS attribute names to return.  Returns all if ``None``.

        Returns
        -------
        pandas.DataFrame
        """
        min_x = float(tile_x_km * 1000)
        max_x = min_x + PNOA_TILE_SIZE_M
        max_y = float(tile_y_km * 1000)
        min_y = max_y - PNOA_TILE_SIZE_M
        logger.debug("query_tile (%d, %d) → bbox %.0f–%.0f / %.0f–%.0f",
                     tile_x_km, tile_y_km, min_x, max_x, min_y, max_y)
        return self.query_bbox(min_x, min_y, max_x, max_y, attributes=attributes)

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        attributes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Alias for :meth:`query_bbox` — returns a :class:`pandas.DataFrame`."""
        return self.query_bbox(min_x, min_y, max_x, max_y, attributes=attributes)

    def to_xarray(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        attributes: Optional[List[str]] = None,
    ):
        """
        Query a bounding box and return an :class:`xarray.Dataset`.

        Parameters
        ----------
        min_x, min_y, max_x, max_y:
            Bounding box in UTM metres.
        attributes:
            Subset of LAS attributes to include.

        Returns
        -------
        xarray.Dataset
        """
        import xarray as xr

        df = self.query_bbox(min_x, min_y, max_x, max_y, attributes=attributes)
        ds = xr.Dataset.from_dataframe(df)
        ds.attrs.update({
            "crs": "EPSG:25830",
            "bbox": [min_x, min_y, max_x, max_y],
        })
        return ds

    def get_available_attributes(self) -> List[str]:
        """Return the list of attribute names present in the array schema."""
        return [self.schema.attr(i).name for i in range(self.schema.nattr)]
