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

    All query methods accept an optional ``year`` parameter.  When provided,
    only points from that survey year are returned.  When ``None``, all years
    are returned and the result DataFrame includes a ``Year`` column.

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _year_range(self) -> tuple[int, int]:
        """Return (year_min, year_max) from the array schema."""
        dim = self.schema.domain.dim("Year")
        return int(dim.domain[0]), int(dim.domain[1])

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
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query all points within a bounding box.

        Parameters
        ----------
        min_x, min_y, max_x, max_y:
            Bounding box in the array's native CRS (UTM metres).
        attributes:
            Subset of LAS attribute names to return.  Returns all if ``None``.
        year:
            If provided, restrict to this survey year only.
            If ``None``, all years are returned (result includes a ``Year`` column).

        Returns
        -------
        pandas.DataFrame
            One row per point; columns are ``X``, ``Y``, ``Year``, and the
            requested attributes.
        """
        attrs = attributes or _ALL_ATTRS
        y0, y1_inc = (year, year) if year is not None else self._year_range()
        # +1: TileDB-Py int-dimension slices are exclusive-end (like Python slices)
        y1 = y1_inc + 1

        with self.open("r") as arr:
            data = arr.query(attrs=attrs)[min_x:max_x, min_y:max_y, y0:y1]

        df = pd.DataFrame(data)
        logger.debug(
            "query_bbox [%.0f–%.0f, %.0f–%.0f, year=%s]: %d points",
            min_x, max_x, min_y, max_y, year or "all", len(df),
        )
        return df

    def query_tile(
        self,
        tile_x_km: int,
        tile_y_km: int,
        attributes: Optional[List[str]] = None,
        year: Optional[int] = None,
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
        year:
            If provided, restrict to this survey year only.

        Returns
        -------
        pandas.DataFrame
        """
        min_x = float(tile_x_km * 1000)
        max_x = min_x + PNOA_TILE_SIZE_M
        max_y = float(tile_y_km * 1000)
        min_y = max_y - PNOA_TILE_SIZE_M
        logger.debug("query_tile (%d, %d, year=%s) → bbox %.0f–%.0f / %.0f–%.0f",
                     tile_x_km, tile_y_km, year or "all", min_x, max_x, min_y, max_y)
        return self.query_bbox(min_x, min_y, max_x, max_y, attributes=attributes, year=year)

    def available_years(self) -> List[int]:
        """
        Return the sorted list of survey years present in the array.

        Reads only the ``Year`` dimension column — no attribute data is fetched.
        """
        with self.open("r") as arr:
            data = arr.query(attrs=[], dims=["Year"])[:]
        return sorted(int(y) for y in np.unique(data["Year"]))

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
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Alias for :meth:`query_bbox` — returns a :class:`pandas.DataFrame`."""
        return self.query_bbox(min_x, min_y, max_x, max_y,
                               attributes=attributes, year=year)

    def to_xarray(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        attributes: Optional[List[str]] = None,
        year: Optional[int] = None,
    ):
        """
        Query a bounding box and return an :class:`xarray.Dataset`.

        Parameters
        ----------
        min_x, min_y, max_x, max_y:
            Bounding box in UTM metres.
        attributes:
            Subset of LAS attributes to include.
        year:
            If provided, restrict to this survey year only.

        Returns
        -------
        xarray.Dataset
        """
        import xarray as xr

        df = self.query_bbox(min_x, min_y, max_x, max_y,
                             attributes=attributes, year=year)
        ds = xr.Dataset.from_dataframe(df)
        ds.attrs.update({
            "crs": "EPSG:25830",
            "bbox": [min_x, min_y, max_x, max_y],
            "year": year,
        })
        return ds

    def get_available_attributes(self) -> List[str]:
        """Return the list of attribute names present in the array schema."""
        return [self.schema.attr(i).name for i in range(self.schema.nattr)]
