# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import numpy as np
import pytest

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.core.alsprovider import ALSProvider
from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig

_SCHEMA_CFG = TileDBSchemaConfig(
    domain_min_x=100_000.0,
    domain_max_x=900_000.0,
    domain_min_y=3_000_000.0,
    domain_max_y=9_999_900.0,
)


@pytest.fixture()
def array_uri(tmp_path) -> str:
    """Populate a local TileDB array with 300 synthetic points and return its URI."""
    uri = str(tmp_path / "als_array")
    db = ALSDatabase(storage_type="local", uri=uri, schema_cfg=_SCHEMA_CFG)

    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(308_000.0, 310_000.0, n)
    y = rng.uniform(4_688_000.0, 4_690_000.0, n)
    attrs = {name: np.zeros(n, dtype=dtype) for name, dtype in LAS_ATTRIBUTES.items()}
    attrs["Z"] = rng.uniform(800.0, 850.0, n).astype(np.float64)
    attrs["ReturnNumber"] = np.ones(n, dtype=np.uint8)
    attrs["NumberOfReturns"] = np.ones(n, dtype=np.uint8)
    db.write(x, y, attrs)
    return uri


def test_query_bbox_returns_all_points(array_uri):
    provider = ALSProvider(storage_type="local", uri=array_uri)
    df = provider.query_bbox(308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0)
    assert len(df) == 300


def test_query_bbox_has_xyz_columns(array_uri):
    provider = ALSProvider(storage_type="local", uri=array_uri)
    df = provider.query_bbox(308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0)
    assert {"X", "Y", "Z"}.issubset(df.columns)


def test_query_bbox_spatial_filter(array_uri):
    provider = ALSProvider(storage_type="local", uri=array_uri)
    df_full = provider.query_bbox(308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0)
    df_half = provider.query_bbox(308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
    assert len(df_half) < len(df_full)


def test_query_tile_matches_bbox(array_uri):
    provider = ALSProvider(storage_type="local", uri=array_uri)
    df_tile = provider.query_tile(308, 4690)
    df_bbox = provider.query_bbox(308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0)
    assert len(df_tile) == len(df_bbox)


def test_query_attribute_subset(array_uri):
    provider = ALSProvider(storage_type="local", uri=array_uri)
    df = provider.query_bbox(
        308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0,
        attributes=["Z", "Classification"],
    )
    assert "Z" in df.columns
    assert "Classification" in df.columns
    assert "Intensity" not in df.columns


def test_to_xarray_returns_dataset(array_uri):
    xr = pytest.importorskip("xarray")
    provider = ALSProvider(storage_type="local", uri=array_uri)
    ds = provider.to_xarray(308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0)
    assert isinstance(ds, xr.Dataset)
    assert "Z" in ds


def test_get_available_attributes(array_uri):
    provider = ALSProvider(storage_type="local", uri=array_uri)
    attrs = provider.get_available_attributes()
    assert "Z" in attrs
    assert "Classification" in attrs
    assert len(attrs) == len(LAS_ATTRIBUTES)
