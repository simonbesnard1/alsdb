import numpy as np
import tiledb
import pytest
from alsdb.config import TileDBConfig
from alsdb.schema import create_schema, LAS_ATTRIBUTES


def test_schema_is_sparse():
    schema = create_schema(TileDBConfig())
    assert schema.sparse


def test_schema_allows_duplicates():
    schema = create_schema(TileDBConfig())
    assert schema.allows_duplicates


def test_schema_dimensions():
    cfg = TileDBConfig()
    schema = create_schema(cfg)
    assert schema.domain.ndim == 2
    dim_names = {schema.domain.dim(i).name for i in range(schema.domain.ndim)}
    assert dim_names == {"X", "Y"}


def test_schema_attribute_names():
    schema = create_schema(TileDBConfig())
    attr_names = {schema.attr(i).name for i in range(schema.nattr)}
    assert attr_names == set(LAS_ATTRIBUTES.keys())


def test_schema_tile_extent():
    cfg = TileDBConfig(tile_extent_x=100.0, tile_extent_y=200.0)
    schema = create_schema(cfg)
    x_dim = schema.domain.dim("X")
    y_dim = schema.domain.dim("Y")
    assert x_dim.tile == pytest.approx(100.0)
    assert y_dim.tile == pytest.approx(200.0)


def test_schema_domain_bounds():
    cfg = TileDBConfig(
        domain_min_x=200_000.0,
        domain_max_x=800_000.0,
        domain_min_y=4_000_000.0,
        domain_max_y=5_000_000.0,
    )
    schema = create_schema(cfg)
    x_dim = schema.domain.dim("X")
    assert x_dim.domain[0] == pytest.approx(200_000.0)
    assert x_dim.domain[1] == pytest.approx(800_000.0)
