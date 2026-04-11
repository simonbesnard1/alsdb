# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for TileDBSchemaConfig and create_schema."""

import pytest
import tiledb

from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig, create_schema

# ---------------------------------------------------------------------------
# TileDBSchemaConfig.for_crs
# ---------------------------------------------------------------------------


def test_for_crs_known_iberian_peninsula():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    assert cfg.domain_min_x == 100_000
    assert cfg.domain_max_x == 900_000


def test_for_crs_known_netherlands():
    cfg = TileDBSchemaConfig.for_crs("EPSG:28992")
    assert cfg.domain_min_x == -7_000
    assert cfg.domain_max_x == 300_000


def test_for_crs_unknown_falls_back_to_global():
    cfg = TileDBSchemaConfig.for_crs("EPSG:99999")
    assert cfg.domain_min_x == -20_000_000
    assert cfg.domain_max_x == 20_000_000


def test_for_crs_kwargs_override_domain():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830", chunk_size=500_000)
    assert cfg.chunk_size == 500_000
    # domain should still come from EPSG:25830
    assert cfg.domain_min_x == 100_000


# ---------------------------------------------------------------------------
# TileDBSchemaConfig.from_bbox
# ---------------------------------------------------------------------------


def test_from_bbox_adds_default_padding():
    bbox = (308_000.0, 4_688_000.0, 310_000.0, 4_690_000.0)
    cfg = TileDBSchemaConfig.from_bbox(bbox)
    assert cfg.domain_min_x == pytest.approx(308_000.0 - 50_000.0)
    assert cfg.domain_max_x == pytest.approx(310_000.0 + 50_000.0)
    assert cfg.domain_min_y == pytest.approx(4_688_000.0 - 50_000.0)
    assert cfg.domain_max_y == pytest.approx(4_690_000.0 + 50_000.0)


def test_from_bbox_custom_padding():
    bbox = (0.0, 0.0, 100.0, 100.0)
    cfg = TileDBSchemaConfig.from_bbox(bbox, padding=10.0)
    assert cfg.domain_min_x == pytest.approx(-10.0)
    assert cfg.domain_max_x == pytest.approx(110.0)


def test_from_bbox_kwargs_override():
    bbox = (0.0, 0.0, 100.0, 100.0)
    cfg = TileDBSchemaConfig.from_bbox(bbox, chunk_size=100_000)
    assert cfg.chunk_size == 100_000


# ---------------------------------------------------------------------------
# create_schema
# ---------------------------------------------------------------------------


def test_create_schema_is_sparse():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    assert schema.sparse


def test_create_schema_allows_duplicates():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    assert schema.allows_duplicates


def test_create_schema_has_three_dimensions():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    assert schema.domain.ndim == 3


def test_create_schema_dimension_names():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    dim_names = {schema.domain.dim(i).name for i in range(3)}
    assert dim_names == {"X", "Y", "Year"}


def test_create_schema_attribute_count():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    assert schema.nattr == len(LAS_ATTRIBUTES)


def test_create_schema_all_attributes_present():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    attr_names = {schema.attr(i).name for i in range(schema.nattr)}
    assert attr_names == set(LAS_ATTRIBUTES.keys())


def test_create_schema_domain_bounds():
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    x_dim = schema.domain.dim("X")
    assert x_dim.domain[0] == pytest.approx(100_000.0)
    assert x_dim.domain[1] == pytest.approx(900_000.0)


def test_create_and_open_array(tmp_path):
    """Schema must be accepted by TileDB when creating a real array."""
    cfg = TileDBSchemaConfig.for_crs("EPSG:25830")
    schema = create_schema(cfg)
    uri = str(tmp_path / "test_array")
    tiledb.Array.create(uri, schema)
    assert tiledb.array_exists(uri)
