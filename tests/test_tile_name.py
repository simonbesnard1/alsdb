# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import pytest

from alsdb.tile.tile_name import PNOATileName, parse_tile_filename

EXAMPLE = "PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz"


def test_parse_year():
    name = parse_tile_filename(EXAMPLE)
    assert name.year == 2021


def test_parse_region():
    name = parse_tile_filename(EXAMPLE)
    assert name.region == "CYL-NW"


def test_parse_tile_coords():
    name = parse_tile_filename(EXAMPLE)
    assert name.tile_x_km == 308
    assert name.tile_y_km == 4690


def test_parse_product():
    name = parse_tile_filename(EXAMPLE)
    assert name.product == "ORT-CLA-RGB"


def test_parse_from_full_path():
    name = parse_tile_filename(f"/data/als/{EXAMPLE}")
    assert name.tile_x_km == 308
    assert name.tile_y_km == 4690


def test_bbox_native_x_range():
    name = parse_tile_filename(EXAMPLE)
    min_x, _, max_x, _ = name.bbox_native
    assert min_x == pytest.approx(308_000.0)
    assert max_x == pytest.approx(310_000.0)


def test_bbox_native_y_range():
    name = parse_tile_filename(EXAMPLE)
    _, min_y, _, max_y = name.bbox_native
    assert max_y == pytest.approx(4_690_000.0)
    assert min_y == pytest.approx(4_688_000.0)


def test_bbox_tile_size():
    name = parse_tile_filename(EXAMPLE)
    min_x, min_y, max_x, max_y = name.bbox_native
    assert (max_x - min_x) == pytest.approx(2000.0)
    assert (max_y - min_y) == pytest.approx(2000.0)


def test_invalid_filename_raises():
    with pytest.raises(ValueError, match="does not match"):
        parse_tile_filename("random_cloud.laz")


def test_frozen_dataclass():
    name = parse_tile_filename(EXAMPLE)
    with pytest.raises(Exception):
        name.year = 2022  # type: ignore[misc]
