# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for GenericTileName and _parse_crs (tile_name.py)."""

import pytest

from alsdb.tile.tile_name import GenericTileName, _parse_crs, parse_tile_filename

# ---------------------------------------------------------------------------
# _parse_crs
# ---------------------------------------------------------------------------


def test_parse_crs_empty_dict_returns_epsg0():
    assert _parse_crs({}) == "EPSG:0"


def test_parse_crs_no_wkt_returns_epsg0():
    assert _parse_crs({"authority": "EPSG"}) == "EPSG:0"


def test_parse_crs_wkt_returned_when_no_pyproj(monkeypatch):
    # If pyproj is absent the function should fall back to the WKT prefix
    import builtins

    real_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("no pyproj")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)
    result = _parse_crs({"wkt": "PROJCRS[...]"})
    assert result.startswith("PROJCRS")


# ---------------------------------------------------------------------------
# GenericTileName.from_pdal_metadata
# ---------------------------------------------------------------------------


def _make_metadata(
    creation_year=0,
    minx=308_000.0,
    miny=4_688_000.0,
    maxx=310_000.0,
    maxy=4_690_000.0,
    wkt="",
):
    """Minimal PDAL metadata dict mimicking the structure PDAL produces."""
    return {
        "metadata": {
            "readers.las[0]": {
                "creation_year": creation_year,
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy,
                "srs": {"wkt": wkt} if wkt else {},
            }
        }
    }


def test_from_pdal_metadata_year_from_filename():
    meta = _make_metadata(creation_year=2019)
    tile = GenericTileName.from_pdal_metadata("ALS_2021_tile.laz", meta)
    # Filename year (2021) takes priority over header year (2019)
    assert tile.year == 2021


def test_from_pdal_metadata_year_from_header_when_no_filename_year():
    meta = _make_metadata(creation_year=2018)
    tile = GenericTileName.from_pdal_metadata("unknown_tile.laz", meta)
    assert tile.year == 2018


def test_from_pdal_metadata_year_zero_for_bad_header():
    meta = _make_metadata(creation_year=0)
    tile = GenericTileName.from_pdal_metadata("no_year.laz", meta)
    assert tile.year == 0


def test_from_pdal_metadata_bbox():
    meta = _make_metadata(minx=100.0, miny=200.0, maxx=300.0, maxy=400.0)
    tile = GenericTileName.from_pdal_metadata("tile_2022.laz", meta)
    assert tile.bbox_native == (100.0, 200.0, 300.0, 400.0)


def test_from_pdal_metadata_filename_stored():
    meta = _make_metadata()
    tile = GenericTileName.from_pdal_metadata("/some/path/tile_2022.laz", meta)
    assert tile.filename == "tile_2022.laz"


def test_from_pdal_metadata_repr_contains_key_info():
    meta = _make_metadata()
    tile = GenericTileName.from_pdal_metadata("tile_2022.laz", meta)
    r = repr(tile)
    assert "tile_2022.laz" in r
    assert "2022" in r


def test_from_pdal_metadata_stats_bbox_preferred():
    """filters.stats bbox, when present, should override readers.las bbox."""
    meta = _make_metadata(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0)
    meta["metadata"]["filters.stats[1]"] = {
        "bbox": {
            "native": {
                "bbox": {
                    "minx": 308_000.0,
                    "miny": 4_688_000.0,
                    "maxx": 310_000.0,
                    "maxy": 4_690_000.0,
                }
            }
        }
    }
    tile = GenericTileName.from_pdal_metadata("tile_2022.laz", meta)
    assert tile.bbox_native[0] == pytest.approx(308_000.0)


def test_from_pdal_metadata_empty_metadata():
    """Must not raise on completely empty metadata dict."""
    tile = GenericTileName.from_pdal_metadata("tile_2020.laz", {})
    assert tile.year == 2020
    assert tile.filename == "tile_2020.laz"


# ---------------------------------------------------------------------------
# PNOATileName — ensure protocol compliance (already tested in test_tile_name)
# ---------------------------------------------------------------------------


def test_pnoa_satisfies_tilebase_protocol():
    from alsdb.tile.tile_name import TileNameBase

    name = parse_tile_filename("PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz")
    assert isinstance(name, TileNameBase)


def test_generic_satisfies_tilebase_protocol():
    from alsdb.tile.tile_name import TileNameBase

    meta = _make_metadata()
    tile = GenericTileName.from_pdal_metadata("tile_2022.laz", meta)
    assert isinstance(tile, TileNameBase)
