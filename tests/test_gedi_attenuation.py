# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""GEDI attenuation pipeline (alsdb.products.gedi_attenuation)."""

import numpy as np
import pandas as pd
import pyproj
import pytest

from alsdb.products import gedi_attenuation as ga
from alsdb.products._tiling import array_crs, array_data_bbox


def _shots(provider, n=40, seed=0):
    """Synthetic gedidb query result: shots scattered over and around the ALS extent."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = array_data_bbox(provider)
    x, y = rng.uniform(x0 - 100, x1 + 100, n), rng.uniform(y0 - 100, y1 + 100, n)
    lon, lat = pyproj.Transformer.from_crs(
        array_crs(provider), "EPSG:4326", always_xy=True
    ).transform(x, y)
    shots = {
        "shot_number": np.arange(n, dtype=np.uint64) + 10**15,
        "longitude": lon,
        "latitude": lat,
        "time": np.array(["2021-06-01"] * n, dtype="datetime64[ns]"),
        "rhov": np.full(n, 0.6, np.float32),
        "rhog": np.full(n, 0.4, np.float32),
        "rossg": np.full(n, 0.5, np.float32),
        "omega": np.ones(n, np.float32),
        "pai": rng.uniform(0, 5, n).astype(np.float32),
        "cover": rng.uniform(0, 1, n).astype(np.float32),
        "sensitivity": np.full(n, 0.98, np.float32),
        "l2a_quality_flag": np.ones(n, np.uint8),
        "l2b_quality_flag": np.ones(n, np.uint8),
        "degrade_flag": np.zeros(n, np.uint8),
        "leaf_off_flag": np.zeros(n, np.uint8),
        "pft_class": np.full(n, 1, np.uint8),
        "cp_pavd": rng.uniform(0, 0.1, (n, 101)).astype(np.float32),
    }
    return shots, x, y


def test_lonlat_box_contains_the_als_extent(provider):
    lon0, lat0, lon1, lat1 = ga.als_lonlat_bbox(provider)
    x0, y0, x1, y1 = array_data_bbox(provider)
    to_lonlat = pyproj.Transformer.from_crs(array_crs(provider), "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    assert lon0 <= min(lon) and max(lon) <= lon1 and lat0 <= min(lat) and max(lat) <= lat1
    wider = ga.als_lonlat_bbox(provider, buffer=500.0)
    assert wider[0] < lon0 and wider[3] > lat1


def test_select_shots_filters_projects_and_sets_the_reflectance_ratio(provider):
    shots, x, y = _shots(provider)
    x0, y0, x1, y1 = array_data_bbox(provider)
    inside = (x >= x0 + 25) & (x <= x1 - 25) & (y >= y0 + 25) & (y <= y1 - 25)
    first = np.flatnonzero(inside)[:6]
    shots["l2b_quality_flag"][first[0]] = 0
    shots["degrade_flag"][first[1]] = 1
    shots["sensitivity"][first[2]] = 0.9
    shots["leaf_off_flag"][first[3]] = 1  # evergreen: kept
    shots["leaf_off_flag"][first[4]], shots["pft_class"][first[4]] = 1, 4  # deciduous: dropped
    shots["rhog"][first[5]] = 0.0  # no ratio: the median of the others
    table = ga.select_shots(shots, provider)
    expected = inside.copy()
    expected[first[[0, 1, 2, 4]]] = False
    assert list(table["gedi_row"]) == list(np.flatnonzero(expected))
    np.testing.assert_allclose(table["center_x"], x[expected], atol=1e-3)
    np.testing.assert_allclose(table["center_y"], y[expected], atol=1e-3)
    assert table["rv_rg"].to_numpy() == pytest.approx(1.5, rel=1e-5)
    # GEDI variables are prefixed, so the simulation's cover and rh do not overwrite them
    assert {"gedi_pai", "gedi_cover", "gedi_sensitivity", "rossg", "omega"} <= set(table.columns)
    assert "cover" not in table.columns and "cp_pavd" not in table.columns
    assert len(ga.select_shots(shots, provider, years=(2019, 2020))) == 0
    assert len(ga.select_shots(shots, provider, quality=False, min_sensitivity=0.0)) == int(
        inside.sum() - 1
    )
    assert len(ga.select_shots({"shot_number": np.empty(0, np.uint64)}, provider)) == 0


def test_query_reads_the_cache_without_gedidb(provider, tmp_path):
    shots, _, _ = _shots(provider)
    np.savez_compressed(tmp_path / "gedi_shots.npz", **shots)
    again = ga.query_gedi_shots(provider, tmp_path / "gedi_shots.npz")
    assert set(again) == set(shots)
    np.testing.assert_array_equal(again["cp_pavd"], shots["cp_pavd"])


def test_query_stores_what_the_database_returns(provider, tmp_path):
    pytest.importorskip("geopandas")
    import xarray as xr

    shots, _, _ = _shots(provider, n=5)
    asked = {}

    class Database:
        def get_data(self, variables, geometry, start_time, end_time, return_type):
            asked.update(
                variables=variables, bounds=geometry.total_bounds, times=(start_time, end_time)
            )
            data = {k: (("shot_number",) if v.ndim == 1 else ("shot_number", "level"), v)
                    for k, v in shots.items() if k not in ("shot_number", "longitude", "latitude", "time")}  # fmt: skip
            data["beam_type"] = ("shot_number", np.array(["full"] * 5, dtype=object))
            coords = {k: ("shot_number", shots[k]) for k in ("longitude", "latitude", "time")}
            return xr.Dataset(data, coords={"shot_number": shots["shot_number"], **coords})

    variables = ("rhov", "rhog", "cp_pavd", "beam_type")
    cache = tmp_path / "area" / "gedi_shots.npz"
    out = ga.query_gedi_shots(provider, cache, gedi_provider=Database(), variables=variables)
    assert cache.exists() and asked["variables"] == list(variables)
    np.testing.assert_allclose(asked["bounds"], ga.als_lonlat_bbox(provider))
    assert out["beam_type"].dtype.kind == "U" and out["cp_pavd"].shape == (5, 101)
    np.testing.assert_array_equal(np.load(cache)["shot_number"], shots["shot_number"])


def test_plant_area_above_sums_the_steps_over_each_level():
    steps = np.array([[0.0, 1.0, 2.0, np.nan, 4.0]])
    np.testing.assert_array_equal(ga._plant_area_above(steps), [[7.0, 6.0, 4.0, 4.0, 0.0]])


def test_attenuation_summary_recovers_a_known_attenuation(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    als = rng.uniform(0.02, 0.06, (n, 101))
    above = ga._plant_area_above(als)
    waveform = als * 1.2 * np.exp(-0.5 * above)  # a = 1.2, beta = 0.5
    gedi = als * 0.9 * np.exp(-0.8 * above)
    first = als * 1.3
    waveform[:5] = np.nan  # footprints without a profile
    np.savez(
        tmp_path / "cp_profiles.npz", cp_pavd_waveform=waveform, cp_pavd_als_all=als,
        cp_pavd_als_first=first, cp_height_als_all=np.tile(np.linspace(0, 20, 101), (n, 1)),
    )  # fmt: skip
    order = rng.permutation(n)  # footprint rows follow processing order, not shot order
    pd.DataFrame({"shot_index": order, "gedi_row": order}).to_parquet(
        tmp_path / "footprints.parquet"
    )
    shots = {"cp_pavd": np.empty((n, 101))}
    shots["cp_pavd"][order] = gedi
    result = ga.attenuation_summary(tmp_path, shots)
    assert result["n_footprints"] == n - 5
    assert result["fits"]["waveform"]["beta"] == pytest.approx(0.5, abs=0.02)
    assert result["fits"]["waveform"]["amplitude"] == pytest.approx(1.2, abs=0.03)
    assert result["fits"]["gedi"]["beta"] == pytest.approx(0.8, abs=0.03)
    levels = result["profiles"]
    assert levels["ratio_als_first"].dropna().to_numpy() == pytest.approx(1.3)
    assert (tmp_path / "attenuation_profiles.csv").exists() and len(levels) == 101
    without = ga.attenuation_summary(tmp_path)
    assert "gedi" not in without["fits"]


def test_canopy_profiles_run_the_gedidb_retrieval(provider, tmp_path):
    pytest.importorskip("gedidb")
    from alsdb.products.profiles import simulate_gedi_profiles

    x0, y0, x1, y1 = array_data_bbox(provider)
    gx, gy = np.meshgrid(np.linspace(x0 + 100, x1 - 100, 3), np.linspace(y0 + 100, y1 - 100, 3))
    shots = pd.DataFrame({"center_x": gx.ravel(), "center_y": gy.ravel(), "rv_rg": 1.5,
                          "rossg": 0.5, "omega": 1.0})  # fmt: skip
    simulate_gedi_profiles(provider, shots, tmp_path, rv_rg_col="rv_rg", year=2021, min_points=1)
    path = ga.canopy_profiles(tmp_path)
    with np.load(path) as cp:
        assert {f"cp_pavd_{kind}" for kind in ga.PROFILE_KINDS} <= set(cp.files)
        assert cp["cp_pavd_waveform"].shape == (9, 101)
