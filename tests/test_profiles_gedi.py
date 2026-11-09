"""alsdb.products.profiles: simulated GEDI canopy profiles and their ALS reference."""

import numpy as np
import pandas as pd
import pytest

from alsdb.products import profiles as gp


def test_height_edges_cover_the_range_in_waveform_bins():
    edges = gp.height_edges((-10.0, 80.0), 0.15)
    assert edges[0] == pytest.approx(-10.0) and edges[-1] >= 80.0
    np.testing.assert_allclose(np.diff(edges), 0.15)
    with pytest.raises(ValueError):
        gp.height_edges((5.0, 1.0))


def test_rebin_keeps_energy_and_shifts_by_the_ground():
    z = 100.0 + np.arange(200) * 0.15  # elevations
    values = np.zeros(200)
    values[[10, 100]] = [0.3, 0.7]  # at 101.5 m and 115.0 m
    edges = gp.height_edges((-5.0, 40.0), 0.15)
    out = gp.rebin(z, values, 100.0, edges)
    assert out.sum() == pytest.approx(1.0)
    centres = (edges[:-1] + edges[1:]) / 2
    assert centres[np.argmax(out)] == pytest.approx(15.0, abs=0.15)


def test_gap_from_waveform_is_one_above_and_one_minus_cover_at_the_ground():
    canopy = np.zeros(100)
    canopy[60:70] = 0.06  # 0.6 of the energy in the canopy
    ground = np.zeros(100)
    ground[5] = 0.4
    gap = gp.gap_from_waveform(canopy, ground, rv_rg=1.0)
    assert gap[-1] == pytest.approx(1.0)
    assert gap[0] == pytest.approx(0.4)  # 1 - cover, cover = 0.6 / (0.6 + 0.4)
    assert np.all(np.diff(gap) >= -1e-12)  # more gaps higher up
    # a brighter canopy than ground (rv_rg > 1) means less cover for the same energy
    assert gp.gap_from_waveform(canopy, ground, rv_rg=2.0)[0] == pytest.approx(
        1 - 0.6 / (0.6 + 0.8)
    )
    # energy outside the grid enters through the totals
    assert gp.gap_from_waveform(canopy, ground, totals=(0.6, 0.8))[0] == pytest.approx(
        1 - 0.6 / 1.4
    )


def test_als_gap_counts_the_share_of_weight_below_each_height():
    heights = np.array([0.0, 0.0, 0.0, 10.0, 20.0])
    weights = np.ones(5)
    centres = np.array([-1.0, 5.0, 15.0, 25.0])
    np.testing.assert_allclose(gp.gap_from_returns(heights, weights, centres), [0.6, 0.6, 0.8, 1.0])


def _stand(n_ground=400, n_canopy=600, seed=0):
    rng = np.random.default_rng(seed)
    n = n_ground + n_canopy
    z = np.r_[np.zeros(n_ground), rng.uniform(15, 20, n_canopy)]
    return {"X": rng.uniform(-12, 12, n), "Y": rng.uniform(-12, 12, n), "Z": z,
            "Intensity": np.ones(n), "ReturnNumber": np.ones(n, int),
            "NumberOfReturns": np.ones(n, int),
            "Classification": np.r_[np.full(n_ground, 2), np.full(n_canopy, 5)],
            "Withheld": np.zeros(n, bool)}  # fmt: skip


def test_gedi_footprint_profiles_waveform_and_als_agree_on_a_simple_stand():
    edges = gp.height_edges((-10.0, 40.0), 0.15)
    options = {"gaussian_beam_weighting": False, "density_cell": None, "min_points": 1}
    stand = _stand()
    result, arrays, top = gp.gedi_footprint_profiles(stand, 0.0, 0.0, edges, rv_rg=1.0, **options)
    inside = np.hypot(stand["X"], stand["Y"]) <= 12.5  # the footprint's points
    share = np.mean(stand["Classification"][inside] == 2)
    centres = (edges[:-1] + edges[1:]) / 2
    assert result.z_ground_true == pytest.approx(0.0)
    assert set(arrays) == set(gp.PROFILE_ARRAYS)
    assert (arrays["canopy"].sum() + arrays["ground"].sum()) == pytest.approx(1.0, abs=1e-3)
    # single-return pulses: first and all returns agree; between ground and
    # canopy the gap is the footprint's ground share, in the waveform and the ALS
    mid = (centres > 3) & (centres < 10)
    np.testing.assert_allclose(arrays["pgap_waveform"][mid], share, atol=1e-3)
    np.testing.assert_allclose(arrays["pgap_als_first"][mid], share, atol=1e-9)
    np.testing.assert_allclose(arrays["pgap_als_all"], arrays["pgap_als_first"])
    # above the signal top (canopy at 15-20 m plus the pulse) the profiles are NaN
    assert 20.0 < top < 24.0
    above = centres > top
    for name in ("pgap_waveform", "pgap_als_first", "pgap_als_all"):
        assert np.isnan(arrays[name][above]).all()
        assert np.isfinite(arrays[name][centres < top - 0.2]).all()


def test_gedi_footprint_profiles_needs_ground_returns():
    stand = _stand()
    stand["Classification"][:] = 5
    edges = gp.height_edges((-10.0, 40.0), 0.15)
    assert gp.gedi_footprint_profiles(stand, 0.0, 0.0, edges, min_points=1) is None


def test_simulate_gedi_profiles_writes_aligned_arrays_and_table(provider, tmp_path):
    import zarr

    shots = pd.DataFrame(
        {"center_x": [308400.0, 308600.0, 308410.0], "center_y": [4688400.0, 4688600.0, 4688410.0],
         "shot_id": ["a", "b", "c"], "ratio": [1.0, 1.5, 2.0]}
    )  # fmt: skip
    options = {"footprint_radius": 150, "min_points": 1, "z_step": 0.3, "batch_tile_size": 1000}
    one = gp.simulate_gedi_profiles(provider, shots, tmp_path / "one", rv_rg_col="ratio",
                               rng=np.random.default_rng(1), **options)  # fmt: skip
    two = gp.simulate_gedi_profiles(provider, shots, tmp_path / "two", rv_rg_col="ratio",
                               n_workers=2, rng=np.random.default_rng(1), **options)  # fmt: skip
    table = pd.read_parquet(one / "footprints.parquet")
    root = zarr.open_group(str(one / "profiles.zarr"), mode="r")
    assert len(table) == 3 and sorted(table.shot_index) == [0, 1, 2]
    # each row carries its own shot's columns and ratio
    for row in table.itertuples():
        assert row.shot_id == shots.shot_id[row.shot_index]
        assert row.rv_rg == shots.ratio[row.shot_index]
    height = root["height"][:]
    for name in gp.PROFILE_ARRAYS:
        assert root[name].shape == (3, len(height))
    assert np.isfinite(root["pgap_waveform"][:]).any()
    # the stored canopy/ground profiles and totals reproduce the stored Pgap
    k = int(np.flatnonzero(np.isfinite(root["pgap_waveform"][:, 0]))[0])
    again = gp.gap_from_waveform(root["canopy"][k], root["ground"][k], table.rv_rg[k],
                               totals=(table.canopy_energy[k], table.ground_energy[k]))  # fmt: skip
    stored = root["pgap_waveform"][k]
    np.testing.assert_allclose(again[np.isfinite(stored)], stored[np.isfinite(stored)], atol=1e-5)
    assert np.isnan(stored[height >= table.signal_top[k]]).all()
    # results do not depend on the number of workers
    other = zarr.open_group(str(two / "profiles.zarr"), mode="r")
    for name in gp.PROFILE_ARRAYS:
        np.testing.assert_array_equal(root[name][:], other[name][:])
    pd.testing.assert_frame_equal(table, pd.read_parquet(two / "footprints.parquet"))


def test_signal_heights_fill_above_each_top():
    height = np.array([-0.5, 0.5, 1.5, 2.5])
    out = gp.signal_heights(height, [2.0, np.nan, 10.0])
    np.testing.assert_array_equal(out[0], [-0.5, 0.5, 1.5, gp.GEDI_FILL])
    assert (out[1] == gp.GEDI_FILL).all()  # no profile
    np.testing.assert_array_equal(out[2], height)
