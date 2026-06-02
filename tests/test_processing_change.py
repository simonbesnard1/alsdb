# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Tests for change.py — compute_change unit and integration tests."""

import numpy as np
import pytest

from alsdb.processing.change import compute_change
from alsdb.processing.chm import compute_chm

BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
RES = 10.0
YEAR_FROM = 2021
YEAR_TO = 2022
CRS = "EPSG:25830"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_two_year_store(store, provider, delta=1.0):
    """
    Write CHM for year 2021 via compute_chm, then write a fake year-2022 slice
    that is uniformly *delta* metres higher.  Returns the 2021 data array.
    """
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    grp = store._root["10m"]
    data_2021 = np.array(grp["chm"][0], dtype=np.float32)
    data_2022 = np.where(np.isnan(data_2021), np.nan, data_2021 + delta).astype(np.float32)
    store.write_tile("chm", RES, YEAR_TO, data_2022, BBOX)
    return data_2021


# ---------------------------------------------------------------------------
# Error / guard tests (no TileDB required)
# ---------------------------------------------------------------------------


def test_compute_change_raises_on_same_year(store, provider):
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    with pytest.raises(ValueError, match="must differ"):
        compute_change(store, "chm", YEAR_FROM, YEAR_FROM, RES)


def test_compute_change_raises_on_missing_resolution(store, provider):
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    with pytest.raises(KeyError):
        compute_change(store, "chm", YEAR_FROM, YEAR_TO, resolution=99.0)


def test_compute_change_raises_on_missing_variable(store, provider):
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    with pytest.raises(KeyError):
        compute_change(store, "nonexistent", YEAR_FROM, YEAR_TO, RES)


def test_compute_change_raises_on_missing_year(store, provider):
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    with pytest.raises(ValueError, match="year_to"):
        compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES)


# ---------------------------------------------------------------------------
# Core output correctness
# ---------------------------------------------------------------------------


def test_compute_change_delta_is_correct(store, provider):
    delta = 2.0
    data_2021 = _setup_two_year_store(store, provider, delta=delta)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES)

    grp = store._root["10m"]
    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    out_delta = np.array(grp["chm_delta"][t_to], dtype=np.float32)

    valid = ~np.isnan(data_2021)
    assert valid.any()
    np.testing.assert_allclose(out_delta[valid], delta, atol=1e-4)


def test_compute_change_delta_pct_is_correct(store, provider):
    delta = 1.0
    data_2021 = _setup_two_year_store(store, provider, delta=delta)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES, pct_min_abs=0.1)

    grp = store._root["10m"]
    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    out_pct = np.array(grp["chm_delta_pct"][t_to], dtype=np.float32)
    out_delta = np.array(grp["chm_delta"][t_to], dtype=np.float32)
    data_2022 = np.array(grp["chm"][t_to], dtype=np.float32)

    valid = ~np.isnan(data_2021) & ~np.isnan(out_pct)
    if valid.any():
        expected = 100.0 * out_delta[valid] / data_2021[valid]
        np.testing.assert_allclose(out_pct[valid], expected, rtol=1e-4)


def test_compute_change_flag_gain(store, provider):
    _setup_two_year_store(store, provider, delta=2.0)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES, min_delta=0.5)

    grp = store._root["10m"]
    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    flag = np.array(grp["chm_change_flag"][t_to], dtype=np.float32)
    valid = ~np.isnan(flag)
    assert valid.any()
    assert np.all(flag[valid] == 1.0), "All valid cells should be gain (+1)"


def test_compute_change_flag_no_change_within_min_delta(store, provider):
    _setup_two_year_store(store, provider, delta=0.2)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES, min_delta=0.5)

    grp = store._root["10m"]
    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    flag = np.array(grp["chm_change_flag"][t_to], dtype=np.float32)
    valid = ~np.isnan(flag)
    assert valid.any()
    assert np.all(flag[valid] == 0.0), "Sub-threshold changes should be flagged 0"


def test_compute_change_pct_min_abs_masks_near_zero(store, provider):
    """Cells where |year_from| < pct_min_abs should have NaN in delta_pct."""
    # Set up store structure with real CHM for YEAR_FROM
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    grp = store._root["10m"]
    data_2021 = np.array(grp["chm"][0], dtype=np.float32)

    # Overwrite YEAR_FROM with tiny values well below pct_min_abs threshold
    tiny_from = np.where(np.isnan(data_2021), np.nan, 0.005).astype(np.float32)
    tiny_to = np.where(np.isnan(data_2021), np.nan, 0.010).astype(np.float32)
    t_from_idx = int(np.where(np.array(grp["time"][:]) == YEAR_FROM)[0][0])
    grp["chm"][t_from_idx] = tiny_from

    # Write YEAR_TO
    store.write_tile("chm", RES, YEAR_TO, tiny_to, BBOX)

    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES, pct_min_abs=0.5)

    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    out_pct = np.array(grp["chm_delta_pct"][t_to], dtype=np.float32)

    # year_from = 0.005 < pct_min_abs = 0.5 → delta_pct must be NaN
    valid_chm = ~np.isnan(data_2021)
    if valid_chm.any():
        assert np.all(np.isnan(out_pct[valid_chm]))


def test_compute_change_nan_propagates(store, provider):
    """NaN in either year propagates to all output products."""
    compute_chm(provider, store, resolution=RES, bbox=BBOX, year=YEAR_FROM)
    grp = store._root["10m"]
    data_2021 = np.array(grp["chm"][0], dtype=np.float32)
    # Write all-NaN year-2022
    all_nan = np.full_like(data_2021, np.nan)
    store.write_tile("chm", RES, YEAR_TO, all_nan, BBOX)

    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES)

    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    delta = np.array(grp["chm_delta"][t_to])
    assert np.all(np.isnan(delta))


def test_compute_change_overwrite_false_skips(store, provider):
    _setup_two_year_store(store, provider, delta=1.0)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES)

    # Corrupt the output
    grp = store._root["10m"]
    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    grp["chm_delta"][t_to] = 999.0

    # Should skip (overwrite=False)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES, overwrite=False)
    assert float(grp["chm_delta"][t_to, 0, 0]) == pytest.approx(999.0)


def test_compute_change_overwrite_true_rewrites(store, provider):
    _setup_two_year_store(store, provider, delta=1.0)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES)

    grp = store._root["10m"]
    t_to = int(np.where(np.array(grp["time"][:]) == YEAR_TO)[0][0])
    grp["chm_delta"][t_to] = 999.0

    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES, overwrite=True)
    result = np.array(grp["chm_delta"][t_to])
    valid = ~np.isnan(result)
    if valid.any():
        assert np.all(result[valid] != 999.0)


def test_compute_change_writes_three_products(store, provider):
    _setup_two_year_store(store, provider, delta=1.0)
    compute_change(store, "chm", YEAR_FROM, YEAR_TO, RES)
    assert store.has_data("chm_delta", RES, YEAR_TO)
    assert store.has_data("chm_delta_pct", RES, YEAR_TO)
    assert store.has_data("chm_change_flag", RES, YEAR_TO)
