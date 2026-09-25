"""Regressions for buffered ownership, shared work and footprint physics."""

import numpy as np
import pandas as pd
import pytest

from alsdb.processing._grid import GridSpec
from alsdb.processing.biomass import (
    _extract_metrics,
    _extract_metrics_baba,
    compute_biomass,
    compute_metrics,
)
from alsdb.processing.gap import compute_gap_fraction, gap_statistics
from alsdb.processing.pavd import (
    _pavd_profile_from_hag,
    compute_als_pavd_batch,
    compute_als_pavd_profile,
    pavd_from_points,
)
from alsdb.processing.waveform import (
    _build_histogram,
    _load_pulse,
    simulate_batch,
    simulate_waveform,
    waveform_from_points,
)


def points(x, y, h, cls=None):
    arr = np.zeros(
        len(x),
        dtype=[
            ("X", "f8"),
            ("Y", "f8"),
            ("Z", "f8"),
            ("HeightAboveGround", "f8"),
            ("Classification", "u1"),
            ("ReturnNumber", "u1"),
            ("NumberOfReturns", "u1"),
            ("Intensity", "f8"),
        ],
    )
    arr["X"], arr["Y"], arr["Z"], arr["HeightAboveGround"] = x, y, h, h
    arr["Classification"] = cls if cls is not None else 5
    arr["ReturnNumber"], arr["NumberOfReturns"], arr["Intensity"] = 1, 1, 1
    return arr


def test_buffer_points_do_not_pollute_edge_cells():
    p = points([0.5, 1.5, -0.2, 2.2], [0.5] * 4, [10, 20, 100, 100], [2, 5, 5, 2])
    full = _extract_metrics(p, 1, (0, 0, 2, 1))
    crop = _extract_metrics(p[:2], 1, (0, 0, 2, 1))
    for name in full:
        np.testing.assert_equal(full[name], crop[name])
    np.testing.assert_equal(gap_statistics(p, 1, (0, 0, 2, 1))["gap"], [[1, 0]])
    np.testing.assert_equal(gap_statistics(p, 1, (0, 0, 2, 1))["gap_saturated"], [[0, 1]])


def test_half_open_boundary_ownership_and_fixed_spacing():
    p = points([0, 1, 2, 1], [1, 1, 1, 0], [1] * 4)
    mask, bins = GridSpec.from_bbox((0, 0, 2, 1), 1).point_bins(p)
    np.testing.assert_equal(mask, [True, True, False, False])
    np.testing.assert_equal(bins, [0, 1])
    grid = GridSpec.from_bbox((0, 0, 2.2, 1.2), 1)
    assert grid.bbox == (0, -0.8, 3, 1.2)


def test_invalid_hag_excluded_from_cover_and_selective_metrics():
    p = points([0.5] * 3, [0.5] * 3, [10, np.nan, 0])
    all_metrics = _extract_metrics(p, 1, (0, 0, 1, 1))
    selected = _extract_metrics(p, 1, (0, 0, 1, 1), metrics=("h95", "cc"))
    assert set(selected) == {"h95", "cc"}
    assert selected["cc"][0, 0] == 0.5
    for name in selected:
        np.testing.assert_equal(selected[name], all_metrics[name])
    circular = _extract_metrics_baba(p, 1, (0, 0, 1, 1), 1, metrics=("cc",))
    assert circular["cc"][0, 0] == 0.5


def test_pointwise_pavd_removes_slope_using_later_ground_returns():
    x, y = np.meshgrid(np.arange(-20, 21, 2.0), np.arange(-20, 21, 2.0))
    ground = points(
        x.ravel(), y.ravel(), 100 + 0.5 * x.ravel() + 0.2 * y.ravel(), np.full(x.size, 2)
    )
    ground["ReturnNumber"] = 2
    rng = np.random.default_rng(42)
    vx, vy = rng.uniform(-5, 5, (2, 150))
    canopy = points(vx, vy, 120 + 0.5 * vx + 0.2 * vy)
    firstground = ground[(ground["X"] ** 2 + ground["Y"] ** 2) < 36][:10].copy()
    firstground["ReturnNumber"] = 1
    p = np.concatenate([ground, canopy, firstground])
    result = pavd_from_points(p, 0, 0, 8, terrain_buffer=22)
    assert result is not None
    assert result.canopy_height == pytest.approx(20, abs=1e-10)
    assert result.n_ground_points == 10
    assert result.n_unsupported == 0
    shifted = p.copy()
    shifted["X"] += 600000
    shifted["Y"] += 8900000
    other = pavd_from_points(shifted, 600000, 8900000, 8, terrain_buffer=22)
    np.testing.assert_allclose(result.total_pai, other.total_pai)


def test_pavd_positive_ground_not_canopy_and_nan_not_cover():
    h = np.r_[np.full(10, 4.0), np.full(20, 10.0), np.full(20, np.nan)]
    cls = np.r_[np.full(10, 2), np.full(40, 5)]
    result = _pavd_profile_from_hag(h, cls)
    assert result.total_pai == pytest.approx(-np.log(1 / 3) / 0.5)
    assert result.n_points == 30


def test_waveform_spacing_and_pulse_physical_width():
    z, h = _build_histogram(np.array([0.0, 0.31]), np.ones(2), 0.15)
    np.testing.assert_allclose(np.diff(z), 0.15)
    assert h.sum() == 2
    widths = []
    for step in (0.075, 0.15, 0.3):
        pulse = _load_pulse("BEAM0000", step)
        assert pulse.sum() == pytest.approx(1.0)
        x = (np.arange(len(pulse)) - len(pulse) // 2) * step
        widths.append(np.sqrt(np.sum(pulse * (x - np.sum(x * pulse)) ** 2)))
    assert max(widths) - min(widths) < 0.06


def test_waveform_keeps_tails_and_reports_ground_support():
    p = points(np.zeros(100), np.zeros(100), np.full(100, 100.0), np.full(100, 2))
    result = waveform_from_points(p, 0, 0, min_points=1, z_step=0.15)
    assert result.z_bins[0] < 100 < result.z_bins[-1]
    assert result.waveform.sum() == pytest.approx(1)
    assert result.ground_supported
    assert abs(result.ground_offset) <= 0.15


def test_forest_fusion_and_cached_model_avoid_duplicate_work(provider, store, monkeypatch):
    from alsdb.processing import forest

    calls = {"query": 0, "normalize": 0}
    query, normalize = forest.query_to_array, forest.normalize_points

    def counted_query(*args, **kwargs):
        calls["query"] += 1
        return query(*args, **kwargs)

    def counted_normalize(*args, **kwargs):
        calls["normalize"] += 1
        return normalize(*args, **kwargs)

    monkeypatch.setattr(forest, "query_to_array", counted_query)
    monkeypatch.setattr(forest, "normalize_points", counted_normalize)
    bbox = (308000.0, 4688000.0, 309000.0, 4689000.0)
    forest.compute_forest_products(
        provider,
        store,
        10,
        bbox,
        2021,
        metrics=("h95", "cc"),
        chm=True,
        gap=True,
        lai=True,
        tile_size=1000,
        ground_outlier_removal=False,
    )
    assert calls == {"query": 1, "normalize": 1}
    compute_biomass(
        provider,
        store,
        10,
        lambda m: m["h95"] * m["cc"],
        bbox,
        2021,
        tile_size=1000,
        ground_outlier_removal=False,
        model_features=("h95", "cc"),
        model_id="test-model",
    )
    assert calls == {"query": 1, "normalize": 1}
    with pytest.raises(ValueError, match="different processing configuration"):
        compute_metrics(
            provider,
            store,
            10,
            bbox,
            2021,
            cc_threshold=5,
            metrics=("cc",),
            tile_size=1000,
            ground_outlier_removal=False,
        )


def test_gap_lai_toggle_preserves_gap(provider, store):
    bbox = (308000.0, 4688000.0, 309000.0, 4689000.0)
    compute_gap_fraction(provider, store, 10, bbox, 2021)
    before = store.to_dataset(10).gap.values.copy()
    compute_gap_fraction(provider, store, 10, bbox, 2021, lai=True)
    np.testing.assert_equal(before, store.to_dataset(10).gap.values)
    assert "gap_n_classified" in store.variables(10)


def test_batched_waveforms_duplicate_indices_and_worker_determinism(provider):
    shots = pd.DataFrame(
        {"center_x": [308400.0, 308600.0, 308410.0], "center_y": [4688400.0, 4688600.0, 4688410.0]},
        index=[7, 7, 8],
    )
    options = {
        "footprint_radius": 150,
        "min_points": 1,
        "noise_std": 0.01,
        "z_step": 0.3,
        "batch_tile_size": 1000,
    }
    a = simulate_batch(provider, shots, n_workers=1, rng=np.random.default_rng(10), **options)
    b = simulate_batch(provider, shots, n_workers=2, rng=np.random.default_rng(10), **options)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.index) == [7, 7, 8]
    assert list(a.center_x) == list(shots.center_x)
    assert a.n_points.min() > 0
    # Identical per-position random streams must also match independent queries.
    seeds = np.random.default_rng(10).integers(
        0, np.iinfo(np.uint64).max, len(shots), dtype=np.uint64
    )
    options.pop("batch_tile_size")
    for index, row in enumerate(shots.itertuples()):
        result = simulate_waveform(
            provider, row.center_x, row.center_y, rng=np.random.default_rng(seeds[index]), **options
        )
        assert a.iloc[index].home == pytest.approx(result.home)


def test_pavd_batch_matches_single(provider):
    shots = pd.DataFrame(
        {"center_x": [308400.0, 308500.0], "center_y": [4688400.0, 4688500.0]}, index=[1, 1]
    )
    options = {
        "footprint_radius": 250,
        "terrain_buffer": 100,
        "min_points": 1,
        "min_ground_points": 1,
        "year": 2021,
    }
    profiles = compute_als_pavd_batch(provider, shots, n_workers=2, **options)
    for row, result in zip(shots.itertuples(), profiles):
        single = compute_als_pavd_profile(provider, row.center_x, row.center_y, **options)
        assert result is not None and single is not None
        np.testing.assert_allclose(result.pavd, single.pavd)


def test_tree_boundary_keeps_whole_crown(monkeypatch):
    from alsdb.processing import trees

    p = points([0.5, 0.9, 1.2, 1.5], [0.5] * 4, [8, 15, 10, 7])
    p["Classification"] = 5
    ground = points([0, 2, 0, 2], [0, 0, 2, 2], [0] * 4, [2] * 4)
    arr = np.concatenate([p, ground])
    monkeypatch.setattr(trees, "query_to_array", lambda *a, **kw: arr)
    monkeypatch.setattr(trees, "_filter_ground_outliers", lambda a: a)

    class Pipeline:
        def __init__(self, stages, arrays):
            array = arrays[0]
            if "litree" in stages:
                from numpy.lib.recfunctions import append_fields

                array = append_fields(
                    array, "TreeID", np.ones(len(array), dtype=np.uint64), usemask=False
                )
            self.arrays = [array]

        def execute(self):
            pass

    monkeypatch.setattr(trees.pdal, "Pipeline", Pipeline)
    result = trees._process_tile(None, (-1, -1, 3, 3), (0, 0, 1, 1), 0, 2021, 1, 3, 2, None)
    assert result is not None
    segmented, records = result
    assert len(segmented) == 4
    assert segmented["X"].max() > 1
    assert records.n_points.iloc[0] == 4
    assert records.apex_x.iloc[0] == 0.9
    other = trees._process_tile(None, (-1, -1, 3, 3), (1, 0, 2, 1), 1, 2021, 1, 3, 2, None)
    assert other is None


def test_exact_boundaries_give_same_metrics_when_tiled():
    p = points([0.5, 1, 1.5, 1, 0.5, 1.5], [0.5, 1, 1.5, 0.5, 1, 1], [4, 5, 6, 7, 8, 9])
    whole = _extract_metrics(p, 1, (0, 0, 2, 2))
    assembled = {name: np.full((2, 2), np.nan) for name in whole}
    for _, crop in GridSpec.from_bbox((0, 0, 2, 2), 1).tiles(1, 1):
        values = _extract_metrics(p, 1, crop)
        row, col = int(2 - crop[3]), int(crop[0])
        for name in values:
            assembled[name][row, col] = values[name][0, 0]
    for name in whole:
        np.testing.assert_equal(whole[name], assembled[name])


def test_tree_streaming_compacts_sparse_ids(monkeypatch):
    from numpy.lib.recfunctions import append_fields

    from alsdb.processing import trees

    def worker(provider, query, crop, index, *args, **kwargs):
        p = points([crop[0] + 0.5], [0.5], [10])
        p = append_fields(p, "TreeID", np.array([2**40], dtype=np.uint64), usemask=False)
        return p, pd.DataFrame(trees._tree_metrics(p))

    monkeypatch.setattr(trees, "_process_tile", worker)
    results = list(trees.iter_segment_trees(None, (0, 0, 2, 1), 2021, tile_size=1, n_workers=2))
    assert [int(p["TreeID"][0]) for p, _ in results] == [1, 2]
    assert [int(df.tree_id.iloc[0]) for _, df in results] == [1, 2]


def test_fractional_returns_preserve_single_pulse_weight():
    single = points([0, 1], [0, 0], [0, 15], [2, 5])
    double = np.repeat(single, 2)
    double["NumberOfReturns"] = 2
    double["ReturnNumber"] = [1, 2, 1, 2]
    options = {"min_points": 1, "gaussian_beam_weighting": False, "return_weighting": "fractional"}
    a = waveform_from_points(single, 0, 0, **options)
    b = waveform_from_points(double, 0, 0, **options)
    np.testing.assert_allclose(a.waveform, b.waveform)


def test_spatial_biomass_validation_holds_out_whole_groups():
    from alsdb.processing.biomass import validate_naesset

    rng = np.random.default_rng(14)
    h = rng.uniform(5, 35, 100)
    cover = rng.uniform(0.2, 0.9, 100)
    agb = 0.4 * h**1.6 * cover**0.7
    groups = np.repeat(np.arange(10), 10)
    result = validate_naesset(h, cover, agb, groups, n_splits=5)
    assert result["rmse"] < 1e-5
    for group in np.unique(groups):
        assert len(np.unique(result["fold"][groups == group])) == 1
    assert result["n_valid"] == 100


def test_failed_biomass_run_restarts_and_clears_stale_data(provider, store):
    from alsdb.processing.forest import compute_forest_products

    options = {
        "resolution": 10,
        "bbox": (308000.0, 4688000.0, 309000.0, 4689000.0),
        "year": 2021,
        "metrics": (),
        "biomass": True,
        "model_features": ("h95",),
        "model_id": "stable-test",
    }

    def fail(metrics):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        compute_forest_products(provider, store, model_fn=fail, **options)
    assert store._root["10m"]["biomass"].attrs["processing_runs"]["2021"]["status"] == "failed"
    compute_forest_products(provider, store, model_fn=lambda m: m["h95"], **options)
    assert store.has_data("biomass", 10, 2021)
    compute_forest_products(
        provider,
        store,
        model_fn=lambda m: np.full_like(m["h95"], np.nan),
        overwrite=True,
        **options,
    )
    assert np.isnan(store.to_dataset(10).biomass.values).all()
    assert store._root["10m"]["biomass"].attrs["processing_runs"]["2021"]["status"] == "complete"


def test_real_pdal_tree_clusters_are_exposed_as_tree_ids(monkeypatch):
    from alsdb.processing import trees

    rng = np.random.default_rng(17)
    gx, gy = np.meshgrid(np.arange(-10, 11, 2), np.arange(-10, 11, 2))
    ground = points(gx.ravel(), gy.ravel(), np.zeros(gx.size), np.full(gx.size, 2))
    crown = points(rng.normal(0, 1, 100), rng.normal(0, 1, 100), rng.uniform(10, 20, 100))
    arr = np.concatenate([ground, crown])
    monkeypatch.setattr(trees, "query_to_array", lambda *a, **kw: arr)
    segmented, records = trees.segment_trees(None, (-5, -5, 5, 5), 2021, min_points=5)
    assert len(records) > 0  # An empty result used to hide ClusterID/TreeID incompatibility.
    assert "TreeID" in segmented.dtype.names
    assert (segmented["TreeID"] > 0).all()
    assert records.height.max() > 15
