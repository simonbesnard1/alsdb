"""Scientific regressions for grids, terrain support, native CHM and uncertainty."""

import numpy as np
import pytest

from alsdb.processing._grid import GridSpec
from alsdb.processing._products import surface_tile
from alsdb.processing._surface import (
    FILLED,
    NO_OBSERVATIONS,
    UNSUPPORTED,
    CHMConfig,
    reconstruct_chm,
)
from alsdb.processing._terrain import TerrainModel, normalize_points
from alsdb.processing.benchmark import compare_rasters, interval_coverage
from alsdb.processing.chm import _fill_pits, compute_chm
from alsdb.processing.uncertainty import (
    CHMErrorModel,
    compute_chm_uncertainty,
    ensemble_summary,
    estimate_chm_uncertainty,
    perturb_points,
)


def scene():
    dtype = [
        ("X", "f8"),
        ("Y", "f8"),
        ("Z", "f8"),
        ("Classification", "u1"),
        ("ReturnNumber", "u1"),
        ("PointSourceId", "u2"),
        ("GpsTime", "f8"),
    ]
    x, y = np.meshgrid(np.arange(-2.0, 13.0, 2.0), np.arange(-2.0, 13.0, 2.0))
    gx, gy = x.ravel(), y.ravel()
    cx, cy = np.meshgrid(np.arange(0.5, 10.0), np.arange(0.5, 10.0))
    cx, cy = cx.ravel(), cy.ravel()
    arr = np.zeros(len(gx) + len(cx), dtype=dtype)
    arr["X"] = np.r_[gx, cx]
    arr["Y"] = np.r_[gy, cy]
    arr["Z"] = 100 + 0.2 * arr["X"] + 0.3 * arr["Y"]
    arr["Classification"][: len(gx)] = 2
    arr["Classification"][len(gx) :] = 5
    arr["Z"][len(gx) :] += 10
    arr["ReturnNumber"] = 1
    arr["PointSourceId"] = 1
    arr["GpsTime"] = np.arange(len(arr)) + 1000.0
    return arr


def test_grid_partial_edge_keeps_requested_resolution():
    grid = GridSpec.from_bbox((0.3, 0.1, 10.6, 9.8), 1.0)
    assert grid.shape == (10, 11)
    assert grid.bbox == pytest.approx((0.3, -0.2, 11.3, 9.8))
    tiles = list(grid.tiles(4))
    assert (
        sum(GridSpec.from_bbox(crop, 1).nx * GridSpec.from_bbox(crop, 1).ny for _, crop in tiles)
        == 110
    )
    for _, crop in tiles:
        assert (grid.y1 - crop[3]) % 1 == pytest.approx(0, abs=1e-12)


def test_planar_terrain_and_normalization_are_analytical():
    arr = scene()
    points, model = normalize_points(arr)
    np.testing.assert_allclose(
        points["HeightAboveGround"][arr["Classification"] == 5], 10, atol=1e-10
    )
    xy = np.array([[2.2, 3.7], [4.0, 8.0]])
    z, _, outside = model.evaluate(xy)
    np.testing.assert_allclose(z, 100 + 0.2 * xy[:, 0] + 0.3 * xy[:, 1])
    assert not outside.any()
    np.testing.assert_allclose(model.slope(xy), np.degrees(np.arctan(np.hypot(0.2, 0.3))))


def test_unsupported_terrain_is_missing_not_zero():
    arr = scene()
    arr = arr[arr["Classification"] == 5]
    result = reconstruct_chm(arr, (0, 0, 10, 10), 1)
    assert np.isnan(result["chm"]).all()
    assert (result["chm_quality"].astype(int) & UNSUPPORTED).all()


def test_extrapolation_is_explicit_and_distance_bounded():
    ground = scene()
    ground = ground[ground["Classification"] == 2]
    model = TerrainModel(ground)
    xy = np.array([[50.0, 50.0]])
    assert np.isnan(model.evaluate(xy)[0]).all()
    z, _, outside = model.evaluate(xy, extrapolate=True)
    assert np.isfinite(z).all() and outside.all()
    assert np.isnan(model.evaluate(xy, extrapolate=True, max_distance=2)[0]).all()


def test_no_canopy_observation_is_not_zero_height():
    arr = scene()
    arr = arr[arr["Classification"] == 2]
    result = reconstruct_chm(arr, (0, 0, 10, 10), 1)
    assert np.isnan(result["chm"]).all()
    assert (result["chm_quality"].astype(int) & NO_OBSERVATIONS).all()


def test_filling_preserves_masks_edges_and_uses_local_values():
    grid = np.full((20, 20), 100.0, np.float32)
    grid[2:7, 2:7] = 5
    grid[4, 4] = np.nan
    grid[0, 0] = np.nan
    assert _fill_pits(grid)[4, 4] == 5
    assert np.isnan(_fill_pits(grid)[0, 0])
    allowed = np.ones_like(grid, bool)
    allowed[4, 4] = False
    assert np.isnan(_fill_pits(grid, eligible=allowed)[4, 4])
    grid[10:13, 10:13] = np.nan
    assert np.isnan(_fill_pits(grid)[11, 11])


def test_filled_cell_has_quality_flag():
    arr = scene()
    arr = arr[~((arr["Classification"] == 5) & (arr["X"] == 4.5) & (arr["Y"] == 5.5))]
    result = reconstruct_chm(arr, (0, 0, 10, 10), 1, CHMConfig(pit_fill=True))
    assert result["chm"][4, 4] == pytest.approx(10)
    assert int(result["chm_quality"][4, 4]) & FILLED


def test_tiling_with_fill_and_boundary_points_matches_whole_scene():
    arr = scene()
    arr["Y"][arr["Classification"] == 5] -= 0.5  # exact pixel edges
    arr = arr[~((arr["Classification"] == 5) & (arr["X"] == 4.5) & (arr["Y"] == 5.0))]
    config = CHMConfig(pit_fill=True)
    full = surface_tile(arr, (0, 0, 10, 10), 1, config)
    for _, crop in GridSpec.from_bbox((0, 0, 10, 10), 1).tiles(5):
        tile = surface_tile(arr, crop, 1, config)
        row, col = round(10 - crop[3]), round(crop[0])
        for name in full:
            np.testing.assert_allclose(
                tile[name], full[name][row : row + 5, col : col + 5], equal_nan=True
            )


def test_ground_outlier_cannot_reenter_canopy_as_unclassified():
    from alsdb.processing._tiling import _filter_ground_outliers

    arr = scene()
    arr["Z"][5] = -500
    filtered = _filter_ground_outliers(arr)
    assert filtered["Classification"][5] == 7


def test_native_freezes_upper_surface_and_buffer_delays_freezing():
    native = pytest.importorskip("alsdb.processing._spikefree_native")
    xyz = np.array([[0, 0, 10], [2, 0, 10], [0, 2, 10], [2, 2, 10], [1, 1, 0]], float)
    frozen = native.rasterize(xyz, 0.0, 2.0, 0.5, 4, 4, 4.0, 0.5, 0.0)
    np.testing.assert_allclose(frozen, 10)
    delayed = native.rasterize(xyz, 0.0, 2.0, 0.5, 4, 4, 4.0, 20.0, 0.0)
    assert delayed.min() < 5


def test_native_duplicates_order_and_collinear_input():
    native = pytest.importorskip("alsdb.processing._spikefree_native")
    xyz = np.array([[0, 0, 10], [2, 0, 10], [0, 2, 10], [2, 2, 10], [0, 0, 0]], float)
    first = native.rasterize(xyz, 0.0, 2.0, 0.5, 4, 4, 4.0, 0.5, 0.0)
    reverse = native.rasterize(xyz[::-1].copy(), 0.0, 2.0, 0.5, 4, 4, 4.0, 0.5, 0.0)
    np.testing.assert_array_equal(first, reverse)
    np.testing.assert_allclose(first, 10)
    line = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]], float)
    assert np.isnan(native.rasterize(line, 0.0, 2.0, 0.5, 4, 4, 4.0, 0.5, 0.0)).all()


def test_native_planar_surface_and_edge_trimming():
    native = pytest.importorskip("alsdb.processing._spikefree_native")
    xyz = np.array([[0, 0, 0], [4, 0, 8], [0, 4, 12], [4, 4, 20]], float)
    raster = native.rasterize(xyz, 0.0, 4.0, 1.0, 4, 4, 0.1, 0.5, 0.0)
    x, y = GridSpec.from_bbox((0, 0, 4, 4), 1).centers()
    np.testing.assert_allclose(raster, 2 * x + 3 * y)
    assert np.isnan(native.rasterize(xyz, 0.0, 4.0, 1.0, 4, 4, 0.1, 0.5, 1.0)).all()


def test_native_surface_pipeline():
    pytest.importorskip("alsdb.processing._spikefree_native")
    result = reconstruct_chm(scene(), (0, 0, 10, 10), 1, CHMConfig(method="spikefree"))
    np.testing.assert_allclose(result["chm"], 10, atol=1e-5)


def test_perturbations_agree_in_overlapping_tiles():
    arr = scene()
    model = CHMErrorModel(
        strip_z_sigma=0.1,
        strip_xy_sigma=0.2,
        pulse_z_sigma=0.1,
        terrain_sigma=0.3,
        pulse_keep_probability=0.9,
    )
    full = perturb_points(arr, model, seed=7, realization=4)
    subset = perturb_points(arr[::2], model, seed=7, realization=4)
    full = full[np.isin(full["GpsTime"], subset["GpsTime"])]
    for name in ("X", "Y", "Z"):
        np.testing.assert_array_equal(full[name], subset[name])


def test_shared_strip_vertical_error_cancels_in_hag():
    arr = scene()
    result = estimate_chm_uncertainty(
        arr,
        (0, 0, 10, 10),
        1,
        error_model=CHMErrorModel(strip_z_sigma=5),
        n_realizations=5,
        ground_outlier_removal=False,
    )
    np.testing.assert_allclose(result["chm_uncertainty_std"], 0, atol=1e-5)
    np.testing.assert_allclose(result["chm_uncertainty_lower"], 10, atol=1e-5)


def test_terrain_error_produces_nonzero_reproducible_uncertainty():
    options = {
        "error_model": CHMErrorModel(terrain_sigma=0.5),
        "n_realizations": 6,
        "seed": 19,
        "ground_outlier_removal": False,
    }
    first = estimate_chm_uncertainty(scene(), (0, 0, 10, 10), 1, **options)
    second = estimate_chm_uncertainty(scene(), (0, 0, 10, 10), 1, **options)
    assert np.nanmedian(first["chm_uncertainty_std"]) > 0.1
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_ensemble_missing_realizations_do_not_create_false_certainty():
    samples = np.array([[[1.0, np.nan]], [[2.0, np.nan]], [[3.0, 5.0]], [[4.0, np.nan]]])
    result = ensemble_summary(samples, interval=0.5)
    assert result["chm_uncertainty_lower"][0, 0] == pytest.approx(1.75)
    assert result["chm_uncertainty_upper"][0, 0] == pytest.approx(3.25)
    assert result["chm_uncertainty_n_valid"][0, 1] == 1
    assert np.isnan(result["chm_uncertainty_std"][0, 1])


def test_missing_pulse_identity_rejected():
    arr = scene()
    arr["GpsTime"] = 0
    with pytest.raises(ValueError, match="pulse identity"):
        perturb_points(arr, CHMErrorModel(pulse_keep_probability=0.5), seed=0, realization=0)


def test_benchmark_and_interval_metrics():
    reference = np.arange(9.0).reshape(3, 3)
    metrics = compare_rasters(reference + 2, reference)
    assert metrics["bias"] == metrics["mae"] == metrics["rmse"] == 2
    assert interval_coverage(reference, reference - 1, reference + 1)["coverage"] == 1


def test_changed_configuration_requires_explicit_overwrite(provider, store):
    opts = {"resolution": 10.0, "year": 2021, "bbox": (308000.0, 4688000.0, 309000.0, 4689000.0)}
    compute_chm(provider, store, **opts)
    with pytest.raises(ValueError, match="different processing"):
        compute_chm(provider, store, max_height=50, **opts)
    compute_chm(provider, store, max_height=50, overwrite=True, **opts)
    run = store._root["10m"]["chm"].attrs["processing_runs"]["2021"]
    assert run["status"] == "complete"
    assert run["configuration"]["settings"]["max_height"] == 50


def test_all_missing_recomputation_clears_old_values(provider, store):
    opts = {"resolution": 10.0, "year": 2021, "bbox": (308000.0, 4688000.0, 309000.0, 4689000.0)}
    compute_chm(provider, store, **opts)
    assert store.has_data("chm", 10.0, 2021)
    compute_chm(provider, store, veg_classes=(31,), overwrite=True, **opts)
    assert np.isnan(store._root["10m"]["chm"][0]).all()
    assert not store.has_data("chm", 10.0, 2021)
    compute_chm(provider, store, veg_classes=(31,), **opts)  # completed empty output is reusable


def test_failed_run_restarts_and_finishes(provider, store, monkeypatch):
    from alsdb.processing import _products

    opts = {"resolution": 10.0, "year": 2021, "bbox": (308000.0, 4688000.0, 309000.0, 4689000.0)}
    original = _products._worker

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(_products, "_worker", fail)
    with pytest.raises(RuntimeError, match="injected"):
        compute_chm(provider, store, **opts)
    assert store._root["10m"]["chm"].attrs["processing_runs"]["2021"]["status"] == "failed"
    monkeypatch.setattr(_products, "_worker", original)
    compute_chm(provider, store, **opts)
    assert store._root["10m"]["chm"].attrs["processing_runs"]["2021"]["status"] == "complete"


def test_tiled_uncertainty_writes_provenance(provider, store):
    compute_chm_uncertainty(
        provider,
        store,
        resolution=10.0,
        year=2021,
        bbox=(308000.0, 4688000.0, 309000.0, 4689000.0),
        n_realizations=3,
        error_model=CHMErrorModel(terrain_sigma=0.1),
        ground_outlier_removal=False,
    )
    group = store._root["10m"]
    assert "chm_uncertainty_n_valid" in group
    run = group["chm_uncertainty_lower"].attrs["processing_runs"]["2021"]
    assert run["status"] == "complete"
    assert run["configuration"]["error_model"]["terrain_sigma"] == 0.1


def test_store_rejects_different_origin(store):
    store.ensure_group("chm", 1.0, (0, 0, 10, 10), "EPSG:25830", 5)
    with pytest.raises(ValueError, match="grid"):
        store.ensure_group("dtm", 1.0, (0.1, 0, 10.1, 10), "EPSG:25830", 5)


def test_native_parallel_calls_are_deterministic():
    from concurrent.futures import ThreadPoolExecutor

    native = pytest.importorskip("alsdb.processing._spikefree_native")
    rng = np.random.default_rng(55)
    xy = rng.uniform(0, 10, (500, 2))
    xyz = np.column_stack((xy, 10 + 0.1 * xy[:, 0] + 0.2 * xy[:, 1]))

    def render(_):
        return native.rasterize(xyz, 0.0, 10.0, 0.5, 20, 20, 1.5, 0.5, 0.0)

    expected = render(0)
    with ThreadPoolExecutor(4) as pool:
        for raster in pool.map(render, range(8)):
            np.testing.assert_array_equal(raster, expected)


def test_lastools_adapter_passes_freezing_parameters_and_checks_grid(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import rasterio
    from rasterio.transform import from_origin

    from alsdb.processing import lastools

    points, _ = normalize_points(scene())
    points = points[points["Classification"] == 5]
    monkeypatch.setattr(lastools.shutil, "which", lambda _: "/test/las2dem64")

    def fake_run(command, **kwargs):
        i = command.index("-spike_free")
        assert command[i + 1 : i + 4] == ["1.5", "0.25", "0.5"]
        assert "-fail" in command
        target = Path(command[command.index("-o") + 1])
        with rasterio.open(
            target,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="float32",
            transform=from_origin(0, 10, 1, 1),
            nodata=-9999,
        ) as ds:
            ds.write(np.full((10, 10), 10, np.float32), 1)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(lastools.subprocess, "run", fake_run)
    result = lastools.rasterize_lastools(points, (0, 0, 10, 10), 1.0)
    np.testing.assert_allclose(result, 10)


def test_source_change_invalidates_cached_product(provider, store, monkeypatch):
    from alsdb.processing import _products

    options = {"resolution": 10.0, "year": 2021, "bbox": (308000.0, 4688000.0, 309000.0, 4689000.0)}
    compute_chm(provider, store, **options)
    monkeypatch.setattr(_products, "input_revision", lambda _: "changed")
    with pytest.raises(ValueError, match="different processing"):
        compute_chm(provider, store, **options)


def test_partial_edge_store_matches_pixel_coordinates(store):
    grid = GridSpec.from_bbox((0.3, 0.1, 10.6, 9.8), 1.0)
    store.ensure_group("chm", 1.0, grid.bbox, "EPSG:25830", 4.0)
    for _, crop in grid.tiles(4):
        local = GridSpec.from_bbox(crop, 1.0)
        x, y = local.centers()
        store.write_tile("chm", 1.0, 2021, (2 * x + 3 * y).astype(np.float32), crop)
    x, y = grid.centers()
    np.testing.assert_allclose(store._root["1m"]["chm"][0], 2 * x + 3 * y, atol=2e-6)
