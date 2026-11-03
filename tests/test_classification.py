"""Tests for ingestion-time classification configuration."""

from types import SimpleNamespace

import numpy as np
import pytest

from alsdb import ALSTile, ClassificationConfig
from alsdb.core.alsdatabase import _classification_neighbors
from alsdb.core.classification import (
    classification_stages,
    denoise_stages,
    ground_stage,
    resolve_classification_config,
)


def test_default_config_matches_existing_classifier_defaults():
    config = ClassificationConfig()
    assert config.ground_method == "csf"
    assert config.vegetation_low == 0.5
    assert config.vegetation_medium == 2.0
    assert config.vegetation_high == 5.0
    assert config.source_classes == (1,)


@pytest.mark.parametrize(
    "options",
    [
        {"vegetation_low": 2, "vegetation_medium": 1},
        {"outlier_radius": 0},
        {"hag_neighbors": 0},
        {"source_classes": ()},
        {"source_classes": (256,)},
        {"buffer": -1},
    ],
)
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        ClassificationConfig(**options)


def test_fingerprint_is_stable_and_sensitive_to_parameters():
    first = ClassificationConfig()
    same = ClassificationConfig()
    changed = ClassificationConfig(vegetation_high=6)
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert first.to_dict()["source_classes"] == [1]


def test_ground_stage_uses_selected_method_and_parameters():
    csf = ground_stage(ClassificationConfig(ground_method="csf", csf_resolution=0.75))
    pmf = ground_stage(ClassificationConfig(ground_method="pmf", pmf_slope=0.8))
    smrf = ground_stage(ClassificationConfig(ground_method="smrf", smrf_window=24))
    assert csf["type"] == "filters.csf" and csf["resolution"] == 0.75
    assert pmf["type"] == "filters.pmf" and pmf["slope"] == 0.8
    assert smrf["type"] == "filters.smrf" and smrf["window"] == 24


def test_pipeline_builders_honor_disabled_features():
    disabled = ClassificationConfig(enabled=False)
    assert not disabled.active
    assert classification_stages(disabled) == []
    assert denoise_stages(disabled) == []


def test_denoising_alone_is_an_active_configuration():
    assert ClassificationConfig(enabled=False, denoise=True).active


def test_pipeline_contains_configured_hag_and_source_classes():
    config = ClassificationConfig(
        hag_neighbors=12,
        allow_hag_extrapolation=False,
        source_classes=(0, 1),
    )
    stages = classification_stages(config)
    assert stages[1] == {
        "type": "filters.hag_nn",
        "count": 12,
        "allow_extrapolation": False,
    }
    assignments = " ".join(stage.get("value", "") for stage in stages)
    assert "Classification == 0 || Classification == 1" in assignments


def test_legacy_arguments_resolve_to_config():
    config = resolve_classification_config(
        None,
        reclassify=True,
        ground_method="pmf",
        denoise=True,
    )
    assert config.enabled and config.denoise and config.ground_method == "pmf"


def test_new_and_legacy_arguments_cannot_be_mixed():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_classification_config(ClassificationConfig(), reclassify=True)


def test_classification_neighbors_find_only_buffer_intersections(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("a.laz", "b.laz", "c.laz")]
    boxes = {
        "a.laz": (0, 0, 10, 10),
        "b.laz": (12, 0, 22, 10),
        "c.laz": (100, 100, 110, 110),
    }

    class FakeTile:
        def __init__(self, path):
            self.native_bbox = boxes[path.name]
            self.name = SimpleNamespace(year=2020, crs="EPSG:25831")

    monkeypatch.setattr("alsdb.core.alsdatabase.Tile", FakeTile)
    neighbors = _classification_neighbors(paths, buffer=3)
    assert neighbors[paths[0]] == (paths[1],)
    assert neighbors[paths[1]] == (paths[0],)
    assert neighbors[paths[2]] == ()


def test_buffered_tile_uses_neighbor_context_but_returns_only_owned_points(monkeypatch):
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")]

    def points(x):
        values = np.zeros(len(x), dtype=dtype)
        values["X"], values["Y"], values["Z"], values["Classification"] = x, 5, 100, 1
        return values

    arrays = {"core.laz": points([0, 10]), "neighbor.laz": points([11, 50])}

    class FakeTile:
        def __init__(self, path):
            self.path = path

        def read(self, chunk_size=None, out_crs=None, bbox=None):
            yield arrays[str(self.path)]

    observed = []

    def classify(self, values):
        observed.append(len(values))
        return values

    monkeypatch.setattr("alsdb.core.alstile.Tile", FakeTile)
    monkeypatch.setattr(ALSTile, "_apply_reclassification", classify)
    tile = ALSTile(
        "core.laz",
        classification_config=ClassificationConfig(buffer=2),
        classification_context=("neighbor.laz",),
    )
    chunks = list(tile.iter_chunks())
    assert observed == [3]
    assert sum(len(x) for x, _, _ in chunks) == 2


def test_ptd_settings_stay_out_of_other_fingerprints():
    csf = ClassificationConfig(ground_method="csf")
    assert "ptd_angle" not in csf.to_dict()
    assert csf.fingerprint == ClassificationConfig(ground_method="csf", ptd_angle=20).fingerprint
    ptd = ClassificationConfig(ground_method="ptd")
    assert ptd.to_dict()["ptd_angle"] == 45.0
    assert ptd.fingerprint != ClassificationConfig(ground_method="ptd", ptd_angle=20).fingerprint
    assert ptd.fingerprint != ClassificationConfig(ground_method="ptd", ptd_votes=4).fingerprint
    assert "ptd_tilt_runs" not in csf.to_dict()
    from alsdb.core.ptd import DEFAULTS

    assert ptd.ptd_parameters == DEFAULTS


def test_ptd_ensemble_settings_are_validated():
    for bad in ({"ptd_tilt_runs": 0}, {"ptd_votes": 9}, {"ptd_votes": 0}, {"ptd_tilt": 0.0},
                {"ptd_tilt": 50.0}, {"ptd_tilt_runs": 2.5}):  # fmt: skip
        with pytest.raises(ValueError):
            ClassificationConfig(ground_method="ptd", **bad)
    ClassificationConfig(ground_method="ptd", ptd_tilt_runs=1, ptd_votes=1)  # plain PTD


def test_ptd_pipeline_has_no_pdal_ground_stage():
    stages = classification_stages(ClassificationConfig(ground_method="ptd"))
    assert stages[0]["type"] == "filters.hag_nn"
    with pytest.raises(ValueError, match="not a PDAL stage"):
        ground_stage(ClassificationConfig(ground_method="ptd"))


def _terrain_and_canopy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    points = np.zeros(n, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1"),
                                ("ReturnNumber", "u1"), ("NumberOfReturns", "u1")])  # fmt: skip
    points["X"], points["Y"] = rng.uniform(0, 60, (2, n))
    ground = rng.random(n) < 0.4
    points["Z"] = 100 + 0.3 * points["X"] + np.where(ground, 0, rng.uniform(3, 20, n))
    points["Classification"] = 1
    points["ReturnNumber"] = points["NumberOfReturns"] = 1
    points["Classification"][0] = 7  # noise is never ground
    points["Z"][0] = 50
    return points, ground


def test_ptd_ground_finds_the_terrain_in_input_order():
    pytest.importorskip("alsdb.processing._ptd_native")
    from alsdb.core.ptd import ptd_ground

    points, ground = _terrain_and_canopy()
    shuffled = points[::-1].copy()
    mask = ptd_ground(shuffled)
    truth = ground[::-1]
    assert not mask[-1]  # the noise point (first in the original order)
    assert (mask == truth)[:-1].mean() > 0.98


def test_tile_reclassification_with_ptd():
    pytest.importorskip("alsdb.processing._ptd_native")
    tile = ALSTile.__new__(ALSTile)
    tile._classification = ClassificationConfig(ground_method="ptd")
    tile._classification_workers = 4
    points, ground = _terrain_and_canopy()
    data = np.zeros(len(points), dtype=[*points.dtype.descr, ("HeightAboveGround", "f8")])
    for name in points.dtype.names:
        data[name] = points[name]
    out = tile._apply_reclassification(data)
    assert out["Classification"][0] == 7
    assert np.mean((out["Classification"] == 2) == ground) > 0.98
    assert np.isin(out["Classification"][~ground][1:], (1, 3, 4, 5)).all()


def test_native_ptd_matches_pylasr():
    # Same r-lidar/PTD code: identical labels on identical coordinates. pyLASR
    # reads them from a 1 mm LAS file, so feed the native module the same values.
    pytest.importorskip("alsdb.processing._ptd_native")
    pytest.importorskip("pylasr")
    from alsdb.core.ptd import ptd_ground

    points, _ = _terrain_and_canopy()
    for name in ("X", "Y", "Z"):
        points[name] = np.round(points[name], 3)
    native = ptd_ground(points, tilt_runs=1, votes=1)
    lasr = ptd_ground(points, backend="pylasr", tilt_runs=1, votes=1)
    assert (native == lasr).mean() > 0.995 and native.any()


def test_native_ptd_parameters_reach_the_library():
    pytest.importorskip("alsdb.processing._ptd_native")
    from alsdb.core.ptd import ptd_ground

    points, _ = _terrain_and_canopy()
    strict = ptd_ground(points, angle=1.0, distance=0.01, tilt_runs=1, votes=1)
    loose = ptd_ground(points, angle=60.0, distance=10.0, tilt_runs=1, votes=1)
    assert strict.sum() < loose.sum()
    with pytest.raises(ValueError, match="unknown PTD"):
        ptd_ground(points, max_angle=30)


def test_native_ptd_does_not_depend_on_point_order():
    pytest.importorskip("alsdb.processing._ptd_native")
    from alsdb.core.ptd import ptd_ground

    points, _ = _terrain_and_canopy(n=6000)
    points["Z"] = np.round(points["Z"], 1)  # ties, as on real tiles
    first = ptd_ground(points)
    perm = np.random.default_rng(5).permutation(len(points))
    again = np.empty(len(points), bool)
    again[perm] = ptd_ground(points[perm])
    np.testing.assert_array_equal(first, again)


def test_ptd_tilt_is_rigid_about_the_centre():
    from alsdb.core.ptd import _tilt_xyz

    points, _ = _terrain_and_canopy(n=50)
    centre = np.array([30.0, 30.0, 110.0])
    xyz = np.column_stack([points[n] for n in ("X", "Y", "Z")])
    new = np.column_stack(_tilt_xyz(*xyz.T, centre, 10.0, 77.0))
    np.testing.assert_allclose(np.linalg.norm(new - centre, axis=1),
                               np.linalg.norm(xyz - centre, axis=1))  # fmt: skip
    # tilted by 10 degrees: the +Z axis now leans 10 degrees off vertical
    up = _tilt_xyz(np.array([30.0]), np.array([30.0]), np.array([111.0]), centre, 10.0, 77.0)
    assert np.degrees(np.arccos(up[2][0] - 110.0)) == pytest.approx(10.0)


def test_ptd_ensemble_votes():
    pytest.importorskip("alsdb.processing._ptd_native")
    from alsdb.core.ptd import ptd_ground

    points, ground = _terrain_and_canopy()
    single = ptd_ground(points, tilt_runs=1, votes=1)
    union, default, unanimous = (ptd_ground(points, votes=k) for k in (1, 3, 8))
    assert (unanimous <= default).all() and (default <= union).all()
    assert (single <= union).all()  # the untilted run is one of the votes
    assert not default[0]  # noise
    assert (default == ground)[1:].mean() > 0.98
    np.testing.assert_array_equal(default, ptd_ground(points, workers=4))


def test_tile_passes_classification_workers_to_ptd(monkeypatch):
    from alsdb.core import ptd

    seen = []

    def fake(points, *, workers=1, **parameters):
        seen.append(workers)
        return np.zeros(len(points), bool)

    monkeypatch.setattr(ptd, "ptd_ground", fake)
    tile = ALSTile.__new__(ALSTile)
    tile._classification = ClassificationConfig(ground_method="ptd")
    tile._classification_workers = 6
    points, _ = _terrain_and_canopy(n=200)
    data = np.zeros(len(points), dtype=[*points.dtype.descr, ("HeightAboveGround", "f8")])
    for name in points.dtype.names:
        data[name] = points[name]
    tile._apply_reclassification(data)
    assert seen == [6]
