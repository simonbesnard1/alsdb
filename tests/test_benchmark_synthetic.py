"""alsdb.benchmark.synthetic: scene and truth construction (no HELIOS++ needed)."""

import json
import math

import numpy as np
import pytest

from alsdb.benchmark import synthetic
from alsdb.benchmark.synthetic import Scenario, make_scene, survey_settings, terrain_height


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    out = tmp_path_factory.mktemp("scene")
    meta = make_scene(Scenario(terrain="rolling", forest="broadleaf", core=40.0, margin=10.0), out)
    return out, meta


def test_truth_is_consistent(scene):
    out, meta = scene
    t = np.load(out / "truth.npz")
    assert t["chm_max"].shape == t["chm_centre"].shape == t["dtm"].shape == (40, 40)
    assert np.all(t["chm_centre"] <= t["chm_max"] + 1e-6)  # centre is one of the cell's columns
    assert np.all(t["chm_centre"] >= 0) and (t["chm_centre"] == 0).any()  # gaps read 0
    assert t["chm_max"].max() <= t["tree_h"].max() + 0.5  # no taller than the tallest crown
    # DTM: exact terrain at the cell centres, north-up
    x = float(t["x0"]) - synthetic.X0 + 0.5
    y = float(t["y1"]) - synthetic.Y0 - 0.5
    assert t["dtm"][0, 0] == pytest.approx(terrain_height("rolling", x, y), abs=0.01)  # mesh vs function
    assert meta["voxels"] > 0 and meta["trees"] > 0


def test_scene_files_reference_each_other(scene):
    out, meta = scene
    assert f"{out / 'forest.vox'}" in (out / "scene.xml").read_text()
    head = (out / "forest.vox").read_text().splitlines()[:6]
    assert head[0] == "VOXEL SPACE" and head[3].startswith("#split:")
    with open(out / "forest.vox") as handle:
        assert sum(1 for _ in handle) - 6 == meta["voxels"]
    assert 'maxNOR="4"' in (out / "scanner.xml").read_text()
    assert 'winSize_ns="2.5"' in (out / "survey.xml").read_text()
    assert "helios_classification 2" in (out / "terrain.mtl").read_text()
    assert "helios_classification 5" in (out / "forest.mtl").read_text()
    assert json.loads((out / "scene.json").read_text())["scenario"]["forest"] == "broadleaf"


def test_survey_settings_follow_swath_geometry():
    prf, scan_freq = survey_settings(2.0)
    swath = 2 * synthetic.FLIGHT_HEIGHT * math.tan(math.radians(synthetic.SCAN_HALF_ANGLE))
    assert prf / (synthetic.SPEED * swath) == pytest.approx(2.0, rel=1e-4)
    assert synthetic.SPEED / (2 * scan_freq) == pytest.approx(1 / math.sqrt(2.0), rel=1e-3)


def test_scenario_validation_and_names():
    assert Scenario().name == "rolling_conifer_d2_p1_r4_s0"
    with pytest.raises(ValueError):
        Scenario(terrain="cliff")
    with pytest.raises(ValueError):
        Scenario(voxel=0.3)


@pytest.mark.parametrize("forest", synthetic.FORESTS)
def test_every_forest_type_builds(forest, tmp_path):
    meta = make_scene(Scenario(forest=forest, terrain="flat", core=20.0, margin=5.0), tmp_path)
    assert meta["voxels"] > 0


def test_valley_crowns_are_not_clipped():
    # Rolling valleys lie below every scene corner; understory trees there used to crash voxelise.
    size, voxel = 210.0, 0.5
    trees = {"x": np.array([125.7]), "y": np.array([188.5]), "h": np.array([5.0]),
             "r": np.array([1.5]), "base": np.array([1.5]), "cone": np.array([False])}  # fmt: skip
    ground = terrain_height("rolling", 125.7, 188.5)
    assert ground < terrain_height("rolling", *np.meshgrid([0, size], [0, size])).min() - 5
    occ, _origin, zmin = synthetic.voxelise(trees, "rolling", size, voxel)
    k = np.nonzero(occ.any(axis=(0, 1)))[0]
    assert zmin + k.min() * voxel <= ground + 1.5 + voxel  # crown base kept
    assert zmin + (k.max() + 1) * voxel >= ground + 5.0 - voxel  # crown top kept


def test_crown_density_gives_the_stand_plant_area_index(scene):
    out, meta = scene
    size, voxel = 60.0, 0.5
    assert meta["pai"] == synthetic.FOREST_TYPES["broadleaf"]["pai"]
    assert meta["pad"] * meta["voxels"] * voxel**3 / size**2 == pytest.approx(meta["pai"])
    first = (out / "forest.vox").read_text().splitlines()[6].split()
    assert float(first[3]) == pytest.approx(meta["pad"], rel=1e-5)


def _hanging_lastools(tmp_path):
    """Fake LAStools: answers -version, otherwise hangs like las2dem did on the grid."""
    bin_dir = tmp_path / "lastools_bin"
    bin_dir.mkdir()
    for name in ("lasheight64", "las2dem64", "lascanopy64"):
        exe = bin_dir / name
        exe.write_text('#!/bin/sh\n[ "$1" = "-version" ] && echo fake && exit 0\nsleep 30\n')
        exe.chmod(0o755)
    return bin_dir


def test_hung_lastools_call_is_killed_and_its_empty_output_removed(tmp_path):
    from alsdb.benchmark import lastools

    target = tmp_path / "out.tif"
    target.touch()
    with pytest.raises(RuntimeError, match="timed out after 0.5 s"):
        lastools._call(["sleep", "30"], None, target, timeout=0.5)
    assert not target.exists()


def test_scene_is_scored_without_a_hung_external_tool(scene, tmp_path):
    import pdal

    out, _ = scene
    laz = out / "laz"
    laz.mkdir(exist_ok=True)
    pts = np.zeros(10, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")])
    pts["X"], pts["Y"], pts["Z"] = synthetic.X0 + 20 + np.arange(10), synthetic.Y0 + 20, 300.0
    pts["Classification"] = 2
    pdal.Pipeline(json.dumps([{"type": "writers.las", "filename": str(laz / "SYN_2020_scene.laz"),
                               "a_srs": synthetic.CRS}]), arrays=[pts]).execute()  # fmt: skip
    preds = synthetic.run_tools(out, ("lastools",), lastools_dir=_hanging_lastools(tmp_path),
                                timeout=1)  # fmt: skip
    assert preds == {}
    failed = json.loads((out / "timings.json").read_text())["failed"]
    assert "timed out" in failed["lastools"]


def test_sensitivity_takes_only_scenes_the_grid_has_scored(tmp_path):
    from alsdb.benchmark import sensitivity

    for name in ("done", "swept", "running"):
        (tmp_path / name / "laz").mkdir(parents=True)
        (tmp_path / name / "truth.npz").write_bytes(b"")
    synthetic.append_results(tmp_path / "results.csv", [{"scenario": n} for n in ("done", "swept")])
    synthetic.append_results(tmp_path / "out.csv", [{"scenario": "swept"}])
    todo, eligible = sensitivity.scenes_to_sweep(tmp_path, tmp_path / "out.csv")
    assert [p.name for p in todo] == ["done"]
    assert [p.name for p in eligible] == ["done", "swept"]
