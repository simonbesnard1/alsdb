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
    occ, origin, zmin = synthetic.voxelise(trees, "rolling", size, voxel)
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
