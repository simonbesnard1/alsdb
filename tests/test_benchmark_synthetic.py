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
    assert t["dtm"][0, 0] == pytest.approx(
        terrain_height("rolling", x, y), abs=0.01
    )  # mesh vs function
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
        Scenario(terrain="canyon")
    with pytest.raises(ValueError):
        Scenario(placement="grid")
    # Newer options only add suffixes, so earlier scenes keep their names.
    assert Scenario(understorey=True, placement="clumped").name == "rolling_conifer_d2_p1_r4_s0_u_c"
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
             "r": np.array([1.5]), "base": np.array([1.5]), "shape": np.array([0])}  # fmt: skip
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


@pytest.mark.parametrize(("kind", "steeper_than_45"), [("cliff", 0.05), ("ravine", 0.05)])
def test_new_terrains_have_sharp_breaks(kind, steeper_than_45):
    g = np.arange(0, 211.0)
    z = terrain_height(kind, *np.meshgrid(g, g))
    gy, gx = np.gradient(z)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    assert (slope > 45).mean() > steeper_than_45
    assert np.isfinite(z).all()


def test_default_trees_are_unchanged_by_the_new_options():
    rng = np.random.default_rng(3)
    plain = synthetic.make_trees("conifer", 60.0, np.random.default_rng(3))
    assert not plain["low"].any()
    with_shrubs = synthetic.make_trees("conifer", 60.0, rng, understorey=True)
    canopy = ~with_shrubs["low"]
    np.testing.assert_array_equal(with_shrubs["x"][canopy], plain["x"])  # same draws first
    shrubs = with_shrubs["h"][with_shrubs["low"]]
    assert shrubs.size > 0 and shrubs.max() <= 3.0


def test_clumped_trees_leave_bigger_gaps():
    from scipy.spatial import cKDTree

    def mean_nearest(placement):
        t = synthetic.make_trees("broadleaf", 200.0, np.random.default_rng(1), placement=placement)
        d, _ = cKDTree(np.column_stack((t["x"], t["y"]))).query(
            np.column_stack((t["x"], t["y"])), k=2
        )
        return d[:, 1].mean()

    assert mean_nearest("clumped") < 0.75 * mean_nearest("regular")


def test_understorey_gets_its_own_plant_area_index(tmp_path):
    meta = make_scene(Scenario(terrain="cliff", forest="conifer", core=40.0, margin=10.0,
                               understorey=True), tmp_path)  # fmt: skip
    assert set(meta["pad"]) == {"canopy", "understorey"} and meta["shrubs"] > 0
    pads = np.array(
        [float(r.split()[3]) for r in (tmp_path / "forest.vox").read_text().splitlines()[6:]]
    )
    low = np.isclose(pads, meta["pad"]["understorey"], rtol=1e-5)
    assert low.any() and (~low).any()
    assert low.sum() * 0.5**3 * meta["pad"]["understorey"] / 60.0**2 == pytest.approx(
        synthetic.UNDERSTOREY["pai"], rel=1e-3)  # fmt: skip
    t = np.load(tmp_path / "truth.npz")
    assert t["tree_h"].min() > 3.0  # shrubs are not scored as trees


def test_beam_divergence_and_voxel_reach_names_and_scanner(tmp_path):
    assert (
        Scenario(beam_divergence=0.5, voxel=0.25).name == "rolling_conifer_d2_p1_r4_s0_bd0.5_v0.25"
    )
    assert Scenario(beam_divergence=0.22).name == Scenario().name  # default: name unchanged
    with pytest.raises(ValueError):
        Scenario(beam_divergence=0)
    make_scene(Scenario(forest="open", core=20.0, margin=5.0, beam_divergence=0.5), tmp_path)
    assert 'beamDivergence_rad="0.0005"' in (tmp_path / "scanner.xml").read_text()


def _tree_csv(path, rows):
    import pandas as pd

    pd.DataFrame(rows, columns=list(synthetic.TREE_LIST_COLUMNS)).to_csv(path, index=False)
    return path


def test_tree_list_crowns_keep_their_leaf_area(tmp_path):
    rows = [(10.0, 10.0, 20.0, 12.0, 3.0, "cylinder", 100.0, 1.5),
            (12.0, 10.0, 18.0, 10.0, 2.0, "cone", 40.0, 0.8),
            (30.0, 30.0, 15.0, 5.0, 2.5, "ellipsoid", 60.0, 0.9),
            (99.0, 99.0, 15.0, 5.0, 2.5, "ellipsoid", 60.0, 0.9)]  # outside a 40 m scene  # fmt: skip
    trees = synthetic.read_tree_list(_tree_csv(tmp_path / "t.csv", rows), 40.0)
    assert len(trees["x"]) == 3 and trees["agb"].sum() == pytest.approx(3.2)
    occ, _, _, pad = synthetic.voxelise(trees, "flat", 40.0, 0.5, with_pad=True)
    assert pad.sum() * 0.5**3 == pytest.approx(200.0, rel=1e-5)  # leaf area summed over crowns
    assert (pad > 0).sum() == occ.sum()
    with pytest.raises(ValueError, match="crown shapes"):
        synthetic.read_tree_list(
            _tree_csv(tmp_path / "bad.csv", [(1, 1, 5, 1, 1, "box", 1, 1)]), 40
        )


def test_formind_tree_list_picks_the_nearest_year_and_crops(tmp_path):
    header = (
        "Full simulation data of each tree.\nlong names\n"
        + "\t".join(synthetic.FORMIND_COLUMNS)
        + "\n"
    )

    def row(t, x, y, h):  # Time ... X Y CLP LAI CD R RespMain RespGrowth ID CR
        vals = [
            t,
            1,
            0,
            1,
            2.0,
            0.3,
            h,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            50,
            1,
            1,
            x,
            y,
            0.4,
            2.0,
            6.0,
            0,
            0,
            0,
            7,
            3.0,
        ]
        return "\t".join(map(str, vals))

    res = tmp_path / "run.res"
    res.write_text(header + "\n".join([row(11, 5, 5, 20), row(51, 5, 5, 30), row(51, 150, 5, 25),
                                       row(101, 5, 5, 35)]) + "\n")  # fmt: skip
    out = tmp_path / "trees.csv"
    assert synthetic.formind_tree_list(res, 50, out, size=100.0) == 51
    trees = synthetic.read_tree_list(out, 100.0)
    assert trees["h"].tolist() == [30.0]  # the tree at x=150 lies outside the crop
    assert trees["base"][0] == pytest.approx(30 * 0.6)
    assert trees["leaf_area"][0] == pytest.approx(2.0 * np.pi * 9.0)
    assert synthetic.SHAPES[trees["shape"][0]] == "cylinder"
    synthetic.formind_tree_list(res, 50, out, size=100.0, shape="ellipsoid")
    assert synthetic.SHAPES[synthetic.read_tree_list(out, 100.0)["shape"][0]] == "ellipsoid"
    # several years in one streamed pass, with snapshots split across chunks
    outs = {10: tmp_path / "a.csv", 100: tmp_path / "b.csv"}
    used = synthetic.formind_tree_lists(res, outs, chunksize=1)
    assert used == {10: 11.0, 100: 101.0}
    assert len(synthetic.read_tree_list(outs[10], 200.0)["h"]) == 1
    res.write_text(header + "\n".join([row(51, 5, 5, 30), row(51, 6, 6, 31), row(51, 7, 7, 32)])
                   + "\n")  # fmt: skip
    synthetic.formind_tree_lists(res, {50: out}, chunksize=2)
    assert synthetic.read_tree_list(out, 100.0)["h"].tolist() == [30.0, 31.0, 32.0]
    # newer FORMIND versions write extra columns (effNPP after BInc): read by name
    names = list(synthetic.FORMIND_COLUMNS)
    names.insert(names.index("DInc"), "effNPP")
    vals = row(51, 5, 5, 30).split("\t")
    vals.insert(names.index("effNPP"), "9.9")
    res.write_text("a\nb\n" + "\t".join(names) + "\tExtra\n" + "\t".join(vals) + "\t0\n")
    synthetic.formind_tree_lists(res, {50: out})
    trees = synthetic.read_tree_list(out, 100.0)
    assert trees["h"].tolist() == [30.0] and trees["base"][0] == pytest.approx(30 * 0.6)
    assert trees["leaf_area"][0] == pytest.approx(2.0 * np.pi * 9.0)


def test_tree_list_scene_records_biomass(tmp_path):
    rows = [(x, y, 20.0, 12.0, 3.0, "cylinder", 50.0, 2.0)
            for x in (12.0, 22.0, 32.0) for y in (12.0, 22.0, 32.0)]  # fmt: skip
    trees = _tree_csv(tmp_path / "stand.csv", rows)
    scenario = Scenario(terrain="flat", forest="stand", trees=str(trees), core=20.0, margin=10.0)
    meta = make_scene(scenario, tmp_path / "scene")
    truth = np.load(tmp_path / "scene" / "truth.npz")
    core = [r for r in rows if 10 <= r[0] < 30 and 10 <= r[1] < 30]
    assert truth["tree_agb"].sum() == pytest.approx(2.0 * len(core))
    assert float(truth["agb_t_ha"]) == pytest.approx(2.0 * len(core) / (20 * 20 / 1e4))
    assert meta["pai"] > 0 and np.nanmax(truth["chm_max"]) == pytest.approx(20.0, abs=0.3)
    # scoring keeps the tree-list path apart from the count of scored tree tops
    np.savez(tmp_path / "scene" / "preds.npz", alsdb_pitfree=truth["chm_max"])
    row = synthetic.score(tmp_path / "scene")[0]
    assert row["tree_list"] == str(trees) and row["trees"] == len(core)
    with pytest.raises(ValueError, match="generated forests"):
        Scenario(forest="stand", trees=str(trees), understorey=True)
