import csv
import json
from dataclasses import asdict

import numpy as np
import pytest

from alsdb.benchmark.classification import (
    PTDParameters,
    SyntheticClassificationScenario,
    score_boundary_strips,
    score_ground_classification,
    synthetic_points,
    terrain_error,
)


def test_classification_scores_known_confusion_matrix():
    score = score_ground_classification(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert (
        score.true_positive,
        score.false_positive,
        score.false_negative,
        score.true_negative,
    ) == (
        1,
        1,
        1,
        1,
    )
    assert score.precision == score.recall == score.f1 == score.balanced_accuracy == 0.5
    assert score.omission == score.commission == score.total_error == score.ground_fraction == 0.5


def test_classification_scores_require_matching_vectors():
    with pytest.raises(ValueError, match="matching"):
        score_ground_classification([True], [True, False])


def test_terrain_error_known_values_and_missing_cells():
    reference = np.array([[1.0, 2.0], [3.0, np.nan]])
    candidate = np.array([[2.0, 2.0], [1.0, 4.0]])
    score = terrain_error(reference, candidate)
    assert score["bias"] == pytest.approx(-1 / 3)
    assert score["mae"] == pytest.approx(1.0)
    assert score["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert score["coverage"] == 1.0


def test_ptd_parameter_defaults_are_alsdb_s():
    from alsdb.core.ptd import DEFAULTS

    assert asdict(PTDParameters()) == DEFAULTS
    with pytest.raises(ValueError, match="votes"):
        PTDParameters(tilt_runs=2, votes=3)


def test_classifier_input_keeps_only_noise_classes():
    from alsdb.benchmark.classification import classifier_input

    points = np.zeros(5, dtype=[("Classification", "u1")])
    points["Classification"] = [2, 5, 7, 18, 1]
    assert classifier_input(points)["Classification"].tolist() == [1, 1, 7, 18, 1]
    assert points["Classification"].tolist() == [2, 5, 7, 18, 1]


def test_classifier_input_numbers_unnumbered_returns():
    from alsdb.benchmark.classification import classifier_input

    points = np.zeros(3, dtype=[("Classification", "u1"), ("ReturnNumber", "u1"),
                                ("NumberOfReturns", "u1")])  # fmt: skip
    points["ReturnNumber"] = [0, 2, 1]
    points["NumberOfReturns"] = [0, 3, 1]
    out = classifier_input(points)
    assert out["ReturnNumber"].tolist() == [1, 2, 1]
    assert out["NumberOfReturns"].tolist() == [1, 3, 1]


def test_reference_scoring_skips_unclassified_and_stratifies_slope():
    from alsdb.benchmark.classification import reference_context, score_against_reference

    points, truth = synthetic_points(
        SyntheticClassificationScenario(terrain="steep", size=40, density=4, noise_fraction=0)
    )
    points["Classification"] = np.where(truth, 2, 5)
    points["Classification"][:10] = 1  # unclassified in the reference
    ref = reference_context(points, (0, 0, 40, 40), boundary_width=5)
    assert not ref.scored[:10].any() and ref.scored[10:].all()
    assert set(ref.point_strata["slope"][ref.interior]) <= {"15-30", "30-45"}

    row, strata = score_against_reference(ref, points, points["Classification"] == 2)
    assert row["f1"] == 1 and row["omission"] == row["commission"] == 0
    assert row["dtm_mae"] == pytest.approx(0, abs=1e-9) and row["dtm_coverage"] > 0.9
    assert row["n_scored"] == len(points) - 10
    assert strata[0]["stratum"] == "all"
    assert {s["stratum"] for s in strata} == {"all", "slope", "cover"}

    # Terrain from a denser original: labels still from the decimated points.
    sparse = points[::4]
    ref_sparse = reference_context(sparse, (0, 0, 40, 40), 5, reference=points)
    np.testing.assert_allclose(ref_sparse.cells_z, ref.cells_z)
    assert ref_sparse.truth.sum() == (sparse["Classification"] == 2).sum()

    # Every reference ground point missed: type I error 1, no DTM.
    row, _ = score_against_reference(ref, points, np.zeros(len(points), bool))
    assert row["omission"] == 1 and np.isnan(row["dtm_mae"])


def test_synthetic_scenarios_are_reproducible_and_have_exact_labels():
    scenario = SyntheticClassificationScenario(terrain="steep", size=20, density=1, seed=4)
    first, first_truth = synthetic_points(scenario)
    second, second_truth = synthetic_points(scenario)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_truth, second_truth)
    assert len(first) == 400 and 0 < first_truth.sum() < len(first)


def test_boundary_scores_partition_points():
    scenario = SyntheticClassificationScenario(terrain="flat", size=20, density=1)
    points, truth = synthetic_points(scenario)
    prediction = truth.copy()
    scores = score_boundary_strips(points, truth, prediction, (0, 0, 20, 20), width=2)
    assert scores["boundary"].f1 == 1
    assert scores["interior"].f1 == 1
    assert scores["boundary"].true_ground + scores["interior"].true_ground == truth.sum()


def test_synthetic_grid_returns_one_row_per_method():
    from alsdb.benchmark.classification import run_synthetic_grid

    rows = run_synthetic_grid(
        [SyntheticClassificationScenario(size=10, density=0.2)],
        methods=("pmf", "smrf"),
        boundary_width=1,
    )
    assert [row["method"] for row in rows] == ["pmf", "smrf"]
    assert all("boundary_f1" in row and "wall_seconds" in row for row in rows)


def test_external_adapters_report_missing_executables():
    from alsdb.benchmark.classification import run_fastgc, run_lastools_lasground

    with pytest.raises(FileNotFoundError, match="LAStools"):
        run_lastools_lasground("input.laz", "output.laz", executable="missing-lasground")
    with pytest.raises(FileNotFoundError, match="FAST-GC"):
        run_fastgc("input.laz", "output", executable="missing-fastgc")


def test_ground_mask_is_aligned_when_classifier_reorders_points(tmp_path, monkeypatch):
    from alsdb.benchmark import classification

    reference, _ = synthetic_points(SyntheticClassificationScenario(size=5, density=0.2))
    classified = reference[[2, 0, 4, 1, 3]].copy()
    classified["Classification"] = [2, 1, 2, 2, 1]
    monkeypatch.setattr(classification, "_read_las", lambda path: classified)
    mask = classification._ground_mask_from_las(tmp_path / "classified.laz", reference)
    expected = np.zeros(5, dtype=bool)
    expected[[reference_index for reference_index in (2, 4, 1)]] = True
    np.testing.assert_array_equal(mask, expected)


def test_file_benchmark_writes_results_and_resumes(tmp_path, monkeypatch):
    from alsdb.benchmark import classification

    laz_dir = tmp_path / "input"
    laz_dir.mkdir()
    (laz_dir / "tile.laz").touch()
    points, truth = synthetic_points(SyntheticClassificationScenario(size=10, density=0.2))
    points["Classification"] = np.where(truth, 2, 5)
    calls = []

    monkeypatch.setattr(classification, "_read_las", lambda path: points)

    def classify(method, input_path, source, work_dir, **kwargs):
        calls.append(method)
        return classification.ClassificationRun(method, truth, 0.25)

    monkeypatch.setattr(classification, "_file_classifier", classify)
    output = tmp_path / "output"
    results = classification.run_file_benchmark(
        laz_dir, output, methods=("pmf", "smrf"), boundary_width=1
    )
    with results.open() as handle:
        rows = list(csv.DictReader(handle))
    assert calls == ["pmf", "smrf"]
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert all(float(row["f1"]) == 1 for row in rows)
    assert all(json.loads(row["parameters"]) for row in rows)
    assert (output / "strata.csv").exists()

    classification.run_file_benchmark(laz_dir, output, methods=("pmf", "smrf"), boundary_width=1)
    assert calls == ["pmf", "smrf"]


def test_file_benchmark_runs_each_ptd_parameter_set(tmp_path, monkeypatch):
    from alsdb.benchmark import classification

    laz_dir = tmp_path / "input"
    laz_dir.mkdir()
    (laz_dir / "tile.laz").touch()
    points, truth = synthetic_points(SyntheticClassificationScenario(size=10, density=0.2))
    points["Classification"] = np.where(truth, 2, 5)
    seen = []
    monkeypatch.setattr(classification, "_read_las", lambda path: points)

    def classify(method, input_path, source, work_dir, ptd=None, **kwargs):
        seen.append(ptd)
        return classification.ClassificationRun(method, truth, 0.1)

    monkeypatch.setattr(classification, "_file_classifier", classify)
    sets = (classification.PTDParameters(angle=20), classification.PTDParameters(angle=40))
    classification.run_file_benchmark(
        laz_dir, tmp_path / "out", methods=("ptd",), boundary_width=1, ptd=sets
    )
    assert seen == list(sets)


def _write_scene(scene, terrain="rolling"):
    import pdal

    from alsdb.benchmark.classification import _terrain_height

    points, truth = synthetic_points(
        SyntheticClassificationScenario(terrain=terrain, size=60, density=4, noise_fraction=0)
    )
    points["Classification"] = np.where(truth, 2, 5)
    (scene / "laz").mkdir(parents=True)
    pdal.Pipeline(
        json.dumps([{"type": "writers.las", "filename": str(scene / "laz" / "scene.laz")}]),
        arrays=[points],
    ).execute()
    # Truth on the 40 m core window from x0 = 10, y1 = 50 (north-up rows).
    cx, cy = np.meshgrid(10 + np.arange(40) + 0.5, 50 - np.arange(40) - 0.5)
    dtm = _terrain_height(cx, cy, terrain)
    np.savez(scene / "truth.npz", dtm=dtm, chm_centre=np.full_like(dtm, 20.0), x0=10.0, y1=50.0)
    (scene / "scene.json").write_text(json.dumps({"scenario": {"terrain": terrain}}))


def test_reference_from_truth_uses_the_core_window(tmp_path):
    from alsdb.benchmark.classification import (
        _read_las,
        reference_from_truth,
        score_against_reference,
    )

    _write_scene(tmp_path / "s")
    points = _read_las(tmp_path / "s" / "laz" / "scene.laz")
    ref = reference_from_truth(points, np.load(tmp_path / "s" / "truth.npz"))
    assert len(ref.cells_z) == 1600
    inside = (points["X"] >= 10) & (points["X"] < 50) & (points["Y"] > 10) & (points["Y"] <= 50)
    np.testing.assert_array_equal(ref.interior, inside)
    assert set(ref.point_strata["cover"][inside]) == {"95-100"}
    row, strata = score_against_reference(ref, points, points["Classification"] == 2)
    assert row["f1"] == 1 and row["dtm_mae"] < 0.1  # TIN of exact ground vs analytic terrain
    assert {s["stratum"] for s in strata} == {"all", "slope", "cover"}


def test_scene_benchmark_scores_and_resumes(tmp_path):
    from alsdb.benchmark.classification import run_scene_benchmark

    _write_scene(tmp_path / "scenes" / "rolling")
    out = run_scene_benchmark(tmp_path / "scenes", tmp_path / "out", methods=("smrf",))
    rows = list(csv.DictReader(out.open()))
    assert [r["status"] for r in rows] == ["ok"]
    assert json.loads(rows[0]["scenario"]) == {"terrain": "rolling"}
    assert float(rows[0]["dtm_mae"]) < 0.5
    run_scene_benchmark(tmp_path / "scenes", tmp_path / "out", methods=("smrf",))
    assert len(list(csv.DictReader(out.open()))) == 1


def test_dtm_references_give_self_and_consensus_scores(tmp_path):
    from alsdb.benchmark.classification import build_dtm_references, run_scene_benchmark

    scene = tmp_path / "scenes" / "area_pd_08"
    _write_scene(scene)
    meta = {"scenario": {"site": "S", "x0": 10.0, "y0": 10.0, "core": 40.0, "margin": 10.0}}
    (scene / "scene.json").write_text(json.dumps(meta))
    # The scene's own points stand in for the full-density survey.
    rows = build_dtm_references(tmp_path / "scenes", {"S": scene / "laz"}, methods=("smrf", "pmf"))
    assert (scene / "dtm_references.npz").exists()
    assert {r["stratum"] for r in rows} == {"all", "slope", "cover"}
    assert 0 <= rows[0]["agreed_share"] <= 1

    out = run_scene_benchmark(tmp_path / "scenes", tmp_path / "out", methods=("smrf",))
    row = next(csv.DictReader(out.open()))
    assert float(row["self_mae"]) < 1e-4  # same points as the reference (float32 storage)
    assert row["consensus_mae"] != ""


def test_ground_masks_round_trip(tmp_path):
    from alsdb.benchmark.classification import _save_ground_masks, load_ground_masks

    path = tmp_path / "m" / "scene.npz"
    a, b = np.array([1, 0, 1], bool), np.array([0, 0, 1], bool)
    _save_ground_masks(path, 3, {"r1": ("smrf", "{}", a)})
    _save_ground_masks(path, 3, {"r2": ("ptd", '{"angle": 30}', b)})  # resumed run adds
    masks = load_ground_masks(path)
    assert set(masks) == {"r1", "r2"}
    np.testing.assert_array_equal(masks["r1"][2], a)
    assert masks["r2"][:2] == ("ptd", '{"angle": 30}')


def test_scene_chm_on_each_classifiers_ground(tmp_path):
    pytest.importorskip("alsdb.processing._spikefree_native")
    from alsdb.benchmark.classification import run_scene_benchmark, run_scene_chm

    _write_scene(tmp_path / "scenes" / "rolling")
    run_scene_benchmark(tmp_path / "scenes", tmp_path / "out", methods=("smrf",))
    assert (tmp_path / "out" / "ground_masks" / "rolling.npz").exists()
    out = run_scene_chm(tmp_path / "scenes", tmp_path / "out")
    rows = list(csv.DictReader(out.open()))
    assert {r["method"] for r in rows} == {"provider", "smrf"}
    assert all(r["chm_mae"] != "" for r in rows)
    run_scene_chm(tmp_path / "scenes", tmp_path / "out")  # resumes: nothing added
    assert len(list(csv.DictReader(out.open()))) == 2


def test_pylasr_runs_with_a_private_temp_dir(tmp_path, monkeypatch):
    import os
    import sys
    import types

    from alsdb.benchmark import classification

    seen = []

    class Pipeline:
        def __add__(self, other):
            return self

        def execute(self, path):
            seen.append(os.environ.get("TMPDIR"))
            (tmp_path / "out.laz").write_bytes(b"x")
            return {"success": True}

    fake = types.SimpleNamespace(classify_with_ptd=lambda **k: Pipeline(), write_las=lambda p: None)
    monkeypatch.setitem(sys.modules, "pylasr", fake)
    monkeypatch.setattr(classification, "pdal", types.SimpleNamespace(
        Pipeline=lambda *a, **k: types.SimpleNamespace(
            execute=lambda: None, arrays=[np.zeros(1, dtype=[("Classification", "u1")])])))  # fmt: skip
    monkeypatch.setenv("TMPDIR", "/original")
    classification.run_pylasr_ptd(tmp_path / "in.laz", tmp_path / "out.laz")
    assert seen[0] != "/original" and "alsdb-pylasr-" in seen[0]
    assert os.environ["TMPDIR"] == "/original"


def test_consensus_ground_labels_an_unclassified_survey(tmp_path):
    import pdal

    from alsdb.benchmark.classification import _read_las, consensus_ground

    points, truth = synthetic_points(
        SyntheticClassificationScenario(terrain="rolling", size=60, density=8, noise_fraction=0)
    )
    points["Classification"] = 0  # delivered without classification
    (tmp_path / "in").mkdir()
    pdal.Pipeline(
        json.dumps([{"type": "writers.las", "filename": str(tmp_path / "in" / "s.laz"),
                     "a_srs": "EPSG:25831", "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001}]),
        arrays=[points],
    ).execute()  # fmt: skip
    out = tmp_path / "out" / "area.laz"
    stats = consensus_ground(tmp_path / "in", (0, 0, 60, 60), out, methods=("csf", "ptd"),
                             max_density=None)  # fmt: skip
    assert stats["methods"] == ["csf", "ptd"] and stats["agreed_share"] > 0.8
    assert json.loads(out.with_suffix(".json").read_text())["points"] == len(points)
    from alsdb.benchmark.classification import _terrain_height

    labelled = _read_las(out)
    assert len(labelled) == len(points)
    assert set(np.unique(labelled["Classification"]).tolist()) <= {1, 2}
    ground = labelled["Classification"] == 2
    height = labelled["Z"] - _terrain_height(labelled["X"], labelled["Y"], "rolling")
    assert ground.mean() == pytest.approx(truth.mean(), abs=0.03)
    assert np.mean(np.abs(height[ground]) < 0.2) > 0.95  # ground labels sit on the terrain
    assert np.mean(height[~ground] > 0.15) > 0.9


def test_ptd_lasr_baseline_is_added_after_the_defaults():
    import argparse

    from alsdb.benchmark.__main__ import _add_ptd_arguments, _ptd_sets

    parser = argparse.ArgumentParser()
    _add_ptd_arguments(parser)
    sets = _ptd_sets(parser.parse_args(["--ptd-lasr"]))
    assert sets[0] == PTDParameters()
    assert sets[1] == PTDParameters(angle=30.0, distance=2.0, tilt_runs=1, votes=1)
    assert _ptd_sets(parser.parse_args([])) == (PTDParameters(),)
