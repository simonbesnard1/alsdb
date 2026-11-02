import csv
import json

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


def test_ptd_parameter_defaults_match_lasr():
    assert PTDParameters() == PTDParameters(10.0, 30.0, 2.0, 0.25)


def test_classifier_input_keeps_only_noise_classes():
    from alsdb.benchmark.classification import classifier_input

    points = np.zeros(5, dtype=[("Classification", "u1")])
    points["Classification"] = [2, 5, 7, 18, 1]
    assert classifier_input(points)["Classification"].tolist() == [1, 1, 7, 18, 1]
    assert points["Classification"].tolist() == [2, 5, 7, 18, 1]


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
