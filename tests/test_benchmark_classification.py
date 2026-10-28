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


def test_ptd_parameter_defaults_match_pylasr():
    assert PTDParameters() == PTDParameters(10.0, 20.0, 3.0, 0.25)


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
