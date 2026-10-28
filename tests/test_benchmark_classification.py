import numpy as np
import pytest

from alsdb.benchmark.classification import score_ground_classification, terrain_error


def test_classification_scores_known_confusion_matrix():
    score = score_ground_classification(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert (score.true_positive, score.false_positive, score.false_negative, score.true_negative) == (
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
