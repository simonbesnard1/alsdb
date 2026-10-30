"""alsdb.benchmark.sensitivity: freeze settings and scene discovery (no tools run)."""

import pytest

from alsdb.benchmark import sensitivity


def test_freeze_settings_follow_spacing_and_floor():
    sparse, dense = sensitivity.freeze_settings(1.5), sensitivity.freeze_settings(0.3)
    assert sparse["k3"] == pytest.approx(4.5) and sparse["floor1.5"] == pytest.approx(4.5)
    assert dense["k3"] == pytest.approx(0.9) and dense["floor1.5"] == 1.5
    assert dense["fixed1.5"] == sparse["fixed1.5"] == 1.5
    assert dense["k2"] < dense["k3"] < dense["k5"]


def test_run_ignores_folders_that_are_not_scenes(tmp_path):
    (tmp_path / "not_a_scene").mkdir()
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "truth.npz").write_bytes(b"")  # no laz/: skipped
    out = sensitivity.run(tmp_path, tmp_path / "out.csv")
    assert out == tmp_path / "out.csv" and not out.exists()
