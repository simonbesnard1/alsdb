"""alsdb.benchmark.sensitivity: freeze settings and scene discovery (no tools run)."""

import pytest

from alsdb.benchmark import sensitivity


def test_freeze_settings_follow_spacing_and_floor():
    sparse, dense = sensitivity.freeze_settings(1.5), sensitivity.freeze_settings(0.3)

    def freeze(settings, name):
        return settings[name]["freeze_distance"]

    assert freeze(sparse, "k3") == pytest.approx(4.5)
    assert freeze(sparse, "floor1.5") == pytest.approx(4.5)
    assert freeze(dense, "k3") == pytest.approx(0.9) and freeze(dense, "floor1.5") == 1.5
    assert freeze(dense, "fixed1.5") == freeze(sparse, "fixed1.5") == 1.5
    assert freeze(dense, "k2") < freeze(dense, "k3") < freeze(dense, "k5")
    # Fischer et al. (2024) at 1 / 0.3^2 = 11 pulses/m2: ~2.0 m
    assert freeze(dense, "lsf_tile") == pytest.approx(2.0, abs=0.05)
    assert dense["lspikefree"] == {"freeze_distance": "local", "spikefree_first_returns": True}
    assert not dense["lsf_local"].get("spikefree_first_returns", False)


def test_run_ignores_folders_that_are_not_scenes(tmp_path):
    (tmp_path / "not_a_scene").mkdir()
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "truth.npz").write_bytes(b"")  # no laz/: skipped
    out = sensitivity.run(tmp_path, tmp_path / "out.csv")
    assert out == tmp_path / "out.csv" and not out.exists()
