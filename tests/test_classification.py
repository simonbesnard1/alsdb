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
    csf = ground_stage(ClassificationConfig(csf_resolution=0.75))
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
