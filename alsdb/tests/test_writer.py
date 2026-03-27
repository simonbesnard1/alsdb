import tempfile
from pathlib import Path

import numpy as np
import pytest
import tiledb

from alsdb.config import TileDBConfig
from alsdb.schema import LAS_ATTRIBUTES, create_schema
from alsdb.writer import ensure_array, write_points


def _synthetic_points(n: int = 100) -> np.ndarray:
    rng = np.random.default_rng(42)
    dtype = [
        ("X", np.float64),
        ("Y", np.float64),
    ] + [(name, dtype) for name, dtype in LAS_ATTRIBUTES.items()]
    data = np.zeros(n, dtype=dtype)
    data["X"] = rng.uniform(300_000, 310_000, n)
    data["Y"] = rng.uniform(4_688_000, 4_690_000, n)
    data["Z"] = rng.uniform(800, 850, n)
    data["Intensity"] = rng.integers(1, 4096, n, dtype=np.uint16)
    data["Classification"] = rng.integers(1, 8, n, dtype=np.uint8)
    data["Red"] = rng.integers(256, 65280, n, dtype=np.uint16)
    data["Green"] = rng.integers(256, 65280, n, dtype=np.uint16)
    data["Blue"] = rng.integers(256, 65280, n, dtype=np.uint16)
    data["ReturnNumber"] = np.ones(n, dtype=np.uint8)
    data["NumberOfReturns"] = np.ones(n, dtype=np.uint8)
    return data


@pytest.fixture()
def tmp_array(tmp_path):
    cfg = TileDBConfig(
        domain_min_x=100_000.0,
        domain_max_x=900_000.0,
        domain_min_y=3_000_000.0,
        domain_max_y=9_999_900.0,
    )
    uri = str(tmp_path / "test_array")
    ensure_array(uri, cfg)
    return uri


def test_write_and_read(tmp_array):
    points = _synthetic_points(500)
    write_points(tmp_array, points)

    with tiledb.open(tmp_array, mode="r") as arr:
        result = arr[:]
    assert len(result["Z"]) == 500


def test_overwrite(tmp_path):
    cfg = TileDBConfig(
        domain_min_x=100_000.0,
        domain_max_x=900_000.0,
        domain_min_y=3_000_000.0,
        domain_max_y=9_999_900.0,
    )
    uri = str(tmp_path / "overwrite_array")
    ensure_array(uri, cfg)
    write_points(uri, _synthetic_points(100))
    ensure_array(uri, cfg, overwrite=True)
    write_points(uri, _synthetic_points(50))
    with tiledb.open(uri, mode="r") as arr:
        result = arr[:]
    assert len(result["Z"]) == 50
