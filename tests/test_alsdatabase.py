# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import numpy as np
import pytest
import tiledb

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA_CFG = TileDBSchemaConfig(
    domain_min_x=100_000.0,
    domain_max_x=900_000.0,
    domain_min_y=3_000_000.0,
    domain_max_y=9_999_900.0,
)


def _make_points(n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(308_000.0, 310_000.0, n)
    y = rng.uniform(4_688_000.0, 4_690_000.0, n)
    attrs = {name: np.zeros(n, dtype=dtype) for name, dtype in LAS_ATTRIBUTES.items()}
    attrs["Z"] = rng.uniform(800.0, 850.0, n).astype(np.float64)
    attrs["Classification"] = rng.integers(1, 8, n, dtype=np.uint8)
    attrs["ReturnNumber"] = np.ones(n, dtype=np.uint8)
    attrs["NumberOfReturns"] = np.ones(n, dtype=np.uint8)
    return x, y, attrs


@pytest.fixture()
def db(tmp_path) -> ALSDatabase:
    return ALSDatabase(
        storage_type="local",
        uri=str(tmp_path / "test_array"),
        schema_cfg=_SCHEMA_CFG,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_array_does_not_exist_initially(db):
    assert not db.array_exists()


def test_create_array(db):
    db.create()
    assert db.array_exists()


def test_write_creates_array_implicitly(db):
    x, y, attrs = _make_points(50)
    db.write(x, y, attrs)
    assert db.array_exists()


def test_write_and_read_point_count(db):
    x, y, attrs = _make_points(200)
    db.write(x, y, attrs)
    with tiledb.open(db.array_uri, mode="r") as arr:
        result = arr[:]
    assert len(result["Z"]) == 200


def test_append_accumulates_points(db):
    x1, y1, a1 = _make_points(100, seed=1)
    x2, y2, a2 = _make_points(150, seed=2)
    db.write(x1, y1, a1)
    db.write(x2, y2, a2)
    with tiledb.open(db.array_uri, mode="r") as arr:
        result = arr[:]
    assert len(result["Z"]) == 250


def test_overwrite_resets_point_count(db):
    x, y, attrs = _make_points(100)
    db.write(x, y, attrs)
    db.create(overwrite=True)
    x2, y2, a2 = _make_points(30)
    db.write(x2, y2, a2)
    with tiledb.open(db.array_uri, mode="r") as arr:
        result = arr[:]
    assert len(result["Z"]) == 30


def test_ingest_returns_point_count(db, tmp_path):
    from unittest.mock import patch
    from alsdb.utils.schema import LAS_ATTRIBUTES

    rng = np.random.default_rng(0)
    n = 200
    dtype = [("X", np.float64), ("Y", np.float64)] + [
        (name, dt) for name, dt in LAS_ATTRIBUTES.items()
    ]
    data = np.zeros(n, dtype=dtype)
    data["X"] = rng.uniform(308_000, 310_000, n)
    data["Y"] = rng.uniform(4_688_000, 4_690_000, n)
    data["Z"] = rng.uniform(800, 850, n)
    data["ReturnNumber"] = np.ones(n, dtype=np.uint8)
    data["NumberOfReturns"] = np.ones(n, dtype=np.uint8)

    def _fake_read(path, chunk_size=None):
        yield data

    with patch("alsdb.tile.Tile.Tile.read", _fake_read):
        count = db.ingest("PNOA_2021_CYL-NW_308-4690_ORT-CLA-RGB.laz")

    assert count == n
