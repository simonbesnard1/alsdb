import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from alsdb.config import TileDBConfig
from alsdb.pipeline import ingest


def _fake_chunks():
    from alsdb.schema import LAS_ATTRIBUTES
    rng = np.random.default_rng(0)
    n = 200
    dtype = [("X", np.float64), ("Y", np.float64)] + [
        (name, dtype) for name, dtype in LAS_ATTRIBUTES.items()
    ]
    data = np.zeros(n, dtype=dtype)
    data["X"] = rng.uniform(300_000, 310_000, n)
    data["Y"] = rng.uniform(4_688_000, 4_690_000, n)
    data["Z"] = rng.uniform(800, 850, n)
    data["ReturnNumber"] = np.ones(n, dtype=np.uint8)
    data["NumberOfReturns"] = np.ones(n, dtype=np.uint8)
    yield data


def test_ingest_local(tmp_path):
    cfg = TileDBConfig(
        domain_min_x=100_000.0,
        domain_max_x=900_000.0,
        domain_min_y=3_000_000.0,
        domain_max_y=9_999_900.0,
        chunk_size=100,
    )
    uri = str(tmp_path / "pipeline_array")
    with patch("alsdb.pipeline.read_laz", return_value=_fake_chunks()):
        result_uri = ingest("fake.laz", uri, cfg)
    assert result_uri == uri

    import tiledb
    with tiledb.open(uri, mode="r") as arr:
        data = arr[:]
    assert len(data["Z"]) == 200
