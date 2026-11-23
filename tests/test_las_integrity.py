# SPDX-License-Identifier: EUPL-1.2
"""Damaged LAS/LAZ files are rejected from their header and end, before PDAL reads points."""

import json
import shutil
import struct

import numpy as np
import pdal
import pytest

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.tile.tile import CorruptLASError, Tile, check_las_file
from tests.conftest import SCHEMA_CFG

N = 120_000  # three LASzip chunks of 50,000 points


def _write(path, compress):
    rng = np.random.default_rng(0)
    pts = np.zeros(N, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")])
    pts["X"] = 500_000 + rng.uniform(0, 100, N)
    pts["Y"] = 4_500_000 + rng.uniform(0, 100, N)
    pts["Z"] = rng.uniform(0, 30, N)
    pts["Classification"] = 2
    stage = {"type": "writers.las", "filename": str(path), "a_srs": "EPSG:25830", "minor_version": 4,
             "dataformat_id": 6, "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01, "compression": compress}  # fmt: skip
    pdal.Pipeline(json.dumps([stage]), arrays=[pts]).execute()
    return path


@pytest.fixture(scope="module")
def laz(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("laz") / "good.laz", True)


@pytest.fixture(scope="module")
def las(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("las") / "good.las", False)


def _copy(src, dst, edit):
    shutil.copyfile(src, dst)
    data = bytearray(dst.read_bytes())
    edit(data)
    dst.write_bytes(bytes(data))
    return dst


def _chunk_table(data):
    data_offset = struct.unpack_from("<I", data, 96)[0]
    return struct.unpack_from("<q", data, data_offset)[0]


def test_complete_files_pass(laz, las):
    check_las_file(laz)
    check_las_file(las)
    assert sum(len(a) for a in Tile(laz).read()) == N


def test_a_zero_filled_end_is_rejected(laz, tmp_path):
    """The PNOA case: the end of an incomplete copy reads back as zeros."""

    def zero_tail(d):
        d[len(d) // 2 :] = bytes(len(d) - len(d) // 2)

    bad = _copy(laz, tmp_path / "zeros.laz", zero_tail)
    with pytest.raises(CorruptLASError, match="chunk table|zero bytes"):
        check_las_file(bad)


def test_an_empty_chunk_table_is_rejected(laz, tmp_path):
    def empty_table(d):
        struct.pack_into("<I", d, _chunk_table(d) + 4, 0)

    bad = _copy(laz, tmp_path / "nochunks.laz", empty_table)
    with pytest.raises(CorruptLASError, match="lists 0 chunks, 3 expected"):
        check_las_file(bad)


def test_a_truncated_laz_is_rejected(laz, tmp_path):
    bad = tmp_path / "cut.laz"
    bad.write_bytes(laz.read_bytes()[: laz.stat().st_size // 2])
    with pytest.raises(CorruptLASError, match="chunk table pointer"):
        check_las_file(bad)


def test_a_truncated_las_and_a_non_las_file_are_rejected(las, tmp_path):
    bad = tmp_path / "cut.las"
    bad.write_bytes(las.read_bytes()[: las.stat().st_size - 1000])
    with pytest.raises(CorruptLASError, match="truncated"):
        check_las_file(bad)
    text = tmp_path / "notes.laz"
    text.write_text("not a point cloud")
    with pytest.raises(CorruptLASError, match="LASF"):
        check_las_file(text)


def test_reading_a_damaged_file_raises_before_pdal(laz, tmp_path, monkeypatch):
    bad = _copy(
        laz, tmp_path / "nochunks.laz", lambda d: struct.pack_into("<I", d, _chunk_table(d) + 4, 0)
    )
    tile = Tile(bad)
    assert tile.name.year  # the header alone is still readable
    executed = []
    monkeypatch.setattr(pdal.Pipeline, "execute", lambda self: executed.append(1))
    with pytest.raises(CorruptLASError):
        next(tile.read())
    assert executed == []


def test_ingest_records_a_damaged_file_as_failed_and_goes_on(laz, tmp_path):
    bad = _copy(
        laz, tmp_path / "nochunks.laz", lambda d: struct.pack_into("<I", d, _chunk_table(d) + 4, 0)
    )
    good = tmp_path / "good.laz"
    shutil.copyfile(laz, good)
    db = ALSDatabase(storage_type="local", uri=str(tmp_path / "arr"), schema_cfg=SCHEMA_CFG)
    db.create(crs="EPSG:25830")
    db.ingest_many([bad, good])
    manifest = db.load_manifest()
    assert manifest["good.laz"]["status"] == "ok"
    assert manifest["nochunks.laz"]["status"] == "failed"
    assert "0 chunks" in manifest["nochunks.laz"]["error"]
