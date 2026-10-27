"""alsdb.benchmark.uls on a small artificial ULS tile (no HELIOS++ or tools needed)."""

import json
import sqlite3
import struct

import numpy as np
import pytest

from alsdb.benchmark import synthetic, uls
from alsdb.benchmark.uls import Area

X0, Y0, GROUND, TOP = 500_000.0, 4_600_000.0, 100.0, 110.0
CROWN = (5.0, 15.0)  # local core coordinates of a square 10 m crown block


def _write_tile(path):
    """Vertical pulses every 0.2 m; half at 5 deg, half at 40 deg scan angle.

    Over the crown block, half the pulses return from the crown top and the
    ground, half from the crown top only; elsewhere one ground return.
    """
    import pdal

    g = np.arange(-21.9, 41.9, 0.2)
    px, py = (a.ravel() for a in np.meshgrid(g, g))
    n = px.size
    crown = (px >= CROWN[0]) & (px < CROWN[1]) & (py >= CROWN[0]) & (py < CROWN[1])
    two = crown & (np.arange(n) % 4 < 2)
    nret = np.where(two, 2, 1)
    pid = np.repeat(np.arange(n), nret)
    rn = np.ones(pid.size, np.uint8)
    rn[1:][pid[1:] == pid[:-1]] = 2
    z = np.where(crown[pid] & (rn == 1), TOP, GROUND)
    pts = np.zeros(pid.size, dtype=[
        ("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("GpsTime", "f8"), ("PointSourceId", "u2"),
        ("ReturnNumber", "u1"), ("NumberOfReturns", "u1"), ("Classification", "u1"),
        ("ScanAngleRank", "f4"), ("ScanChannel", "u1")])  # fmt: skip
    pts["X"], pts["Y"], pts["Z"] = X0 + px[pid], Y0 + py[pid], z
    pts["GpsTime"] = 4.5e8 + pid * 1e-5
    pts["PointSourceId"] = 1
    pts["ReturnNumber"], pts["NumberOfReturns"] = rn, nret[pid]
    pts["Classification"] = np.where(z == GROUND, 2, 1)
    pts["ScanAngleRank"] = np.where(pid % 2 == 0, 5.0, 40.0)
    rng = np.random.default_rng(0)
    pts = pts[rng.permutation(pts.size)]  # returns of a pulse are not adjacent in real tiles
    pdal.Pipeline(json.dumps([
        {"type": "writers.las", "filename": str(path), "a_srs": "EPSG:25831", "minor_version": 4,
         "dataformat_id": 6, "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001,
         "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}]), arrays=[pts]).execute()  # fmt: skip


@pytest.fixture(scope="module")
def area(tmp_path_factory):
    d = tmp_path_factory.mktemp("uls")
    _write_tile(d / "tile.laz")
    return Area(site="TEST", laz_dir=str(d), x0=X0, y0=Y0, core=20.0, margin=10.0, year=2024)


@pytest.fixture(scope="module")
def scan(area):
    return uls.scan_area(area, candidate_fraction=1.0, workers=1)


def test_scan_reads_crs_and_keeps_only_near_nadir_pulses(scan):
    assert scan.crs == "EPSG:25831"
    assert scan.nadir_pulses_per_m2 == pytest.approx(25 / 2, rel=0.02)  # 0.2 m grid, half at 5 deg
    assert np.all(np.abs(scan.candidates["ScanAngleRank"]) <= 20)


def test_reference_recovers_ground_and_crown(scan, area):
    truth, zgrid = uls.reference(scan, area)
    assert np.allclose(truth["dtm"], GROUND, atol=1e-3)
    assert np.allclose(zgrid, GROUND, atol=1e-3)
    chm = truth["chm_max"][::-1]  # back to rows south to north
    assert np.allclose(chm[6:14, 6:14], TOP - GROUND, atol=1e-3)  # crown: core 5-15 m
    assert np.allclose(chm[0:4, 0:4], 0.0, atol=1e-3)
    assert np.allclose(truth["chm_centre"][::-1][6:14, 6:14], TOP - GROUND, atol=1e-3)


@pytest.mark.parametrize("density", [0.5, 2.0, 5.0])
def test_thinning_keeps_whole_pulses_at_the_target_density(scan, density):
    pts, achieved = uls.thin_pulses(scan.candidates, scan.box, density)
    assert achieved == pytest.approx(density, rel=0.05)
    t, _ = uls.pulse_keys(pts)
    _, counts = np.unique(t, return_counts=True)
    first = pts[pts["ReturnNumber"] == 1]
    nr = dict(zip(np.round(first["GpsTime"] * 1e8).astype(np.int64), first["NumberOfReturns"]))
    assert all(nr[k] == c for k, c in zip(np.unique(t), counts))  # every return of each pulse
    again, _ = uls.thin_pulses(scan.candidates, scan.box, density)
    assert np.array_equal(np.sort(again["GpsTime"]), np.sort(pts["GpsTime"]))  # deterministic


def test_pulse_hash_is_per_pulse_and_seeded():
    t = np.array([5, 5, 6], np.int64)
    tag = np.array([64, 64, 64], np.int64)
    h = uls.pulse_hash(t, tag)
    assert h[0] == h[1] != h[2] and np.all((h >= 0) & (h < 1))
    assert uls.pulse_hash(t, tag, seed=1)[0] != h[0]


def test_plant_area_density_follows_macarthur_horn():
    veg = np.zeros((1, 1, 4))
    veg[0, 0, 3] = 30.0  # top layer intercepts 30 of 100
    veg[0, 0, 1] = 20.0  # lower layer intercepts 20 of the 70 that pass the top
    pad = uls.plant_area_density(veg, np.array([[50.0]]))
    expect = lambda below, inside: -np.log((below + 0.5) / (below + inside + 0.5)) / (0.5 * 0.5)
    assert pad[0, 0, 3] == pytest.approx(expect(70, 30))
    assert pad[0, 0, 1] == pytest.approx(expect(50, 20))
    assert pad[0, 0, 0] == pad[0, 0, 2] == 0.0


def test_voxel_scene_truth_matches_the_crown(scan, area, tmp_path):
    truth_ref, zgrid = uls.reference(scan, area)
    truth, n_vox, pad = uls.voxel_scene(scan, zgrid, area, 2.0, tmp_path)
    assert n_vox > 0 and pad > uls.PAD_MIN
    chm = truth["chm_max"][::-1]
    assert np.allclose(chm[6:14, 6:14], TOP - GROUND, atol=uls.VOXEL + 1e-6)
    assert np.allclose(chm[0:4, 0:4], 0.0)
    assert np.allclose(truth["dtm"], truth_ref["dtm"])
    assert "helios_classification 5" in (tmp_path / "forest.mtl").read_text()
    assert f"{X0 - area.margin};{Y0 - area.margin};0" in (tmp_path / "scene.xml").read_text()


def test_mesh_height_is_the_triangulated_grid():
    z = np.array([[0.0, 1.0], [2.0, 4.0]])  # z[j, i]
    assert synthetic.mesh_height(z, 1.0, 1.0) == 4.0
    assert synthetic.mesh_height(z, 0.5, 0.5) == pytest.approx(2.0)  # on the diagonal
    assert synthetic.mesh_height(z, 0.75, 0.25) == pytest.approx(0.75 * 1 + 0.25 * 3)
    assert synthetic.mesh_height(z, 0.25, 0.75) == pytest.approx(0.75 * 2 + 0.25 * 2)


def test_read_trees_from_geopackage(tmp_path):
    path = tmp_path / "trees.gpkg"
    con = sqlite3.connect(path)
    con.execute("create table gpkg_contents (table_name text)")
    con.execute("create table gpkg_geometry_columns (table_name text, column_name text)")
    con.execute("create table trees (geom blob, tls_treeheight real)")
    con.execute("insert into gpkg_contents values ('trees')")
    con.execute("insert into gpkg_geometry_columns values ('trees', 'geom')")
    for x, y, h in [(10.0, 20.0, 25.5), (500.0, 20.0, 30.0), (12.0, 22.0, None)]:
        blob = b"GP\x00\x01" + struct.pack("<i", 32632) + struct.pack("<BIdd", 1, 1, x, y)
        con.execute("insert into trees values (?, ?)", (blob, h))
    con.commit()
    t = uls.read_trees(str(path), "tls_treeheight", (0, 0, 100, 100))
    assert t.tolist() == [[10.0, 20.0, 25.5]]
