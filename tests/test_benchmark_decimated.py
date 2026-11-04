"""alsdb.benchmark.decimated on a small artificial dense survey and a decimated copy."""

import json

import numpy as np
import pytest

from alsdb.benchmark import decimated
from alsdb.benchmark.uls import Area

X0, Y0, GROUND, TOP = 500_000.0, 4_600_000.0, 100.0, 120.0


def _write(path, step, classes_seen):
    """Pulses every *step* m; a 10 m crown block (class 4) at core 5-15 m; one class-1 point."""
    import pdal

    g = np.arange(-24.9, 44.9, step)
    px, py = (a.ravel() for a in np.meshgrid(g, g))
    crown = (px >= 5) & (px < 15) & (py >= 5) & (py < 15)
    pts = np.zeros(px.size + 1, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("GpsTime", "f8"),
        ("ReturnNumber", "u1"), ("NumberOfReturns", "u1"), ("Classification", "u1"),
        ("ScanAngleRank", "f4"), ("PointSourceId", "u2")])  # fmt: skip
    pts["X"][:-1], pts["Y"][:-1] = X0 + px, Y0 + py
    pts["Z"][:-1] = np.where(crown, TOP, GROUND)
    pts["Classification"][:-1] = np.where(crown, 4, 2)
    pts[-1] = (X0 + 30, Y0 + 30, GROUND + 500, 0, 1, 1, 1, 0, 1)  # unclassified high outlier
    pts["GpsTime"][:-1] = 4.5e8 + np.arange(px.size) * 1e-5
    pts["ReturnNumber"], pts["NumberOfReturns"], pts["PointSourceId"] = 1, 1, 1
    classes_seen.update(np.unique(pts["Classification"]).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    pdal.Pipeline(json.dumps([{"type": "writers.las", "filename": str(path), "a_srs": "EPSG:25831",
        "minor_version": 4, "dataformat_id": 6, "scale_x": 0.001, "scale_y": 0.001,
        "scale_z": 0.001, "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}]),
        arrays=[pts]).execute()  # fmt: skip


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("site")
    seen = set()
    _write(root / "original" / "tile.laz", 0.2, seen)  # 25 pulses/m2
    _write(root / "pd_01" / "tile.laz", 1.0, seen)  # 1 pulse/m2
    return root


def test_parse_version():
    assert decimated.parse_version("pd_00p5") == (0.5, "")
    assert decimated.parse_version("pd_04_ft_0.75") == (4.0, "ft_0.75")
    assert decimated.parse_version("original") == (None, "original")


def test_build_area_writes_scenes_with_full_density_reference(site, tmp_path):
    area = Area(site="T", laz_dir=str(site / "original"), x0=X0, y0=Y0, core=20.0, margin=5.0)
    assert list(decimated.versions_next_to(area.laz_dir)) == ["pd_01"]
    (scene,) = decimated.build_area(tmp_path, area)
    assert scene.name == "T_500000_4600000_20_pd_01"
    meta = json.loads((scene / "scene.json").read_text())
    assert meta["scenario"]["density"] == 1.0 and meta["crs"] == "EPSG:25831"
    assert meta["first_returns_per_m2"] == pytest.approx(1.0, rel=0.05)
    t = np.load(scene / "truth.npz")
    chm = t["chm_max"][::-1]  # rows south to north
    assert np.allclose(chm[6:14, 6:14], TOP - GROUND, atol=1e-3)
    assert np.allclose(chm[0:4, 0:4], 0.0, atol=1e-3)  # the class-1 outlier is not in the reference
    assert np.isfinite(t["chm_centre"]).mean() > 0.95  # 0.4 m windows hold a return
    import pdal

    laz = scene / "laz" / f"SYN_{meta['year']}_scene.laz"
    p = pdal.Pipeline(f'["{laz}"]')
    p.execute()
    assert set(np.unique(p.arrays[0]["Classification"]).tolist()) == {2, 5}
    assert decimated.build_area(tmp_path, area) == [scene]  # rerun: nothing rebuilt


def test_mixed_scene_has_dense_strips(tmp_path_factory, tmp_path):
    root = tmp_path_factory.mktemp("mixed")
    seen = set()
    _write(root / "original" / "tile.laz", 0.2, seen)
    _write(root / "pd_01" / "tile.laz", 1.0, seen)  # 1 pulse/m2
    _write(root / "pd_04" / "tile.laz", 0.5, seen)  # 4 pulses/m2
    area = Area(site="T", laz_dir=str(root / "original"), x0=X0, y0=Y0, core=20.0, margin=5.0)
    strips = (10.0, 4.0, 2.0)  # dense where (x - west edge - 2) mod 10 < 4
    scenes = decimated.build_area(tmp_path, area, mixed=("pd_01:pd_04",), strips=strips)
    scene = tmp_path / "T_500000_4600000_20_mix_pd_01_pd_04"
    assert scene in scenes
    meta = json.loads((scene / "scene.json").read_text())
    assert meta["scenario"]["density"] == 1.0 and meta["scenario"]["variant"] == "mix4"
    t = np.load(scene / "truth.npz")
    dense = t["dense"]
    assert dense.shape == (20, 20) and (dense == dense[0]).all()
    expected = decimated.in_strips(X0 + np.arange(20) + 0.5, X0 - 5, strips)
    np.testing.assert_array_equal(dense[0], expected)
    assert meta["dense_share"] == pytest.approx(expected.mean())
    import pdal

    p = pdal.Pipeline(f'["{scene / "laz" / "SYN_2024_scene.laz"}"]')
    p.execute()
    x = p.arrays[0]["X"]
    inside = decimated.in_strips(x, X0 - 5, strips)
    width = strips[1] / strips[0]
    per_m2_in = inside.sum() / (30 * 30 * width)
    per_m2_out = (~inside).sum() / (30 * 30 * (1 - width))
    assert per_m2_in == pytest.approx(4, rel=0.25) and per_m2_out == pytest.approx(1, rel=0.25)
    with pytest.raises(ValueError, match="both copies"):
        decimated.mixed_versions({"pd_01": root}, ["pd_01:pd_08"])
