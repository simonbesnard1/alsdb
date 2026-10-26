"""alsdb.benchmark: reference runners (external tools faked) and matched comparison."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alsdb.processing.chm import compute_chm

BBOX = (308000.0, 4688000.0, 309000.0, 4689000.0)
OPTS = {"resolution": 10.0, "year": 2021, "bbox": BBOX}


def _write_reference(path, data, x0, y1, res, nodata=-9999.0):
    import rasterio
    from rasterio.transform import from_origin

    data = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
        dtype="float32", transform=from_origin(x0, y1, res, res), nodata=nodata,
    ) as ds:  # fmt: skip
        ds.write(data, 1)


def _chm(store):
    return store._root["10m"]["chm"][0]


def _offset_reference(tmp_path, chm, offset):
    """Reference = CHM - offset, split into two tiles on a different tiling."""
    ref = tmp_path / "ref"
    ref.mkdir()
    x0, y1 = BBOX[0], BBOX[3]
    _write_reference(ref / "a.tif", chm[:, :30] - offset, x0, y1, 10.0)
    _write_reference(ref / "b.tif", chm[:, 30:] - offset, x0 + 300, y1, 10.0)
    return ref


def test_compare_scores_known_offset_on_matched_pixels(provider, store, tmp_path):
    from alsdb.benchmark.compare import compare

    compute_chm(provider, store, **OPTS)
    chm = _chm(store)
    ref = _offset_reference(tmp_path, chm, 1.5)
    result = compare({"alsdb": store.path}, {"ref": ref}, year=2021, resolution=10.0)
    stats = result.stats()["ref"]["alsdb"]["all"]
    assert stats["n"] == np.isfinite(chm).sum()
    assert stats["bias"] == pytest.approx(1.5, abs=1e-5)
    assert stats["mae"] == pytest.approx(1.5, abs=1e-5)
    bins = result.stats()["ref"]["alsdb"]["bins"]
    assert sum(b["n"] for b in bins) == stats["n"]
    assert (tmp_path / "ref" / ".alsdb_index.json").is_file()  # index cached for reruns


def test_compare_uses_only_tiles_finished_in_every_candidate(provider, store, tmp_path):
    from alsdb.benchmark.compare import compare

    compute_chm(provider, store, **OPTS)  # 1000 m bbox, 500 m tiles: 2x2 tiles of 50 px
    chm = _chm(store)
    arr = store._root["10m"]["chm"]
    runs = dict(arr.attrs["processing_runs"])
    runs["2021"] = {**runs["2021"], "status": "running"}
    arr.attrs["processing_runs"] = runs
    markers = Path(store.path) / "_alsdb_tiles" / "10m" / "chm" / "2021"
    for name in ("1", "2", "3"):
        (markers / name).unlink()
    ref = _offset_reference(tmp_path, chm, 0.0)
    result = compare({"alsdb": store.path}, {"ref": ref}, year=2021, resolution=10.0)
    assert result.tiles_compared == 1
    assert result.stats()["ref"]["alsdb"]["all"]["n"] == np.isfinite(chm[:50, :50]).sum()


def test_compare_rejects_misaligned_reference(provider, store, tmp_path):
    from alsdb.benchmark.compare import compare

    compute_chm(provider, store, **OPTS)
    ref = tmp_path / "ref"
    ref.mkdir()
    _write_reference(ref / "a.tif", _chm(store), BBOX[0] + 3.0, BBOX[3], 10.0)
    with pytest.raises(ValueError, match="aligned"):
        compare({"alsdb": store.path}, {"ref": ref}, year=2021, resolution=10.0)


def test_compare_cli_prints_summary(provider, store, tmp_path, capsys):
    from alsdb.benchmark.__main__ import main

    compute_chm(provider, store, **OPTS)
    ref = _offset_reference(tmp_path, _chm(store), 0.5)
    npz = tmp_path / "out.npz"
    main(["compare", "--store", f"v3={store.path}", "--ref", f"ref={ref}",
          "--year", "2021", "--resolution", "10", "--npz", str(npz)])  # fmt: skip
    assert "bias=+0.500" in capsys.readouterr().out
    saved = np.load(npz)
    assert json.loads(str(saved["stats_json"]))["ref"]["v3"]["all"]["n"] > 0


# ---------------------------------------------------------------------------
# Synthetic LAS tile: ground plane + 12 m canopy blocks with gaps
# ---------------------------------------------------------------------------


def _write_las(path, seed=0):
    import pdal

    rng = np.random.default_rng(seed)
    n = 30_000
    x, y = rng.uniform(0, 100, (2, n))
    canopy = ((x // 20) % 2 == 0) & (rng.random(n) < 0.7)
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1"),
             ("ReturnNumber", "u1"), ("NumberOfReturns", "u1")]  # fmt: skip
    pts = np.zeros(n, dtype=dtype)
    pts["X"], pts["Y"] = 500_000 + x, 4_600_000 + y
    pts["Z"] = 100 + 0.05 * x + np.where(canopy, 12 + rng.normal(0, 0.3, n), 0)
    pts["Classification"] = np.where(canopy, 5, 2)
    pts["ReturnNumber"] = pts["NumberOfReturns"] = 1
    writer = {"type": "writers.las", "filename": str(path), "a_srs": "EPSG:25831",
              "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01}  # fmt: skip
    pdal.Pipeline(json.dumps([writer]), arrays=[pts]).execute()
    return pts


# ---------------------------------------------------------------------------
# LAStools runner (executables faked)
# ---------------------------------------------------------------------------


def _fake_lastools(tmp_path, monkeypatch):
    from alsdb.benchmark import lastools

    tools = tmp_path / "lastools"
    tools.mkdir()
    for name in ("lasheight64", "las2dem64", "lascanopy64"):
        (tools / name).write_text("#!/bin/sh\n")
        (tools / name).chmod(0o755)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "-version":
            return SimpleNamespace(returncode=0, stdout="LAStools (fake) 260101", stderr="")
        target = Path(command[command.index("-o") + 1])
        if Path(command[0]).name == "lascanopy64":
            target = target.with_name(target.stem + "_max.tif")
        target.write_bytes(b"x")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lastools.subprocess, "run", fake_run)
    return tools, calls


def test_lastools_run_builds_commands_with_auto_freeze_and_resumes(tmp_path, monkeypatch):
    from alsdb.benchmark import lastools
    from alsdb.benchmark._common import auto_parameters

    laz = tmp_path / "laz"
    laz.mkdir()
    _write_las(laz / "tile.las")
    tools, calls = _fake_lastools(tmp_path, monkeypatch)
    out = tmp_path / "out"
    lastools.run(laz, out, lastools_dir=tools)
    by_tool = {Path(c[0]).name: c for c in calls if c[-1] != "-version"}
    dem = by_tool["las2dem64"]
    assert dem[dem.index("-keep_class") + 1 : dem.index("-spike_free")] == ["2", "3", "4", "5", "9"]
    freeze, _ = auto_parameters(laz / "tile.las", (3, 4, 5))
    assert dem[dem.index("-spike_free") + 1] == f"{freeze:.2f}"
    assert (
        "-fail" in dem and "-fail" in by_tool["lasheight64"] and "-fail" in by_tool["lascanopy64"]
    )
    assert "-keep_first" in by_tool["lascanopy64"]
    assert (out / "max" / "tile.tif").is_file() and not (out / "max" / "tile_max.tif").exists()
    assert (out / "spikefree" / "tile.tif").is_file()
    assert "tile.las" in (out / "params.csv").read_text()
    assert json.loads((out / "runs.json").read_text())[0]["tool"] == "lastools"
    calls.clear()
    lastools.run(laz, out, lastools_dir=tools)
    assert [c for c in calls if c[-1] != "-version"] == []  # finished tiles are skipped


def test_lastools_run_reuses_existing_hag(tmp_path, monkeypatch):
    from alsdb.benchmark import lastools

    laz = tmp_path / "laz"
    laz.mkdir()
    _write_las(laz / "tile.las")
    hag = tmp_path / "HAG_classfixed"
    hag.mkdir()
    (hag / "tile.laz").write_bytes(b"x")
    tools, calls = _fake_lastools(tmp_path, monkeypatch)
    lastools.run(laz, tmp_path / "out", lastools_dir=tools, hag_dir=hag, products=("spikefree",))
    names = [Path(c[0]).name for c in calls if c[-1] != "-version"]
    assert names == ["las2dem64"]
    dem = calls[-1]
    assert dem[dem.index("-i") + 1] == str(hag / "tile.laz")


def test_auto_parameters_match_alsdb_rules(tmp_path):
    from alsdb.benchmark._common import auto_parameters
    from alsdb.processing._surface import auto_freeze_distance
    from alsdb.processing.chm import _pulse_spacing, _thin_highest_per_subcell

    pts = _write_las(tmp_path / "t.las")
    freeze, edge = auto_parameters(tmp_path / "t.las", (3, 4, 5))
    pts["X"], pts["Y"] = np.round(pts["X"], 2), np.round(pts["Y"], 2)  # LAS 1 cm quantization
    assert freeze == pytest.approx(auto_freeze_distance(pts), rel=1e-3)
    thinned = _thin_highest_per_subcell(pts, 1.0, value_field="Z")
    assert edge == pytest.approx(3 * _pulse_spacing(thinned, cell=5.0), rel=1e-3)


# ---------------------------------------------------------------------------
# lasR runner
# ---------------------------------------------------------------------------


def test_lasr_run_passes_options_and_records_version(tmp_path, monkeypatch):
    from alsdb.benchmark import lasr

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "-e":
            return SimpleNamespace(returncode=0, stdout="0.21.2\n", stderr="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lasr.shutil, "which", lambda name: "/fake/Rscript")
    monkeypatch.setattr(lasr.subprocess, "run", fake_run)
    lasr.run("/data/laz", tmp_path, veg_classes=(20,), tiles=3, n_workers=2)
    command = calls[-1]
    assert command[1] == str(lasr.SCRIPT) and Path(command[1]).is_file()
    assert {"--veg=20", "--tiles=3", "--ncores=2", "--freeze=auto", "--edge=auto"} <= set(command)
    run = json.loads((tmp_path / "runs.json").read_text())[0]
    assert run["tool"] == "lasR" and run["version"] == "0.21.2"


def test_lasr_missing_rscript_has_actionable_error(monkeypatch):
    from alsdb.benchmark import lasr

    monkeypatch.setattr(lasr.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="pixi install -e lasr"):
        lasr.lasr_version()


def _lasr_available():
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(["Rscript", "-e", "library(lasR)"], capture_output=True, check=False)
    return probe.returncode == 0


@pytest.mark.skipif(not _lasr_available(), reason="needs R + lasR (pixi -e lasr)")
def test_lasr_pitfree_matches_alsdb_on_synthetic_tile(tmp_path):
    """Independent implementations, same algorithm and parameters: same raster."""
    import rasterio

    from alsdb.benchmark import lasr
    from alsdb.processing._surface import CHMConfig, reconstruct_chm

    laz = tmp_path / "laz"
    laz.mkdir()
    pts = _write_las(laz / "tile.las")
    lasr.run(laz, tmp_path / "out", n_workers=1)
    with rasterio.open(tmp_path / "out" / "pitfree" / "tile.tif") as ds:
        x0, y1 = ds.transform.c, ds.transform.f
        theirs = ds.read(1, masked=True).filled(np.nan)
    pts["X"], pts["Y"], pts["Z"] = (np.round(pts[k], 2) for k in ("X", "Y", "Z"))
    crop = (x0 + 10, y1 - theirs.shape[0] + 10, x0 + theirs.shape[1] - 10, y1 - 10)
    ours = reconstruct_chm(pts, crop, 1.0, CHMConfig(method="pitfree"))["chm"]
    theirs = theirs[10:-10, 10:-10]
    both = np.isfinite(ours) & np.isfinite(theirs)
    assert both.mean() > 0.95
    assert np.mean(np.abs(ours[both] - theirs[both]) < 0.05) > 0.95


# ---------------------------------------------------------------------------
# Moved modules: point benchmark CLI and old import paths
# ---------------------------------------------------------------------------


def test_old_import_paths_still_work():
    from alsdb.processing.benchmark import compare_rasters  # noqa: F401
    from alsdb.processing.lastools import rasterize_lastools  # noqa: F401


def test_points_cli_runs_tin_methods(tmp_path):
    """The point benchmark still passed the removed max_distance to pitfree."""
    from alsdb.benchmark.points import main
    from alsdb.processing._terrain import normalize_points
    from tests.test_surface_pipeline import scene

    points, _ = normalize_points(scene())
    np.save(tmp_path / "points.npy", points)
    main(["--input", str(tmp_path / "points.npy"), "--bbox", "0", "0", "10", "10",
          "--tile-size", "5", "--tile-buffer", "2", "--output", str(tmp_path / "r.json"),
          "--methods", "max", "pitfree", "highest_subcell_tin"])  # fmt: skip
    report = json.loads((tmp_path / "r.json").read_text())
    assert set(report["methods"]) == {"max", "pitfree", "highest_subcell_tin"}


def test_lastools_failure_reports_exit_code_and_stray_outputs(tmp_path, monkeypatch):
    from alsdb.benchmark import lastools

    target = tmp_path / "tile.tif"
    (tmp_path / "tile_1.tif").write_bytes(b"x")  # output written under another name
    monkeypatch.setattr(
        lastools.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="WARNING: something"),
    )  # fmt: skip
    with pytest.raises(RuntimeError) as err:
        lastools._call(["las2dem64", "-o", str(target)], {}, target)
    message = str(err.value)
    assert "exit code 0" in message and "output missing" in message
    assert "tile_1.tif" in message and "command: las2dem64 -o" in message


@pytest.mark.parametrize(
    ("code", "text", "ok"),
    [
        (0, "", True),
        (1, "WARNING: there were 500 duplicate rasters", True),  # real las2dem 260123
        (1, "ERROR: cannot understand argument '-x'", False),
        (2, "WARNING: something", False),
    ],
)
def test_lastools_exit_code_one_with_only_warnings_is_success(
    tmp_path, monkeypatch, code, text, ok
):
    from alsdb.benchmark import lastools

    target = tmp_path / "tile.tif"
    target.write_bytes(b"x")
    monkeypatch.setattr(
        lastools.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=code, stdout="", stderr=text),
    )  # fmt: skip
    if ok:
        lastools._call(["las2dem64"], {}, target)
    else:
        with pytest.raises(RuntimeError, match=f"exit code {code}"):
            lastools._call(["las2dem64"], {}, target)
