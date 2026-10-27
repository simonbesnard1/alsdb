"""alsdb.benchmark.timing: tile selection, measurement and a full small run."""

import csv
import json
import sys

import numpy as np
import pytest

from alsdb.benchmark import timing

TILE = 100.0


def _write_tile(path, x0, y0, seed):
    import pdal

    rng = np.random.default_rng(seed)
    n = 4000
    x, y = rng.uniform(0, TILE, (2, n))
    canopy = ((x // 20) % 2 == 0) & (rng.random(n) < 0.7)
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1"),
             ("ReturnNumber", "u1"), ("NumberOfReturns", "u1")]  # fmt: skip
    pts = np.zeros(n, dtype=dtype)
    pts["X"], pts["Y"] = x0 + x, y0 + y
    pts["Z"] = 100 + np.where(canopy, 12 + rng.normal(0, 0.3, n), 0)
    pts["Classification"] = np.where(canopy, 5, 2)
    pts["ReturnNumber"] = pts["NumberOfReturns"] = 1
    writer = {"type": "writers.las", "filename": str(path), "a_srs": "EPSG:25831",
              "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
              "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"}  # fmt: skip
    pdal.Pipeline(json.dumps([writer]), arrays=[pts]).execute()


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    """3 x 3 synthetic 100 m tiles named like PNOA (year in the name)."""
    folder = tmp_path_factory.mktemp("laz")
    for i in range(3):
        for j in range(3):
            x0, y0 = 500_000 + i * TILE, 4_600_000 + j * TILE
            _write_tile(folder / f"SYN_2016_{int(x0)}-{int(y0)}.las", x0, y0, seed=i * 3 + j)
    return folder


def test_select_blocks_nested_and_centred(tiles):
    files = sorted(tiles.iterdir())
    blocks, size = timing.select_blocks(files, [1, 4, 9])
    assert size == TILE
    assert [len(blocks[n]) for n in (1, 4, 9)] == [1, 4, 9]
    assert blocks[1][0].name == "SYN_2016_500100-4600100.las"  # the centre tile
    assert set(blocks[1]) <= set(blocks[4]) <= set(blocks[9])
    with pytest.raises(ValueError, match="square"):
        timing.select_blocks(files, [2])
    with pytest.raises(ValueError, match="fully covered"):
        timing.select_blocks(files, [16])


def test_measure_reports_tree_memory_and_failures(tmp_path):
    grow = "import time; b = bytearray(300_000_000); time.sleep(1.5)"
    child = [
        sys.executable,
        "-c",
        f"import subprocess, sys; subprocess.run([sys.executable, '-c', {grow!r}])",
    ]
    result = timing.measure(child, interval=0.2, log=tmp_path / "log.txt")
    assert result["exit_code"] == 0
    assert result["peak_rss_gb"] > 0.25  # memory of the grandchild counts
    assert result["wall_s"] >= 1.5
    failed = timing.measure([sys.executable, "-c", "raise SystemExit(3)"], cpus=[0])
    assert failed["exit_code"] == 3


def test_measure_counts_detached_workers(tmp_path):
    # Like lasR's PSOCK workers: a shell starts the worker in the background and exits.
    burn = "import time; b = bytearray(300_000_000); t = time.process_time()\nwhile time.process_time() - t < 1.5: pass"
    detach = f"({sys.executable} -c '{burn}' &)"
    result = timing.measure(["sh", "-c", detach], interval=0.2, log=tmp_path / "log.txt")
    assert result["exit_code"] == 0
    assert result["user_s"] + result["sys_s"] >= 1.4  # the worker's CPU counts
    assert result["wall_s"] >= 1.4  # the step lasts until the worker exits
    assert result["peak_rss_gb"] > 0.25  # and so does its memory


def _have_native_spikefree():
    try:
        from alsdb.processing.spikefree import require_backend

        require_backend()
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_native_spikefree(), reason="needs the native spikefree build")
def test_full_alsdb_run_resumes_and_reports(tiles, tmp_path, capsys):
    work, local = tmp_path / "work", tmp_path / "local"
    argv = [str(tiles), str(work), "--areas", "1", "4", "--cores", "1", "--repeats", "1",
            "--tools", "alsdb", "--launcher", "direct", "--local-dir", str(local)]  # fmt: skip
    timing.main(argv)
    rows = list(csv.DictReader((work / "timings.csv").open()))
    assert [(r["n_tiles"], r["step"]) for r in rows] == [
        ("1", "ingest"),
        ("1", "pitfree"),
        ("1", "spikefree"),
        ("4", "ingest"),
        ("4", "pitfree"),
        ("4", "spikefree"),
    ]
    assert all(r["exit_code"] == "0" for r in rows), [r for r in rows if r["exit_code"] != "0"]
    assert float(rows[0]["area_km2"]) == pytest.approx(0.01)
    assert (local / "tiles").is_dir() and not (local / "runs" / "local_1t_1c_r1_alsdb").exists()
    timing.main(argv)  # everything done: nothing rerun
    assert len(list(csv.DictReader((work / "timings.csv").open()))) == 6
    text = timing.report(work / "timings.csv")
    assert "alsdb" in text and "km2/core-h" in text


FAKE_TOOL = """#!/bin/sh
if [ "$1" = "-version" ]; then echo "fake 260123"; exit 0; fi
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
case "$(basename "$0")" in lascanopy64) out="${out%.tif}_max.tif";; esac
touch "$out"
echo "WARNING: there were 3 duplicate rasters"
exit 1
"""


def test_lastools_steps_with_stand_in_executables(tiles, tmp_path):
    """Setup (lasheight only) and run stages, exit code 1 with warnings as real LAStools."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("lasheight64", "las2dem64", "lascanopy64"):
        (bin_dir / name).write_text(FAKE_TOOL)
        (bin_dir / name).chmod(0o755)
    work = tmp_path / "work"
    timing.main([str(tiles), str(work), "--areas", "1", "--cores", "1", "--repeats", "1",
                 "--tools", "lastools", "--launcher", "direct", "--storage", "network",
                 "--lastools-dir", str(bin_dir), "--keep-outputs"])  # fmt: skip
    rows = list(csv.DictReader((work / "timings.csv").open()))
    assert [(r["stage"], r["step"], r["exit_code"]) for r in rows] == [
        ("setup", "lasheight", "0"),
        ("run", "spikefree+max", "0"),
    ]
    out = work / "runs" / "network_1t_1c_r1_lastools"
    assert len(list((out / "hag").glob("*.laz"))) == 1
    assert len(list((out / "spikefree").glob("*.tif"))) == 1
    assert len(list((out / "max").glob("*.tif"))) == 1
