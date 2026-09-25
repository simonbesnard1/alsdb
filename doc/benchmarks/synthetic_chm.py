"""Generate a reproducible known canopy and run the isolated-process benchmark.

Usage: pixi run -e spikefree python doc/benchmarks/synthetic_chm.py /tmp/chm-benchmark
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    x, y = np.meshgrid(np.arange(-2.0, 22.0, 0.5), np.arange(-2.0, 22.0, 0.5))
    xy = np.column_stack((x.ravel(), y.ravel()))
    lower_xy = rng.uniform(-2, 22, size=(1200, 2))
    points = np.empty(
        len(xy) + len(lower_xy),
        dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("HeightAboveGround", "f8")],
    )
    points["X"] = np.r_[xy[:, 0], lower_xy[:, 0]]
    points["Y"] = np.r_[xy[:, 1], lower_xy[:, 1]]
    points["Z"] = 15 + 0.1 * points["X"] + 0.15 * points["Y"]
    points["Z"][len(xy) :] -= rng.uniform(2, 10, len(lower_xy))
    points["HeightAboveGround"] = points["Z"]
    np.save(directory / "points.npy", points)
    gx, gy = np.meshgrid(np.arange(0.5, 20), np.arange(19.5, 0, -1))
    np.save(directory / "truth.npy", 15 + 0.1 * gx + 0.15 * gy)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "alsdb.processing.benchmark",
            "--input",
            str(directory / "points.npy"),
            "--reference",
            str(directory / "truth.npy"),
            "--bbox",
            "0",
            "0",
            "20",
            "20",
            "--tile-size",
            "10",
            "--tile-buffer",
            "2",
            "--freeze-distance",
            "1.5",
            "--output",
            str(directory / "report.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
