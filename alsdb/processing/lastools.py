"""Optional explicit LAStools executable adapter for reference comparisons."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from alsdb.processing._grid import GridSpec


def executable_version(executable="las2dem64"):
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"LAStools executable not found: {executable}")
    result = subprocess.run(
        [resolved, "-version"], capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or "LAStools version query failed")
    return (result.stdout + result.stderr).strip()


def rasterize_lastools(
    points,
    bbox,
    resolution,
    *,
    executable="las2dem64",
    freeze_distance=1.5,
    height_buffer=0.5,
    freeze_interval=0.25,
    max_triangle_edge=100.0,
    value_field="HeightAboveGround",
    demo=False,
    timeout=600,
):
    """Run las2dem on exactly the provided eligible points and height field.

    Uses a temporary ASCII point cloud to preserve input precision without LAS
    quantization. Demo mode is explicit and limited to fewer than 1.5M points,
    as documented by rapidlasso. No executable is downloaded automatically.
    """
    import rasterio
    from rasterio.transform import from_origin

    grid = GridSpec.from_bbox(bbox, resolution)
    if not np.isfinite([freeze_distance, height_buffer, freeze_interval, max_triangle_edge]).all():
        raise ValueError("LAStools parameters must be finite")
    if min(freeze_distance, freeze_interval, max_triangle_edge) <= 0 or height_buffer < 0:
        raise ValueError("Invalid spike-free parameters")
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"LAStools executable not found: {executable}")
    xyz = np.column_stack((points["X"], points["Y"], points[value_field]))
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if demo and len(xyz) >= 1_500_000:
        raise ValueError("Demo comparison requires fewer than 1.5 million points")
    if len(xyz) < 3:
        return np.full(grid.shape, np.nan, np.float32)
    with tempfile.TemporaryDirectory(prefix="alsdb-lastools-") as directory:
        source, target = Path(directory) / "points.txt", Path(directory) / "surface.tif"
        np.savetxt(source, xyz, fmt="%.17g")
        command = [
            resolved,
            "-i",
            str(source),
            "-iparse",
            "xyz",
            "-o",
            str(target),
            "-spike_free",
            str(freeze_distance),
            str(freeze_interval),
            str(height_buffer),
            "-step",
            str(resolution),
            "-ll",
            str(grid.bbox[0]),
            str(grid.bbox[1]),
            "-ncols",
            str(grid.nx),
            "-nrows",
            str(grid.ny),
            "-nbits",
            "32",
            "-kill",
            str(max_triangle_edge),
            "-no_kml",
            "-demo" if demo else "-fail",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode or not target.is_file():
            raise RuntimeError("LAStools failed: " + (result.stderr + result.stdout)[-4000:])
        with rasterio.open(target) as dataset:
            expected = from_origin(grid.x0, grid.y1, resolution, resolution)
            if dataset.shape != grid.shape or not dataset.transform.almost_equals(expected):
                raise ValueError("LAStools output grid differs from the requested pixel lattice")
            return dataset.read(1, masked=True).astype(np.float32).filled(np.nan)
