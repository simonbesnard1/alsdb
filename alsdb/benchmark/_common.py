"""Helpers shared by the file-based reference runners."""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from alsdb.processing._terrain import GROUND_CLASSES


def list_tiles(laz_dir, tiles=None):
    """Sorted LAS/LAZ files in *laz_dir*; *tiles* keeps only the first N."""
    files = sorted(
        p for p in Path(laz_dir).iterdir() if p.is_file() and p.suffix.lower() in (".las", ".laz")
    )
    if not files:
        raise FileNotFoundError(f"No .las/.laz files in {laz_dir}")
    return files if tiles is None else files[:tiles]


def surface_classes(veg_classes, ground_classes=GROUND_CLASSES):
    return tuple(sorted({*map(int, veg_classes), *map(int, ground_classes)}))


def auto_parameters(path, veg_classes, ground_classes=GROUND_CLASSES, resolution=1.0):
    """alsdb's ``"auto"`` freeze distance and pitfree edge for one LAS/LAZ tile.

    Same rule as :func:`alsdb.processing._surface.auto_freeze_distance` and the
    pitfree ``"auto"`` edge, on the tile's surface-class first returns. PNOA
    overlap-strip returns (class 0) are not surface classes, so they are left
    out here exactly as in alsdb; header point counts would include them.
    """
    import pdal

    from alsdb.processing.chm import _pulse_spacing, _thin_highest_per_subcell

    pipeline = pdal.Pipeline(json.dumps([str(path)]))
    pipeline.execute()
    points = pipeline.arrays[0]
    keep = (points["ReturnNumber"] == 1) & np.isin(
        points["Classification"], surface_classes(veg_classes, ground_classes)
    )
    first = points[keep]
    if len(first) < 3:
        return None, None
    thinned = _thin_highest_per_subcell(first, resolution, value_field="Z")
    return (
        3.0 * _pulse_spacing(first),
        3.0 * _pulse_spacing(thinned, cell=5 * resolution),
    )


def write_params(out_dir, rows, fields):
    """Append per-tile parameters to ``<out_dir>/params.csv``."""
    path = Path(out_dir) / "params.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def write_provenance(out_dir, record):
    """Append one run record to ``<out_dir>/runs.json`` (a JSON list)."""
    path = Path(out_dir) / "runs.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record})
    path.write_text(json.dumps(runs, indent=2, default=str))


def run_parallel(fn, items, n_workers):
    """Run *fn* over *items* with *n_workers* threads (subprocess-bound work)."""
    with ThreadPoolExecutor(max(1, n_workers)) as pool:
        return list(pool.map(fn, items))
