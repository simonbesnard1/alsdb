"""Reproducible CHM comparison on one pre-normalized, pre-filtered point set.

Run ``python -m alsdb.processing.benchmark --help``. Each method executes in
its own process so peak RSS is comparable (includes Python/library overhead).
LAStools comparisons require a separately installed executable.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label, maximum_filter, median_filter
from scipy.spatial import cKDTree

from alsdb.processing._grid import GridSpec
from alsdb.processing._products import software_versions


def _peaks(grid):
    valid = np.isfinite(grid)
    data = np.where(valid, grid, -np.inf)
    peaks = valid & (data == maximum_filter(data, size=3))
    components, _ = label(peaks)
    # One representative per equal-height plateau, in O(pixels log pixels).
    coordinates = np.argwhere(peaks)
    _, first = np.unique(components[peaks], return_index=True)
    return coordinates[first]


def compare_rasters(candidate, reference, *, resolution=1.0, peak_tolerance=2.0):
    """Agreement metrics; reference agreement is not independent accuracy."""
    if candidate.shape != reference.shape:
        raise ValueError("Candidate and reference must share the same grid")
    valid, ref_valid = np.isfinite(candidate), np.isfinite(reference)
    joint = valid & ref_valid
    difference = candidate[joint].astype(float) - reference[joint]
    cp, rp = _peaks(candidate), _peaks(reference)
    # Greedy one-to-one peak matching avoids counting one reference twice.
    pairs = []
    if len(cp) and len(rp):
        for i, neighbours in enumerate(
            cKDTree(rp * resolution).query_ball_point(cp * resolution, peak_tolerance)
        ):
            pairs.extend(
                (float(np.linalg.norm(cp[i] - rp[j]) * resolution), i, j) for j in neighbours
            )
    used_c, used_r = set(), set()
    for distance, i, j in sorted(pairs):
        if i not in used_c and j not in used_r:
            used_c.add(i)
            used_r.add(j)
    return {
        "shared_pixels": int(joint.sum()),
        "coverage": float(valid.mean()),
        "reference_coverage": float(ref_valid.mean()),
        "coverage_disagreement": float((valid ^ ref_valid).mean()),
        "bias": float(difference.mean()) if len(difference) else None,
        "mae": float(np.abs(difference).mean()) if len(difference) else None,
        "rmse": float(np.sqrt(np.mean(difference**2))) if len(difference) else None,
        "absolute_error_p95": float(np.percentile(np.abs(difference), 95))
        if len(difference)
        else None,
        "candidate_peaks": len(cp),
        "reference_peaks": len(rp),
        "matched_peaks": len(used_c),
        "peak_precision": len(used_c) / len(cp) if len(cp) else None,
        "peak_recall": len(used_r) / len(rp) if len(rp) else None,
    }


def interval_coverage(reference, lower, upper):
    """Empirical coverage against independent, spatially matched observations."""
    reference, lower, upper = map(np.asarray, (reference, lower, upper))
    if reference.shape != lower.shape or reference.shape != upper.shape:
        raise ValueError("Reference and interval bounds must have matching shapes")
    valid = np.isfinite(reference) & np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[valid] > upper[valid]):
        raise ValueError("Lower bound exceeds upper bound")
    return {
        "n": int(valid.sum()),
        "coverage": float(
            np.mean((reference[valid] >= lower[valid]) & (reference[valid] <= upper[valid]))
        )
        if valid.any()
        else None,
        "mean_width": float(np.mean(upper[valid] - lower[valid])) if valid.any() else None,
    }


def _render(points, bbox, resolution, method, args):
    from alsdb.processing.chm import _pitfree_rasterise, _rasterise, _spikefree_rasterise

    if method == "spikefree":
        from alsdb.processing.spikefree import rasterize_spikefree

        return rasterize_spikefree(
            points,
            bbox,
            resolution,
            freeze_distance=args.freeze_distance,
            height_buffer=args.height_buffer,
            max_triangle_edge=args.max_edge,
        )
    if method == "lastools":
        from alsdb.processing.lastools import rasterize_lastools

        return rasterize_lastools(
            points,
            bbox,
            resolution,
            executable=args.lastools,
            freeze_distance=args.freeze_distance,
            height_buffer=args.height_buffer,
            freeze_interval=args.freeze_interval,
            max_triangle_edge=args.max_edge,
            demo=args.demo,
        )
    if method == "pitfree":
        return _pitfree_rasterise(points, bbox, resolution, max_distance=args.max_distance)
    if method == "highest_subcell_tin":
        return _spikefree_rasterise(
            points,
            bbox,
            resolution,
            subcell_resolution=resolution / 3,
            max_distance=args.max_distance,
        )
    inside = (
        (points["X"] >= bbox[0])
        & (points["X"] < bbox[2])
        & (points["Y"] > bbox[1])
        & (points["Y"] <= bbox[3])
    )
    points = points[inside]
    if not len(points):
        return np.full(GridSpec.from_bbox(bbox, resolution).shape, np.nan, np.float32)
    return _rasterise(
        points["X"],
        np.minimum(np.nextafter(points["Y"], -np.inf), bbox[3]),
        points["HeightAboveGround"],
        bbox,
        resolution,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Structured .npy with X/Y/Z/HeightAboveGround; eligible buffered points only",
    )
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="JSON report path; rasters saved beside it")
    parser.add_argument(
        "--methods", nargs="+", default=["max", "pitfree", "highest_subcell_tin", "spikefree"]
    )
    parser.add_argument("--reference", help="Optional independent .npy raster on the same grid")
    parser.add_argument("--lastools", help="Path to las2dem64; adds a LAStools reference run")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--freeze-distance", type=float, default=1.5)
    parser.add_argument("--height-buffer", type=float, default=0.5)
    parser.add_argument("--freeze-interval", type=float, default=0.25)
    parser.add_argument("--max-edge", type=float, default=100.0)
    parser.add_argument("--max-distance", type=float, default=3.0)
    parser.add_argument("--tile-size", type=float, default=50.0)
    parser.add_argument("--tile-buffer", type=float, default=10.0)
    parser.add_argument(
        "--worker",
        choices=["max", "pitfree", "highest_subcell_tin", "spikefree", "lastools"],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    grid = GridSpec.from_bbox(args.bbox, args.resolution)
    if args.worker:
        import resource

        points = np.load(args.input, allow_pickle=False)
        start = time.perf_counter()
        full = _render(points, grid.bbox, args.resolution, args.worker, args)
        seconds = time.perf_counter() - start
        tiled = np.full(grid.shape, np.nan, np.float32)
        for query, crop in grid.tiles(args.tile_size, args.tile_buffer):
            mask = (
                (points["X"] >= query[0])
                & (points["X"] <= query[2])
                & (points["Y"] >= query[1])
                & (points["Y"] <= query[3])
            )
            tile = _render(points[mask], crop, args.resolution, args.worker, args)
            row = round((grid.y1 - crop[3]) / args.resolution)
            col = round((crop[0] - grid.x0) / args.resolution)
            tiled[row : row + tile.shape[0], col : col + tile.shape[1]] = tile
        metrics = {
            "seconds": seconds,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "tiled_comparison": compare_rasters(tiled, full, resolution=args.resolution),
        }
        # Pit indicator uses only complete 3x3 neighbourhoods.
        from scipy.ndimage import minimum_filter

        complete = minimum_filter(np.isfinite(full).astype(int), size=3, mode="constant") > 0
        pits = complete & (median_filter(np.nan_to_num(full), size=3) - full > 1.0)
        metrics["pit_fraction_above_1m"] = (
            float(pits.sum() / complete.sum()) if complete.any() else None
        )
        np.save(str(args.output) + ".npy", full)
        Path(args.output).write_text(json.dumps(metrics, indent=2))
        return
    import hashlib

    report = {
        "parameters": vars(args),
        "software": software_versions(),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "methods": {},
    }
    if args.lastools:
        from alsdb.processing.lastools import executable_version

        report["lastools_version"] = executable_version(args.lastools)
    methods = list(dict.fromkeys(args.methods + (["lastools"] if args.lastools else [])))
    if not set(methods) <= {"max", "pitfree", "highest_subcell_tin", "spikefree", "lastools"}:
        parser.error("Unknown benchmark method")
    rasters = {}
    with tempfile.TemporaryDirectory(prefix="alsdb-benchmark-") as temporary:
        for method in methods:
            output = str(Path(temporary) / f"{method}.json")
            command = [
                sys.executable,
                "-m",
                __name__,
                *sys.argv[1:],
                "--worker",
                method,
                "--output",
                output,
            ]
            # __name__ is __main__ under -m, so use the importable module name.
            command[2] = "alsdb.processing.benchmark"
            subprocess.run(command, check=True)
            report["methods"][method] = json.loads(Path(output).read_text())
            rasters[method] = np.load(output + ".npy")
    reference = np.load(args.reference) if args.reference else rasters.get("lastools")
    if reference is not None:
        for method, raster in rasters.items():
            report["methods"][method]["reference_comparison"] = compare_rasters(
                raster, reference, resolution=args.resolution
            )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(str(args.output) + ".npz", **rasters)


if __name__ == "__main__":
    main()
