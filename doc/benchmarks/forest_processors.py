"""Reproducible processor timings on a bounded real LAZ sample.

Run: pixi run python doc/benchmarks/forest_processors.py INPUT.laz OUTPUT.json
Add --database to measure actual TileDB footprint batching and fused Zarr writes.
The first 500,000 file records are a bounded sample, not a representative survey.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pdal
from numpy.lib import recfunctions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alsdb.processing._terrain import TerrainModel, normalize_points
from alsdb.processing._tiling import _filter_ground_outliers
from alsdb.processing.biomass import _extract_metrics
from alsdb.processing.gap import _compute_gap_grid


def timed(fn, repeats=3):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return result, {"seconds": samples, "median_seconds": float(np.median(samples))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=500000)
    parser.add_argument("--database", action="store_true")
    args = parser.parse_args()
    pipeline = pdal.Pipeline(
        json.dumps([{"type": "readers.las", "filename": str(args.input), "count": args.count}])
    )
    start = time.perf_counter()
    pipeline.execute()
    raw = pipeline.arrays[0]
    with args.input.open("rb") as source:
        input_hash = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "input_sha256": input_hash,
        "platform": platform.platform(),
        "python": sys.version,
        "input": str(args.input),
        "sample": "first records in file order",
        "points": len(raw),
        "read_seconds": time.perf_counter() - start,
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    cleaned, _ = timed(lambda: _filter_ground_outliers(raw), 1)
    normalized, report["normalization_single_allocation"] = timed(
        lambda: normalize_points(cleaned)[0]
    )
    names = ("HeightAboveGround", "GroundDistance", "GroundExtrapolated", "NegativeHAG")

    def legacy_normalization():
        # Identical interpolation; reconstruct the previous four-copy append path.
        terrain = TerrainModel(cleaned[cleaned["Classification"] == 2])
        ground, distance, extrapolated = terrain.evaluate(
            np.column_stack((cleaned["X"], cleaned["Y"]))
        )
        heights = cleaned["Z"] - ground
        values = (
            np.maximum(heights, 0),
            distance,
            extrapolated.astype(np.uint8),
            (np.isfinite(heights) & (heights < 0)).astype(np.uint8),
        )
        output = cleaned
        for name, value in zip(names, values):
            output = recfunctions.append_fields(output, name, value, usemask=False)
        return output

    legacy, report["normalization_four_appends"] = timed(legacy_normalization)
    for name in legacy.dtype.names:
        np.testing.assert_equal(legacy[name], normalized[name])
    center = np.array([np.median(raw["X"]), np.median(raw["Y"])])
    bbox = (*(center - 50), *(center + 50))
    # Reorder xy-low, xy-high gives xmin,ymin,xmax,ymax.
    inside = (
        (normalized["X"] >= bbox[0])
        & (normalized["X"] < bbox[2])
        & (normalized["Y"] > bbox[1])
        & (normalized["Y"] <= bbox[3])
    )
    crop = normalized[inside]
    report["bbox"] = list(bbox)
    report["owned_points"] = len(crop)
    full, report["metrics_all"] = timed(lambda: _extract_metrics(normalized, 10, bbox))
    selected, report["metrics_h95_cc"] = timed(
        lambda: _extract_metrics(normalized, 10, bbox, metrics=("h95", "cc"))
    )
    for name in selected:
        np.testing.assert_allclose(selected[name], full[name], equal_nan=True)
    _, report["gap"] = timed(lambda: _compute_gap_grid(normalized, 10, bbox))
    # Load only the historical pure metric functions, for the same-input baseline.
    import types

    old = types.ModuleType("legacy_biomass_benchmark")
    exec(  # noqa: S102 - explicitly benchmark trusted repository HEAD
        subprocess.check_output(["git", "show", "HEAD:alsdb/processing/biomass.py"], text=True),
        old.__dict__,
    )
    _, report["legacy_metrics_buffered"] = timed(lambda: old._extract_metrics(normalized, 10, bbox))
    old_crop, report["legacy_metrics_owned"] = timed(lambda: old._extract_metrics(crop, 10, bbox))
    # LAS XY is quantized: exact internal grid boundaries are common. The new
    # north-up half-open ownership rule intentionally changes their old bins.
    report["legacy_owned_max_differences"] = {
        name: float(np.nanmax(np.abs(full[name] - old_crop[name]))) for name in full
    }
    oracle = _extract_metrics(crop, 10, bbox)
    for name in full:
        np.testing.assert_equal(full[name], oracle[name])
    if args.database:
        database_bench(raw, center, bbox, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def database_bench(raw, center, bbox, report):
    from alsdb.core.alsdatabase import ALSDatabase
    from alsdb.core.alsprovider import ALSProvider
    from alsdb.processing._tiling import query_to_array
    from alsdb.processing.biomass import compute_metrics
    from alsdb.processing.chm import compute_chm
    from alsdb.processing.forest import compute_forest_products
    from alsdb.processing.gap import compute_gap_fraction
    from alsdb.processing.pavd import compute_als_pavd_batch, compute_als_pavd_profile
    from alsdb.processing.waveform import simulate_batch, simulate_waveform
    from alsdb.storage.zarr_store import ALSZarrStore
    from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig

    with tempfile.TemporaryDirectory(prefix="alsdb-benchmark-") as tmp:
        uri = str(Path(tmp) / "points")
        config = TileDBSchemaConfig(
            domain_min_x=float(raw["X"].min() - 100),
            domain_max_x=float(raw["X"].max() + 100),
            domain_min_y=float(raw["Y"].min() - 100),
            domain_max_y=float(raw["Y"].max() + 100),
        )
        db = ALSDatabase(storage_type="local", uri=uri, schema_cfg=config)
        attrs = {
            name: raw[name].astype(dtype)
            if name in raw.dtype.names
            else np.zeros(len(raw), dtype=dtype)
            for name, dtype in LAS_ATTRIBUTES.items()
        }
        db.write(raw["X"], raw["Y"], 2014, attrs, crs="EPSG:31983")
        provider = ALSProvider(storage_type="local", uri=uri)
        _, report["query_all_attributes"] = timed(lambda: query_to_array(provider, bbox, year=2014))
        _, report["query_selected_attributes"] = timed(
            lambda: query_to_array(
                provider,
                bbox,
                year=2014,
                attributes=("Z", "Classification", "ReturnNumber", "Withheld"),
            )
        )
        rng = np.random.default_rng(47)
        centers = center + rng.uniform(-20, 20, (30, 2))
        shots = pd.DataFrame(centers, columns=["center_x", "center_y"])
        singles, report["waveforms_30_single_queries"] = timed(
            lambda: [simulate_waveform(provider, *xy, year=2014) for xy in centers], 1
        )
        batch, report["waveforms_30_batched"] = timed(
            lambda: simulate_batch(provider, shots, year=2014, batch_tile_size=100), 1
        )
        for i, result in enumerate(singles):
            if result is not None:
                np.testing.assert_allclose(batch.iloc[i].home, result.home, rtol=0, atol=1e-10)
        singles, report["pavd_30_single_queries"] = timed(
            lambda: [compute_als_pavd_profile(provider, *xy, 12.5, year=2014) for xy in centers], 1
        )
        batch, report["pavd_30_batched"] = timed(
            lambda: compute_als_pavd_batch(provider, shots, year=2014), 1
        )
        for one, two in zip(singles, batch):
            if one is not None:
                np.testing.assert_allclose(one.pavd, two.pavd, rtol=0, atol=1e-10)
        from alsdb.processing.trees import segment_trees

        tree_bbox = (*(center - 7.5), *(center + 7.5))
        (tree_points, trees), report["trees_15m_with_10m_buffer_sample_0_5m"] = timed(
            lambda: segment_trees(
                provider, bbox=tree_bbox, year=2014, tile_buffer=10, voxel_size=0.5
            ),
            1,
        )
        assert len(trees) > 0, "Real sample unexpectedly produced no owned trees"
        report["tree_count"] = len(trees)
        report["segmented_tree_points"] = len(tree_points)
        options = {
            "resolution": 10.0,
            "bbox": bbox,
            "year": 2014,
            "tile_size": 100.0,
            "ground_outlier_removal": False,
        }
        separate = ALSZarrStore(str(Path(tmp) / "separate.zarr"))

        def run_separate():
            compute_metrics(provider, separate, metrics=("h95", "cc"), **options)
            compute_gap_fraction(provider, separate, lai=True, **options)
            compute_chm(provider, separate, **options)

        _, report["rasters_separate"] = timed(run_separate, 1)
        fused = ALSZarrStore(str(Path(tmp) / "fused.zarr"))
        _, report["rasters_fused"] = timed(
            lambda: compute_forest_products(
                provider, fused, metrics=("h95", "cc"), gap=True, lai=True, chm=True, **options
            ),
            1,
        )
        for name in ("chm", "h95", "cc", "gap", "lai"):
            np.testing.assert_allclose(
                separate.to_dataset(10)[name], fused.to_dataset(10)[name], equal_nan=True
            )


if __name__ == "__main__":
    main()
