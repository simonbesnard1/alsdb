"""Command-line entry point for alsdb validation benchmarks."""

import argparse
import json
import sys


def _number_or_auto(value):
    return value if value == "auto" else float(value)


def _classes(value):
    return tuple(int(c) for c in value.split(","))


def _pairs(values):
    out = {}
    for item in values:
        name, sep, path = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {item!r}")
        out[name] = path
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "points":
        from alsdb.benchmark.points import main as points_main

        return points_main(argv[1:])
    if argv and argv[0] in ("timing", "timing-report"):
        from alsdb.benchmark.timing import main as timing_main

        return timing_main(argv[1:] if argv[0] == "timing" else ["report", *argv[1:]])

    parser = argparse.ArgumentParser(prog="python -m alsdb.benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("points", help="all alsdb methods on one prepared point set (see points --help)")
    sub.add_parser("timing", help="compute-performance benchmark (see timing --help)")
    sub.add_parser("timing-report", help="summarise a timing CSV: timing-report <timings.csv>")

    p = sub.add_parser("classification", help="synthetic ground-classification benchmark")
    p.add_argument("--terrain", nargs="+", default=["flat", "rolling", "steep", "rough"])
    p.add_argument("--density", type=float, nargs="+", default=[0.5, 2.0, 10.0])
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--size", type=float, default=100.0)
    p.add_argument("--methods", nargs="+", default=["csf", "pmf", "smrf"])
    p.add_argument("--boundary-width", type=float, default=5.0)
    p.add_argument("--output", help="write JSON results here")

    p = sub.add_parser(
        "classification-files", help="compare ground classifiers on classified LAS/LAZ files"
    )
    p.add_argument("out_dir")
    p.add_argument("--laz-dir", required=True, help="folder containing reference LAS/LAZ files")
    p.add_argument(
        "--tools",
        nargs="+",
        default=["csf", "pmf", "smrf"],
        choices=["csf", "pmf", "smrf", "ptd", "fastgc", "lastools"],
    )
    p.add_argument("--boundary-width", type=float, default=5.0)
    p.add_argument("--tiles", type=int, help="only process the first N files")
    p.add_argument("--lastools-dir", help="folder containing lasground64")
    p.add_argument("--fastgc-executable", default="fastgc")

    p = sub.add_parser("lastools", help="LAStools reference CHM tiles from LAS/LAZ")
    p.add_argument("laz_dir")
    p.add_argument("out_dir")
    p.add_argument("--products", nargs="*", default=["spikefree", "max"],
                   help="none given: only lasheight normalization")  # fmt: skip
    p.add_argument("--veg", type=_classes, default=(3, 4, 5), help="e.g. 3,4,5")
    p.add_argument("--freeze", type=_number_or_auto, default="auto")
    p.add_argument("--lastools-dir", help="folder with las2dem64/lasheight64/lascanopy64")
    p.add_argument("--hag-dir", help="reuse existing lasheight -replace_z outputs")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tiles", type=int, help="only the first N tiles (pilot)")

    p = sub.add_parser("lasr", help="lasR pitfree + spike-free reference tiles from LAS/LAZ")
    p.add_argument("laz_dir")
    p.add_argument("out_dir")
    p.add_argument("--veg", type=_classes, default=(3, 4, 5), help="e.g. 3,4,5 (DE-Hai: 20)")
    p.add_argument("--freeze", default="auto")
    p.add_argument("--edge", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tiles", type=int, help="only the first N tiles (pilot)")
    p.add_argument("--normalized", action="store_true", help="input Z is already height")

    p = sub.add_parser("synthetic", help="HELIOS++ scenes with exact truth; score all tools")
    p.add_argument("out_dir")
    p.add_argument("--terrain", nargs="+", default=["rolling"])
    p.add_argument("--forest", nargs="+", default=["conifer"])
    p.add_argument("--density", type=float, nargs="+", default=[2.0], help="pulses per m2")
    p.add_argument("--pad-scale", type=float, nargs="+", default=[1.0],
                   help="multiplies the forest type's plant area index")  # fmt: skip
    p.add_argument("--max-returns", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--tools", nargs="+", default=["alsdb", "lasr"],
                   choices=["alsdb", "lasr", "lastools"])  # fmt: skip
    p.add_argument("--lastools-dir")
    p.add_argument("--threads", type=int, default=8, help="HELIOS++ threads")
    p.add_argument("--keep", action="store_true", help="keep HELIOS++ output and tool products")

    p = sub.add_parser("uls", help="real ULS canopies at ALS density: thinned and HELIOS++")
    p.add_argument("out_dir")
    p.add_argument("--laz-dir", required=True, help="folder of ULS LAS/LAZ tiles")
    p.add_argument("--site", required=True)
    p.add_argument("--x0", type=float, required=True, help="core lower-left x (whole metres)")
    p.add_argument("--y0", type=float, required=True)
    p.add_argument("--core", type=float, default=100.0)
    p.add_argument("--margin", type=float, default=15.0)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--crs", help="e.g. EPSG:25833; default: from the LAZ headers")
    p.add_argument("--trees", help="GeoPackage of field trees with heights")
    p.add_argument("--tree-height-field", default="tls_treeheight")
    p.add_argument("--density", type=float, nargs="+", default=[0.5, 2.0, 10.0])
    p.add_argument("--variants", nargs="+", default=["thinned", "helios"],
                   choices=["thinned", "helios"])  # fmt: skip
    p.add_argument("--max-angle", type=float, default=20.0, help="near-nadir limit, degrees")
    p.add_argument("--tools", nargs="+", default=["alsdb", "lasr"],
                   choices=["alsdb", "lasr", "lastools"])  # fmt: skip
    p.add_argument("--lastools-dir")
    p.add_argument("--threads", type=int, default=8, help="HELIOS++ threads")
    p.add_argument("--keep", action="store_true", help="keep HELIOS++ output and tool products")

    p = sub.add_parser("sensitivity", help="spike-free freeze distances on finished scenes")
    p.add_argument("scenes_dir", help="folder of synthetic or uls scenes")
    p.add_argument("out_csv")
    p.add_argument("--workers", type=int, default=1, help="scenes processed in parallel")
    p.add_argument("--keep", action="store_true", help="keep the per-scene stores")

    p = sub.add_parser("compare", help="score alsdb stores against reference tile folders")
    p.add_argument("--store", nargs="+", required=True, metavar="NAME=PATH")
    p.add_argument("--ref", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="stop after N tiles (smoke test)")
    p.add_argument("--npz", help="save stats, residual samples and maps here")

    args = parser.parse_args(argv)
    if args.command == "classification":
        import itertools

        from alsdb.benchmark.classification import (
            SyntheticClassificationScenario,
            run_synthetic_grid,
        )

        scenarios = [
            SyntheticClassificationScenario(terrain=t, density=d, size=args.size, seed=s)
            for t, d, s in itertools.product(args.terrain, args.density, args.seed)
        ]
        rows = run_synthetic_grid(
            scenarios,
            methods=tuple(args.methods),
            boundary_width=args.boundary_width,
        )
        payload = json.dumps(rows, indent=2)
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(payload)
        else:
            print(payload)
    elif args.command == "classification-files":
        from pathlib import Path

        from alsdb.benchmark.classification import run_file_benchmark

        lasground = (
            str(Path(args.lastools_dir) / "lasground64") if args.lastools_dir else "lasground64"
        )
        run_file_benchmark(
            args.laz_dir,
            args.out_dir,
            methods=tuple(args.tools),
            boundary_width=args.boundary_width,
            tiles=args.tiles,
            lastools_executable=lasground,
            fastgc_executable=args.fastgc_executable,
        )
    elif args.command == "lastools":
        from alsdb.benchmark import lastools

        lastools.run(
            args.laz_dir, args.out_dir, products=args.products, veg_classes=args.veg,
            freeze_distance=args.freeze, lastools_dir=args.lastools_dir, hag_dir=args.hag_dir,
            n_workers=args.workers, tiles=args.tiles,
        )  # fmt: skip
    elif args.command == "lasr":
        from alsdb.benchmark import lasr

        lasr.run(
            args.laz_dir, args.out_dir, veg_classes=args.veg, freeze_distance=args.freeze,
            max_edge=args.edge, n_workers=args.workers, tiles=args.tiles,
            normalized=args.normalized,
        )  # fmt: skip
    elif args.command == "synthetic":
        import itertools

        from alsdb.benchmark.synthetic import Scenario, run_grid

        scenarios = [
            Scenario(
                terrain=t, forest=f, density=d, pad_scale=p, max_returns=args.max_returns, seed=s
            )
            for t, f, d, p, s in itertools.product(
                args.terrain, args.forest, args.density, args.pad_scale, args.seeds
            )
        ]
        print(run_grid(args.out_dir, scenarios, tools=args.tools, lastools_dir=args.lastools_dir,
                       threads=args.threads, keep=args.keep))  # fmt: skip
    elif args.command == "uls":
        from alsdb.benchmark.uls import Area, run_area

        area = Area(
            site=args.site, laz_dir=args.laz_dir, x0=args.x0, y0=args.y0, core=args.core,
            margin=args.margin, year=args.year, crs=args.crs, trees=args.trees,
            tree_height_field=args.tree_height_field,
        )  # fmt: skip
        print(run_area(args.out_dir, area, densities=args.density, variants=args.variants,
                       tools=args.tools, lastools_dir=args.lastools_dir,
                       max_angle=args.max_angle, threads=args.threads, keep=args.keep))  # fmt: skip
    elif args.command == "sensitivity":
        from alsdb.benchmark.sensitivity import run

        print(run(args.scenes_dir, args.out_csv, workers=args.workers, keep=args.keep))
    elif args.command == "compare":
        from alsdb.benchmark.compare import compare

        result = compare(
            _pairs(args.store), _pairs(args.ref), year=args.year, resolution=args.resolution,
            limit=args.limit, progress=True,
        )  # fmt: skip
        print(result.summary())
        if args.npz:
            result.to_npz(args.npz)


if __name__ == "__main__":
    main()
