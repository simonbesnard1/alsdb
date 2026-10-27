"""``python -m alsdb.benchmark {lastools,lasr,compare,points} ...``"""

import argparse
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

    p = sub.add_parser("compare", help="score alsdb stores against reference tile folders")
    p.add_argument("--store", nargs="+", required=True, metavar="NAME=PATH")
    p.add_argument("--ref", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="stop after N tiles (smoke test)")
    p.add_argument("--npz", help="save stats, residual samples and maps here")

    args = parser.parse_args(argv)
    if args.command == "lastools":
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
