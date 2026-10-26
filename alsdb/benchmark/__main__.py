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

    parser = argparse.ArgumentParser(prog="python -m alsdb.benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("points", help="all alsdb methods on one prepared point set (see points --help)")

    p = sub.add_parser("lastools", help="LAStools reference CHM tiles from LAS/LAZ")
    p.add_argument("laz_dir")
    p.add_argument("out_dir")
    p.add_argument("--products", nargs="+", default=["spikefree", "max"])
    p.add_argument("--veg", type=_classes, default=(3, 4, 5), help="e.g. 3,4,5")
    p.add_argument("--freeze", type=_number_or_auto, default="auto")
    p.add_argument("--lastools-dir", help="folder with las2dem64/lasheight64/lascanopy64")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tiles", type=int, help="only the first N tiles (pilot)")

    p = sub.add_parser("lasr", help="lasR pitfree + spike-free reference tiles from LAS/LAZ")
    p.add_argument("laz_dir")
    p.add_argument("out_dir")
    p.add_argument("--veg", type=_classes, default=(3, 4, 5), help="e.g. 3,4,5 (DE-Hai: 20)")
    p.add_argument("--freeze", default="auto")
    p.add_argument("--edge", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tiles", type=int, help="only the first N unfinished tiles (pilot)")
    p.add_argument("--normalized", action="store_true", help="input Z is already height")

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
            freeze_distance=args.freeze, lastools_dir=args.lastools_dir,
            n_workers=args.workers, tiles=args.tiles,
        )  # fmt: skip
    elif args.command == "lasr":
        from alsdb.benchmark import lasr

        lasr.run(
            args.laz_dir, args.out_dir, veg_classes=args.veg, freeze_distance=args.freeze,
            max_edge=args.edge, n_workers=args.workers, tiles=args.tiles,
            normalized=args.normalized,
        )  # fmt: skip
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
