"""Command-line entry point for alsdb validation benchmarks."""

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


def _add_ptd_arguments(p):
    """PTD (Axelsson 2000) settings; several values run every combination.

    Defaults: alsdb's (tilt ensemble); ``--ptd-tilt-runs 1 --ptd-votes 1
    --ptd-angle 30 --ptd-distance 2`` is lasR's single run."""
    from alsdb.methods.ptd import DEFAULTS as d

    for flag, key, kind, unit in (
        ("--ptd-res", "seed_resolution", float, "seed cell (m)"),
        ("--ptd-angle", "angle", float, "degrees"),
        ("--ptd-distance", "distance", float, "m"),
        ("--ptd-spacing", "spacing", float, "m"),
        ("--ptd-tilt-runs", "tilt_runs", int, "runs of the tilt ensemble (1: plain PTD)"),
        ("--ptd-tilt", "tilt", float, "ensemble tilt (degrees)"),
        ("--ptd-votes", "votes", int, "runs that must call a point ground"),
    ):
        p.add_argument(flag, type=kind, nargs="+", default=[d[key]], help=unit)
    p.add_argument("--ptd-lasr", action="store_true",
                   help="also run lasR's single PTD (angle 30, distance 2) as a baseline")  # fmt: skip


def _ptd_sets(args):
    import itertools

    from alsdb.benchmark.classification import PTDParameters
    from alsdb.methods.ptd import LASR_DEFAULTS

    sets = [
        PTDParameters(*values)
        for values in itertools.product(
            args.ptd_res, args.ptd_angle, args.ptd_distance, args.ptd_spacing,
            args.ptd_tilt_runs, args.ptd_tilt, args.ptd_votes,
        )
    ]  # fmt: skip
    lasr = PTDParameters(**LASR_DEFAULTS, tilt_runs=1, votes=1)
    if args.ptd_lasr and lasr not in sets:
        sets.append(lasr)  # after the defaults: the first set is each method's default run
    return tuple(sets)


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

    p = sub.add_parser(
        "classification-files", help="compare ground classifiers on classified LAS/LAZ files"
    )
    p.add_argument("out_dir")
    p.add_argument("--laz-dir", required=True, help="folder containing reference LAS/LAZ files")
    p.add_argument(
        "--tools",
        nargs="+",
        default=["csf", "pmf", "smrf"],
        choices=["csf", "pmf", "smrf", "ptd", "lastools"],
    )
    p.add_argument("--boundary-width", type=float, default=5.0)
    p.add_argument("--tiles", type=int, help="only process the first N files")
    p.add_argument("--reference-dir", help="full-density originals (same file names) for terrain")
    p.add_argument("--lastools-dir", help="folder containing lasground64")
    _add_ptd_arguments(p)

    p = sub.add_parser(
        "classification-scenes",
        help="classify synthetic / ULS / decimated benchmark scenes; DTM against their truth",
    )
    p.add_argument("out_dir")
    p.add_argument("--scenes-dir", nargs="+", required=True, help="folders of benchmark scenes")
    p.add_argument(
        "--tools",
        nargs="+",
        default=["csf", "pmf", "smrf", "ptd"],
        choices=["csf", "pmf", "smrf", "ptd", "lastools"],
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--lastools-dir", help="folder containing lasground64")
    _add_ptd_arguments(p)

    p = sub.add_parser(
        "classification-references",
        help="full-density DTM of every classifier for ULS / decimated scene areas",
    )
    p.add_argument("--scenes-dir", nargs="+", required=True)
    p.add_argument("--source", nargs="+", required=True, metavar="SITE=DIR",
                   help="full-density survey per site, e.g. PetitBuech=<site>/original")  # fmt: skip
    p.add_argument(
        "--tools",
        nargs="+",
        default=["csf", "pmf", "smrf", "ptd"],
        choices=["csf", "pmf", "smrf", "ptd", "lastools"],
    )
    p.add_argument("--tolerance", type=float, default=0.25, help="agreement tolerance (m)")
    p.add_argument("--max-density", type=float, help="thin denser surveys to this (points/m2)")
    p.add_argument("--out-csv", help="agreement table per area and stratum")
    _add_ptd_arguments(p)

    p = sub.add_parser(
        "classification-chm",
        help="alsdb spike-free CHM on each classifier's ground (after classification-scenes)",
    )
    p.add_argument("out_dir", help="the classification-scenes output folder")
    p.add_argument("--scenes-dir", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=1)

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
    p.add_argument("--understorey", nargs="+", default=["no"], choices=["no", "yes"],
                   help="low vegetation 0.5-3 m under the forest")  # fmt: skip
    p.add_argument("--placement", nargs="+", default=["regular"], choices=["regular", "clumped"])
    p.add_argument(
        "--beam-divergence",
        type=float,
        nargs="+",
        default=[0.22],
        help="mrad (footprint at 1000 m: divergence in m)",
    )
    p.add_argument("--voxel", type=float, nargs="+", default=[0.5],
                   help="crown voxel size (m): 0.5 or 0.25")  # fmt: skip
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
    p.add_argument("--classes", type=int, nargs="+",
                   help="read only these classes (e.g. 2 3 4 5 9 for national ALS); default: all")  # fmt: skip
    p.add_argument(
        "--candidate-fraction",
        type=float,
        default=0.05,
        help="share of pulses kept for thinning: 0.05 for ULS, 1 for aircraft ALS",
    )
    p.add_argument("--centre", type=float, default=0.2,
                   help="cell-centre reference window (m): 0.2 for ULS, 0.4 for ALS density")  # fmt: skip
    p.add_argument("--tools", nargs="+", default=["alsdb", "lasr"],
                   choices=["alsdb", "lasr", "lastools"])  # fmt: skip
    p.add_argument("--lastools-dir")
    p.add_argument("--threads", type=int, default=8, help="HELIOS++ threads")
    p.add_argument("--keep", action="store_true", help="keep HELIOS++ output and tool products")

    p = sub.add_parser(
        "decimated", help="decimated copies of a dense ALS survey vs its full density"
    )
    p.add_argument("out_dir")
    p.add_argument("--original", required=True, help="folder of the dense LAZ tiles")
    p.add_argument("--versions", nargs="*", metavar="NAME=DIR",
                   help="decimated copies; default: the pd_* folders beside --original")  # fmt: skip
    p.add_argument("--site", required=True)
    p.add_argument("--x0", type=float, required=True, help="core lower-left x (whole metres)")
    p.add_argument("--y0", type=float, required=True)
    p.add_argument("--core", type=float, default=200.0)
    p.add_argument("--margin", type=float, default=15.0)
    p.add_argument("--year", type=int, default=2020, help="label only")
    p.add_argument("--crs", help="default: from the LAZ headers")
    p.add_argument("--tools", nargs="*", default=["alsdb", "lasr"],
                   choices=["alsdb", "lasr", "lastools"], help="none given: only build the scenes")  # fmt: skip
    p.add_argument("--lastools-dir")
    p.add_argument("--keep", action="store_true", help="keep tool products")
    p.add_argument(
        "--mixed",
        nargs="*",
        default=[],
        metavar="SPARSE:DENSE",
        help="uneven-density scenes: SPARSE copy with DENSE in strips, e.g. pd_01:pd_02",
    )
    p.add_argument("--strips", type=float, nargs=3, default=[100.0, 30.0, 35.0],
                   metavar=("PERIOD", "WIDTH", "OFFSET"), help="dense strips (m)")  # fmt: skip

    p = sub.add_parser("lidarhd-frame", help="tree cover and slope of every 2 km block of France")
    p.add_argument("out_csv")
    p = sub.add_parser("lidarhd-index", help="every Lidar HD tile: classification method, dates")
    p.add_argument("out_csv")
    p = sub.add_parser("lidarhd-select", help="stratified random draw of Lidar HD forest blocks")
    p.add_argument("frame_csv")
    p.add_argument("out_csv")
    p.add_argument("--per-stratum", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-distance", type=float, default=50_000.0, help="between blocks (m)")
    p.add_argument("--cores", type=int, default=4, help="scored 200 m cores per block")
    p.add_argument("--index", help="tile index (lidarhd-index): only fully published blocks")
    p.add_argument("--any-classification", action="store_true",
                   help="also blocks with IGN's automatic classification (default: hand-corrected)")  # fmt: skip
    p = sub.add_parser("lidarhd-download", help="download the tiles of the selected blocks")
    p.add_argument("selection_csv")
    p.add_argument("data_dir")

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
    if args.command == "classification-files":
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
            reference_dir=args.reference_dir,
            lastools_executable=lasground,
            ptd=_ptd_sets(args),
        )
    elif args.command == "classification-scenes":
        from pathlib import Path

        from alsdb.benchmark.classification import run_scene_benchmark

        run_scene_benchmark(
            args.scenes_dir,
            args.out_dir,
            methods=tuple(args.tools),
            workers=args.workers,
            lastools_executable=(
                str(Path(args.lastools_dir) / "lasground64") if args.lastools_dir else "lasground64"
            ),
            ptd=_ptd_sets(args),
        )
    elif args.command == "classification-references":
        from alsdb.benchmark.classification import build_dtm_references

        build_dtm_references(
            args.scenes_dir,
            dict(item.split("=", 1) for item in args.source),
            methods=tuple(args.tools),
            tolerance=args.tolerance,
            out_csv=args.out_csv,
            max_density=args.max_density,
            ptd=_ptd_sets(args),
        )
    elif args.command == "classification-chm":
        from alsdb.benchmark.classification import run_scene_chm

        run_scene_chm(args.scenes_dir, args.out_dir, workers=args.workers)
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
                terrain=t, forest=f, density=d, pad_scale=p, max_returns=args.max_returns, seed=s,
                understorey=u == "yes", placement=pl, beam_divergence=bd, voxel=v,
            )
            for t, f, d, p, s, u, pl, bd, v in itertools.product(
                args.terrain, args.forest, args.density, args.pad_scale, args.seeds,
                args.understorey, args.placement, args.beam_divergence, args.voxel,
            )
        ]  # fmt: skip
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
                       max_angle=args.max_angle, threads=args.threads, keep=args.keep,
                       keep_classes=tuple(args.classes) if args.classes else None,
                       candidate_fraction=args.candidate_fraction, centre=args.centre))  # fmt: skip
    elif args.command == "decimated":
        from alsdb.benchmark import decimated
        from alsdb.benchmark.uls import Area

        area = Area(site=args.site, laz_dir=args.original, x0=args.x0, y0=args.y0,
                    core=args.core, margin=args.margin, year=args.year, crs=args.crs)  # fmt: skip
        versions = _pairs(args.versions) if args.versions else None
        mix = {"mixed": tuple(args.mixed), "strips": tuple(args.strips)}
        if args.tools:
            print(decimated.run_area(args.out_dir, area, versions, tools=args.tools,
                                     lastools_dir=args.lastools_dir, keep=args.keep, **mix))  # fmt: skip
        else:
            for scene in decimated.build_area(args.out_dir, area, versions, **mix):
                print(scene)
    elif args.command == "lidarhd-frame":
        from alsdb.benchmark.lidarhd import build_frame

        print(build_frame(args.out_csv))
    elif args.command == "lidarhd-index":
        from alsdb.benchmark.lidarhd import tile_index

        print(tile_index(args.out_csv))
    elif args.command == "lidarhd-select":
        from alsdb.benchmark.lidarhd import add_cores, draw

        draw(args.frame_csv, args.out_csv, per_stratum=args.per_stratum, seed=args.seed,
             min_distance=args.min_distance, index_csv=args.index,
             hand_corrected=not args.any_classification)  # fmt: skip
        add_cores(args.out_csv, n_cores=args.cores, seed=args.seed)
        print(args.out_csv)
    elif args.command == "lidarhd-download":
        from alsdb.benchmark.lidarhd import download

        print(download(args.selection_csv, args.data_dir))
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
