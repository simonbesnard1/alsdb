"""Compute-performance benchmark: alsdb vs lasR vs LAStools on identical work.

Every tool turns the same LAS/LAZ tiles into the same 1 m CHMs with the same
settings (surface algorithm 3, ``"auto"`` freeze/edge). One-off preparation
and per-run work are timed separately so the result answers both questions:

- setup (one-off): alsdb ingest into TileDB; LAStools ``lasheight``; lasR none
- run (per product run): alsdb ``compute_chm`` pitfree + spikefree; lasR
  pitfree + spikefree; LAStools spikefree + max on the existing heights

Factors: area (nested, compact n x n tile blocks), cores (each step pinned to
that many CPUs with ``taskset``), storage (input on its original, usually
network, filesystem vs copied to local disk) and repeats. Every step runs in
its own process; wall time, user/system CPU (``wait4`` rusage, children
included) and peak resident memory of the whole process tree (sampled from
``/proc``) go to one CSV row. The harness is a child subreaper, so processes
a step detaches (lasR's PSOCK workers are started in the background by a
shell) stay in the tree and are counted, and the step ends when they exit.
Linux only.

    python -m alsdb.benchmark timing <laz_dir> <work_dir> --dry-run
    python -m alsdb.benchmark timing <laz_dir> <work_dir> --areas 1 4 16 --cores 1 4 8
    python -m alsdb.benchmark timing-report <work_dir>/timings.csv

Steps run through ``pixi run -e <env>`` in the alsdb checkout (``--repo``):
alsdb in ``spikefree`` (native spike-free), lasR in ``lasr``, LAStools in
``default``. ``--launcher direct`` uses the current Python and ``Rscript`` on
PATH instead (tests). The CSV is appended as steps finish; rerunning skips
completed steps, so an interrupted benchmark resumes.
"""

import argparse
import csv
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOLS = ("alsdb", "lasr", "lastools")
FIELDS = [
    "host", "storage", "n_tiles", "area_km2", "cores", "repeat", "tool", "stage", "step",
    "wall_s", "user_s", "sys_s", "peak_rss_gb", "exit_code", "started",
]  # fmt: skip
ENV = {"alsdb": "spikefree", "lasr": "lasr", "lastools": "default"}


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


def las_bounds(path):
    """(min_x, min_y, max_x, max_y) from a LAS/LAZ public header (uncompressed in LAZ too)."""
    with open(path, "rb") as handle:
        head = handle.read(227)
    if head[:4] != b"LASF":
        raise ValueError(f"{path} is not a LAS/LAZ file")
    max_x, min_x, max_y, min_y = struct.unpack_from("<4d", head, 179)
    return min_x, min_y, max_x, max_y


def select_blocks(files, levels):
    """Nested compact square blocks of tiles, one per level in *levels* (tile counts).

    Levels must be square numbers. The largest block is the fully covered
    k x k block closest to the centre of the survey; smaller blocks are
    centred inside it, so every level is a subset of the next.
    Returns ``({n_tiles: [paths]}, tile_size_m)``.
    """
    sides = {}
    for n in levels:
        s = math.isqrt(n)
        if s * s != n:
            raise ValueError(f"area levels must be square tile counts (1, 4, 9, 16, ...), got {n}")
        sides[n] = s
    bounds = {Path(f): las_bounds(f) for f in files}
    widths = sorted(b[2] - b[0] for b in bounds.values())
    size = round(widths[len(widths) // 2])  # nominal tile size, e.g. 2000 m
    grid = {}
    for path, (x0, y0, _, _) in bounds.items():
        grid[(round(x0 / size), round(y0 / size))] = path
    k = max(sides.values())
    cols = sorted(i for i, _ in grid)
    rows = sorted(j for _, j in grid)
    centre = (cols[len(cols) // 2], rows[len(rows) // 2])
    anchors = [
        (i, j) for i, j in grid if all((i + a, j + b) in grid for a in range(k) for b in range(k))
    ]
    if not anchors:
        raise ValueError(f"no fully covered {k}x{k} block of tiles in the input")
    ai, aj = min(
        anchors, key=lambda a: (a[0] + k / 2 - centre[0]) ** 2 + (a[1] + k / 2 - centre[1]) ** 2
    )
    blocks = {}
    for n, s in sorted(sides.items()):
        off = (k - s) // 2
        blocks[n] = [grid[(ai + off + a, aj + off + b)] for b in range(s) for a in range(s)]
    return blocks, size


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _children_map():
    parents = {}
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                with open(f"/proc/{entry.name}/stat") as handle:
                    stat = handle.read()
                parents[int(entry.name)] = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
    return parents


def _descendants(root):
    parents = _children_map()
    tree, frontier = set(), [root]
    while frontier:
        pid = frontier.pop()
        for child, parent in parents.items():
            if parent == pid and child not in tree:
                tree.add(child)
                frontier.append(child)
    return tree


def _tree_rss(root):
    """Resident memory (bytes) summed over all descendants of *root* (not *root* itself)."""
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for pid in _descendants(root):
        try:
            with open(f"/proc/{pid}/statm") as handle:
                total += int(handle.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total


def _become_subreaper():
    """Re-parent orphaned descendants to this process instead of init (Linux prctl)."""
    import ctypes

    PR_SET_CHILD_SUBREAPER = 36
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")


def _reap_orphans(timeout):
    """Wait for re-parented descendants; return their summed user and system CPU seconds.

    Orphans still running after *timeout* seconds are killed.
    """
    user = system = 0.0
    deadline = time.monotonic() + timeout
    while True:
        try:
            pid, _, usage = os.wait4(-1, os.WNOHANG)
        except ChildProcessError:
            return user, system
        if pid:
            user += usage.ru_utime
            system += usage.ru_stime
            continue
        if time.monotonic() > deadline:
            for left in _descendants(os.getpid()):
                try:
                    os.kill(left, 9)
                except ProcessLookupError:
                    pass
            deadline = math.inf
        time.sleep(0.05)


def measure(command, *, cpus=None, env=None, log=None, interval=0.5, orphan_timeout=60.0):
    """Run *command*; return wall, user, sys seconds, peak tree RSS (GB), exit code.

    Processes the command leaves behind (detached workers) are counted and
    waited for; still running after *orphan_timeout* s, they are killed.
    """
    _become_subreaper()
    if cpus:
        command = ["taskset", "-c", ",".join(map(str, cpus)), *command]
    with open(log if log else os.devnull, "w") as out:
        started = time.perf_counter()
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env)
        peak, done = [0], threading.Event()

        def sample():  # the whole tree below this process: the step plus re-parented orphans
            while not done.is_set():
                peak[0] = max(peak[0], _tree_rss(os.getpid()))
                done.wait(interval)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        _, status, usage = os.wait4(proc.pid, 0)
        orphan_user, orphan_sys = _reap_orphans(orphan_timeout)
        wall = time.perf_counter() - started
        done.set()
        sampler.join()
        proc.returncode = os.waitstatus_to_exitcode(status)
    return {
        "wall_s": round(wall, 3),
        "user_s": round(usage.ru_utime + orphan_user, 3),
        "sys_s": round(usage.ru_stime + orphan_sys, 3),
        "peak_rss_gb": round(peak[0] / 1e9, 3),
        "exit_code": proc.returncode,
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _python(tool, args):
    if args.launcher == "direct":
        return [sys.executable]
    return ["pixi", "run", "--manifest-path", str(args.repo), "-e", ENV[tool], "python"]


def steps_for(tool, subset, out, cores, args):
    """[(stage, step, command)] for one tool on one tile subset."""
    py = _python(tool, args)
    if tool == "alsdb":
        tiledb, worker = out / "tiledb", [*py, "-m", "alsdb.benchmark.timing", "worker"]
        return [
            ("setup", "ingest", [*worker, "ingest", str(subset), str(tiledb), "--crs", args.crs,
                                 "--workers", str(cores)]),
            *[("run", method, [*worker, "chm", str(tiledb), str(out / f"{method}.zarr"),
                               "--method", method, "--year", str(args.year),
                               "--workers", str(cores)])
              for method in ("pitfree", "spikefree")],
        ]  # fmt: skip
    if tool == "lasr":
        command = [*py, "-m", "alsdb.benchmark", "lasr", str(subset), str(out),
                   "--workers", str(cores), "--veg", args.veg]  # fmt: skip
        return [("run", "pitfree+spikefree", command)]
    lastools = [*py, "-m", "alsdb.benchmark", "lastools", str(subset), str(out),
                "--workers", str(cores), "--veg", args.veg]  # fmt: skip
    if args.lastools_dir:
        lastools += ["--lastools-dir", str(args.lastools_dir)]
    return [
        ("setup", "lasheight", [*lastools, "--products"]),
        ("run", "spikefree+max", [*lastools, "--products", "spikefree", "max",
                                  "--hag-dir", str(out / "hag")]),
    ]  # fmt: skip


def _worker(argv):
    """In-process alsdb steps, run inside the timed subprocess."""
    ap = argparse.ArgumentParser(prog="timing worker")
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("laz_dir")
    p.add_argument("tiledb")
    p.add_argument("--crs", required=True)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("chm")
    p.add_argument("tiledb")
    p.add_argument("zarr")
    p.add_argument("--method", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    from alsdb.core.alsdatabase import ALSDatabase

    db = ALSDatabase(storage_type="local", uri=args.tiledb)
    if args.what == "ingest":
        files = sorted(
            p for p in Path(args.laz_dir).iterdir() if p.suffix.lower() in (".las", ".laz")
        )
        db.create(crs=args.crs)
        db.ingest_many(files, n_workers=args.workers)
        return
    from alsdb.processing.chm import compute_chm
    from alsdb.storage.zarr_store import ALSZarrStore

    # Same settings as the production Barcelona runs (16_reprocess_barcelona_v3.py).
    compute_chm(
        db, ALSZarrStore(args.zarr), resolution=1.0, year=args.year, method=args.method,
        tile_size=1000.0, n_workers=args.workers, first_returns_only=True, pit_fill=False,
        quality=False,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _done_keys(csv_path):
    if not csv_path.exists():
        return set()
    with csv_path.open() as handle:
        return {
            (r["storage"], r["n_tiles"], r["cores"], r["repeat"], r["tool"], r["step"])
            for r in csv.DictReader(handle)
            if r["exit_code"] == "0"
        }


def _append(csv_path, row):
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _subsets(blocks, storage, args):
    """{n_tiles: folder} of symlinks; 'local' first copies the largest block to local disk."""
    root = (args.local_dir if storage == "local" else args.work_dir) / "subsets" / storage
    sources = {}
    largest = blocks[max(blocks)]
    if storage == "local":
        tiles = args.local_dir / "tiles"
        tiles.mkdir(parents=True, exist_ok=True)
        for path in largest:
            target = tiles / path.name
            if not target.exists() or target.stat().st_size != path.stat().st_size:
                shutil.copy2(path, target)
            sources[path] = target
    else:
        sources = {p: p for p in largest}
    folders = {}
    for n, paths in blocks.items():
        folder = root / f"{n}tiles"
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            link = folder / path.name
            if not link.exists():
                link.symlink_to(sources[path])
        folders[n] = folder
    return folders


def run(args):
    files = sorted(p for p in Path(args.laz_dir).iterdir() if p.suffix.lower() in (".las", ".laz"))
    blocks, size = select_blocks(files, args.areas)
    cpus_available = sorted(os.sched_getaffinity(0))
    cores = [c for c in args.cores if c <= len(cpus_available)]
    if len(cores) < len(args.cores):
        print(f"only {len(cpus_available)} CPUs available; skipping cores > that", flush=True)
    plan = [
        (storage, n, c, r, tool)
        for storage in args.storage
        for n in sorted(blocks)
        for c in cores
        for r in range(1, args.repeats + 1)
        for tool in args.tools
    ]
    print(f"tile size {size} m; blocks: " + ", ".join(
        f"{n} tiles ({n * size * size / 1e6:g} km2)" for n in sorted(blocks)))  # fmt: skip
    print(f"{len(plan)} tool runs: storage {args.storage} x areas {sorted(blocks)} x cores "
          f"{cores} x {args.repeats} repeats x {args.tools}", flush=True)  # fmt: skip
    if args.dry_run:
        for n, paths in sorted(blocks.items()):
            print(f"  {n:3d} tiles: {[p.name for p in paths[:4]]}{' ...' if n > 4 else ''}")
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.work_dir / "timings.csv"
    done = _done_keys(csv_path)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    host = platform.node()
    for storage in args.storage:
        folders = _subsets(blocks, storage, args)
        for n, c, r, tool in [(n, c, r, t) for s, n, c, r, t in plan if s == storage]:
            keys = [(storage, str(n), str(c), str(r), tool, step)
                    for _, step, _ in steps_for(tool, Path("."), Path("."), c, args)]  # fmt: skip
            if all(k in done for k in keys):
                continue
            out = (
                (args.local_dir if storage == "local" else args.work_dir)
                / "runs"
                / (f"{storage}_{n}t_{c}c_r{r}_{tool}")
            )
            shutil.rmtree(out, ignore_errors=True)  # never let resume logic skip work
            out.mkdir(parents=True)
            logs = args.work_dir / "logs"
            logs.mkdir(exist_ok=True)
            for stage, step, command in steps_for(tool, folders[n], out, c, args):
                started = time.strftime("%Y-%m-%dT%H:%M:%S")
                print(f"{started} {storage} {n:3d} tiles {c:2d} cores r{r} {tool:8s} {step} ...",
                      end=" ", flush=True)  # fmt: skip
                result = measure(command, cpus=cpus_available[:c], env=env,
                                 log=logs / f"{out.name}_{step.replace('+', '_')}.log")  # fmt: skip
                print(f"{result['wall_s']:.1f} s, {result['peak_rss_gb']:.2f} GB"
                      + ("" if result["exit_code"] == 0 else f"  FAILED ({result['exit_code']})"),
                      flush=True)  # fmt: skip
                _append(csv_path, {
                    "host": host, "storage": storage, "n_tiles": n,
                    "area_km2": round(n * size * size / 1e6, 3), "cores": c, "repeat": r,
                    "tool": tool, "stage": stage, "step": step, "started": started, **result,
                })  # fmt: skip
                if result["exit_code"]:
                    break  # later steps depend on this one
            if not args.keep_outputs:
                shutil.rmtree(out, ignore_errors=True)
    print(f"-> {csv_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(csv_path):
    """Medians per configuration, throughput and alsdb break-even, as text."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[df["exit_code"] == 0]
    per_stage = (
        df.groupby(["storage", "n_tiles", "area_km2", "cores", "tool", "stage", "repeat"])
        [["wall_s", "user_s", "sys_s", "peak_rss_gb"]]
        .agg({"wall_s": "sum", "user_s": "sum", "sys_s": "sum", "peak_rss_gb": "max"})
        .groupby(level=[0, 1, 2, 3, 4, 5]).median()
        .reset_index()
    )  # fmt: skip
    keys = ["storage", "n_tiles", "area_km2", "cores", "tool"]
    wall = per_stage.pivot_table(index=keys, columns="stage", values="wall_s", aggfunc="first")
    peak = per_stage.groupby(keys)["peak_rss_gb"].max()
    lines = ["Median over repeats. setup = one-off preparation, run = one product run.",
             "km2/core-h uses the run stage only.", ""]  # fmt: skip
    header = (f"{'storage':8s} {'tiles':>5s} {'km2':>7s} {'cores':>5s} {'tool':9s} "
              f"{'setup_s':>9s} {'run_s':>9s} {'end2end_s':>10s} {'km2/core-h':>11s} {'peak_GB':>8s}")  # fmt: skip
    lines.append(header)
    rows = {}
    for key, row in wall.iterrows():
        storage, n, km2, cores, tool = key
        setup = row.get("setup", 0.0)
        setup = 0.0 if math.isnan(setup) else setup  # lasR has no setup stage
        runs = row.get("run", math.nan)
        rate = km2 / (cores * runs / 3600) if runs > 0 else math.nan  # NaN > 0 is False
        rows[(storage, n, cores, tool)] = (setup, runs)
        lines.append(f"{storage:8s} {n:5d} {km2:7.1f} {cores:5d} {tool:9s} {setup:9.1f} "
                     f"{runs:9.1f} {setup + runs:10.1f} {rate:11.2f} {peak[key]:8.2f}")  # fmt: skip
    lines += ["", "alsdb break-even: product runs after which alsdb (setup + N runs) is",
              "faster in total than the other tool (its own setup + N runs).", ""]  # fmt: skip
    for (storage, n, cores, tool), (setup, runs) in sorted(rows.items()):
        if tool == "alsdb" or (storage, n, cores, "alsdb") not in rows:
            continue
        a_setup, a_run = rows[(storage, n, cores, "alsdb")]
        if runs > a_run:
            n_star = max(0.0, (a_setup - setup) / (runs - a_run))
            verdict = f"after {math.ceil(n_star)} run(s)" if n_star > 0 else "from the first run"
        else:
            verdict = "never (alsdb run is not faster)"
        lines.append(f"  {storage:8s} {n:3d} tiles {cores:2d} cores vs {tool:9s}: {verdict}")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "worker":
        return _worker(argv[1:])
    if argv and argv[0] == "report":
        print(report(argv[1]))
        return
    ap = argparse.ArgumentParser(prog="python -m alsdb.benchmark timing", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("laz_dir", type=Path)
    ap.add_argument("work_dir", type=Path, help="CSV, logs, network-storage runs")
    ap.add_argument(
        "--areas", type=int, nargs="+", default=[1, 4, 16], help="tile counts (squares)"
    )
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--storage", nargs="+", choices=["network", "local"], default=["local"])
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=Path(f"/tmp/{os.environ.get('USER', 'user')}/alsdb_timing"),
    )
    ap.add_argument("--tools", nargs="+", choices=TOOLS, default=list(TOOLS))
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2],
                    help="alsdb checkout with the pixi environments")  # fmt: skip
    ap.add_argument("--launcher", choices=["pixi", "direct"], default="pixi")
    ap.add_argument("--lastools-dir", type=Path)
    ap.add_argument("--crs", default="EPSG:25831")
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--veg", default="3,4,5")
    ap.add_argument("--keep-outputs", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="show the selected tiles and plan")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
