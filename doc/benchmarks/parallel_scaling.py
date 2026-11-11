"""Time threads against worker processes for the tiled products on this machine.

Usage::

    python doc/benchmarks/parallel_scaling.py ARRAY_URI YEAR OUT_DIR \
        --bbox X0 Y0 X1 Y1 --workers 1 8 16 32

For each worker count the DTM + DSM + pit-free CHM run and the 10 m metrics
are timed with ``parallel="threads"`` and ``parallel="processes"``, with the
number of cores kept busy (CPU time over wall time) and the peak memory of
the main process and of the largest worker (the highest so far in this
script, so read the memory of a mode from its first lines or run it alone).
Outputs go to OUT_DIR and are overwritten by each run. Choose a bbox of a few hundred tiles (tile size
500 m) so every worker has work.
"""

import argparse
import os
import resource
import shutil
import time
from functools import partial
from pathlib import Path

from alsdb import ALSDatabase
from alsdb.products.biomass import compute_metrics
from alsdb.products.chm import compute_all
from alsdb.storage.zarr_store import ALSZarrStore


def measure(run):
    """Return wall time, busy cores and peak memory (GB) of *run*."""
    wall, cpu = time.perf_counter(), os.times()
    run()
    wall, after = time.perf_counter() - wall, os.times()
    used = sum(
        getattr(after, name) - getattr(cpu, name)
        for name in ("user", "system", "children_user", "children_system")
    )
    main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    worker = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return wall, used / wall, main, worker


def main():
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("array_uri")
    parser.add_argument("year", type=int)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--bbox", type=float, nargs=4, default=None)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16])
    args = parser.parse_args()
    bbox = tuple(args.bbox) if args.bbox else None
    products = {
        "DTM+DSM+pit-free CHM": lambda db, store, **kw: compute_all(
            db, store, 1.0, bbox, args.year, method="pitfree", **kw
        ),
        "metrics 10 m": lambda db, store, **kw: compute_metrics(
            db, store, 10.0, bbox, args.year, **kw
        ),
    }
    cpus = len(os.sched_getaffinity(0))
    print(f"{cpus} CPUs available to this job")
    for label, product in products.items():
        for workers in args.workers:
            for parallel in ("threads", "processes") if workers > 1 else ("threads",):
                db = ALSDatabase(
                    storage_type="local",
                    uri=args.array_uri,
                    max_reader_threads=max(1, cpus // workers),
                )
                out = args.out_dir / "parallel_scaling.zarr"
                shutil.rmtree(out, ignore_errors=True)
                store = ALSZarrStore(str(out))
                wall, cores, main_gb, worker_gb = measure(
                    partial(product, db, store, n_workers=workers, parallel=parallel)
                )
                print(
                    f"{label:22s} {workers:3d} {parallel:9s} {wall:8.1f} s  "
                    f"{cores:5.1f} cores busy  main {main_gb:5.1f} GB  "
                    f"largest worker {worker_gb:4.1f} GB",
                    flush=True,
                )


if __name__ == "__main__":
    main()
