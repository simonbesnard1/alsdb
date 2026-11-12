"""GEDI attenuation over many ALS blocks (alsdb.products.gedi_attenuation at scale).

For each block of a Lidar HD selection (``lidarhd-select``): ingest its tiles
once, query the GEDI L2A/L2B shots over it from gedidb, keep the usable ones
acquired within ``year_window`` years of the ALS survey, simulate the same
footprints from the ALS points and compare real GEDI, simulated and ALS
canopy profiles. Blocks flown leaf-off are skipped unless asked for: their
ALS profile is not comparable with leaf-on GEDI shots.

GEDI's own ``leaf_off_flag`` is not applied by default: it comes from a
phenology climatology per 1 km cell, and over Petit Buëch it flagged whole
late-August orbits of deciduous forest as leaf-off. Every good-quality shot
is simulated; ``gedi_leaf_off_flag`` and ``time`` stay in each block's
``footprints.parquet`` for the analysis to filter on (``leaf_flag=True``
filters at selection instead).

    pixi run -e gedi python -m alsdb.benchmark gedi-attenuation <out_dir>
        --selection selection_manual.csv selection_all.csv --data-dir <lidarhd>

Outputs: ``<out_dir>/<block>/`` (the pipeline's files and ``tiledb/``),
``<out_dir>/blocks.csv`` (one row per block: status, shots, fits) and
``<out_dir>/profiles.csv`` (every block's per-height table, with the block's
columns). Finished blocks are skipped on rerun.

gedidb: GFZ, https://github.com/simonbesnard1/gedidb; GEDI: Dubayah et al.
(2020), https://doi.org/10.1016/j.srs.2020.100002.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_COLUMNS = ("block", "year", "leaf_season", "stratum", "slope_mean", "tree_share", "sensor")


def read_selection(paths):
    """Blocks of one or more ``lidarhd-select`` CSVs (first occurrence of a block wins)."""
    blocks = {}
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                blocks.setdefault(row["block"], row)
    return list(blocks.values())


def _done(blocks_csv):
    if not Path(blocks_csv).exists():
        return set()
    return set(pd.read_csv(blocks_csv, dtype={"block": str})["block"])


def _append(path, row):
    """Add one row, rewriting the table: rows of skipped and finished blocks differ in columns."""
    path = Path(path)
    old = pd.read_csv(path, dtype={"block": str}) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def run_blocks(out_dir, selection, data_dir, *, crs="EPSG:2154", year_window=2,
               include_leaf_off=False, leaf_flag=False, n_workers=4, ingest_workers=4,
               seed=0, keep_store=False):  # fmt: skip
    """Run the attenuation pipeline on every block; see the module docstring.

    Each block's TileDB store (a few GB for a 4 km2 Lidar HD block) is deleted
    once the block is recorded, unless *keep_store*; reruns do not need it.
    """
    import shutil

    from alsdb import ALSDatabase
    from alsdb.products.gedi_attenuation import run_gedi_attenuation

    out_dir, data_dir = Path(out_dir), Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_csv, profiles_csv = out_dir / "blocks.csv", out_dir / "profiles.csv"
    done = _done(blocks_csv)
    for meta in read_selection(selection):
        name = meta["block"]
        if name in done:
            continue
        info = {k: meta.get(k) for k in BLOCK_COLUMNS}
        year = int(meta["year"])
        if meta.get("leaf_season") == "leaf-off" and not include_leaf_off:
            _append(blocks_csv, {**info, "status": "skipped: ALS leaf-off"})
            print(f"{name}: skipped (ALS leaf-off)", flush=True)
            continue
        files = sorted((data_dir / name).glob("*.la[sz]"))
        if not files:
            _append(blocks_csv, {**info, "status": "skipped: no tiles"})
            continue
        t0 = time.perf_counter()
        block_dir = out_dir / name
        block_dir.mkdir(parents=True, exist_ok=True)
        uri = block_dir / "tiledb"
        db = ALSDatabase(storage_type="local", uri=str(uri))
        if not (block_dir / "ingested").exists():
            db.create(crs=crs, overwrite=True)
            db.ingest_many(files, n_workers=ingest_workers)
            (block_dir / "ingested").touch()
        try:
            # One survey per block store: simulate from all of it (the year an ingest
            # reads from a header can be the processing date); the selection's survey
            # year only sets the GEDI time window.
            result = run_gedi_attenuation(
                db, block_dir, year=None, years=(year - year_window, year + year_window),
                leaf_on=leaf_flag,
                n_workers=n_workers, rng=np.random.default_rng(seed),
            )  # fmt: skip
        except ValueError as exc:  # e.g. no usable shot over the block
            _append(blocks_csv, {**info, "status": f"no result: {exc}",
                                 "minutes": round((time.perf_counter() - t0) / 60, 1)})  # fmt: skip
            print(f"{name}: {exc}", flush=True)
            if not keep_store:
                shutil.rmtree(uri, ignore_errors=True)
            continue
        fits = {f"{src}_{k}": v for src, fit in result["fits"].items() for k, v in fit.items()}
        prof = result["profiles"].copy()
        for k, v in info.items():
            prof[k] = v
        _append_frame(profiles_csv, prof)  # before blocks.csv, which marks the block done
        _append(blocks_csv, {**info, "status": "ok", "shots": len(result["shots"]),
                             "footprints": result["n_footprints"], **fits,
                             "minutes": round((time.perf_counter() - t0) / 60, 1)})  # fmt: skip
        if not keep_store:
            shutil.rmtree(uri, ignore_errors=True)
        print(f"{name}: {len(result['shots'])} shots, {result['n_footprints']} usable footprints, "
              f"{json.dumps({k: round(v, 3) for k, v in fits.items() if isinstance(v, float)})}, "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)  # fmt: skip
    return blocks_csv


def _append_frame(path, frame):
    path = Path(path)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
