"""Spike-free freeze-distance sensitivity on finished benchmark scenes

Every scene's LAZ is ingested once and alsdb spike-free is run with each
freeze setting below (``spacing``: mean spacing of first surface returns);
outputs are scored with :func:`~alsdb.benchmark.synthetic.score` and
appended to one CSV, and the CHMs are kept in ``preds_sensitivity.npz``.
Finished scenes are skipped on rerun.

- ``k3``: 3 spacings (``"auto"``)
- ``floor1.5``: ``max(3 * spacing, 1.5 m)``
- ``fixed1.5``: 1.5 m
- ``lsf_local``: Fischer et al. (2024) per 5 m cell (``"local"``)
- ``k3_first``: ``k3`` on first returns only
- ``lspikefree``: ``lsf_local`` on first returns only, the full method of
  Fischer et al. (2024), https://doi.org/10.1111/2041-210X.14416

    python -m alsdb.benchmark sensitivity <scenes_dir> <out.csv> --workers 8
"""

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from alsdb.benchmark import synthetic

SURFACE_CLASSES = (2, 5, 9)


def freeze_settings(spacing):
    """{name: spike-free options of ``compute_chm``} for a scene with this pulse spacing."""
    scalar = {
        "k3": 3 * spacing,
        "floor1.5": max(3 * spacing, 1.5),
        "fixed1.5": 1.5,
    }
    settings = {name: {"freeze_distance": d} for name, d in scalar.items()}
    settings["lsf_local"] = {"freeze_distance": "local"}
    settings["k3_first"] = {"freeze_distance": 3 * spacing, "spikefree_first_returns": True}
    settings["lspikefree"] = {"freeze_distance": "local", "spikefree_first_returns": True}
    return settings


def first_return_spacing(laz):
    import pdal

    from alsdb.methods.rasterise import _pulse_spacing

    p = pdal.Pipeline(f'["{laz}"]')
    p.execute()
    a = p.arrays[0]
    first = a[np.isin(a["Classification"], SURFACE_CLASSES) & (a["ReturnNumber"] == 1)]
    return _pulse_spacing(first)


def sweep_scene(scene, keep=False):
    """Run every freeze setting on one scene; return score rows."""
    import zarr

    from alsdb.core.alsdatabase import ALSDatabase
    from alsdb.products.chm import compute_chm
    from alsdb.storage.zarr_store import ALSZarrStore

    scene = Path(scene)
    crs, year = synthetic._scene_crs_year(scene)
    truth = np.load(scene / "truth.npz")
    bounds, n = synthetic._window(truth)
    laz = sorted((scene / "laz").glob("*.laz"))
    spacing = first_return_spacing(laz[0])
    work = scene / "sensitivity"
    shutil.rmtree(work, ignore_errors=True)
    db = ALSDatabase(storage_type="local", uri=str(work / "tiledb"))
    db.create(crs=crs)
    db.ingest_many(laz)
    settings = freeze_settings(spacing)
    preds = {}
    for name, options in settings.items():
        store = work / f"spikefree_{name}.zarr"
        compute_chm(db, ALSZarrStore(str(store)), resolution=1.0, year=year, bbox=bounds,
                    method="spikefree", tile_size=float(n), quality=False,
                    **options)  # fmt: skip
        chm = zarr.open(str(store), mode="r")["1m"]["chm"][0, :n, :n]
        preds[f"spikefree_{name}"] = np.asarray(chm, np.float32)
    np.savez_compressed(work / "preds.npz", **preds)
    # Kept beside the scene for later analyses (density strata, derived metrics).
    np.savez_compressed(scene / "preds_sensitivity.npz", **preds)
    for f in ("truth.npz", "scene.json"):
        shutil.copy(scene / f, work / f)
    rows = synthetic.score(work)
    for r in rows:
        name = r["source"].removeprefix("spikefree_")
        freeze = settings[name]["freeze_distance"]
        r.update({"scenario": scene.name, "setting": name,
                  "freeze_m": freeze if freeze != "local" else np.nan,
                  "first_returns": settings[name].get("spikefree_first_returns", False),
                  "spacing": spacing})  # fmt: skip
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return rows


def scenes_to_sweep(scenes_dir, out_csv):
    """(scenes still to sweep, all eligible scenes) under *scenes_dir*."""
    scenes = sorted(
        p for p in Path(scenes_dir).iterdir() if (p / "truth.npz").exists() and (p / "laz").is_dir()
    )
    # A benchmark still running in the same folder writes scenes as it goes:
    # take only those it has finished and scored.
    scored = Path(scenes_dir) / "results.csv"
    if scored.exists():
        finished = synthetic.finished_scenarios(scored)
        scenes = [s for s in scenes if s.name in finished]
    done = synthetic.finished_scenarios(out_csv)
    return [s for s in scenes if s.name not in done], scenes


def run(scenes_dir, out_csv, *, workers=1, keep=False):
    """Sweep every unfinished scene under *scenes_dir*; append rows to *out_csv*."""
    todo, scenes = scenes_to_sweep(scenes_dir, out_csv)
    print(f"{len(todo)} of {len(scenes)} scenes to sweep", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sweep_scene, s, keep): s for s in todo}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                synthetic.append_results(out_csv, future.result())
                print(f"{scene.name}: done", flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad scene must not stop the sweep
                print(f"{scene.name}: FAILED {exc}", flush=True)
    return out_csv
