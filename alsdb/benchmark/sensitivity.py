"""Spike-free freeze-distance sensitivity on finished benchmark scenes.

Works on any folder of scenes written by :mod:`alsdb.benchmark.synthetic` or
:mod:`alsdb.benchmark.uls` (``laz/``, ``truth.npz``, ``scene.json`` each).
Every scene's LAZ is ingested once and alsdb spike-free is run with each
freeze distance below; the outputs are scored with
:func:`~alsdb.benchmark.synthetic.score` and appended to one CSV (one row per
scene and setting). Finished scenes are skipped on rerun.

Freeze settings (``spacing`` is the mean spacing of first surface returns of
the scene, as alsdb's ``"auto"`` rule computes it):

- ``k2``, ``k3``, ``k5``: 2, 3 (the current ``"auto"``) and 5 spacings
- ``floor1.5``: ``max(3 * spacing, 1.5 m)``, the candidate rule
- ``fixed1.5``: 1.5 m, the old default

    python -m alsdb.benchmark sensitivity <scenes_dir> <out.csv> --workers 8
"""

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from alsdb.benchmark import synthetic

SURFACE_CLASSES = (2, 5, 9)


def freeze_settings(spacing):
    """{name: freeze distance in m} for a scene with this pulse spacing."""
    return {
        "k2": 2 * spacing,
        "k3": 3 * spacing,
        "k5": 5 * spacing,
        "floor1.5": max(3 * spacing, 1.5),
        "fixed1.5": 1.5,
    }


def first_return_spacing(laz):
    import pdal

    from alsdb.processing.chm import _pulse_spacing

    p = pdal.Pipeline(f'["{laz}"]')
    p.execute()
    a = p.arrays[0]
    first = a[np.isin(a["Classification"], SURFACE_CLASSES) & (a["ReturnNumber"] == 1)]
    return _pulse_spacing(first)


def sweep_scene(scene, keep=False):
    """Run every freeze setting on one scene; return score rows."""
    import zarr

    from alsdb.core.alsdatabase import ALSDatabase
    from alsdb.processing.chm import compute_chm
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
    for name, freeze in settings.items():
        store = work / f"spikefree_{name}.zarr"
        compute_chm(db, ALSZarrStore(str(store)), resolution=1.0, year=year, bbox=bounds,
                    method="spikefree", tile_size=float(n), quality=False,
                    freeze_distance=freeze)  # fmt: skip
        chm = zarr.open(str(store), mode="r")["1m"]["chm"][0, :n, :n]
        preds[f"spikefree_{name}"] = np.asarray(chm, np.float32)
    np.savez_compressed(work / "preds.npz", **preds)
    for f in ("truth.npz", "scene.json"):
        shutil.copy(scene / f, work / f)
    rows = synthetic.score(work)
    for r in rows:
        name = r["source"].removeprefix("spikefree_")
        r.update({"scenario": scene.name, "setting": name, "freeze_m": settings[name],
                  "spacing": spacing})  # fmt: skip
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return rows


def run(scenes_dir, out_csv, *, workers=1, keep=False):
    """Sweep every unfinished scene under *scenes_dir*; append rows to *out_csv*."""
    scenes = sorted(
        p for p in Path(scenes_dir).iterdir() if (p / "truth.npz").exists() and (p / "laz").is_dir()
    )
    done = synthetic.finished_scenarios(out_csv)
    todo = [s for s in scenes if s.name not in done]
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
