"""Real ALS at reduced density: decimated copies of a dense survey vs its full density.

For sites delivered as one dense ALS survey plus decimated copies of it (e.g.
``original/`` and ``pd_00p5/`` ... ``pd_08/``, ``pd_04_ft_*/``), every copy of
an :class:`~alsdb.benchmark.uls.Area` becomes one benchmark scene that
:func:`~alsdb.benchmark.synthetic.run_tools`, :func:`~alsdb.benchmark.synthetic.score`
and :mod:`alsdb.benchmark.sensitivity` process like the synthetic ones:

- reference from the original, as for ULS (:func:`alsdb.benchmark.uls.reference`):
  TIN of the lowest ground return per 25 cm, highest return per 10 cm above
  it (cell maximum, and the 0.4 m around the cell centre);
- input: the decimated copy cropped to the scene, ground (2) and water (9)
  kept, vegetation classes 3-5 written as 5.

Unlike thinned ULS, the input keeps the aircraft sensor and footprint; only
the density changes. The reference uses the original's classification, which
the tools also see.

    python -m alsdb.benchmark decimated <out_dir> --site PetitBuech \\
        --original <site>/original --x0 937650 --y0 6396650 --core 200
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from alsdb.benchmark import synthetic, uls

SURFACE = (2, 3, 4, 5, 9)
VEG = (3, 4, 5)


def versions_next_to(original):
    """{name: folder} of the decimated copies beside *original* (``pd_*``)."""
    original = Path(original)
    return {p.name: p for p in sorted(original.parent.glob("pd_*")) if p.is_dir()}


def parse_version(name):
    """``pd_00p5`` -> (0.5, ""), ``pd_04_ft_0.75`` -> (4.0, "ft_0.75")."""
    m = re.fullmatch(r"pd_(\d+)(?:p(\d+))?(?:_(.+))?", name)
    if not m:
        return None, name
    density = float(f"{int(m.group(1))}.{m.group(2) or 0}")
    return density, m.group(3) or ""


def crop(laz_dir, box):
    """Surface-class points of *laz_dir* inside *box* to *out*; vegetation as class 5."""
    import pdal

    files, _, _ = uls._files_and_header(laz_dir, box)
    bounds = f"([{box[0]},{box[2]}],[{box[1]},{box[3]}])"
    parts = []
    for f in files:
        p = pdal.Pipeline(json.dumps([str(f), {"type": "filters.crop", "bounds": bounds}]))
        p.execute()
        a = p.arrays[0]
        parts.append(a[np.isin(a["Classification"], SURFACE)])
    pts = np.concatenate(parts)
    pts["Classification"][np.isin(pts["Classification"], VEG)] = 5
    return pts


def build_area(out_dir, area, versions=None, *, centre=0.4):
    """Write one scene per decimated copy of *area*; return their folders."""
    out_dir = Path(out_dir)
    versions = versions or versions_next_to(area.laz_dir)
    todo = {n: d for n, d in versions.items()
            if not (out_dir / f"{area.name}_{n}" / "truth.npz").exists()}  # fmt: skip
    scenes = [out_dir / f"{area.name}_{n}" for n in versions]
    if not todo:
        return scenes
    t0 = time.perf_counter()
    scan = uls.scan_area(area, max_angle=90.0, candidate_fraction=0.0, voxels=False,
                         keep_classes=SURFACE)  # fmt: skip
    if scan.crs is None:
        raise ValueError("no CRS in the LAZ headers; pass Area(crs=...)")
    truth, _ = uls.reference(scan, area, centre=centre)
    print(f"{area.name}: reference from {scan.points:,} points in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    trees = {"tree_x": np.zeros(0), "tree_y": np.zeros(0), "tree_h": np.zeros(0)}
    for name, folder in todo.items():
        scene = out_dir / f"{area.name}_{name}"
        pts = crop(folder, scan.box)
        uls.write_laz(pts, scene / "laz" / f"SYN_{area.year}_scene.laz", scan.crs)
        density, variant = parse_version(name)
        first = np.sum(pts["ReturnNumber"] == 1) / (scan.box[2] - scan.box[0]) ** 2
        meta = {
            "scenario": {"site": area.site, "x0": area.x0, "y0": area.y0, "core": area.core,
                         "margin": area.margin, "version": name, "density": density,
                         "variant": variant},
            "crs": scan.crs, "year": area.year, "points": int(pts.size),
            "first_returns_per_m2": float(first), "reference_centre_m": centre,
        }  # fmt: skip
        (scene / "scene.json").write_text(json.dumps(meta, indent=2))
        np.savez_compressed(scene / "truth.npz", **truth, x0=area.x0,
                            y1=area.y0 + area.core, **trees)  # fmt: skip
    return scenes


def run_area(out_dir, area, versions=None, *, tools=("alsdb", "lasr"), lastools_dir=None,
             keep=False):  # fmt: skip
    """Build the scenes of *area*, run the tools on each, append to results.csv."""
    import shutil

    out_dir = Path(out_dir)
    results = out_dir / "results.csv"
    done = synthetic.finished_scenarios(results)
    for scene in build_area(out_dir, area, versions):
        if scene.name in done:
            continue
        t = time.perf_counter()
        synthetic.run_tools(scene, tools, lastools_dir=lastools_dir)
        synthetic.append_results(results, synthetic.score(scene))
        print(f"{scene.name}: {time.perf_counter() - t:.0f} s", flush=True)
        if not keep:
            for sub in ("alsdb", "lasr", "lastools"):
                shutil.rmtree(scene / sub, ignore_errors=True)
    return results
