"""Decimated copies of a dense ALS survey, scored against its full density.

Each decimated copy (``pd_*`` folders beside ``original/``) of an
:class:`~alsdb.benchmark.uls.Area` becomes one benchmark scene. The
reference comes from the original as for ULS (:func:`alsdb.benchmark.uls.reference`);
the input is the copy cropped to the scene, vegetation written as class 5.
``mixed`` scenes take a sparse copy everywhere and a denser one in
north-south strips, as flight-line overlaps do; ``truth.npz`` marks the
dense cells (``dense``).

The Chapitre Petit Buech, Vizzavona (both IGN Lidar HD, classification
corrected by hand) and Robson Creek (TERN) surveys and their decimated
copies were provided by Fabian Fischer (University of Bristol), made with
the workflow of Fischer et al. (2024): ``pd_X`` keeps X pulses/m2 by
removing pulses in acquisition order, ``pd_X_ft_F`` also removes a
fraction F of higher-order returns.

References
----------
Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. (2024). Robust
    characterisation of forest structure from airborne laser scanning - A
    systematic assessment and sample workflow for ecologists. Methods in
    Ecology and Evolution, 15, 1873-1888.
    https://doi.org/10.1111/2041-210X.14416

    python -m alsdb.benchmark decimated <out_dir> --site PetitBuech \
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
STRIPS = (100.0, 30.0, 35.0)  # period, width, offset from the scene's west edge (m)


def in_strips(x, xmin, strips=STRIPS):
    """Return where *x* lies in a dense strip of a scene starting at *xmin*."""
    period, width, offset = strips
    return ((np.asarray(x) - xmin - offset) % period) < width


def mixed_versions(versions, pairs):
    """{``mix_<sparse>_<dense>``: (sparse folder, dense folder)} for ``SPARSE:DENSE`` names."""
    out = {}
    for pair in pairs:
        sparse, _, dense = pair.partition(":")
        if sparse not in versions or dense not in versions:
            raise ValueError(f"{pair}: both copies must be among {sorted(versions)}")
        out[f"mix_{sparse}_{dense}"] = (versions[sparse], versions[dense])
    return out


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
    names = set.intersection(*(set(a.dtype.names) for a in parts))
    pts = np.concatenate([a[[n for n in parts[0].dtype.names if n in names]] for a in parts])
    pts["Classification"][np.isin(pts["Classification"], VEG)] = 5
    return pts


def crop_mixed(sparse_dir, dense_dir, box, strips=STRIPS):
    """Combine sparse points outside strips with dense points inside them."""
    sparse, dense = crop(sparse_dir, box), crop(dense_dir, box)
    names = [n for n in sparse.dtype.names if n in dense.dtype.names]
    return np.concatenate((sparse[names][~in_strips(sparse["X"], box[0], strips)],
                           dense[names][in_strips(dense["X"], box[0], strips)]))  # fmt: skip


def build_area(out_dir, area, versions=None, *, centre=0.4, mixed=(), strips=STRIPS):
    """Write one scene per decimated copy of *area*; return their folders.

    *mixed*: ``SPARSE:DENSE`` pairs of copy names, each one more scene with
    the dense copy in *strips* (period, width, offset in m).
    """
    out_dir = Path(out_dir)
    versions = versions or versions_next_to(area.laz_dir)
    versions = {**versions, **mixed_versions(versions, mixed)}
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
        extra, strip_meta = {}, {}
        if isinstance(folder, tuple):  # mixed: (sparse, dense)
            pts = crop_mixed(*folder, scan.box, strips)
            density, variant = parse_version(Path(folder[0]).name)
            dense_density, _ = parse_version(Path(folder[1]).name)
            variant = f"mix{dense_density:g}" + (f"_{variant}" if variant else "")
            n = round(area.core)
            columns = in_strips(area.x0 + np.arange(n) + 0.5, scan.box[0], strips)
            extra["dense"] = np.broadcast_to(columns, (n, n)).copy()
            strip_meta = {"dense_density": dense_density, "strips_period_width_offset_m": strips,
                          "dense_share": float(columns.mean())}  # fmt: skip
        else:
            pts = crop(folder, scan.box)
            density, variant = parse_version(name)
        uls.write_laz(pts, scene / "laz" / f"SYN_{area.year}_scene.laz", scan.crs)
        first = np.sum(pts["ReturnNumber"] == 1) / (scan.box[2] - scan.box[0]) ** 2
        meta = {
            "scenario": {"site": area.site, "x0": area.x0, "y0": area.y0, "core": area.core,
                         "margin": area.margin, "version": name, "density": density,
                         "variant": variant},
            "crs": scan.crs, "year": area.year, "points": int(pts.size),
            "first_returns_per_m2": float(first), "reference_centre_m": centre, **strip_meta,
        }  # fmt: skip
        (scene / "scene.json").write_text(json.dumps(meta, indent=2))
        np.savez_compressed(scene / "truth.npz", **truth, x0=area.x0,
                            y1=area.y0 + area.core, **trees, **extra)  # fmt: skip
    return scenes


def run_area(out_dir, area, versions=None, *, tools=("alsdb", "lasr"), lastools_dir=None,
             keep=False, mixed=(), strips=STRIPS):  # fmt: skip
    """Build the scenes of *area*, run the tools on each, append to results.csv."""
    import shutil

    out_dir = Path(out_dir)
    results = out_dir / "results.csv"
    done = synthetic.finished_scenarios(results)
    for scene in build_area(out_dir, area, versions, mixed=mixed, strips=strips):
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
