"""ULS benchmark: real canopies from UAV laser scanning, at ALS pulse densities.

Each :class:`Area` (a square of a ULS survey) is read once, streaming, and
turned into two kinds of scenes that the synthetic benchmark's
:func:`~alsdb.benchmark.synthetic.run_tools` and
:func:`~alsdb.benchmark.synthetic.score` process like any other:

- ``thinned``: the real ULS returns, sampled to ALS density by pulse. Only
  near-nadir pulses are used; one pulse per cell of 1/density m2 is kept,
  with all its returns, so the sample is as regular as an ALS scan pattern.
  The reference comes from the full-density cloud: a TIN of the lowest
  ground return per 25 cm (DTM) and the highest return per 10 cm above it
  (CHM: the cell maximum, and the 20 cm around the cell centre). Real
  canopy and returns, but the sensor stays a UAV one (small footprint,
  short range).
- ``helios``: plant area density per 0.5 m voxel estimated from the ULS
  (MacArthur-Horn on near-nadir returns weighted 1 / number of returns,
  per voxel column, as in leafR), scanned with HELIOS++ by the synthetic
  benchmark's aircraft survey over the ULS terrain. Aircraft geometry and an
  exact truth (the voxels), but the canopy is a voxel model.

Classes: ground (2) is kept, every other non-noise class is written as
vegetation (5), so the tools need no site-specific class settings.

Field tree heights (e.g. TLS heights in a GeoPackage) are optional; tree tops
are then scored against them too.

    python -m alsdb.benchmark uls <out_dir> --laz-dir <tiles> --site DE-Eco \\
        --x0 416700 --y0 5346650 --core 100 --year 2026 --density 0.5 2 10
"""

import json
import shutil
import sqlite3
import struct
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from alsdb.benchmark import synthetic

VOXEL = 0.5
GROUND_CELL = 0.25
SURFACE_CELL = 0.1
SPREAD = 12.0  # read beyond the scene so pulses near its edge keep all their returns
NOISE_CLASSES = (7, 18)
G_FUNCTION = 0.5  # spherical leaf angle distribution
PAD_MIN = 0.05  # m2/m3; sparser voxels (a stray return among thousands of pulses) are dropped
CHUNK = 2_000_000


@dataclass(frozen=True)
class Area:
    """A square of a ULS survey: *core* m scored, *margin* m simulated around it."""

    site: str
    laz_dir: str
    x0: float  # core lower-left corner, whole metres
    y0: float
    core: float = 100.0
    margin: float = 15.0
    year: int = 2024
    crs: str | None = None  # from the LAZ headers when not given
    trees: str | None = None  # GeoPackage with field tree heights
    tree_height_field: str = "tls_treeheight"

    def __post_init__(self):
        for v in (self.x0, self.y0, self.core, self.margin):
            if v != round(v):
                raise ValueError("x0, y0, core and margin must be whole metres")

    @property
    def name(self):
        return f"{self.site}_{self.x0:.0f}_{self.y0:.0f}_{self.core:g}"

    @property
    def box(self):
        """Scene extent: core plus margin (xmin, ymin, xmax, ymax)."""
        m = self.margin
        return (self.x0 - m, self.y0 - m, self.x0 + self.core + m, self.y0 + self.core + m)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _header(path):
    import pdal

    p = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(path), "count": 0}]))
    p.execute()
    return p.metadata["metadata"]["readers.las"]


def _files_and_header(laz_dir, bounds, workers=8):
    """LAZ files overlapping *bounds*, their z range, and the horizontal EPSG code."""
    files = sorted(Path(laz_dir).glob("*.la[sz]"))
    with ThreadPoolExecutor(workers) as pool:
        headers = list(pool.map(_header, files))
    x0, y0, x1, y1 = bounds
    keep = [(f, h) for f, h in zip(files, headers)
            if h["minx"] < x1 and h["maxx"] > x0 and h["miny"] < y1 and h["maxy"] > y0]  # fmt: skip
    if not keep:
        raise ValueError(f"no LAZ in {laz_dir} overlaps {bounds}")
    import pyproj

    wkt = keep[0][1].get("srs", {}).get("horizontal")
    epsg = pyproj.CRS.from_wkt(wkt).to_epsg(min_confidence=20) if wkt else None
    zr = (min(h["minz"] for _, h in keep), max(h["maxz"] for _, h in keep))
    return [f for f, _ in keep], zr, (f"EPSG:{epsg}" if epsg else None)


def _splitmix(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def pulse_keys(a):
    """(time in 10 ns, line/channel tag) identifying the pulse of each point."""
    t = np.round(a["GpsTime"] * 1e8).astype(np.int64)
    tag = a["PointSourceId"].astype(np.int64) * 64
    if "ScanChannel" in a.dtype.names:
        tag = tag + a["ScanChannel"].astype(np.int64)
    return t, tag


def pulse_hash(t, tag, seed=0):
    """Uniform [0, 1) per pulse, the same for all its returns wherever they are stored."""
    z = _splitmix(t.view(np.uint64) ^ _splitmix(tag.astype(np.uint64) + np.uint64(seed)))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


@dataclass
class Scan:
    """Everything one pass over an area's points produces."""

    box: tuple
    zmin: float
    ground: np.ndarray  # lowest ground return per 25 cm: (n, 3) x, y, z
    surface: np.ndarray  # highest return per 10 cm cell, NaN where empty
    veg: np.ndarray  # near-nadir vegetation return weight per voxel (x, y, z)
    ground_weight: np.ndarray  # near-nadir ground return weight per voxel column
    candidates: np.ndarray  # points of preselected near-nadir pulses
    crs: str | None
    points: int
    nadir_pulses_per_m2: float


def scan_area(area, *, max_angle=20.0, candidate_fraction=0.05, seed=0, workers=8, voxels=True,
              keep_classes=None):  # fmt: skip
    """Stream the area's points once; see :class:`Scan`.

    ``voxels=False`` skips the voxel profiles (only the HELIOS++ variant needs
    them; on steep terrain they take gigabytes) and ``candidate_fraction=0``
    the pulse candidates (only the thinned variant needs them).
    *keep_classes* restricts every product to those classes (default: all
    but noise).
    """
    import pdal

    bx0, by0, bx1, by1 = area.box
    read = (bx0 - SPREAD, by0 - SPREAD, bx1 + SPREAD, by1 + SPREAD)
    files, (zlo, zhi), crs = _files_and_header(area.laz_dir, read, workers)
    size = bx1 - bx0
    zmin = float(np.floor(zlo) - 1)
    ns = round(size / SURFACE_CELL)
    ng = round(size / GROUND_CELL)
    nv = round(size / VOXEL) + 1  # voxel columns centred on multiples of VOXEL from bx0
    nz = int(np.ceil((zhi + 1 - zmin) / VOXEL)) if voxels else 0
    if not voxels:
        nv = 0
    surface = np.full(ns * ns, -np.inf)
    veg = np.zeros(nv * nv * nz)
    ground_w = np.zeros(nv * nv)
    ground_parts, cand_parts = [], []
    points = nadir_first = 0
    bounds = f"([{read[0]},{read[2]}],[{read[1]},{read[3]}])"
    for f in files:
        pipe = pdal.Pipeline(json.dumps([str(f), {"type": "filters.crop", "bounds": bounds}]))
        for a in pipe.iterator(chunk_size=CHUNK):
            if keep_classes is None:
                a = a[~np.isin(a["Classification"], NOISE_CLASSES)]
            else:
                a = a[np.isin(a["Classification"], keep_classes)]
            points += a.size
            x, y, z = a["X"], a["Y"], a["Z"]
            inbox = (x >= bx0) & (x < bx1) & (y >= by0) & (y < by1)
            ground = a["Classification"] == 2
            # surface: highest return per 10 cm
            i = ((x[inbox] - bx0) / SURFACE_CELL).astype(np.int64)
            j = ((y[inbox] - by0) / SURFACE_CELL).astype(np.int64)
            np.maximum.at(surface, np.minimum(j, ns - 1) * ns + np.minimum(i, ns - 1), z[inbox])
            # ground: lowest return per 25 cm, with its position
            g = inbox & ground
            if g.any():
                cell = (np.minimum(((y[g] - by0) / GROUND_CELL).astype(np.int64), ng - 1) * ng
                        + np.minimum(((x[g] - bx0) / GROUND_CELL).astype(np.int64), ng - 1))  # fmt: skip
                o = np.lexsort((z[g], cell))
                first = np.r_[True, cell[o][1:] != cell[o][:-1]]
                ground_parts.append(np.column_stack((cell[o][first], x[g][o][first],
                                                     y[g][o][first], z[g][o][first])))  # fmt: skip
            nadir = np.abs(a["ScanAngleRank"]) <= max_angle
            nadir_first += int(np.sum(nadir & inbox & (a["ReturnNumber"] == 1)))
            if voxels:
                # voxel profiles from near-nadir returns, each weighted 1 / its pulse's returns
                v = nadir & (x >= bx0 - VOXEL / 2) & (x < bx1 + VOXEL / 2) \
                    & (y >= by0 - VOXEL / 2) & (y < by1 + VOXEL / 2)  # fmt: skip
                iv = np.floor((x[v] - bx0) / VOXEL + 0.5).astype(np.int64)
                jv = np.floor((y[v] - by0) / VOXEL + 0.5).astype(np.int64)
                kv = np.clip(((z[v] - zmin) / VOXEL).astype(np.int64), 0, nz - 1)
                w = 1.0 / np.maximum(a["NumberOfReturns"][v], 1)
                gv = ground[v]
                col = iv * nv + jv
                ground_w += np.bincount(col[gv], w[gv], minlength=nv * nv)
                veg += np.bincount(col[~gv] * nz + kv[~gv], w[~gv], minlength=nv * nv * nz)
            if candidate_fraction > 0:
                # pulses preselected by hash: all their returns, wherever they are stored
                t, tag = pulse_keys(a[nadir])
                keep = pulse_hash(t, tag, seed) < candidate_fraction
                cand_parts.append(a[nadir][keep])
    ground = np.concatenate(ground_parts) if ground_parts else np.zeros((0, 4))
    o = np.lexsort((ground[:, 3], ground[:, 0]))  # lowest per cell across chunks
    ground = ground[o][np.r_[True, ground[o][1:, 0] != ground[o][:-1, 0]]][:, 1:]
    surface[~np.isfinite(surface)] = np.nan
    return Scan(
        box=area.box, zmin=zmin, ground=ground, surface=surface.reshape(ns, ns),
        veg=veg.reshape(nv, nv, nz), ground_weight=ground_w.reshape(nv, nv),
        candidates=np.concatenate(cand_parts) if cand_parts else np.zeros(0),
        crs=area.crs or crs, points=points,
        nadir_pulses_per_m2=nadir_first / size**2,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Terrain and reference
# ---------------------------------------------------------------------------


def _terrain(ground, box):
    """Linear interpolation of the ground points; nearest outside their hull."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    lin = LinearNDInterpolator(ground[:, :2] - box[:2], ground[:, 2])
    near = NearestNDInterpolator(ground[:, :2] - box[:2], ground[:, 2])

    def at(x, y):  # local coordinates
        z = lin(x, y)
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = near(x[bad], y[bad])
        return z

    return at


def reference(scan, area, centre=0.2):
    """Full-density DTM and CHM truths on the core (1 m, north-up) and the 1 m terrain grid.

    ``chm_centre`` is the highest return within the *centre* x *centre* m
    around each cell centre (0.2 m for ULS; ~0.4 m for 30 pulses/m2 ALS, so
    that most windows hold a return).
    """
    at = _terrain(scan.ground, scan.box)
    size = scan.box[2] - scan.box[0]
    g = np.arange(round(size) + 1, dtype=float)
    zgrid = at(*np.meshgrid(g, g))  # [y, x], the mesh HELIOS++ gets
    ns = scan.surface.shape[0]
    c = (np.arange(ns) + 0.5) * SURFACE_CELL
    cx, cy = np.meshgrid(c, c)
    hag = scan.surface - at(cx, cy)  # [y, x], 10 cm
    m0, nc, per = round(area.margin), round(area.core), round(1 / SURFACE_CELL)
    core = hag[m0 * per : (m0 + nc) * per, m0 * per : (m0 + nc) * per].reshape(nc, per, nc, per)
    half = round(centre / SURFACE_CELL / 2)
    if half < 1 or 2 * half > per:
        raise ValueError(f"centre must be between {2 * SURFACE_CELL} and 1 m")
    mid = slice(per // 2 - half, per // 2 + half)  # the centre x centre m around the cell centre
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty cells stay NaN
        chm_max = np.nanmax(core, axis=(1, 3))
        chm_centre = np.nanmax(core[:, mid, :, mid], axis=(1, 3))
    ccx = m0 + np.arange(nc) + 0.5
    dtm = at(*np.meshgrid(ccx, ccx))

    def to_north_up(a):  # [y, x] rows south to north -> north-up
        return a[::-1].astype(np.float32)

    return {"dtm": to_north_up(dtm), "chm_max": to_north_up(np.maximum(chm_max, 0)),
            "chm_centre": to_north_up(np.maximum(chm_centre, 0))}, zgrid  # fmt: skip


def read_trees(path, height_field, bounds):
    """Field trees (x, y, height) inside *bounds* from a point GeoPackage."""
    con = sqlite3.connect(path)
    for (table,) in con.execute("select table_name from gpkg_contents"):
        cols = [r[1] for r in con.execute(f'pragma table_info("{table}")')]
        if height_field in cols:
            (geom,) = con.execute(
                "select column_name from gpkg_geometry_columns where table_name = ?", (table,)
            ).fetchone()
            rows = con.execute(f'select "{geom}", "{height_field}" from "{table}"').fetchall()
            break
    else:
        raise ValueError(f"no table with {height_field!r} in {path}")
    xyh = []
    for blob, h in rows:
        if blob is None or h is None:
            continue
        env = (blob[3] >> 1) & 7
        off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
        bo = "<" if blob[off] == 1 else ">"
        x, y = struct.unpack(bo + "dd", blob[off + 5 : off + 21])
        xyh.append((x, y, float(h)))
    t = np.array(xyh).reshape(-1, 3)
    x0, y0, x1, y1 = bounds
    return t[(t[:, 0] >= x0) & (t[:, 0] < x1) & (t[:, 1] >= y0) & (t[:, 1] < y1)]


# ---------------------------------------------------------------------------
# Variant 1: thinned ULS
# ---------------------------------------------------------------------------


def thin_pulses(candidates, box, density, seed=0):
    """One pulse per cell of about 1/*density* m2 (the lowest hash), with all its returns.

    A pulse's position is its first stored return. Returns the selected
    points and the achieved pulses per m2.
    """
    t, tag = pulse_keys(candidates)
    keys = np.rec.fromarrays([t, tag])
    uniq, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    o = np.lexsort((candidates["ReturnNumber"], inv))
    first = o[np.r_[True, inv[o][1:] != inv[o][:-1]]]  # first stored return of each pulse
    px, py = candidates["X"][first], candidates["Y"][first]
    h = pulse_hash(uniq["f0"], uniq["f1"], seed)
    bx0, by0, bx1, by1 = box
    inside = (px >= bx0) & (px < bx1) & (py >= by0) & (py < by1)
    n = max(1, round((bx1 - bx0) * np.sqrt(density)))  # cells per side, tiling the box exactly
    s = (bx1 - bx0) / n
    cell = ((py - by0) / s).astype(np.int64) * n + ((px - bx0) / s).astype(np.int64)
    ids = np.nonzero(inside)[0]
    o = np.lexsort((h[ids], cell[ids]))
    win = ids[o][np.r_[True, cell[ids][o][1:] != cell[ids][o][:-1]]]
    keep = np.zeros(uniq.size, bool)
    keep[win] = True
    return candidates[keep[inv]], win.size / (bx1 - bx0) ** 2


def write_laz(points, path, crs):
    import pdal

    pts = points.copy()
    veg = pts["Classification"] != 2
    pts["Classification"][veg] = 5
    path.parent.mkdir(parents=True, exist_ok=True)
    pdal.Pipeline(json.dumps([
        {"type": "writers.las", "filename": str(path), "a_srs": crs, "minor_version": 4,
         "dataformat_id": 6, "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001,
         "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"},
    ]), arrays=[pts]).execute()  # fmt: skip


# ---------------------------------------------------------------------------
# Variant 2: ULS voxels scanned by HELIOS++
# ---------------------------------------------------------------------------


def plant_area_density(veg, ground_weight):
    """MacArthur-Horn plant area density per voxel from near-nadir return weights.

    Per column, the share of pulse energy entering a voxel that passes it is
    T = below / (below + inside) (half a return added to both, so opaque
    voxels stay finite); PAD = -ln(T) / (G * voxel). Empty voxels get 0.
    """
    below = np.cumsum(veg, axis=2) - veg + ground_weight[:, :, None]
    t = (below + 0.5) / (below + veg + 0.5)
    pad = -np.log(t) / (G_FUNCTION * VOXEL)
    pad[veg == 0] = 0.0
    return pad


def voxel_scene(scan, zgrid, area, density, out, *, max_returns=4, seed=0):
    """Write a HELIOS++ scene of the ULS voxels over the ULS terrain, and its truth."""
    pad = plant_area_density(scan.veg, scan.ground_weight)
    nv = pad.shape[0]
    centres = (np.arange(nv)) * VOXEL  # local column centres
    cz = scan.zmin + (np.arange(pad.shape[2]) + 0.5) * VOXEL
    ground = synthetic.mesh_height(zgrid, *np.meshgrid(centres, centres, indexing="ij"))
    occ = (pad >= PAD_MIN) & (cz[None, None, :] > ground[:, :, None])
    origin = -VOXEL / 2
    truth = synthetic.voxel_truth(occ, origin, scan.zmin, VOXEL, zgrid, area.margin, area.core)
    voxels = np.nonzero(occ)
    synthetic.write_helios_scene(
        out, name=out.name, zgrid=zgrid, voxels=voxels, pad=pad[voxels], origin=origin,
        zmin=scan.zmin, shape=occ.shape, voxel=VOXEL, xy0=scan.box[:2], density=density,
        max_returns=max_returns, seed=seed,
    )  # fmt: skip
    return truth, int(occ.sum()), float(np.median(pad[voxels])) if voxels[0].size else 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_area(out_dir, area, *, densities=(0.5, 2.0, 10.0), variants=("thinned", "helios"),
             tools=("alsdb", "lasr"), lastools_dir=None, max_angle=20.0, candidate_fraction=0.05,
             threads=8, seed=0, keep=False):  # fmt: skip
    """Scan *area*, build every variant x density scene, run the tools, append to results.csv."""
    out_dir = Path(out_dir)
    results = out_dir / "results.csv"
    done = synthetic.finished_scenarios(results)
    todo = [(v, d) for v in variants for d in densities if f"{area.name}_{v}_d{d:g}" not in done]
    if not todo:
        return results
    t0 = time.perf_counter()
    scan = scan_area(area, max_angle=max_angle, candidate_fraction=candidate_fraction, seed=seed)
    if scan.crs is None:
        raise ValueError("no CRS in the LAZ headers; pass Area(crs=...)")
    truth_ref, zgrid = reference(scan, area)
    x0, y1 = area.x0, area.y0 + area.core
    trees = {"tree_x": np.zeros(0), "tree_y": np.zeros(0), "tree_h": np.zeros(0)}
    if area.trees:
        t = read_trees(area.trees, area.tree_height_field,
                       (area.x0, area.y0, area.x0 + area.core, area.y0 + area.core))  # fmt: skip
        trees = {"tree_x": t[:, 0], "tree_y": t[:, 1], "tree_h": t[:, 2], "tree_h_field": t[:, 2]}
    print(f"{area.name}: scanned {scan.points:,} points in {time.perf_counter() - t0:.0f} s, "
          f"{scan.nadir_pulses_per_m2:,.0f} near-nadir pulses/m2, "
          f"{scan.candidates.size:,} candidate returns", flush=True)  # fmt: skip
    base = {"site": area.site, "x0": area.x0, "y0": area.y0, "core": area.core,
            "margin": area.margin, "max_angle": max_angle, "seed": seed}  # fmt: skip
    for variant, density in todo:
        name = f"{area.name}_{variant}_d{density:g}"
        scene = out_dir / name
        shutil.rmtree(scene, ignore_errors=True)
        scene.mkdir(parents=True)
        t = time.perf_counter()
        meta = {"scenario": {**base, "variant": variant, "density": density},
                "crs": scan.crs, "year": area.year, "area": asdict(area)}  # fmt: skip
        if variant == "thinned":
            pts, achieved = thin_pulses(scan.candidates, scan.box, density, seed)
            write_laz(pts, scene / "laz" / f"SYN_{area.year}_scene.laz", scan.crs)
            truth = truth_ref
            meta.update({"pulses_per_m2": achieved, "points": int(pts.size)})
        elif variant == "helios":
            truth, n_vox, pad_median = voxel_scene(scan, zgrid, area, density, scene, seed=seed)
            meta.update({"voxels": n_vox, "pad_median": pad_median})
            (scene / "scene.json").write_text(json.dumps(meta, indent=2))
            synthetic.simulate(scene, threads=threads)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        np.savez_compressed(scene / "truth.npz", **truth, x0=x0, y1=y1, **trees)
        (scene / "scene.json").write_text(json.dumps(meta, indent=2))
        synthetic.run_tools(scene, tools, lastools_dir=lastools_dir)
        synthetic.append_results(results, synthetic.score(scene))
        print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
        if not keep:
            for sub in ("helios", "alsdb", "lasr", "lastools"):
                shutil.rmtree(scene / sub, ignore_errors=True)
            (scene / "forest.vox").unlink(missing_ok=True)
    return results
