"""Synthetic benchmark: simulated ALS over known forests, scored against exact truth.

Each :class:`Scenario` (terrain x forest type x pulse density x canopy
penetration) becomes a scene with an analytic terrain and voxelised crowns.
HELIOS++ (Winiwarter et al. 2022, LGPL-3, run as an external program from
the optional ``helios`` pixi environment) flies a survey over it and writes
classified LAZ. alsdb, lasR and LAStools then process that LAZ, and every
output is scored against the truth of the same scene:

- DTM: the 1 m terrain mesh HELIOS++ scans, exact.
- CHM: the top of the voxel crowns HELIOS++ sees, relative to the terrain, at
  each 1 m cell centre (what TIN-based CHMs estimate) and as the cell maximum
  (what binned maxima estimate). The voxel grid is offset so cell centres
  fall on voxel-column centres.
- Trees: nominal apex heights and the voxel top near each stem.

    python -m alsdb.benchmark synthetic <out_dir> --terrain flat rolling \\
        --forest conifer broadleaf --density 2 10 --tools alsdb lasr

Run it from the ``spikefree`` environment (native spike-free); HELIOS++ and
lasR are called in their own environments. Results go to
``<out_dir>/results.csv`` (one row per scenario and method); finished
scenarios are skipped on rerun.
"""

import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

X0, Y0 = 500_000.0, 4_600_000.0  # scene origin; written as EPSG:25831
CRS = "EPSG:25831"
YEAR = 2020
TERRAINS = ("flat", "rolling", "steep", "rough")
FORESTS = ("open", "conifer", "broadleaf", "tropical")
METHODS = ("pitfree", "spikefree")
FLIGHT_HEIGHT = 1000.0  # m above the highest terrain point
SCAN_HALF_ANGLE = 15.0  # degrees; HELIOS++ scanAngle_deg is the half angle
SPEED = 60.0  # m/s


@dataclass(frozen=True)
class Scenario:
    terrain: str = "rolling"
    forest: str = "conifer"
    density: float = 2.0  # pulses per m2
    pad_scale: float = 1.0  # multiplies the forest type's plant area index
    max_returns: int = 4
    core: float = 150.0  # scored square, m
    margin: float = 30.0  # simulated around the core, not scored
    voxel: float = 0.5
    seed: int = 0

    def __post_init__(self):
        if self.terrain not in TERRAINS or self.forest not in FORESTS:
            raise ValueError(f"terrain in {TERRAINS}, forest in {FORESTS}")
        if self.density <= 0 or self.pad_scale <= 0 or self.max_returns < 1:
            raise ValueError("density, pad_scale and max_returns must be positive")
        if abs(1.0 / self.voxel - round(1.0 / self.voxel)) > 1e-9 or round(1.0 / self.voxel) % 2:
            raise ValueError("voxel must divide 1 m into an even number of columns (0.5, 0.25)")

    @property
    def name(self):
        return (f"{self.terrain}_{self.forest}_d{self.density:g}_p{self.pad_scale:g}"
                f"_r{self.max_returns}_s{self.seed}")  # fmt: skip

    @property
    def size(self):
        return self.core + 2 * self.margin


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def terrain_height(kind, x, y):
    """Terrain elevation (m) at scene-local coordinates *x*, *y* (m)."""
    if kind == "flat":
        return np.full(np.broadcast(x, y).shape, 300.0)
    rolling = 300.0 + 15.0 * np.sin(x / 80.0) * np.cos(y / 60.0)
    if kind == "rolling":
        return rolling
    if kind == "steep":  # ~35 degree slope along x, gentle cross-slope undulation
        return 300.0 + math.tan(math.radians(35.0)) * x + 3.0 * np.sin(y / 40.0)
    # rough: rolling hills plus metre-scale micro-topography
    return rolling + 0.6 * np.sin(x / 3.1 + 1.0) * np.cos(y / 2.7) + 0.4 * np.sin((x + y) / 1.9)


# Forest types: list of layers (stems per ha, height mean/sd, height bounds,
# crown radius / height, crown base / height, shape, placement), and the
# stand plant area index (m2 plant area per m2 ground, gaps included; leaf
# area index values after Asner et al. 2003, Global Ecol. Biogeogr., plus
# wood). The plant area density inside crowns follows from the crown volume
# (see crown_pad): fixing it directly made deep, filled crowns far too opaque.
FOREST_TYPES = {
    "open": {"pai": 1.0, "layers": [
        {"per_ha": 60, "h": (10.0, 3.0), "hb": (4.0, 20.0), "r": 0.35, "base": 0.3, "shape": "ellipsoid", "place": "random"}]},
    "conifer": {"pai": 4.0, "layers": [
        {"per_ha": 340, "h": (18.0, 4.0), "hb": (6.0, 32.0), "r": 0.18, "base": 0.4, "shape": "cone", "place": "grid"}]},
    "broadleaf": {"pai": 5.5, "layers": [
        {"per_ha": 200, "h": (22.0, 5.0), "hb": (8.0, 35.0), "r": 0.30, "base": 0.45, "shape": "ellipsoid", "place": "grid"}]},
    "tropical": {"pai": 6.0, "layers": [
        {"per_ha": 8, "h": (45.0, 3.0), "hb": (38.0, 52.0), "r": 0.18, "base": 0.6, "shape": "ellipsoid", "place": "random"},
        {"per_ha": 160, "h": (30.0, 4.0), "hb": (20.0, 38.0), "r": 0.22, "base": 0.55, "shape": "ellipsoid", "place": "grid"},
        {"per_ha": 400, "h": (10.0, 2.5), "hb": (5.0, 16.0), "r": 0.25, "base": 0.3, "shape": "ellipsoid", "place": "random"}]},
}  # fmt: skip


def make_trees(forest, size, rng):
    """Stems of every layer: dict of arrays x, y (scene-local), h, r, base, cone (bool)."""
    out = {k: [] for k in ("x", "y", "h", "r", "base", "cone")}
    for layer in FOREST_TYPES[forest]["layers"]:
        n = round(layer["per_ha"] * size * size / 1e4)
        if layer["place"] == "grid":
            spacing = math.sqrt(1e4 / layer["per_ha"])
            g = np.arange(spacing / 2, size, spacing)
            x, y = (a.ravel() for a in np.meshgrid(g, g))
            x = x + rng.uniform(-0.3, 0.3, x.size) * spacing
            y = y + rng.uniform(-0.3, 0.3, y.size) * spacing
            keep = rng.random(x.size) < 0.85  # some gaps
            x, y = x[keep], y[keep]
        else:
            x, y = rng.uniform(0, size, (2, n))
        h = np.clip(rng.normal(*layer["h"], x.size), *layer["hb"])
        out["x"].append(x)
        out["y"].append(y)
        out["h"].append(h)
        out["r"].append(layer["r"] * h * rng.uniform(0.8, 1.2, x.size))
        out["base"].append(layer["base"] * h)
        out["cone"].append(np.full(x.size, layer["shape"] == "cone"))
    return {k: np.concatenate(v) for k, v in out.items()}


def voxelise(trees, terrain, size, voxel):
    """Occupied voxels of all crowns, and the grid geometry.

    Columns are centred on multiples of *voxel* (grid origin at -voxel/2), so
    every 1 m cell centre lies on a column centre. A voxel is occupied when
    its centre lies inside a crown; crown heights are relative to the ground
    at the stem.
    """
    n_xy = round(size / voxel) + 1
    origin = -voxel / 2
    ground_at_stem = terrain_height(terrain, trees["x"], trees["y"])
    # Lowest ground anywhere in the scene (valleys lie inside it, not at the corners).
    g = np.arange(0.0, size + 1.0)
    ground_min = min(terrain_height(terrain, *np.meshgrid(g, g)).min(), ground_at_stem.min())
    zmin = float(np.floor(ground_min) - 1)
    ztop = float((ground_at_stem + trees["h"]).max()) + 1
    nz = math.ceil((ztop - zmin) / voxel)
    occ = np.zeros((n_xy, n_xy, nz), dtype=bool)
    for x, y, h, r, cb, cone, g in zip(*(trees[k] for k in ("x", "y", "h", "r", "base", "cone")),
                                       ground_at_stem):  # fmt: skip
        i0, i1 = max(0, int((x - r - origin) / voxel)), min(n_xy, int((x + r - origin) / voxel) + 1)
        j0, j1 = max(0, int((y - r - origin) / voxel)), min(n_xy, int((y + r - origin) / voxel) + 1)
        k0, k1 = max(0, int((g + cb - zmin) / voxel)), min(nz, int((g + h - zmin) / voxel) + 1)
        cx = origin + (np.arange(i0, i1) + 0.5) * voxel
        cy = origin + (np.arange(j0, j1) + 0.5) * voxel
        cz = zmin + (np.arange(k0, k1) + 0.5) * voxel - g  # height above the stem's ground
        dx, dy, hz = np.meshgrid(cx - x, cy - y, cz, indexing="ij")
        d2 = dx * dx + dy * dy
        if cone:
            rad = r * (h - hz) / (h - cb)
            inside = (hz >= cb) & (hz <= h) & (d2 <= rad * rad)
        else:
            a = (h - cb) / 2
            inside = d2 / (r * r) + ((hz - cb - a) / a) ** 2 <= 1
        occ[i0:i1, j0:j1, k0:k1] |= inside
    return occ, origin, zmin


def crown_pad(pai, occ, size, voxel):
    """Plant area density (m2/m3) giving stand plant area index *pai* over *size* x *size* m."""
    return pai * size * size / (occ.sum() * voxel**3)


def mesh_height(zgrid, x, y):
    """Height of the 1 m terrain mesh written for HELIOS++ at local *x*, *y*.

    Vertices are ``zgrid[j, i]`` at local (i, j); each square is split along
    its (i, j)-(i+1, j+1) diagonal, as in ``terrain.obj``.
    """
    ny, nx = zgrid.shape
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = np.clip(np.floor(x).astype(int), 0, nx - 2)
    j = np.clip(np.floor(y).astype(int), 0, ny - 2)
    u, v = x - i, y - j
    za, zb, zc, zd = zgrid[j, i], zgrid[j, i + 1], zgrid[j + 1, i + 1], zgrid[j + 1, i]
    return np.where(u >= v, za + u * (zb - za) + v * (zc - zb), za + v * (zd - za) + u * (zc - zd))


def voxel_truth(occ, origin, zmin, voxel, zgrid, margin, core):
    """DTM and CHM truths on the core (1 m cells, north-up) from occupied voxels.

    The DTM is the terrain mesh at cell centres. CHMs are the top of the
    voxel columns above the mesh: the column at each cell centre, and the
    highest column touching the cell.
    """
    nx, ny, nz = occ.shape
    has = occ.any(axis=2)
    top = np.where(has, zmin + (nz - np.argmax(occ[:, :, ::-1], axis=2)) * voxel, -np.inf)
    cx = origin + (np.arange(nx) + 0.5) * voxel  # column centres (multiples of voxel)
    cy = origin + (np.arange(ny) + 0.5) * voxel
    col_hag = np.maximum(top - mesh_height(zgrid, *np.meshgrid(cx, cy, indexing="ij")), 0.0)
    per_m = round(1 / voxel)
    m0, nc = round(margin), round(core)
    centre_idx = np.round((np.arange(nc) + m0 + 0.5) * per_m).astype(int)  # column of each centre
    chm_centre = col_hag[np.ix_(centre_idx, centre_idx)]
    half = per_m // 2  # columns from the centre to the cell edge (edge columns straddle)
    chm_max = np.full((nc, nc), -np.inf)
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            chm_max = np.maximum(chm_max, col_hag[np.ix_(centre_idx + di, centre_idx + dj)])
    ccx = m0 + np.arange(nc) + 0.5
    dtm = mesh_height(zgrid, *np.meshgrid(ccx, ccx, indexing="ij"))

    def to_north_up(a):  # (x, y) -> rows north to south
        return a.T[::-1].astype(np.float32)

    return {"dtm": to_north_up(dtm), "chm_centre": to_north_up(chm_centre),
            "chm_max": to_north_up(chm_max)}  # fmt: skip


def write_helios_scene(out, *, name, zgrid, voxels, pad, origin, zmin, shape, voxel, xy0, density,
                       max_returns, seed):  # fmt: skip
    """Write terrain mesh, voxel crowns, scene, scanner and survey for HELIOS++.

    *zgrid* is the 1 m terrain (local coordinates, ``zgrid[j, i]`` at x=i,
    y=j); *voxels* the (i, j, k) indices of occupied voxels and *pad* their
    plant area density (scalar or one per voxel). Everything is translated by
    *xy0*. One flight line along x over the middle of the scene.
    """
    out = Path(out)
    ny, nx = zgrid.shape
    gx, gy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    idx = np.arange(nx * ny).reshape(ny, nx) + 1
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = np.stack([a, b, c, a, c, d], -1).reshape(-1, 3)
    with open(out / "terrain.obj", "w") as f:
        f.write("mtllib terrain.mtl\nusemtl ground\n")
        np.savetxt(
            f, np.column_stack((gx.ravel(), gy.ravel(), zgrid.ravel())), fmt="v %.3f %.3f %.3f"
        )
        np.savetxt(f, faces, fmt="f %d %d %d")
    (out / "terrain.mtl").write_text(
        "newmtl ground\nKd 0.3 0.3 0.3\nhelios_isGround 1\nhelios_classification 2\n"
    )

    # Crowns as transmittive voxels (AMAPVox format read by HELIOS++).
    ii, jj, kk = voxels
    vx, vy, vz = shape
    with open(out / "forest.vox", "w") as f:
        f.write(f"VOXEL SPACE\n#min_corner: {origin} {origin} {zmin}\n")
        f.write(f"#max_corner: {origin + vx * voxel} {origin + vy * voxel} {zmin + vz * voxel}\n")
        f.write(f"#split: {vx} {vy} {vz}\n#res: {voxel}\n")
        f.write("i j k PadBVTotal angleMean bsEntering bsIntercepted bsPotential ground_distance "
                "lMeanTotal lgTotal nbEchos nbSampling transmittance attenuation "
                "attenuationBiasCorrection\n")  # fmt: skip
        rows = np.column_stack((ii, jj, kk, np.broadcast_to(pad, np.shape(ii))))
        np.savetxt(f, rows, fmt="%d %d %d %.6g 0 0 0 0 0 0 0 0 0 0 0 0")
    (out / "forest.mtl").write_text(
        "newmtl foliage\nKd 0.05 0.4 0.05\nhelios_spectra deciduous\nhelios_classification 5\n"
    )

    x0, y0 = xy0
    (out / "scene.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <scene id="syn" name="{name}">
    <part>
      <filter type="objloader"><param type="string" key="filepath" value="{out / "terrain.obj"}"/></filter>
      <filter type="translate"><param type="vec3" key="offset" value="{x0};{y0};0"/></filter>
    </part>
    <part>
      <filter type="detailedvoxels">
        <param type="string" key="filepath" value="{out / "forest.vox"}"/>
        <param type="string" key="matfile" value="{out / "forest.mtl"}"/>
        <param type="string" key="matname" value="foliage"/>
      </filter>
      <filter type="translate"><param type="vec3" key="offset" value="{x0};{y0};0"/></filter>
    </part>
  </scene>
</document>
""")
    prf, scan_freq = survey_settings(density)
    (out / "scanner.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <!-- Leica ALS50-II geometry; maxNOR and pulse rate set per scenario -->
  <scanner id="syn_als" accuracy_m="0.05" beamDivergence_rad="0.00022" name="synthetic ALS"
           optics="oscillating" pulseFreqs_Hz="{prf}" pulseLength_ns="10" rangeMin_m="200"
           scanAngleMax_deg="37.5" scanFreqMin_Hz="0" scanFreqMax_Hz="400"
           maxNOR="{max_returns}"/>
</document>
""")
    alt = float(zgrid.max()) + FLIGHT_HEIGHT
    size_x = nx - 1
    cy_line = y0 + (ny - 1) / 2
    (out / "survey.xml").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <scannerSettings id="set" active="true" pulseFreq_hz="{prf}" scanAngle_deg="{SCAN_HALF_ANGLE}"
                   scanFreq_hz="{scan_freq}" trajectoryTimeInterval_s="0.05"/>
  <survey name="syn" seed="{seed + 1}" scene="{out / "scene.xml"}#syn"
          platform="data/platforms.xml#sr22" scanner="{out / "scanner.xml"}#syn_als">
    <!-- winSize_ns: HELIOS++ default peak window. 1 ns split filled crowns into returns
         0.8 m apart, so 94% of broadleaf pulses spent all four returns in the crown top. -->
    <FWFSettings beamSampleQuality="3" binSize_ns="0.25" winSize_ns="2.5"/>
    <leg><platformSettings x="{x0 - 300}" y="{cy_line}" z="{alt}" onGround="false" movePerSec_m="{SPEED}"/><scannerSettings template="set"/></leg>
    <leg><platformSettings x="{x0 + size_x + 300}" y="{cy_line}" z="{alt}" onGround="false" movePerSec_m="{SPEED}"/><scannerSettings template="set" active="false"/></leg>
  </survey>
</document>
""")  # fmt: skip
    return prf, scan_freq


def make_scene(scenario, out_dir):
    """Write the HELIOS++ scene and survey, and the truth, for *scenario*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(scenario.seed)
    size, voxel = scenario.size, scenario.voxel
    trees = make_trees(scenario.forest, size, rng)
    occ, origin, zmin = voxelise(trees, scenario.terrain, size, voxel)
    pai = FOREST_TYPES[scenario.forest]["pai"] * scenario.pad_scale
    pad = crown_pad(pai, occ, size, voxel)
    g = np.arange(int(size) + 1, dtype=float)
    zgrid = terrain_height(scenario.terrain, *np.meshgrid(g, g))  # 1 m mesh, [y, x]

    truth = voxel_truth(occ, origin, zmin, voxel, zgrid, scenario.margin, scenario.core)
    m0, nc = round(scenario.margin), round(scenario.core)
    in_core = ((trees["x"] >= m0) & (trees["x"] < m0 + nc)
               & (trees["y"] >= m0) & (trees["y"] < m0 + nc))  # fmt: skip
    np.savez_compressed(
        out / "truth.npz", **truth, x0=X0 + m0, y1=Y0 + m0 + nc,
        tree_x=X0 + trees["x"][in_core], tree_y=Y0 + trees["y"][in_core],
        tree_h=trees["h"][in_core],
    )  # fmt: skip
    prf, scan_freq = write_helios_scene(
        out, name=scenario.name, zgrid=zgrid, voxels=np.nonzero(occ), pad=pad, origin=origin,
        zmin=zmin, shape=occ.shape, voxel=voxel, xy0=(X0, Y0), density=scenario.density,
        max_returns=scenario.max_returns, seed=scenario.seed,
    )  # fmt: skip
    meta = {"scenario": asdict(scenario), "crs": CRS, "year": YEAR, "trees": int(trees["x"].size),
            "voxels": int(occ.sum()), "pai": pai, "pad": pad, "pulse_rate_hz": prf,
            "scan_freq_hz": scan_freq}  # fmt: skip
    (out / "scene.json").write_text(json.dumps(meta, indent=2))
    return meta


def survey_settings(density):
    """Pulse rate and mirror frequency for *density* pulses/m2 (calibrated in HELIOS++).

    density = rate / (speed * swath), swath = 2 * height * tan(half angle); on
    flat ground HELIOS++ gave 4.59 pulses/m2 for a predicted 4.66. The mirror
    frequency makes along-track line spacing (speed / 2f) equal to the
    point spacing.
    """
    swath = 2 * FLIGHT_HEIGHT * math.tan(math.radians(SCAN_HALF_ANGLE))
    prf = round(density * SPEED * swath)
    scan_freq = round(SPEED * math.sqrt(density) / 2, 2)
    return prf, scan_freq


# ---------------------------------------------------------------------------
# Simulation and tools
# ---------------------------------------------------------------------------


def _pixi(env, *args, repo=None):
    repo = repo or Path(__file__).resolve().parents[2]
    return ["pixi", "run", "--manifest-path", str(repo), "-e", env, *map(str, args)]


def _scene_crs_year(scene_dir):
    meta = json.loads((Path(scene_dir) / "scene.json").read_text())
    return meta.get("crs", CRS), meta.get("year", YEAR)


def simulate(scene_dir, *, threads=8, repo=None):
    """Run HELIOS++ and write ``<scene_dir>/laz/SYN_<year>_scene.laz`` (CRS, 1 mm scale)."""
    import pdal

    scene_dir = Path(scene_dir)
    crs, year = _scene_crs_year(scene_dir)
    shutil.rmtree(scene_dir / "helios", ignore_errors=True)
    with open(scene_dir / "helios.log", "w") as log:
        subprocess.run(
            _pixi("helios", "helios", scene_dir / "survey.xml", "--output", scene_dir / "helios",
                  "--lasOutput", "--zipOutput", "--seed", 1, "-j", threads, repo=repo),
            check=True, stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip
    legs = sorted(str(p) for p in (scene_dir / "helios").rglob("leg*_points.laz"))
    laz = scene_dir / "laz"
    laz.mkdir(exist_ok=True)
    pdal.Pipeline(json.dumps([
        *legs, {"type": "filters.merge"},
        {"type": "writers.las", "filename": str(laz / f"SYN_{year}_scene.laz"), "a_srs": crs,
         "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001, "offset_x": "auto",
         "offset_y": "auto", "offset_z": "auto", "minor_version": 4},
    ])).execute()  # fmt: skip
    return laz


def _window(truth):
    n = truth["chm_max"].shape[0]
    x0, y1 = float(truth["x0"]), float(truth["y1"])
    return (x0, y1 - n, x0 + n, y1), n


def run_tools(scene_dir, tools=("alsdb", "lasr"), *, lastools_dir=None, repo=None):
    """Process the scene's LAZ with each tool; save core-window rasters to ``preds.npz``."""
    import rasterio
    import zarr
    from rasterio.windows import from_bounds

    scene_dir = Path(scene_dir)
    crs, year = _scene_crs_year(scene_dir)
    truth = np.load(scene_dir / "truth.npz")
    bounds, n = _window(truth)
    laz = scene_dir / "laz"
    preds, timings = {}, {}

    def tif(path):
        with rasterio.open(path) as ds:
            a = ds.read(1, window=from_bounds(*bounds, transform=ds.transform), masked=True)
        return np.asarray(a.astype(np.float32).filled(np.nan))

    if "alsdb" in tools:
        from alsdb.core.alsdatabase import ALSDatabase
        from alsdb.processing.chm import compute_all
        from alsdb.storage.zarr_store import ALSZarrStore

        uri = scene_dir / "alsdb" / "tiledb"
        shutil.rmtree(scene_dir / "alsdb", ignore_errors=True)
        db = ALSDatabase(storage_type="local", uri=str(uri))
        db.create(crs=crs)
        db.ingest_many(sorted(laz.glob("*.laz")))
        for method in METHODS:
            store = scene_dir / "alsdb" / f"{method}.zarr"
            t = time.perf_counter()
            compute_all(db, ALSZarrStore(str(store)), resolution=1.0, year=year, bbox=bounds,
                        method=method, tile_size=float(n), quality=False)  # fmt: skip
            timings[f"alsdb_{method}"] = time.perf_counter() - t
            g = zarr.open(str(store), mode="r")["1m"]
            preds[f"alsdb_{method}"] = np.asarray(g["chm"][0, :n, :n], np.float32)
            preds[f"alsdb_dtm_{method}"] = np.asarray(g["dtm"][0, :n, :n], np.float32)
    if "lasr" in tools:
        out = scene_dir / "lasr"
        shutil.rmtree(out, ignore_errors=True)
        t = time.perf_counter()
        subprocess.run(_pixi("lasr", "python", "-m", "alsdb.benchmark", "lasr", laz, out,
                             "--workers", 1, "--veg", 5, repo=repo),
                       check=True, capture_output=True)  # fmt: skip
        timings["lasr"] = time.perf_counter() - t
        for method in METHODS:
            preds[f"lasr_{method}"] = tif(out / method / f"SYN_{year}_scene.tif")
    if "lastools" in tools:
        from alsdb.benchmark import lastools

        out = scene_dir / "lastools"
        shutil.rmtree(out, ignore_errors=True)
        t = time.perf_counter()
        lastools.run(laz, out, veg_classes=(5,), lastools_dir=lastools_dir, products=("spikefree",))
        timings["lastools"] = time.perf_counter() - t
        preds["lastools_spikefree"] = tif(out / "spikefree" / f"SYN_{year}_scene.tif")
    np.savez_compressed(scene_dir / "preds.npz", **preds)
    (scene_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    return preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _peaks(a, rows, cols, k=1):
    return np.array(
        [np.nanmax(a[r - k : r + k + 1, c - k : c + k + 1]) for r, c in zip(rows, cols)]
    )


def score(scene_dir):
    """One row per CHM output: errors against both truths, cover, mean height, tree tops.

    Cells where the truth is missing (a real-data reference can have gaps)
    are left out. When the truth carries ``tree_h_field`` (e.g. TLS heights),
    tree tops are also scored against those.
    """
    scene_dir = Path(scene_dir)
    truth = np.load(scene_dir / "truth.npz")
    preds = dict(np.load(scene_dir / "preds.npz"))
    meta = json.loads((scene_dir / "scene.json").read_text())
    n = truth["chm_max"].shape[0]
    rows = (float(truth["y1"]) - truth["tree_y"]).astype(int)
    cols = (truth["tree_x"] - float(truth["x0"])).astype(int)
    ok = (rows >= 1) & (cols >= 1) & (rows < n - 1) & (cols < n - 1)
    rows, cols = rows[ok], cols[ok]
    vox_top = _peaks(truth["chm_max"], rows, cols)
    field = truth["tree_h_field"][ok] if "tree_h_field" in truth else None
    has_truth = np.isfinite(truth["chm_centre"]) & np.isfinite(truth["chm_max"])
    out = []
    for name, p in preds.items():
        if "_dtm_" in name:
            continue
        m = np.isfinite(p) & has_truth
        row = {"scenario": Path(scene_dir).name, **meta["scenario"], "source": name,
               "valid": float(np.isfinite(p)[has_truth].mean())}  # fmt: skip
        for label in ("centre", "max"):
            d = p[m] - truth[f"chm_{label}"][m]
            row.update({f"bias_{label}": float(d.mean()), f"mae_{label}": float(np.abs(d).mean()),
                        f"within05_{label}": float(np.mean(np.abs(d) < 0.5))})  # fmt: skip
        t = truth["chm_centre"]
        row.update({
            "mean_height": float(np.nanmean(p)), "mean_height_truth": float(np.nanmean(t)),
            "cover2": float(np.mean(p[m] > 2)), "cover2_truth": float(np.mean(t[has_truth] > 2)),
        })  # fmt: skip
        if rows.size:
            peaks = _peaks(p, rows, cols)
            e = peaks - vox_top
            row.update({"treetop_bias": float(np.nanmean(e)),
                        "treetop_mae": float(np.nanmean(np.abs(e))),
                        "treetop_within1": float(np.nanmean(np.abs(e) < 1)),
                        "trees": int(ok.sum())})  # fmt: skip
            if field is not None:
                f = peaks - field
                row.update({"treetop_field_bias": float(np.nanmean(f)),
                            "treetop_field_mae": float(np.nanmean(np.abs(f)))})  # fmt: skip
        dtm = preds.get(name.replace("alsdb_", "alsdb_dtm_")) if name.startswith("alsdb_") else None
        if dtm is not None:
            dm = np.isfinite(dtm) & np.isfinite(truth["dtm"])
            dd = dtm[dm] - truth["dtm"][dm]
            row.update({"dtm_bias": float(dd.mean()), "dtm_mae": float(np.abs(dd).mean())})
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def finished_scenarios(results):
    """Scenario names already in *results* (a results.csv), so reruns skip them."""
    results = Path(results)
    if not results.exists():
        return set()
    with results.open() as handle:
        return {r["scenario"] for r in csv.DictReader(handle)}


def append_results(results, rows):
    """Append score rows to *results*, widening the header when new columns appear."""
    results = Path(results)
    results.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    new = not results.exists()
    if not new:
        with results.open() as handle:
            old = next(csv.reader(handle))
        fields = list(dict.fromkeys([*old, *fields]))
        if fields != old:  # rewrite with the wider header
            with results.open() as handle:
                kept = list(csv.DictReader(handle))
            with results.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
    with results.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def run_grid(out_dir, scenarios, *, tools=("alsdb", "lasr"), lastools_dir=None, threads=8,
             keep=False):  # fmt: skip
    """Make, simulate, process and score every scenario; append rows to results.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.csv"
    done = finished_scenarios(results)
    for scenario in scenarios:
        if scenario.name in done:
            continue
        scene = out_dir / scenario.name
        t = time.perf_counter()
        meta = make_scene(scenario, scene)
        simulate(scene, threads=threads)
        run_tools(scene, tools, lastools_dir=lastools_dir)
        append_results(results, score(scene))
        print(f"{scenario.name}: {meta['trees']} trees, {meta['voxels']:,} voxels, "
              f"{time.perf_counter() - t:.0f} s", flush=True)  # fmt: skip
        if not keep:
            for sub in ("helios", "alsdb", "lasr", "lastools"):
                shutil.rmtree(scene / sub, ignore_errors=True)
            (scene / "forest.vox").unlink(missing_ok=True)
    return results
