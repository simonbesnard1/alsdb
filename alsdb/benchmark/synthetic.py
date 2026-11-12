"""Simulated ALS over generated forests, scored against exact truth.

Each :class:`Scenario` (terrain, forest type, pulse density, ...) becomes a
scene with analytic terrain and voxel crowns. HELIOS++ (Winiwarter et al.
2022, https://doi.org/10.1016/j.rse.2021.112772; LGPL-3, run from the
optional ``helios`` environment) scans it; alsdb, lasR and LAStools process
the result, and every output is scored against the scene's truth: the exact
terrain, the voxel canopy top (at cell centres and as cell maxima) and the
tree apices. Results go to ``<out_dir>/results.csv``; finished scenarios are
skipped on rerun.

Forests can also come from tree lists (:func:`read_tree_list`), e.g. stands
simulated by the forest gap model FORMIND (:func:`formind_tree_list`; Fischer
et al. 2016, https://doi.org/10.1016/j.ecolmodel.2015.11.018), whose tree
biomass then enters the truth.

    python -m alsdb.benchmark synthetic <out_dir> --terrain flat rolling \
        --forest conifer broadleaf --density 2 10 --tools alsdb lasr
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
TERRAINS = ("flat", "rolling", "steep", "rough", "cliff", "ravine")
PLACEMENTS = ("regular", "clumped")
SHAPES = ("ellipsoid", "cone", "cylinder")  # crown shapes
TREE_LIST_COLUMNS = ("x", "y", "h", "crown_base", "crown_radius", "shape", "leaf_area", "agb")
FORESTS = ("open", "conifer", "broadleaf", "tropical")
METHODS = ("pitfree", "spikefree")  # run by alsdb and lasR
# alsdb outputs: name -> compute_all options. "tin" is the first-return TIN
# and "lspikefree" the locally adaptive first-return spike-free of Fischer et
# al. (2024), https://doi.org/10.1111/2041-210X.14416.
ALSDB_METHODS = {
    "pitfree": {"method": "pitfree"},
    "spikefree": {"method": "spikefree"},
    "tin": {"method": "tin"},
    "lspikefree": {
        "method": "spikefree",
        "freeze_distance": "local",
        "spikefree_first_returns": True,
    },
}
FLIGHT_HEIGHT = 1000.0  # m above the highest terrain point
BEAM_DIVERGENCE = 0.22  # mrad, Leica ALS50-II: a 0.22 m footprint at 1000 m
SCAN_HALF_ANGLE = 15.0  # degrees; HELIOS++ scanAngle_deg is the half angle
SPEED = 60.0  # m/s


@dataclass(frozen=True)
class Scenario:
    """Configuration for one synthetic terrain and forest scene."""

    terrain: str = "rolling"
    forest: str = "conifer"
    density: float = 2.0  # pulses per m2
    pad_scale: float = 1.0  # multiplies the forest type's plant area index
    max_returns: int = 4
    core: float = 150.0  # scored square, m
    margin: float = 30.0  # simulated around the core, not scored
    voxel: float = 0.5
    seed: int = 0
    understorey: bool = False  # add low vegetation (UNDERSTOREY)
    placement: str = "regular"  # "clumped": trees in groups (CLUMP_SIZE, CLUMP_SPREAD)
    beam_divergence: float = BEAM_DIVERGENCE  # mrad; at 1000 m the footprint is this many m wide
    trees: str | None = None  # tree list (read_tree_list) instead of a generated forest type

    def __post_init__(self):
        """Validate scene geometry and simulation settings."""
        if self.terrain not in TERRAINS:
            raise ValueError(f"terrain in {TERRAINS}")
        if self.trees is None and self.forest not in FORESTS:
            raise ValueError(f"forest in {FORESTS}, or give a tree list")
        if self.trees is not None and (self.understorey or self.placement != "regular"):
            raise ValueError("understorey and placement apply to generated forests only")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement in {PLACEMENTS}")
        if self.density <= 0 or self.pad_scale <= 0 or self.max_returns < 1:
            raise ValueError("density, pad_scale and max_returns must be positive")
        if not self.beam_divergence > 0:
            raise ValueError("beam_divergence must be positive (mrad)")
        if abs(1.0 / self.voxel - round(1.0 / self.voxel)) > 1e-9 or round(1.0 / self.voxel) % 2:
            raise ValueError("voxel must divide 1 m into an even number of columns (0.5, 0.25)")

    @property
    def name(self):
        """Return the stable directory name for this scenario."""
        # Suffixes only for the newer options, so earlier scenes keep their names.
        extra = ("_u" if self.understorey else "") + ("_c" if self.placement == "clumped" else "")
        if self.beam_divergence != BEAM_DIVERGENCE:
            extra += f"_bd{self.beam_divergence:g}"
        if self.voxel != 0.5:
            extra += f"_v{self.voxel:g}"
        return (f"{self.terrain}_{self.forest}_d{self.density:g}_p{self.pad_scale:g}"
                f"_r{self.max_returns}_s{self.seed}{extra}")  # fmt: skip

    @property
    def size(self):
        """Return the simulated side length, including margins."""
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
    if kind == "rough":  # rolling hills plus metre-scale micro-topography
        return (rolling + 0.6 * np.sin(x / 3.1 + 1.0) * np.cos(y / 2.7)
                + 0.4 * np.sin((x + y) / 1.9))  # fmt: skip
    if kind == "cliff":
        # Repeated 70--80 degree scarps create convex and concave terrain breaks.
        x = np.asarray(x, dtype=float) + 10.0 - 2.0 * np.sin(np.asarray(y, dtype=float) / 25.0)
        period, width = 45.0, 3.0
        step = np.floor(x / period)
        heights = np.array([6.0, 10.0, 14.0])
        below = np.cumsum(np.r_[0.0, np.resize(heights, 64)])  # scarps fully climbed
        k = np.clip(step.astype(int), 0, 63)
        ramp = np.clip((x - step * period - (period - width)) / width, 0.0, 1.0)
        return 300.0 + math.tan(math.radians(20.0)) * x + below[k] + heights[k % 3] * ramp
    if kind == "ravine":
        # Rolling hills cut by two meandering V-shaped channels 10 and 7 m deep
        # with ~50 degree sides: sharp concave breaks at the channel floor.
        y = np.asarray(y, dtype=float)
        side = math.tan(math.radians(50.0))
        out = rolling
        for centre, depth in ((60.0 + 15.0 * np.sin(y / 40.0), 10.0),
                              (150.0 + 12.0 * np.cos(y / 50.0), 7.0)):  # fmt: skip
            out = out - np.maximum(0.0, depth - np.abs(np.asarray(x) - centre) * side)
        return out
    raise ValueError(f"unknown terrain {kind!r}")


# Forest layers define stem density, crown geometry, placement, and stand PAI.
# PAI values follow Asner et al. (2003), doi:10.1046/j.1466-822X.2003.00026.x.
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
# Optional low vegetation under any forest type (shrubs, regeneration, scrub):
# 0.5-3 m tall, ~60% cover, its own plant area index. The classic ground-filter
# error: returns from it read as ground lift the DTM.
UNDERSTOREY = {"pai": 1.0, "layers": [
    {"per_ha": 2500, "h": (1.8, 0.6), "hb": (0.5, 3.0), "r": 0.5, "base": 0.05, "shape": "ellipsoid", "place": "random"}]}  # fmt: skip
# Clumped placement (a Thomas cluster process): trees in groups of about 8
# around random centres, spread over 1.2 times the regular spacing, leaving
# gaps of varied size instead of the regular grid's uniform ones.
CLUMP_SIZE, CLUMP_SPREAD = 8, 1.2


def make_trees(forest, size, rng, *, placement="regular", understorey=False):
    """Return the stems of every layer: arrays x, y, h, r, base, shape (SHAPES index) and low (understorey)."""
    out = {k: [] for k in ("x", "y", "h", "r", "base", "shape", "low")}
    layers = [(layer, False) for layer in FOREST_TYPES[forest]["layers"]]
    if understorey:
        layers += [(layer, True) for layer in UNDERSTOREY["layers"]]
    for layer, low in layers:
        n = round(layer["per_ha"] * size * size / 1e4)
        if placement == "clumped" and not low:
            spacing = math.sqrt(1e4 / layer["per_ha"])
            centres = rng.uniform(0, size, (2, max(1, round(n / CLUMP_SIZE))))
            pick = rng.integers(0, centres.shape[1], n)
            offset = rng.normal(0.0, CLUMP_SPREAD * spacing, (2, n))
            x, y = (centres[:, pick] + offset) % size
        elif layer["place"] == "grid":
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
        out["shape"].append(np.full(x.size, SHAPES.index(layer["shape"])))
        out["low"].append(np.full(x.size, low))
    return {k: np.concatenate(v) for k, v in out.items()}


def voxelise(trees, terrain, size, voxel, *, with_pad=False):
    """Occupied voxels of all crowns, and the grid geometry.

    Columns are centred on multiples of *voxel* (grid origin at -voxel/2), so
    every 1 m cell centre lies on a column centre. A voxel is occupied when
    its centre lies inside a crown; crown heights are relative to the ground
    at the stem. With *with_pad*, also returns the plant area density of every
    voxel: each tree's ``leaf_area`` spread evenly over its crown voxels,
    summed where crowns overlap.
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
    pad = np.zeros((n_xy, n_xy, nz), dtype=np.float32) if with_pad else None
    leaf_area = trees.get("leaf_area")
    if with_pad and leaf_area is None:
        raise ValueError("with_pad needs a leaf_area per tree")
    for t, (x, y, h, r, cb, shape, g) in enumerate(zip(
            *(trees[k] for k in ("x", "y", "h", "r", "base", "shape")), ground_at_stem)):  # fmt: skip
        i0, i1 = max(0, int((x - r - origin) / voxel)), min(n_xy, int((x + r - origin) / voxel) + 1)
        j0, j1 = max(0, int((y - r - origin) / voxel)), min(n_xy, int((y + r - origin) / voxel) + 1)
        k0, k1 = max(0, int((g + cb - zmin) / voxel)), min(nz, int((g + h - zmin) / voxel) + 1)
        cx = origin + (np.arange(i0, i1) + 0.5) * voxel
        cy = origin + (np.arange(j0, j1) + 0.5) * voxel
        cz = zmin + (np.arange(k0, k1) + 0.5) * voxel - g  # height above the stem's ground
        dx, dy, hz = np.meshgrid(cx - x, cy - y, cz, indexing="ij")
        d2 = dx * dx + dy * dy
        if SHAPES[shape] == "cone":
            rad = r * (h - hz) / (h - cb)
            inside = (hz >= cb) & (hz <= h) & (d2 <= rad * rad)
        elif SHAPES[shape] == "cylinder":
            inside = (hz >= cb) & (hz <= h) & (d2 <= r * r)
        else:
            a = (h - cb) / 2
            inside = d2 / (r * r) + ((hz - cb - a) / a) ** 2 <= 1
        occ[i0:i1, j0:j1, k0:k1] |= inside
        if with_pad and inside.any():
            pad[i0:i1, j0:j1, k0:k1] += inside * (leaf_area[t] / (inside.sum() * voxel**3))
    if with_pad:
        return occ, origin, zmin, pad
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
                       max_returns, seed, beam_divergence=BEAM_DIVERGENCE):  # fmt: skip
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
  <!-- Leica ALS50-II geometry; maxNOR, pulse rate and beam divergence set per scenario -->
  <scanner id="syn_als" accuracy_m="0.05" beamDivergence_rad="{beam_divergence / 1000:g}" name="synthetic ALS"
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


def read_tree_list(path, size):
    """Trees of a tree-list CSV in the arrays :func:`voxelise` takes.

    Columns (scene-local metres, origin at the scene's south-west corner):
    ``x, y, h, crown_base, crown_radius, shape`` (ellipsoid, cone or cylinder),
    ``leaf_area`` (m2 per tree) and optionally ``agb`` (t per tree). Trees with
    their stem outside the *size* x *size* m scene are left out.
    """
    import pandas as pd

    table = pd.read_csv(path)
    missing = set(TREE_LIST_COLUMNS[:-1]) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    unknown = set(table["shape"]) - set(SHAPES)
    if unknown:
        raise ValueError(f"{path}: unknown crown shapes {sorted(unknown)}")
    table = table[(table.x >= 0) & (table.x < size) & (table.y >= 0) & (table.y < size)
                  & (table.h > table.crown_base) & (table.crown_radius > 0)]  # fmt: skip
    trees = {"x": table.x.to_numpy(float), "y": table.y.to_numpy(float),
             "h": table.h.to_numpy(float), "r": table.crown_radius.to_numpy(float),
             "base": table.crown_base.to_numpy(float),
             "shape": table["shape"].map(SHAPES.index).to_numpy(int),
             "leaf_area": table.leaf_area.to_numpy(float), "low": np.zeros(len(table), bool)}  # fmt: skip
    if "agb" in table:
        trees["agb"] = table.agb.to_numpy(float)
    return trees


FORMIND_COLUMNS = ("Time", "Grp", "Species", "N", "BT", "D", "H", "SV", "LAITREE", "IR", "IR_rel",
                   "PB", "BInc", "DInc", "OBA", "AGE", "Plot", "Hec", "X", "Y", "CLP", "LAI", "CD",
                   "R", "RespMain", "RespGrowth", "ID", "CR")  # fmt: skip
TREE_FIELDS = ("Time", "BT", "H", "X", "Y", "CLP", "LAI", "CR")  # what a tree list needs


def formind_tree_list(res_path, year, out_csv, *, size=None, x0=0.0, y0=0.0, shape="cylinder"):
    """Tree list of one year of a FORMIND ``.res`` file (per-tree output).

    FORMIND (Fischer et al. 2016, https://doi.org/10.1016/j.ecolmodel.2015.11.018;
    UFZ Leipzig, AGPL-3) writes every tree with its position, height (H),
    crown length share (CLP), crown radius (CR), leaf area index per crown
    area (LAI) and above-ground biomass (BT, t). FORMIND treats crowns as
    cylinders from H * (1 - CLP) to H; *shape* can replace them by an
    ellipsoid or cone of the same height, base, radius and leaf area. The
    snapshot nearest *year* is used; with *size*,
    trees with stems in [x0, x0 + size) x [y0, y0 + size) are kept and shifted
    to the scene origin. Returns the snapshot year used.
    """
    return formind_tree_lists(res_path, {year: out_csv}, size=size, x0=x0, y0=y0, shape=shape)[year]


def formind_tree_lists(res_path, outputs, *, size=None, x0=0.0, y0=0.0, shape="cylinder",
                       chunksize=500_000):  # fmt: skip
    """:func:`formind_tree_list` for several years in one pass over the file.

    *outputs* maps each wanted year to its CSV path. The file is streamed, so
    multi-GB ``.res`` files (yearly output of many hectares) fit in memory.
    Returns the snapshot year used for each wanted year.
    """
    import pandas as pd

    if shape not in SHAPES:
        raise ValueError(f"shape in {SHAPES}")
    best = {}  # wanted year -> [snapshot year, row blocks]
    # Columns by the short names of the third header line: FORMIND versions
    # differ in which columns they write (newer ones add e.g. effNPP).
    chunks = pd.read_csv(res_path, sep="\t", skiprows=2, header=0, encoding="latin-1",
                         usecols=lambda c: c.strip() in TREE_FIELDS, chunksize=chunksize)  # fmt: skip
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip()
        if missing := set(TREE_FIELDS) - set(chunk.columns):
            raise ValueError(f"{res_path} lacks FORMIND columns {sorted(missing)}")
        chunk = chunk.apply(pd.to_numeric, errors="coerce").dropna(subset=["Time", "X", "Y", "H"])
        times = chunk.Time.unique()  # before cropping: a snapshot may have no trees in the crop
        if size is not None:
            chunk = chunk[(chunk.X >= x0) & (chunk.X < x0 + size) & (chunk.Y >= y0)
                          & (chunk.Y < y0 + size)]  # fmt: skip
        for snap in times:
            rows = chunk[chunk.Time == snap]
            for year in outputs:
                kept = best.get(year)
                if kept is None or abs(snap - year) < abs(kept[0] - year):
                    best[year] = [snap, [rows]]
                elif snap == kept[0]:
                    kept[1].append(rows)
    used = {}
    for year, out_csv in outputs.items():
        if year not in best:
            raise ValueError(f"no trees in {res_path}")
        snap, blocks = best[year]
        t = pd.concat(blocks)
        out = pd.DataFrame({"x": t.X - x0, "y": t.Y - y0, "h": t.H, "crown_base": t.H * (1 - t.CLP),
                            "crown_radius": t.CR, "shape": shape,
                            "leaf_area": t.LAI * np.pi * t.CR**2, "agb": t.BT})  # fmt: skip
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        used[year] = float(snap)
    return used


def make_scene(scenario, out_dir):
    """Write the HELIOS++ scene and survey, and the truth, for *scenario*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(scenario.seed)
    size, voxel = scenario.size, scenario.voxel
    if scenario.trees is not None:
        trees = read_tree_list(scenario.trees, size)
        occ, origin, zmin, pad_grid = voxelise(trees, scenario.terrain, size, voxel, with_pad=True)
    else:
        trees = make_trees(scenario.forest, size, rng, placement=scenario.placement,
                           understorey=scenario.understorey)  # fmt: skip
        occ, origin, zmin = voxelise(trees, scenario.terrain, size, voxel)
    canopy = {k: v[~trees["low"]] for k, v in trees.items()}
    g = np.arange(int(size) + 1, dtype=float)
    zgrid = terrain_height(scenario.terrain, *np.meshgrid(g, g))  # 1 m mesh, [y, x]
    # Crowns are placed relative to the ground at the stem; on cliffs and in
    # ravines they can reach into the terrain: drop voxels below the mesh.
    centres = origin + (np.arange(occ.shape[0]) + 0.5) * voxel
    ground = mesh_height(zgrid, *np.meshgrid(centres, centres, indexing="ij"))
    cz = zmin + (np.arange(occ.shape[2]) + 0.5) * voxel
    occ &= cz[None, None, :] > ground[:, :, None]
    if scenario.trees is not None:
        pad = pad_grid[np.nonzero(occ)] * scenario.pad_scale
        pai = float(pad.sum() * voxel**3 / (size * size))
        pad_meta = {"mean": float(pad.mean()), "max": float(pad.max())}
    elif scenario.understorey:
        pai = FOREST_TYPES[scenario.forest]["pai"] * scenario.pad_scale
        # The canopy keeps its stand PAI over its own voxels, the understorey gets
        # UNDERSTOREY["pai"] over the voxels only it fills.
        upper, _, upper_zmin = voxelise(canopy, scenario.terrain, size, voxel)
        if upper.shape != occ.shape or upper_zmin != zmin:  # the shrubs are never the tallest
            raise RuntimeError("canopy and understorey voxel grids differ")
        upper &= occ
        lower = occ & ~upper
        pad_canopy = crown_pad(pai, upper, size, voxel)
        pad_low = crown_pad(UNDERSTOREY["pai"] * scenario.pad_scale, lower, size, voxel)
        pad_grid = np.where(upper, pad_canopy, pad_low)
        pad = pad_grid[np.nonzero(occ)]
        pad_meta = {"canopy": pad_canopy, "understorey": pad_low}
    else:
        pai = FOREST_TYPES[scenario.forest]["pai"] * scenario.pad_scale
        pad = crown_pad(pai, occ, size, voxel)
        pad_meta = pad

    truth = voxel_truth(occ, origin, zmin, voxel, zgrid, scenario.margin, scenario.core)
    m0, nc = round(scenario.margin), round(scenario.core)
    in_core = ((canopy["x"] >= m0) & (canopy["x"] < m0 + nc)
               & (canopy["y"] >= m0) & (canopy["y"] < m0 + nc))  # fmt: skip
    biomass = {}
    if "agb" in canopy:  # tree biomass (t) and plot biomass (t/ha) of the core
        biomass = {"tree_agb": canopy["agb"][in_core],
                   "agb_t_ha": np.float64(canopy["agb"][in_core].sum() / (nc * nc / 1e4))}  # fmt: skip
    np.savez_compressed(
        out / "truth.npz", **truth, x0=X0 + m0, y1=Y0 + m0 + nc,
        tree_x=X0 + canopy["x"][in_core], tree_y=Y0 + canopy["y"][in_core],
        tree_h=canopy["h"][in_core], **biomass,
    )  # fmt: skip
    prf, scan_freq = write_helios_scene(
        out, name=scenario.name, zgrid=zgrid, voxels=np.nonzero(occ), pad=pad, origin=origin,
        zmin=zmin, shape=occ.shape, voxel=voxel, xy0=(X0, Y0), density=scenario.density,
        max_returns=scenario.max_returns, seed=scenario.seed,
        beam_divergence=scenario.beam_divergence,
    )  # fmt: skip
    meta = {"scenario": asdict(scenario), "crs": CRS, "year": YEAR, "trees": int(canopy["x"].size),
            "shrubs": int(trees["low"].sum()),
            "voxels": int(occ.sum()), "pai": pai, "pad": pad_meta, "pulse_rate_hz": prf,
            "scan_freq_hz": scan_freq}  # fmt: skip
    (out / "scene.json").write_text(json.dumps(meta, indent=2))
    return meta


def survey_settings(density):
    """Pulse rate and mirror frequency giving *density* pulses/m2.

    density = rate / (speed * swath), with swath = 2 * height * tan(half
    angle); the mirror frequency makes the line spacing equal the point spacing.
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


def run_tools(scene_dir, tools=("alsdb", "lasr"), *, lastools_dir=None, repo=None,
              timeout=3600):  # fmt: skip
    """Process the scene's LAZ with each tool; save core-window rasters to ``preds.npz``.

    A failing or hung external tool (lasR, LAStools; killed after *timeout*
    seconds) is recorded under ``"failed"`` in ``timings.json`` and the
    scene is scored without it. alsdb errors still raise.
    """
    import rasterio
    import zarr
    from rasterio.windows import from_bounds

    scene_dir = Path(scene_dir)
    crs, year = _scene_crs_year(scene_dir)
    truth = np.load(scene_dir / "truth.npz")
    bounds, n = _window(truth)
    laz = scene_dir / "laz"
    preds, timings, failed = {}, {}, {}

    def tif(path):
        with rasterio.open(path) as ds:
            a = ds.read(1, window=from_bounds(*bounds, transform=ds.transform), masked=True)
        return np.asarray(a.astype(np.float32).filled(np.nan))

    if "alsdb" in tools:
        from alsdb.core.alsdatabase import ALSDatabase
        from alsdb.products.chm import compute_all
        from alsdb.storage.zarr_store import ALSZarrStore

        uri = scene_dir / "alsdb" / "tiledb"
        shutil.rmtree(scene_dir / "alsdb", ignore_errors=True)
        db = ALSDatabase(storage_type="local", uri=str(uri))
        db.create(crs=crs)
        db.ingest_many(sorted(laz.glob("*.laz")))
        for method, options in ALSDB_METHODS.items():
            store = scene_dir / "alsdb" / f"{method}.zarr"
            t = time.perf_counter()
            compute_all(db, ALSZarrStore(str(store)), resolution=1.0, year=year, bbox=bounds,
                        tile_size=float(n), quality=False, **options)  # fmt: skip
            timings[f"alsdb_{method}"] = time.perf_counter() - t
            g = zarr.open(str(store), mode="r")["1m"]
            preds[f"alsdb_{method}"] = np.asarray(g["chm"][0, :n, :n], np.float32)
            preds[f"alsdb_dtm_{method}"] = np.asarray(g["dtm"][0, :n, :n], np.float32)

    def external(name, work):
        t = time.perf_counter()
        try:
            work()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            # The tool's own error output, when it was captured.
            output = getattr(exc, "stderr", None) or getattr(exc, "output", None) or b""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            detail = output.strip().splitlines()[-1] if output.strip() else ""
            failed[name] = (str(exc) + "\n" + output)[-4000:]
            print(f"{scene_dir.name}: {name} failed, scored without it: "
                  f"{str(exc).splitlines()[0]} {detail}", flush=True)  # fmt: skip
            return
        timings[name] = time.perf_counter() - t

    if "lasr" in tools:
        out = scene_dir / "lasr"
        shutil.rmtree(out, ignore_errors=True)

        def lasr():
            subprocess.run(_pixi("lasr", "python", "-m", "alsdb.benchmark", "lasr", laz, out,
                                 "--workers", 1, "--veg", 5, repo=repo),
                           check=True, capture_output=True, timeout=timeout)  # fmt: skip
            for method in METHODS:
                preds[f"lasr_{method}"] = tif(out / method / f"SYN_{year}_scene.tif")

        external("lasr", lasr)
    if "lastools" in tools:
        from alsdb.benchmark import lastools

        out = scene_dir / "lastools"
        shutil.rmtree(out, ignore_errors=True)

        def lastools_spikefree():
            lastools.run(laz, out, veg_classes=(5,), lastools_dir=lastools_dir,
                         products=("spikefree",), timeout=timeout)  # fmt: skip
            preds["lastools_spikefree"] = tif(out / "spikefree" / f"SYN_{year}_scene.tif")

        external("lastools", lastools_spikefree)
    np.savez_compressed(scene_dir / "preds.npz", **preds)
    (scene_dir / "timings.json").write_text(json.dumps({**timings, "failed": failed}, indent=2))
    return preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _peaks(a, rows, cols, k=1):
    return np.array(
        [np.nanmax(a[r - k : r + k + 1, c - k : c + k + 1]) for r, c in zip(rows, cols)]
    )


def score(scene_dir, truth_file="truth.npz"):
    """One row per CHM output: errors against both truths, cover, mean height, tree tops.

    Cells where the truth is missing (a real-data reference can have gaps)
    are left out. When the truth carries ``tree_h_field`` (e.g. TLS heights),
    tree tops are also scored against those. *truth_file* scores against
    another truth in the scene folder (e.g. ``truth_nadir.npz``).
    """
    scene_dir = Path(scene_dir)
    truth = np.load(scene_dir / truth_file)
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
        # the tree-list path as "tree_list": "trees" is the count of scored tree tops
        setup = {("tree_list" if k == "trees" else k): v for k, v in meta["scenario"].items()}
        row = {"scenario": Path(scene_dir).name, **setup, "source": name,
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
