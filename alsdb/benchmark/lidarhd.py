"""A stratified random sample of forest blocks from IGN Lidar HD, for the benchmark.

IGN's Lidar HD covers France with classified ALS (about 10 pulses/m2 or more);
tiles whose classification was corrected by hand (``ST_AUTO_MANUEL``) give a
ground reference of the quality of the Chapitre Petit Buech data, which is
itself Lidar HD (F. Fischer, pers. comm., suggested using it systematically).
This module draws 2 x 2 km blocks (four 1 km tiles) from mainland France and
Corsica so that every combination of slope class and leaf type is
represented, then picks scored areas inside each block. The benchmark then
treats each area like a ULS area (:mod:`alsdb.benchmark.uls`, ``thinned``
variant): pulses thinned to lower densities against the full-density
reference.

Steps (each writes a CSV; the draw is reproducible from the seed):

1. :func:`build_frame` - every 2 km block of the Lambert-93 grid with its
   tree cover (ESA WorldCover 2021, read at ~80 m) and mean slope
   (Copernicus DEM GLO-30, read at ~60 m). Both are coarse and only sort
   blocks into strata; the scenes measure terrain and canopy from the lidar.
2. :func:`draw` - for each slope class, eligible blocks in random order; a
   block is accepted when its leaf type (IGN BD Foret v2: broadleaf, conifer
   or mixed by area of typed forest) still has room in that slope class, it
   lies at least *min_distance* from the blocks already accepted, and all
   four tiles are published with a hand-corrected classification. Strata
   with too few eligible blocks are reported, not filled from others.
3. :func:`choose_cores` - inside each block, *n_cores* random 250 m cells
   with high tree cover (WorldCover at 10 m), each holding a 200 m scored
   core and its 15 m margin.
4. :func:`download` - the tiles of the selected blocks (run on a machine
   with internet access, e.g. the cluster).

Sources (all open):

- IGN Lidar HD, classified point clouds (COPC), Licence Ouverte Etalab 2.0,
  https://geoservices.ign.fr/lidarhd; tile metadata and links from the
  STAC catalogue https://api.stac.teledetection.fr (collection ``lidarhd``).
- IGN BD Foret v2 (Licence Ouverte Etalab 2.0), Geoplateforme WFS layer
  ``LANDCOVER.FORESTINVENTORY.V2:formation_vegetale``; IGN ADMIN EXPRESS
  regions (Etalab 2.0) for the outline of France.
- ESA WorldCover 10 m 2021 v200, Zanaga et al. (2022),
  https://doi.org/10.5281/zenodo.7254221 (CC BY 4.0).
- Copernicus DEM GLO-30, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and
  Space GmbH 2014-2018, provided under COPERNICUS by the European Union and
  ESA; all rights reserved.

    python -m alsdb.benchmark lidarhd-frame <frame.csv>
    python -m alsdb.benchmark lidarhd-select <frame.csv> <selection.csv>
    python -m alsdb.benchmark lidarhd-download <selection.csv> <data_dir>
"""

from __future__ import annotations

import csv
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BLOCK = 2000.0  # m, 2 x 2 Lidar HD tiles
TILE = 1000.0
GRID = (90_000.0, 6_030_000.0, 1_250_000.0, 7_130_000.0)  # Lambert-93 extent of the frame
LONLAT = (-5.5, 41.2, 9.8, 51.2)  # the same in degrees (lon, lat)
CRS = "EPSG:2154"
SLOPE_BREAKS = (0.0, 10.0, 25.0, 90.0)  # degrees, block mean
SLOPE_CLASSES = ("gentle", "moderate", "steep")
LEAF_TYPES = ("broadleaf", "conifer", "mixed")
LEAF_SHARE = 0.75  # of typed forest area: broadleaf or conifer above this, else mixed
MIN_TREE_COVER = 0.6  # WorldCover tree share of the block
MIN_FOREST = 0.6  # BD Foret forest share of the block
CELL = 250.0  # m, candidate cells for scored cores
CORE, MARGIN = 200.0, 15.0
MIN_CORE_TREE_COVER = 0.8
CLASSIFIED = "ST_AUTO_MANUEL"  # automatic, then corrected by hand

WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")  # fmt: skip
COPDEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
          "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif")  # fmt: skip
STAC = "https://api.stac.teledetection.fr/collections/lidarhd/items"
STAC_SEARCH = "https://api.stac.teledetection.fr/search"
INDEX_FIELDS = ("x0", "y0", "method", "acquired", "edition", "id")
WFS = "https://data.geopf.fr/wfs/ows"
BDFORET = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
USER_AGENT = "alsdb-benchmark (https://github.com/simonbesnard1/alsdb)"

FRAME_FIELDS = ("block", "x0", "y0", "france_share", "land_share", "tree_share", "slope_mean",
                "slope_class")  # fmt: skip
ADMIN = "ADMINEXPRESS-COG-CARTO.LATEST:region"  # IGN ADMIN EXPRESS (Etalab 2.0)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def _shape():
    return (round((GRID[3] - GRID[1]) / BLOCK), round((GRID[2] - GRID[0]) / BLOCK))


def _block_index(lon, lat):
    """Flat block index (or -1 outside the grid) of points given in degrees."""
    import pyproj

    # One transformer per call: pyproj transformers are not thread-safe.
    x, y = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    ny, nx = _shape()
    col = np.floor((x - GRID[0]) / BLOCK).astype(np.int64)
    row = np.floor((y - GRID[1]) / BLOCK).astype(np.int64)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    return np.where(ok, row * nx + col, -1)


def _open(url, overview_level):
    import rasterio

    try:
        return rasterio.open("/vsicurl/" + url, overview_level=overview_level)
    except rasterio.errors.RasterioIOError:  # no tile (sea)
        return None


def _pixel_lonlat(ds):
    rows, cols = np.mgrid[0 : ds.height, 0 : ds.width]
    lon = ds.transform.c + (cols + 0.5) * ds.transform.a
    lat = ds.transform.f + (rows + 0.5) * ds.transform.e
    return lon.ravel(), lat.ravel()


def _worldcover_tile(name):
    """(all, land, tree pixels) per block from one WorldCover tile at ~80 m."""
    n = int(np.prod(_shape()))
    total, land, tree = np.zeros(n), np.zeros(n), np.zeros(n)
    ds = _open(WORLDCOVER.format(tile=name), overview_level=2)  # 1/8: ~80 m
    if ds is None:
        return total, land, tree
    with ds:
        a = ds.read(1).ravel()
        lon, lat = _pixel_lonlat(ds)
    idx = _block_index(lon, lat)
    ok = idx >= 0
    a, idx = a[ok], idx[ok]
    is_land = (a != 0) & (a != 80)  # 0 no data (sea), 80 permanent water
    total += np.bincount(idx, minlength=n)
    land += np.bincount(idx, weights=is_land.astype(float), minlength=n)
    tree += np.bincount(idx, weights=(a == 10).astype(float), minlength=n)
    return total, land, tree


def _dem_tile(lat0, lon0):
    """(slope sum, count) per block from one Copernicus DEM tile at ~60 m."""
    n = int(np.prod(_shape()))
    lat = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
    lon = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    total, count = np.zeros(n), np.zeros(n)
    ds = _open(COPDEM.format(lat=lat, lon=lon), overview_level=0)  # 1/2: ~60 m
    if ds is None:
        return total, count
    with ds:
        z = ds.read(1).astype(float)
        plon, plat = _pixel_lonlat(ds)
        dy = abs(ds.transform.e) * 110_574.0
        lats = plat.reshape(z.shape)[:, 0]
        dx = ds.transform.a * 111_320.0 * np.cos(np.radians(lats))[:, None]
    gy, gx = np.gradient(z)
    slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy))).ravel()
    land = z.ravel() > 0  # sea is 0 in GLO-30
    idx = _block_index(plon[land], plat[land])
    ok = idx >= 0
    total += np.bincount(idx[ok], weights=slope[land][ok], minlength=n)
    count += np.bincount(idx[ok], minlength=n)
    return total, count


def _france_share(cell=250.0):
    """Share of every block inside France (IGN ADMIN EXPRESS regions), at *cell* m."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    page = _get_json(WFS, {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                           "TYPENAMES": ADMIN, "OUTPUTFORMAT": "application/json",
                           "SRSNAME": CRS, "COUNT": 1000})  # fmt: skip
    ny, nx = _shape()
    k = round(BLOCK / cell)
    grid = rasterize([(f["geometry"], 1) for f in page["features"]], out_shape=(ny * k, nx * k),
                     transform=from_origin(GRID[0], GRID[3], cell, cell), fill=0, dtype="uint8")  # fmt: skip
    share = grid.reshape(ny, k, nx, k).mean(axis=(1, 3))[::-1]  # rows south to north
    return share.ravel()


def tile_index(out_csv, *, page_size=1000, retries=4):
    """Every Lidar HD tile in the catalogue: origin, classification method, dates; write *out_csv*.

    One paged search (about 3 s per 1000 tiles); each tile's latest edition.
    """
    body = {"collections": ["lidarhd"], "limit": page_size,
            "fields": {"include": ["id", "properties.proj:bbox", "properties.lidarhd:procede_classement",
                                   "properties.lidarhd:date_debut_acquisition",
                                   "properties.lidarhd:date_edition"],
                       "exclude": ["assets", "links", "geometry"]}}  # fmt: skip
    tiles, pages = {}, 0
    while body is not None:
        request = urllib.request.Request(STAC_SEARCH, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": USER_AGENT})  # fmt: skip
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    page = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
                if attempt == retries - 1:
                    raise
                time.sleep(5 * 2**attempt)
        for item in page["features"]:
            props = item["properties"]
            bx = props.get("proj:bbox")
            if not bx:
                continue
            key = (round(bx[0] / TILE) * round(TILE), round(bx[1] / TILE) * round(TILE))
            row = {"x0": key[0], "y0": key[1], "method": props.get("lidarhd:procede_classement", ""),
                   "acquired": props.get("lidarhd:date_debut_acquisition", ""),
                   "edition": props.get("lidarhd:date_edition", ""), "id": item["id"]}  # fmt: skip
            if key not in tiles or row["edition"] > tiles[key]["edition"]:
                tiles[key] = row
        nxt = [link for link in page.get("links", []) if link.get("rel") == "next"]
        body = nxt[0]["body"] if nxt else None
        pages += 1
        if pages % 50 == 0:
            print(f"{pages} pages, {len(tiles):,} tiles", flush=True)
    _write_csv(out_csv, sorted(tiles.values(), key=lambda r: (r["y0"], r["x0"])), INDEX_FIELDS)
    return out_csv


def slope_class(slope):
    """Slope class name of a block mean slope (degrees)."""
    i = int(np.searchsorted(SLOPE_BREAKS, slope, side="right")) - 1
    return SLOPE_CLASSES[min(max(i, 0), len(SLOPE_CLASSES) - 1)]


def build_frame(out_csv, *, workers=4):
    """Tree cover and mean slope of every 2 km block of the grid; write *out_csv*."""
    lon0, lat0, lon1, lat1 = LONLAT
    wc = [f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
          for la in range(39, int(lat1) + 1, 3) for lo in range(-6, int(lon1) + 1, 3)]  # fmt: skip
    dem = [(la, lo) for la in range(math.floor(lat0), math.ceil(lat1))
           for lo in range(math.floor(lon0), math.ceil(lon1))]  # fmt: skip
    n = int(np.prod(_shape()))
    pixels, land, tree, total, count = (np.zeros(n) for _ in range(5))
    with ThreadPoolExecutor(workers) as pool:
        for i, (a, b, c) in enumerate(pool.map(_worldcover_tile, wc)):
            pixels += a
            land += b
            tree += c
            print(f"WorldCover {i + 1}/{len(wc)}", flush=True)
        for i, (a, b) in enumerate(pool.map(lambda t: _dem_tile(*t), dem)):
            total += a
            count += b
            if (i + 1) % 20 == 0 or i + 1 == len(dem):
                print(f"DEM {i + 1}/{len(dem)}", flush=True)
    france = _france_share()
    nx = _shape()[1]
    rows = []
    for k in np.flatnonzero(land > 0):
        r, c = divmod(int(k), nx)
        x0, y0 = GRID[0] + c * BLOCK, GRID[1] + r * BLOCK
        mean = total[k] / count[k] if count[k] else float("nan")
        rows.append({"block": f"{x0 / 1000:.0f}_{y0 / 1000:.0f}", "x0": x0, "y0": y0,
                     "france_share": round(float(france[k]), 3),
                     "land_share": round(land[k] / pixels[k], 3),
                     "tree_share": round(tree[k] / land[k], 3),
                     "slope_mean": round(mean, 2),
                     "slope_class": slope_class(mean) if np.isfinite(mean) else ""})  # fmt: skip
    _write_csv(out_csv, rows, FRAME_FIELDS)
    return out_csv


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------


def _get_json(url, params, *, retries=4):
    query = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(query, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2**attempt)
    return None


def leaf_code(code_tfv):
    """Leaf type of a BD Foret v2 formation (``code_tfv``), None when untyped.

    FF1/FO1 broadleaf (closed/open forest) and FP poplar; FF2/FO2 conifer;
    FF3/FO3 mixed; FF0 (closed forest without cover, e.g. clear-cut), heath
    (LA) and grassland are not typed forest.
    """
    if code_tfv.startswith(("FF1", "FO1", "FP")):
        return "broadleaf"
    if code_tfv.startswith(("FF2", "FO2")):
        return "conifer"
    if code_tfv.startswith(("FF3", "FO3")):
        return "mixed"
    return None


def leaf_shares(features, x0, y0, size=BLOCK, cell=10.0):
    """Area shares of the block (forest, broadleaf, conifer, mixed) from BD Foret features."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    n = round(size / cell)
    transform = from_origin(x0, y0 + size, cell, cell)
    codes = {None: 1, "broadleaf": 2, "conifer": 3, "mixed": 4}
    shapes = [(f["geometry"], codes[leaf_code(f["properties"].get("code_tfv") or "")])
              for f in features
              if (f["properties"].get("code_tfv") or "").startswith(("FF", "FO", "FP"))]  # fmt: skip
    if not shapes:
        return {"forest": 0.0, "broadleaf": 0.0, "conifer": 0.0, "mixed": 0.0}
    grid = rasterize(shapes, out_shape=(n, n), transform=transform, fill=0, dtype="uint8")
    out = {"forest": float(np.mean(grid > 0))}
    for name, code in codes.items():
        if name:
            out[name] = float(np.mean(grid == code))
    return out


def leaf_type(shares):
    """broadleaf / conifer when one holds LEAF_SHARE of the typed forest, else mixed."""
    typed = shares["broadleaf"] + shares["conifer"] + shares["mixed"]
    if typed <= 0:
        return None
    for name in ("broadleaf", "conifer"):
        if shares[name] / typed >= LEAF_SHARE:
            return name
    return "mixed"


def bdforet(x0, y0, size=BLOCK):
    """BD Foret v2 formations intersecting the block (GeoJSON features, Lambert-93)."""
    features, start = [], 0
    while True:
        page = _get_json(WFS, {
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": BDFORET,
            "OUTPUTFORMAT": "application/json", "SRSNAME": CRS, "COUNT": 1000,
            "STARTINDEX": start, "BBOX": f"{x0},{y0},{x0 + size},{y0 + size},{CRS}",
        })  # fmt: skip
        features += page["features"]
        if len(page["features"]) < 1000:
            return features
        start += 1000


def lidarhd_tiles(x0, y0, size=BLOCK):
    """{(x0, y0) of each 1 km tile: STAC item} of the block, the latest edition of each."""
    import pyproj

    to_deg = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_deg.transform([x0 + 100, x0 + size - 100], [y0 + 100, y0 + size - 100])
    page = _get_json(STAC, {"bbox": f"{min(lon)},{min(lat)},{max(lon)},{max(lat)}", "limit": 100})
    tiles = {}
    for item in page["features"]:
        bx = item["properties"].get("proj:bbox")
        if not bx:
            continue
        key = (round(bx[0]), round(bx[1]))
        if not (x0 <= key[0] < x0 + size and y0 <= key[1] < y0 + size):
            continue
        old = tiles.get(key)
        edition = item["properties"].get("lidarhd:date_edition", "")
        if old is None or edition > old["properties"].get("lidarhd:date_edition", ""):
            tiles[key] = item
    return tiles


def block_tiles_ok(tiles, x0, y0, size=BLOCK):
    """True when every 1 km tile of the block is published and corrected by hand."""
    n = round(size / TILE)
    needed = {(round(x0 + i * TILE), round(y0 + j * TILE)) for i in range(n) for j in range(n)}
    if set(tiles) != needed:
        return False
    return all(
        t["properties"].get("lidarhd:procede_classement") == CLASSIFIED for t in tiles.values()
    )


def corrected_blocks(index_csv):
    """{(x0, y0): first acquisition date} of the blocks whose four tiles are all
    published and corrected by hand, from :func:`tile_index`."""
    per_block = {}
    with open(index_csv) as handle:
        for r in csv.DictReader(handle):
            x, y = float(r["x0"]), float(r["y0"])
            key = (math.floor(x / BLOCK) * BLOCK, math.floor(y / BLOCK) * BLOCK)
            per_block.setdefault(key, []).append(r)
    n = round(BLOCK / TILE) ** 2
    return {key: min(r["acquired"] for r in rows) for key, rows in per_block.items()
            if len(rows) == n and all(r["method"] == CLASSIFIED for r in rows)}  # fmt: skip


def leaf_season(date):
    """ "leaf-on" for acquisitions from May to October, else "leaf-off" (deciduous trees)."""
    month = int(date[5:7]) if len(date) >= 7 else 0
    return "leaf-on" if 5 <= month <= 10 else "leaf-off"


def draw(frame_csv, out_csv, *, per_stratum=2, seed=0, min_distance=50_000.0, max_checks=400,
         index_csv=None, lookup_forest=bdforet, lookup_tiles=lidarhd_tiles):  # fmt: skip
    """Draw *per_stratum* blocks per (slope class, leaf type); write *out_csv*.

    Blocks with enough tree cover (and, with *index_csv* from
    :func:`tile_index`, only those whose four tiles are all corrected by hand)
    are visited in a random order (seeded) per slope class; each is checked
    against BD Foret (leaf type, forest share), the distance to blocks already
    accepted and the Lidar HD tiles. At most *max_checks* blocks are checked
    per slope class. Returns the selected rows and a per-stratum summary
    (also printed).
    """
    corrected = corrected_blocks(index_csv) if index_csv else None
    with open(frame_csv) as handle:
        frame = [r for r in csv.DictReader(handle)
                 if float(r["tree_share"]) >= MIN_TREE_COVER and float(r["land_share"]) >= 0.95
                 and float(r["france_share"]) >= 0.95 and r["slope_class"]
                 and (corrected is None or (float(r["x0"]), float(r["y0"])) in corrected)]  # fmt: skip
    rng = np.random.default_rng(seed)
    selected, summary = [], {}
    for slope in SLOPE_CLASSES:
        pool = [r for r in frame if r["slope_class"] == slope]
        order = rng.permutation(len(pool))
        counts = dict.fromkeys(LEAF_TYPES, 0)
        checked = 0
        for i in order:
            if all(c >= per_stratum for c in counts.values()) or checked >= max_checks:
                break
            r = pool[i]
            x0, y0 = float(r["x0"]), float(r["y0"])
            if any(math.hypot(x0 - s["x0"], y0 - s["y0"]) < min_distance for s in selected):
                continue
            checked += 1
            shares = leaf_shares(lookup_forest(x0, y0), x0, y0)
            print(f"{slope} {r['block']}: {leaf_type(shares)}, forest {shares['forest']:.2f}",
                  flush=True)  # fmt: skip
            leaf = leaf_type(shares)
            if leaf is None or shares["forest"] < MIN_FOREST or counts[leaf] >= per_stratum:
                continue
            tiles = lookup_tiles(x0, y0)
            if not block_tiles_ok(tiles, x0, y0):
                continue
            counts[leaf] += 1
            props = [t["properties"] for t in tiles.values()]
            selected.append({
                "block": r["block"], "x0": x0, "y0": y0, "slope_class": slope, "leaf_type": leaf,
                "stratum": f"{slope}_{leaf}", "slope_mean": float(r["slope_mean"]),
                "tree_share": float(r["tree_share"]),
                **{f"bdforet_{k}": round(v, 3) for k, v in shares.items()},
                "sensor": ";".join(sorted({s for p in props for s in p.get("lidarhd:capteur", [])})),
                "acquired": ";".join(sorted({p.get("lidarhd:date_debut_acquisition", "") for p in props})),
                "leaf_season": leaf_season(min(p.get("lidarhd:date_debut_acquisition", "") for p in props)),
                "mission": ";".join(sorted({p.get("lidarhd:code_mission", "") for p in props})),
                "classified_by": ";".join(sorted({p.get("lidarhd:moe_classement_manuel", "") for p in props})),
                "points": sum(int(p.get("pc:count", 0)) for p in props),
                "tiles": ";".join(tiles[k]["assets"]["data"]["href"] for k in sorted(tiles)),
            })  # fmt: skip
        for leaf in LEAF_TYPES:
            summary[f"{slope}_{leaf}"] = {"selected": counts[leaf], "eligible_pool": len(pool),
                                          "checked": checked}  # fmt: skip
    for stratum, s in summary.items():
        note = "" if s["selected"] >= per_stratum else "  <- fewer than asked"
        print(f"{stratum:20s} {s['selected']} selected ({s['checked']} of {s['eligible_pool']} "
              f"blocks checked){note}", flush=True)  # fmt: skip
    fields = list(dict.fromkeys(k for r in selected for k in r))
    _write_csv(out_csv, selected, fields)
    return selected, summary


# ---------------------------------------------------------------------------
# Scored cores and download
# ---------------------------------------------------------------------------


def choose_cores(x0, y0, *, n_cores=4, seed=0, tree_cover=None):
    """Lower-left corners of *n_cores* scored 200 m cores inside the block.

    The block is split into 250 m cells; a core (200 m plus its 15 m margin)
    sits in the middle of a cell, so cores never overlap and stay inside the
    downloaded tiles. Cells are drawn at random (seeded) among those with
    WorldCover tree cover of at least MIN_CORE_TREE_COVER (*tree_cover*:
    function (x0, y0, size) -> share, for tests).
    """
    tree_cover = tree_cover or _worldcover_share
    n = round(BLOCK / CELL)
    cells = [(x0 + i * CELL, y0 + j * CELL) for j in range(n) for i in range(n)]
    good = [c for c in cells if tree_cover(c[0], c[1], CELL) >= MIN_CORE_TREE_COVER]
    rng = np.random.default_rng(seed)
    picked = [
        good[i] for i in sorted(rng.choice(len(good), min(n_cores, len(good)), replace=False))
    ]
    offset = (CELL - CORE) / 2
    return [(cx + offset, cy + offset) for cx, cy in picked]


def _worldcover_share(x0, y0, size):
    """WorldCover 2021 tree share of a Lambert-93 square, at 10 m."""
    import pyproj
    import rasterio
    from rasterio.windows import from_bounds

    to_deg = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs, ys = np.meshgrid(
        x0 + (np.arange(25) + 0.5) * size / 25, y0 + (np.arange(25) + 0.5) * size / 25
    )
    lon, lat = to_deg.transform(xs.ravel(), ys.ravel())
    la, lo = 3 * math.floor(min(lat) / 3), 3 * math.floor(min(lon) / 3)
    name = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    with rasterio.open("/vsicurl/" + WORLDCOVER.format(tile=name)) as ds:
        window = from_bounds(min(lon), min(lat), max(lon), max(lat), ds.transform)
        a = ds.read(1, window=window, boundless=True)
        rows, cols = ds.index(lon, lat)
        r0, c0 = int(window.row_off), int(window.col_off)
        r = np.clip(np.asarray(rows) - r0, 0, a.shape[0] - 1)
        c = np.clip(np.asarray(cols) - c0, 0, a.shape[1] - 1)
    return float(np.mean(a[r, c] == 10))


def add_cores(selection_csv, *, n_cores=4, seed=0):
    """Add the scored cores (``x,y;x,y;...``) to every block of *selection_csv*."""
    with open(selection_csv) as handle:
        rows = list(csv.DictReader(handle))
    for i, r in enumerate(rows):
        cores = choose_cores(float(r["x0"]), float(r["y0"]), n_cores=n_cores, seed=seed + i)
        r["cores"] = ";".join(f"{x:.0f},{y:.0f}" for x, y in cores)
        r["year"] = r["acquired"].split(";")[0][:4]
    _write_csv(selection_csv, rows, list(rows[0]) if rows else ["cores"])
    return rows


def download(selection_csv, data_dir, *, retries=4):
    """Download the tiles of every selected block to ``<data_dir>/<block>/`` (resumable)."""
    with open(selection_csv) as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        folder = Path(data_dir) / r["block"]
        folder.mkdir(parents=True, exist_ok=True)
        for url in r["tiles"].split(";"):
            target = folder / url.rsplit("/", 1)[-1]
            if target.exists():
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            for attempt in range(retries):
                try:
                    with (
                        urllib.request.urlopen(request, timeout=600) as response,
                        open(partial, "wb") as out,
                    ):
                        while chunk := response.read(1 << 22):
                            out.write(chunk)
                    partial.replace(target)
                    break
                except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
                    if attempt == retries - 1:
                        raise
                    time.sleep(10 * 2**attempt)
            print(f"{r['block']}: {target.name}", flush=True)
    return data_dir


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
