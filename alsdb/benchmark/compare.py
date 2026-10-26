"""Matched comparison of alsdb raster stores against reference raster tiles.

Every reference (LAStools, lasR, or any folder of GeoTIFF tiles on the same
pixel lattice) is scored against every candidate store on identical pixels:
inside tiles that all candidates have finished, where all candidates and the
reference are valid. Comparing populations that differ in coverage confounds
"different method" with "different area", so nothing is scored otherwise.

    from alsdb.benchmark import compare
    result = compare.compare(
        {"v3": "alsdb_out_v3/zarr_25831.zarr"},
        {"lasr_pitfree": "lasr/pitfree", "lastools_spikefree": "lastools/spikefree"},
        year=2016,
    )
    print(result.summary())
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEIGHT_BINS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, math.inf)


class Moments:
    """Streaming bias/MAE/RMSE/R2/r between a reference x and a candidate y."""

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = self.sad = 0.0

    def update(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()
        self.sad += np.abs(y - x).sum()

    def stats(self):
        """bias, MAE, RMSE, R2 (of y as predictor of x) and Pearson r; NaN when empty."""
        if not self.n:
            return {"n": 0, "bias": math.nan, "mae": math.nan, "rmse": math.nan,
                    "r2": math.nan, "r": math.nan}  # fmt: skip
        n = self.n
        mx, my = self.sx / n, self.sy / n
        mse = (self.syy - 2 * self.sxy + self.sxx) / n
        vx, vy = self.sxx / n - mx * mx, self.syy / n - my * my
        cov = self.sxy / n - mx * my
        return {
            "n": n,
            "bias": my - mx,
            "mae": self.sad / n,
            "rmse": math.sqrt(max(mse, 0.0)),
            "r2": 1 - mse / vx if vx > 0 else math.nan,
            "r": cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else math.nan,
        }


class Reservoir:
    """Uniform random sample of at most *cap* values from a stream of any length.

    Each value gets a random priority and the *cap* lowest are kept, so the
    sample does not depend on tile order and memory stays bounded.
    """

    def __init__(self, cap, rng):
        self.cap, self.rng = cap, rng
        self.values = np.empty(0, np.float32)
        self.keys = np.empty(0, np.float64)

    def add(self, values):
        if not values.size:
            return
        self.values = np.concatenate((self.values, values.astype(np.float32)))
        self.keys = np.concatenate((self.keys, self.rng.random(values.size)))
        if self.values.size > self.cap:
            keep = np.argpartition(self.keys, self.cap)[: self.cap]
            self.values, self.keys = self.values[keep], self.keys[keep]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    name: str
    array: object  # zarr array (time, y, x)
    t: int
    done: np.ndarray  # (tile_row, tile_col) finished
    tile_px: int


def _open_candidate(name, path, variable, resolution, year):
    import zarr

    from alsdb.storage.zarr_store import ALSZarrStore, _res_str

    key = _res_str(resolution)
    group = zarr.open(str(path), mode="r")[key]
    attrs = dict(group.attrs)
    times = group["time"][:]
    if year not in times:
        raise ValueError(f"{name}: year {year} not in {path}")
    array = group[variable]
    run = dict(array.attrs).get("processing_runs", {}).get(str(year))
    ny, nx = int(attrs["ny"]), int(attrs["nx"])
    if run is None:  # legacy store without provenance: assume complete
        tile_px = array.chunks[1]
        done = np.ones((-(-ny // tile_px), -(-nx // tile_px)), bool)
    else:
        tile_px = round(run["configuration"]["tile_size"] / resolution)
        shape = (-(-ny // tile_px), -(-nx // tile_px))
        if run["status"] == "complete":
            done = np.ones(shape, bool)
        else:
            flat = np.zeros(shape[0] * shape[1], bool)
            finished = ALSZarrStore(str(path), mode="r").completed_tiles(variable, resolution, year)
            flat[[i for i in finished if i < flat.size]] = True
            done = flat.reshape(shape)
    grid = (float(attrs["x_origin"]), float(attrs["y_origin"]), ny, nx)
    return _Candidate(name, array, int(np.flatnonzero(times == year)[0]), done, tile_px), grid


def reference_index(ref_dir):
    """Bounds/resolution/nodata of every GeoTIFF in *ref_dir*, cached on disk.

    Opening thousands of files over a network mount is slow, so the index is
    written to ``<ref_dir>/.alsdb_index.json`` and reused while each file's
    size and modification time are unchanged.
    """
    import rasterio

    ref_dir = Path(ref_dir)
    cache_path = ref_dir / ".alsdb_index.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    index, changed = {}, False
    for tif in sorted(ref_dir.glob("*.tif")):
        stat = tif.stat()
        stamp = [stat.st_size, stat.st_mtime]
        entry = cache.get(tif.name)
        if entry is None or entry["stamp"] != stamp:
            with rasterio.open(tif) as ds:
                entry = {
                    "stamp": stamp,
                    "bounds": list(ds.bounds),
                    "res": [ds.transform.a, -ds.transform.e],
                    "nodata": ds.nodata,
                }
            changed = True
        index[tif.name] = entry
    if changed or set(index) != set(cache):
        try:
            cache_path.write_text(json.dumps(index))
        except OSError:
            pass  # read-only reference folder: just don't cache
    return {ref_dir / name: entry for name, entry in index.items()}


def read_reference(index, bounds, resolution):
    """Mosaic of reference tiles over *bounds* on the candidate lattice (NaN outside)."""
    import rasterio
    from rasterio.windows import from_bounds

    x0, y0, x1, y1 = bounds
    ny, nx = round((y1 - y0) / resolution), round((x1 - x0) / resolution)
    out = np.full((ny, nx), np.nan, np.float32)
    for path, entry in index.items():
        bx0, by0, bx1, by1 = entry["bounds"]
        if bx0 >= x1 or bx1 <= x0 or by0 >= y1 or by1 <= y0:
            continue
        if not np.allclose(entry["res"], resolution):
            raise ValueError(f"{path}: resolution {entry['res']} != {resolution}")
        offset = (bx0 - x0) / resolution
        if abs(offset - round(offset)) > 1e-6:
            raise ValueError(f"{path}: not aligned with the candidate pixel lattice")
        with rasterio.open(path) as ds:
            window = from_bounds(x0, y0, x1, y1, transform=ds.transform)
            data = ds.read(1, window=window.round_offsets().round_lengths(), boundless=True,
                           fill_value=np.nan if ds.nodata is None else ds.nodata)  # fmt: skip
            data = data.astype(np.float32)
            if ds.nodata is not None:
                data[data == ds.nodata] = np.nan
        fill = np.isnan(out)
        out[fill] = data[:ny, :nx][fill]
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    candidates: list
    references: list
    height_bins: tuple
    overall: dict = field(default_factory=dict)  # (ref, cand) -> Moments
    binned: dict = field(default_factory=dict)  # (ref, cand, bin) -> Moments
    samples: dict = field(default_factory=dict)  # (ref, cand, bin) -> Reservoir
    maps: dict = field(default_factory=dict)  # (ref, cand) -> block-mean residual
    tiles_compared: int = 0

    def stats(self):
        """Nested plain-dict statistics: stats()[ref][cand] = {"all": ..., "bins": [...]}."""
        out = {}
        for ref in self.references:
            out[ref] = {}
            for cand in self.candidates:
                out[ref][cand] = {
                    "all": self.overall[(ref, cand)].stats(),
                    "bins": [
                        {"lo": lo, "hi": hi, **self.binned[(ref, cand, b)].stats()}
                        for b, (lo, hi) in enumerate(zip(self.height_bins, self.height_bins[1:]))
                    ],
                }
        return out

    def summary(self):
        lines = [
            f"Tiles compared: {self.tiles_compared} (finished in every candidate, with valid pixels)"
        ]
        for ref, by_cand in self.stats().items():
            lines.append(f"\nvs {ref} (reference = x, candidate - reference):")
            for cand, s in by_cand.items():
                a = s["all"]
                lines.append(
                    f"  {cand:12s} n={a['n']:,}  bias={a['bias']:+.3f}  MAE={a['mae']:.3f}  "
                    f"RMSE={a['rmse']:.3f}  R2={a['r2']:.3f}  r={a['r']:.3f}"
                )
            lines.append("  bias by reference height: " + "   ".join(
                f"{b['lo']:g}-{b['hi']:g}" for b in next(iter(by_cand.values()))["bins"]))  # fmt: skip
            for cand, s in by_cand.items():
                lines.append(f"  {cand:12s} " + "  ".join(
                    f"{b['bias']:+.3f}" if b["n"] else "   n/a" for b in s["bins"]))  # fmt: skip
        return "\n".join(lines)

    def to_npz(self, path):
        """Save stats (JSON), residual samples and maps for plotting."""
        data = {"stats_json": np.array(json.dumps(self.stats(), default=float))}
        for (ref, cand, b), res in self.samples.items():
            data[f"res__{ref}__{cand}__{b}"] = res.values
        for (ref, cand), m in self.maps.items():
            data[f"map__{ref}__{cand}"] = m
        np.savez_compressed(path, **data)


def compare(
    candidates,
    references,
    *,
    year,
    resolution=1.0,
    variable="chm",
    height_bins=HEIGHT_BINS,
    map_block=100,
    sample_cap=200_000,
    limit=None,
    seed=0,
    progress=False,
):
    """Score each alsdb store in *candidates* against each folder in *references*.

    *candidates* and *references* map display names to paths. Candidate stores
    must share one grid. Height classes use the reference value. ``limit``
    stops after that many candidate tiles (smoke test). Residual samples (up to
    *sample_cap* per reference/candidate/class) and *map_block*-pixel block
    means of the residual are kept for plotting.
    """
    if not candidates or not references:
        raise ValueError("Need at least one candidate store and one reference folder")
    opened, grids = [], set()
    for name, path in candidates.items():
        cand, grid = _open_candidate(name, path, variable, resolution, year)
        opened.append(cand)
        grids.add(grid)
    if len(grids) != 1:
        raise ValueError(f"Candidate stores are on different grids: {sorted(grids)}")
    x_origin, y_origin, ny, nx = grids.pop()
    indexes = {name: reference_index(path) for name, path in references.items()}
    result = Comparison(list(candidates), list(references), tuple(height_bins))
    rng = np.random.default_rng(seed)
    nbins = len(height_bins) - 1
    mshape = (-(-ny // map_block), -(-nx // map_block))
    sums, counts = {}, {}
    for ref in references:
        for cand in candidates:
            result.overall[(ref, cand)] = Moments()
            sums[(ref, cand)], counts[(ref, cand)] = np.zeros(mshape), np.zeros(mshape)
            for b in range(nbins):
                result.binned[(ref, cand, b)] = Moments()
                result.samples[(ref, cand, b)] = Reservoir(sample_cap, rng)

    tile_px = opened[0].tile_px
    for row0 in range(0, ny, tile_px):
        for col0 in range(0, nx, tile_px):
            row1, col1 = min(row0 + tile_px, ny), min(col0 + tile_px, nx)
            rows, cols = np.arange(row0, row1), np.arange(col0, col1)
            finished = np.ones((rows.size, cols.size), bool)
            for c in opened:
                finished &= c.done[np.ix_(rows // c.tile_px, cols // c.tile_px)]
            if not finished.any():
                continue
            values = {
                c.name: np.asarray(c.array[c.t, row0:row1, col0:col1], np.float32) for c in opened
            }
            valid = finished.copy()
            for v in values.values():
                valid &= np.isfinite(v)
            if not valid.any():  # e.g. bounding-box corners outside the survey
                continue
            bounds = (
                x_origin + col0 * resolution,
                y_origin - row1 * resolution,
                x_origin + col1 * resolution,
                y_origin - row0 * resolution,
            )
            brow = np.broadcast_to((rows // map_block)[:, None], valid.shape)
            bcol = np.broadcast_to((cols // map_block)[None, :], valid.shape)
            for ref, index in indexes.items():
                reference = read_reference(index, bounds, resolution)
                m = valid & np.isfinite(reference)
                if not m.any():
                    continue
                x = reference[m]
                bin_of = np.digitize(x, height_bins[1:-1])
                for cand, v in values.items():
                    y = v[m]
                    result.overall[(ref, cand)].update(x, y)
                    np.add.at(sums[(ref, cand)], (brow[m], bcol[m]), y - x)
                    np.add.at(counts[(ref, cand)], (brow[m], bcol[m]), 1)
                    for b in range(nbins):
                        sel = bin_of == b
                        result.binned[(ref, cand, b)].update(x[sel], y[sel])
                        result.samples[(ref, cand, b)].add((y - x)[sel])
            result.tiles_compared += 1
            if progress and result.tiles_compared % 50 == 0:
                print(f"  {result.tiles_compared} tiles", end="\r", flush=True)
            if limit and result.tiles_compared >= limit:
                break
        else:
            continue
        break
    with np.errstate(invalid="ignore", divide="ignore"):
        for key, total in sums.items():
            result.maps[key] = (total / counts[key]).astype(np.float32)
    return result
