# Benchmarking against reference implementations

`alsdb.benchmark` produces reference CHMs with independent software and scores
alsdb outputs against them on identical pixels. External tools run as separate
processes; alsdb never installs or links them.

| Module | What it does | Needs |
|---|---|---|
| `alsdb.benchmark.lasr` | lasR pitfree + spike-free tiles from LAS/LAZ | `pixi install -e lasr`, then `pixi run -e lasr install-lasr` once |
| `alsdb.benchmark.lastools` | LAStools spike-free + binned-max tiles from LAS/LAZ; `rasterize_lastools` in-memory adapter | licensed LAStools executables |
| `alsdb.benchmark.compare` | matched scoring of alsdb stores against reference tile folders | – |
| `alsdb.benchmark.points` | every alsdb method on one prepared point set | – |

## Reference tiles

Both runners write one GeoTIFF per input tile, skip tiles that already exist
(rerun to resume), log per-tile parameters in `params.csv` and the tool
version and settings in `runs.json`.

```bash
# lasR: own ground + water normalization, 50 m buffer from neighbouring tiles
pixi run -e lasr python -m alsdb.benchmark lasr <laz_dir> <out_dir> --workers 8
# LAStools: lasheight + las2dem -spike_free / lascanopy -max, per tile without buffer
python -m alsdb.benchmark lastools <laz_dir> <out_dir> --lastools-dir /path/to/lastools/bin --workers 8
```

Both mirror alsdb surface algorithm 3, so differences reflect implementation,
not settings: noise (7, 18) dropped, surface = `--veg` classes (default 3,4,5)
plus ground and water (2, 9), and `"auto"` freeze distance / pitfree edge
computed per tile with alsdb's rule (three pulse spacings of surface first
returns). Use `--tiles N` for a pilot and `--veg 20` for surveys with
vegetation in class 20 (e.g. DE-Hai).

Returns outside the surface classes are ignored. On PNOA this includes
overlap-strip returns (class 12, read by PDAL as class 0 with the Overlap
flag), often most of the points. LAZ header point counts include them, so the
runners measure density from the classified returns instead.

## Scoring

```python
from alsdb.benchmark.compare import compare

result = compare(
    {"v3": "alsdb_out_v3/zarr_25831.zarr"},
    {"lasr_pitfree": "lasr/pitfree", "lastools_spikefree": "lastools/spikefree"},
    year=2016,
)
print(result.summary())
result.to_npz("comparison.npz")  # residual samples and block-mean maps for plots
```

or `python -m alsdb.benchmark compare --store v3=... --ref lasr_pitfree=... --year 2016`.

Only tiles finished in every candidate store are used, and only pixels valid in
all candidates and the reference, so a partly processed run can be scored
without mixing coverage differences into the result. Reference tiles may use
any tiling but must share the candidates' resolution and pixel lattice.
Statistics are bias, MAE, RMSE, R² and r overall and per reference height class.
The first run indexes each reference folder into `.alsdb_index.json`.

## What agreement to expect

On a Barcelona PNOA tile (inner 1 km², same auto parameters), alsdb pitfree
matched lasR on 99.3% of cells to within 1 cm; spike-free was 80% identical,
alsdb +0.18 m on average (lasR seeds freezing with the highest point of a
coarse grid; alsdb freezes at each insertion). Agreement between
implementations is not accuracy: field data (e.g. NFI plots) remain necessary.
