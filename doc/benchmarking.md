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
| `alsdb.benchmark.synthetic` | HELIOS++ surveys of generated forests, scored against exact truth | `pixi install -e helios` |
| `alsdb.benchmark.uls` | real canopies from UAV laser scanning at ALS densities, thinned and via HELIOS++ | ULS tiles with GPS time; `helios` for the HELIOS++ variant |
| `alsdb.benchmark.timing` | compute time of all three tools on the same work | licensed LAStools, `lasr` |
| `alsdb.benchmark.decimated` | decimated copies of a dense ALS survey scored against its full density | the survey and its decimated copies |
| `alsdb.benchmark.sensitivity` | spike-free freeze distances on finished synthetic or ULS scenes | `spikefree` |
| `alsdb.benchmark.classification` | ground classifiers against reference classifications, by slope and cover | pyLASR for PTD (`pixi run -e spikefree install-pylasr` once) |

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

The lasR runner processes tiles in separate R worker processes, each tile with
its own auto parameters, reading only its neighbours for the buffer. Missing
`.lax` spatial indexes are written next to the input files first. lasR's
spike-free is not bit-reproducible: repeated runs differ on ~0.1-0.4% of
cells (mean 1-2 cm); pitfree is identical.

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

## Compute performance

`python -m alsdb.benchmark timing <laz_dir> <work_dir>` times all three tools
doing the same work (the same tiles to the same 1 m CHMs with the same
settings), each step in its own process pinned to N cores with `taskset`.
One-off preparation and per-run work are kept apart:

| Tool | setup (one-off) | run (per product run) |
|---|---|---|
| alsdb | ingest into TileDB | `compute_chm` pitfree + spikefree |
| LAStools | `lasheight` | `las2dem -spike_free` + `lascanopy -max` |
| lasR | – | normalization + pitfree + spikefree |

Factors are area (nested square blocks of tiles near the survey centre,
`--areas 1 4 16`), cores (`--cores 1 4 8`), storage (`--storage local` copies
the input to `--local-dir`; `network` reads it in place) and `--repeats`.
Each step adds a CSV row with wall time, user/system CPU and peak memory of
the whole process tree. Rerunning skips finished steps. `--dry-run` shows the
selected tiles and the number of runs first.

`python -m alsdb.benchmark timing-report <work_dir>/timings.csv` summarises
medians, throughput (km² per core-hour of the run stage), end-to-end time and
the alsdb break-even: the number of product runs after which alsdb's ingest
plus N runs is faster than the other tool's setup plus N runs. alsdb's store
also serves other products, years and areas; that benefit is real but is not
part of these timings.

Process start-up (pixi, Python imports, R) is a few seconds per step and is
included for every tool; it matters only for very small areas. Run on a
quiet node: other jobs on the pinned cores show up as slower timings.

## Accuracy against known truth

Agreement between tools says nothing about which is right. Two benchmarks
score all tools against a truth instead; both write one scene folder per case
(`laz/`, `truth.npz`, `scene.json`), run alsdb (pitfree, spike-free), lasR
(both) and optionally LAStools (spike-free) on it, and append one row per
output to `<out_dir>/results.csv`. Finished cases are skipped on rerun.
Run them from the `spikefree` environment.

| | canopy | sensor | truth |
|---|---|---|---|
| `synthetic` | generated crowns (voxels) | HELIOS++ aircraft | exact |
| `uls` thinned | real | real UAV, thinned | full-density ULS |
| `uls` helios | real, as 0.5 m voxels | HELIOS++ aircraft | exact (the voxels) |

A method that ranks the same in all three is robust; where rankings differ,
the column that changed (canopy, sensor or truth) is the likely cause.

Scores per output: bias, MAE and share within 0.5 m against the CHM at the
cell centre and the cell maximum, cover above 2 m, and tree-top errors (3 x 3
cell peak at each stem against the truth; against field heights when given).

### Synthetic forests (HELIOS++)

```bash
pixi run -e spikefree python -m alsdb.benchmark synthetic <out_dir> \
    --terrain flat rolling steep rough --forest open conifer broadleaf tropical \
    --density 0.5 2 10 --tools alsdb lasr lastools --lastools-dir /path/to/bin
```

Each scenario is 150 m scored plus a 30 m margin. Forest types set stems,
crown shapes and a stand plant area index (open 1, conifer 4, broadleaf 5.5,
tropical 6); the plant area density inside crowns follows from the crown
volume. One flight line at 1000 m, Leica ALS50-II geometry, up to four
returns per pulse. HELIOS++ needs about 4.5 KB of memory per voxel: a 210 m
broadleaf or tropical scene needs more than 22 GB.

Terrains: `flat`; `rolling` (±15 m hills); `steep` (a smooth 35° plane);
`rough` (hills plus ±1 m micro-relief); `cliff` (a 20° slope broken every 45 m
by 6-14 m scarps with 70-80° faces); `ravine` (hills cut by two V-shaped
channels 7 and 10 m deep with 50° sides). Only `cliff` and `ravine` have
slopes above 45° and sharp breaks, which is where ground filters differ.
`--understorey yes` adds low vegetation (0.5-3 m shrubs, ~60% cover, its own
plant area index of 1) under any forest type, the classic source of a DTM
lifted by vegetation read as ground. `--placement clumped` places the trees in
groups (a Thomas cluster process) instead of a jittered grid, so canopy gaps
vary in size. Both only add suffixes (`_u`, `_c`) to the scene names; the
default scenes are unchanged.

### ULS canopies

```bash
pixi run -e spikefree python -m alsdb.benchmark uls <out_dir> \
    --laz-dir <ULS tiles> --site DE-Eco --x0 416700 --y0 5346650 --core 100 \
    --year 2026 --trees inventory.gpkg --density 0.5 2 10
```

The tiles are read once, streaming. Only near-nadir returns (`--max-angle`,
default 20°) feed the samples and the voxel profiles; the reference uses all
returns.

- `thinned`: one pulse per cell of 1/density m², with all its returns.
  Pulses are identified by GPS time, flight line and scanner channel, and
  preselected by a hash of that key, so a pulse is kept or dropped whole even
  when its returns sit in different tiles. Reference: DTM from a TIN of the
  lowest ground return per 25 cm; CHM from the highest return per 10 cm above
  it (the 1 m cell maximum, and the 20 cm around the cell centre).
- `helios`: plant area density per 0.5 m voxel from the near-nadir returns
  (MacArthur-Horn per voxel column, returns weighted 1 / number of returns, as
  in leafR); voxels below 0.05 m²/m³ are dropped. HELIOS++ scans these voxels
  over the reference terrain with the synthetic survey; the truth is computed
  from the same voxels.

Classes: ground (2) stays, every other non-noise class becomes vegetation (5).
The tools reuse the ULS ground classification; it is not under test.

Caveats. The thinned samples keep the UAV sensor: centimetre footprints and
low flying height hit crown tops and gaps more easily than aircraft ALS.
Density-harmonised ULS (one point per 2 cm voxel) has lost some returns of
many pulses (only 32% of pulses complete on DE-Eco), so the thinned samples
have fewer late returns than a real survey. Existing CHMs delivered with ULS
data are often made with one of the tools under test (e.g. LAStools
spike-free) and must not be used as truth.

### Decimated aircraft ALS

```bash
pixi run -e spikefree python -m alsdb.benchmark decimated <out_dir> \
    --original <site>/original --site PetitBuech --x0 937650 --y0 6396650 --core 200 \
    --crs EPSG:2154 --tools alsdb lasr
```

For sites delivered as a dense survey plus decimated copies (`pd_00p5/` ...
`pd_08/`, `pd_04_ft_*/` next to `original/`; or `--versions NAME=DIR ...`),
each copy of the area becomes one scene. The reference comes from the
original as for ULS, with the centre truth over 0.4 m (enough returns at ~30
pulses/m2); only ground, water and vegetation classes are used, vegetation
written as class 5. Unlike thinned ULS, the input keeps the aircraft sensor
and footprint. `--tools` without names only builds the scenes (for
`sensitivity`).

### Robustness

lasR and LAStools run as external programs. Each LAStools call is killed after
an hour (`timeout`): `las2dem -spike_free` once looped on a synthetic scene
(`WARNING: using brute_locate`). A failed or hung external tool is recorded
under `"failed"` in the scene's `timings.json` and the scene is scored without
it, so compare tools only on scenes where all of them succeeded.

### Freeze-distance sensitivity

```bash
pixi run -e spikefree python -m alsdb.benchmark sensitivity <scenes_dir> <out.csv> --workers 8
```

reruns alsdb spike-free on every finished scene of a `synthetic`, `uls` or
`decimated` folder with freeze distances of 2, 3 (`"auto"`) and 5 first-return
spacings, `max(3 spacings, 1.5 m)`, a fixed 1.5 m, the formula of Fischer et
al. (2024) at the scene's mean density (`lsf_tile`) and per 5 m cell
(`lsf_local`), and on first returns only `k3_first` and `lspikefree` (Fischer
et al.'s full method), and scores each like the benchmark (one row per scene and setting; resumable). Scenes that the
benchmark in the same folder has not scored yet are skipped. Results:
[chm_methods.md](chm_methods.md#how-sensitive-is-the-freeze-distance).

## Ground classification

```bash
pixi run -e spikefree python -m alsdb.benchmark classification-files <out_dir> \
    --laz-dir <site> --reference-dir <site>/original --tools csf pmf smrf ptd \
    --ptd-angle 20 30 40
```

compares ground classifiers with the reference classification of real LAS/LAZ
files: CSF (Zhang et al. 2016), PMF (Zhang et al. 2003) and SMRF (Pingel et al.
2013) as in PDAL, PTD (Axelsson 2000) via pyLASR, and optionally LAStools
`lasground` and FAST-GC (Fareed et al.) as external programs. Each classifier
sees every class reset to 1 except noise (7, 18), which it skips. Unclassified
(0, 1) and noise points of the reference are not scored.

Per file, method and parameter set, `results.csv` holds:

- point scores: type I error (ground rejected / true ground), type II error
  (non-ground accepted / true non-ground) and total error (Sithole & Vosselman
  2004), F1, ground fraction;
- DTM error: a TIN of the predicted ground against a TIN of the reference
  ground on 1 m cells (bias, MAE, RMSE, NMAD, 95th percentile, coverage). For
  canopy heights this is what matters: a few wrong points under a cliff move
  the terrain by metres, many wrong points next to the ground hardly at all;
- the settings used (`parameters`) and the runtime.

`strata.csv` splits the same scores by terrain slope (0-15, 15-30, 30-45, >45°)
and canopy cover (share of first returns above 2 m: 0-50, 50-80, 80-95,
95-100%) in 5 m cells, on points and cells inside the 5 m edge strip only.
Several `--ptd-*` values run every combination as separate rows. With
`--laz-dir` at a site folder, every version (`original/`, `pd_*/`) is scored;
`--reference-dir` takes terrain and cover from the full-density original, so
DTM errors at all densities are against the same terrain.

The same classifiers also run on the benchmark scenes, scored against each
scene's truth DTM:

```bash
pixi run -e spikefree python -m alsdb.benchmark classification-scenes <out_dir> \
    --scenes-dir <synthetic_grid> <uls_out> <decimated_out> --tools csf pmf smrf ptd --workers 8
```

| scenes | reference DTM | point labels scored against |
|---|---|---|
| `synthetic` | exact terrain mesh | HELIOS++ labels (exact) |
| `uls` thinned / helios | full-density ULS terrain / exact | ULS provider's classes / exact |
| `decimated` | full-density ALS terrain | survey provider's classes |

Strata use the truth averaged over 5 m blocks (slope from the DTM, cover from
the share of the CHM above 2 m), inside the scored core window only. The
synthetic scenes give an unbiased DTM error (no reference classifier), but
their terrain has no cliffs and their forests no understorey; the real scenes
have both, against a measured reference.

The reference DTM of real scenes rests on the survey provider's ground
classification. At full density the classifiers do not agree everywhere
(Petit Buech: 73% of cells within 0.25 m of each other, 45% on slopes over
45°; Robson Creek: 27%), so no single classifier, nor their median, is a
truth. Two more references separate the questions:

```bash
pixi run -e spikefree python -m alsdb.benchmark classification-references \
    --scenes-dir <decimated_out> <uls_out> --source PetitBuech=<site>/original \
    RobsonCreek=<site>/original DE-Tha=<ULS tiles> --out-csv agreement.csv
```

classifies the full-density survey of every scene area with each run and
stores the DTMs with the scenes (`dtm_references.npz`). `classification-scenes`
then also reports

- `self_*`: each run against its own full-density DTM — robustness to density
  without any external truth;
- `consensus_*`: against the median of the methods' default runs, on the cells
  where they agree within 0.25 m (`agreed_share`) — accuracy where a
  defensible truth exists.

`agreement.csv` lists, per area and slope / cover class, the share of agreeing
cells, the median spread and every method's distance from the provider's DTM.
On the disputed cells (steep slopes, tropical forest) only an independent
truth can rank the methods: exact synthetic terrain or field measurements.

Caveat: the reference classification is itself the output of some ground
filter, often a PTD variant in commercial software; a classifier close to it
in kind may score better partly for that reason. Ask the data provider how
the reference was made.

## What agreement to expect

On a Barcelona PNOA tile (inner 1 km², same auto parameters), alsdb pitfree
matched lasR on 99.3% of cells to within 1 cm; spike-free was 80% identical,
alsdb +0.18 m on average (lasR seeds freezing with the highest point of a
coarse grid; alsdb freezes at each insertion). Agreement between
implementations is not accuracy: field data (e.g. NFI plots) remain necessary.

## References

- Axelsson, P. (2000). DEM generation from laser scanner data using adaptive
  TIN models. *International Archives of Photogrammetry and Remote Sensing*,
  33(B4), 110–117.
- Asner, G. P., Scurlock, J. M. O., & Hicke, J. A. (2003). Global synthesis of
  leaf area index observations. *Global Ecology and Biogeography*, 12, 191–205.
  <https://doi.org/10.1046/j.1466-822X.2003.00026.x>
- Butler, H., Chambers, B., Hartzell, P., & Glennie, C. (2021). PDAL: An open
  source library for the processing and analysis of point clouds. *Computers &
  Geosciences*, 148, 104680. <https://doi.org/10.1016/j.cageo.2020.104680>
- de Almeida, D. R. A., et al. (2019). Optimizing the remote detection of
  tropical rainforest structure with airborne lidar: leaf area profile
  sensitivity to pulse density and spatial sampling. *Remote Sensing*, 11(1),
  92. <https://doi.org/10.3390/rs11010092> (leafR)
- Fareed, N., et al. FAST-GC (preprint, Preprints.org 202609.1631),
  <https://github.com/nadeemfareed/FAST-GC>
- Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. (2024). Robust
  characterisation of forest structure from airborne laser scanning — A
  systematic assessment and sample workflow for ecologists. *Methods in Ecology
  and Evolution*, 15, 1873–1888. <https://doi.org/10.1111/2041-210X.14416>
- Khosravipour, A., Skidmore, A. K., Isenburg, M., Wang, T., & Hussin, Y. A.
  (2014). Generating pit-free canopy height models from airborne lidar.
  *Photogrammetric Engineering & Remote Sensing*, 80(9), 863–872.
  <https://doi.org/10.14358/PERS.80.9.863>
- Khosravipour, A., Skidmore, A. K., & Isenburg, M. (2016). Generating
  spike-free digital surface models using LiDAR raw point clouds.
  *International Journal of Applied Earth Observation and Geoinformation*, 52,
  104–114. <https://doi.org/10.1016/j.jag.2016.06.005>
- MacArthur, R. H., & Horn, H. S. (1969). Foliage profile by vertical
  measurements. *Ecology*, 50(5), 802–804. <https://doi.org/10.2307/1933693>
- Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). An improved simple
  morphological filter for the terrain classification of airborne LIDAR data.
  *ISPRS Journal of Photogrammetry and Remote Sensing*, 77, 21–30.
  <https://doi.org/10.1016/j.isprsjprs.2012.12.002>
- Roussel, J.-R., et al. (2020). lidR: An R package for analysis of Airborne
  Laser Scanning (ALS) data. *Remote Sensing of Environment*, 251, 112061.
  <https://doi.org/10.1016/j.rse.2020.112061>; lasR and pyLASR:
  <https://github.com/r-lidar/lasR>
- Sithole, G., & Vosselman, G. (2004). Experimental comparison of filter
  algorithms for bare-Earth extraction from airborne laser scanning point
  clouds. *ISPRS Journal of Photogrammetry and Remote Sensing*, 59(1–2), 85–101.
  <https://doi.org/10.1016/j.isprsjprs.2004.05.004>
- Winiwarter, L., et al. (2022). Virtual laser scanning with HELIOS++: A novel
  take on ray tracing-based simulation of topographic full-waveform 3D laser
  scanning. *Remote Sensing of Environment*, 269, 112772.
  <https://doi.org/10.1016/j.rse.2021.112772>
- Zhang, K., Chen, S.-C., Whitman, D., Shyu, M.-L., Yan, J., & Zhang, C. (2003).
  A progressive morphological filter for removing nonground measurements from
  airborne LIDAR data. *IEEE Transactions on Geoscience and Remote Sensing*,
  41(4), 872–882. <https://doi.org/10.1109/TGRS.2003.810682>
- Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016). An
  easy-to-use airborne LiDAR data filtering method based on cloth simulation.
  *Remote Sensing*, 8(6), 501. <https://doi.org/10.3390/rs8060501>
- LAStools: rapidlasso GmbH, <https://rapidlasso.de>

Data: Chapitre Petit Buëch and Robson Creek surveys and their decimated copies
provided by Fabian Fischer (University of Bristol), decimated with the workflow
of Fischer et al. (2024); the Robson Creek original is TERN data. ULS surveys
DE-Tha and DE-Eco: ICOS; credit them as the ICOS data licence requires.
