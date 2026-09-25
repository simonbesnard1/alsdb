# CHM methods, quality and uncertainty

Surface processing uses one north-up pixel lattice, a buffered ground TIN, and
explicit quality layers. CHM is a statistic/interpolation of normalized canopy
returns; it is not generally equal to the independently rasterized DSM minus DTM.
All horizontal and vertical coordinates must use compatible metric units.

## Choosing a method

| `method` | Behavior | Important parameters |
|---|---|---|
| `max` | Bin normalized eligible returns; `height_statistic` can select another statistic | `first_returns_only`, `height_statistic` |
| `pitfree` | Maximum of height-threshold TIN layers | `pitfree_thresholds`, required `pitfree_max_distance` |
| `highest_subcell_tin` | Highest return per small subcell, followed by a TIN | `spikefree_subcell_resolution`, required `spikefree_max_distance` |
| `spikefree` | Incremental constrained Delaunay with triangle freezing | `freeze_distance`, `height_buffer`, optional `max_triangle_edge` |
| `lastools` | Explicit external las2dem spike-free backend | `lastools_executable`, `freeze_interval`, `freeze_distance`, `height_buffer` |

Distance masks for the two approximation methods accept metres or `"auto"`.
Auto distances are estimated from spacing within each buffered tile/layer and
therefore are not guaranteed to be invariant to tile size. Check seams explicitly.
The native method uses all eligible return numbers; classification selection
still applies (default vegetation classes 3–5). For surveys with vegetation in
class 1, explicitly choose appropriate `veg_classes` after checking classification.
Withheld, nonfinite, and classified noise returns are excluded from CHM.

## Native spike-free installation

```bash
pixi install -e spikefree
pixi run -e spikefree build-spikefree
pixi run -e spikefree python your_processing_script.py
```

The optional environment supplies CGAL, pybind11 and a C++ compiler. The compiled
module is local to this checkout/Python ABI: rebuild after changing Python or the
C++ source. It is not required for the other methods. CGAL supplies robust
orientation/incircle predicates and constrained triangulation; alsdb implements
the height-ordered insertion and freezing logic. The native loop releases the
Python GIL, so independent tiles can execute concurrently.

```python
from alsdb.processing.chm import compute_chm

compute_chm(
    provider, store, year=2021, resolution=1.0,
    method="spikefree", freeze_distance=1.5, height_buffer=0.5,
    max_triangle_edge=10.0,
    max_ground_distance=20.0,
    tile_size=500.0, tile_buffer=50.0, n_workers=4,
)
```

These example distances are illustrative; choose them from acquisition spacing,
terrain support, and validation. The freeze distance limits all three horizontal
triangle edges. Freezing is delayed until all vertices are above the current
insertion elevation plus `height_buffer`. Frozen edges are constrained; later
points inside the face or on its constrained boundary are rejected. Equal-height
points are ordered deterministically by XY. Duplicate XY retains the highest
height. Degenerate/collinear input produces NoData. `max_triangle_edge` trims
long triangles in the final mesh independently of the freezing rule.

The input elevation for native CHM is normalized HAG. The low-level
`rasterize_spikefree(..., value_field="Z")` can instead reconstruct a raw DSM.
Normalization before triangulation can change insertion order on slopes, so it
must match the reference workflow when comparing algorithms.

This is an independent implementation of the published algorithm, not a promise
of identical LAStools pixels. In particular, freezing is evaluated at each
insertion elevation rather than LAStools' configurable height intervals. High
outliers still need classification/filtering; triangle freezing prevents downward
pits rather than identifying all erroneous high returns.

## Terrain, holes, and pixel support

Ground TIN interpolation is shared between CHM normalization and TIN DTM output.
Duplicate ground XY locations use mean elevation. Outside the convex hull, or
with insufficient noncollinear ground, heights remain missing by default.
`ground_extrapolation=True` explicitly enables nearest-ground fallback (IDW uses
its neighbourhood outside the hull). `max_ground_distance` limits support for
both normalized points and output cells. `dtm_method="min"` is ground-point
binning and does not interpolate. IDW and TIN retain query-buffer points.

`ground_outlier_removal=True` applies the existing ground cleaning before fitting;
rejected ground is class 7 rather than class 1, preventing it from re-entering a
class-1-inclusive canopy selection. Disable it for controlled comparisons to
already cleaned inputs. The separate canopy `remove_outliers` option remains
available. Negative normalized values are clamped to zero and flagged.

`pit_fill=False` is now the default. Enabling it fills only enclosed holes of at
most `fill_max_cells` pixels, using neighbouring observed heights. It never uses a
tile-wide median, fills raster edges, or restores a rejected-support/height cell.
A raster halo preserves local context at tile boundaries. TIN distance/hull gaps
are deliberately excluded from post-filling. True canopy gaps and absent returns
cannot always be distinguished from points alone; missing vegetation is not
silently converted to zero.

Output extents preserve the requested upper-left corner and extend east/south to
whole pixels. Tile windows are integer offsets on that grid, with half-open point
ownership at shared boundaries. `tile_size` must be a whole multiple of resolution.
Use adequate terrain buffers; triangulation with a finite buffer can still differ
from a whole-scene TIN in sparse areas. Mismatched store grids/CRSs are rejected.

## Quality outputs

`quality=True` writes:

- `chm_canopy_count`: eligible, supported canopy returns contributing locally.
- `chm_ground_count`: local retained ground returns.
- `chm_ground_distance`: nearest retained ground point from the cell centre, metres.
- `chm_terrain_slope`: local ground-TIN slope, degrees; missing outside the TIN.
- `chm_quality`: bit mask, stored as exactly representable float32 integers.

| Bit | Meaning |
|---:|---|
| 1 | Valid surface with eligible canopy observations in the pixel |
| 2 | Interpolated surface (may also have local observations) |
| 4 | Locally filled hole |
| 8 | Unsupported terrain at cell centre or for a candidate canopy return |
| 16 | No candidate canopy observations in the pixel |
| 32 | Cell-centre terrain outside the ground TIN / degenerate ground support |
| 64 | A candidate canopy return had negative HAG before clamping |
| 128 | Surface rejected by `max_height` or all local supported returns rejected as outliers |

Flags may coexist. They describe evidence and processing, not a categorical
land-cover map or uncertainty in metres. Undefined nearest-ground distance is
stored as NaN rather than infinity.

## Provenance and reruns

Every surface product records software/algorithm versions, parameters, source URI,
visible-fragment fingerprint, optional `source_version`, and run status under
`processing_runs[year]` in the variable attributes. Changed configurations or
source fragments require `overwrite=True`; metadata-free legacy products cannot
be silently reused. Consolidation also changes the conservative source fingerprint.
Input fragment changes during a run mark the attempt failed. Do not ingest while
processing a survey.

A failed/interrupted run restarts from the beginning and clears stale slices,
including old quality layers. It does not claim partially written outputs are
complete. An all-NoData completed run is recorded and can be reused. `has_data`
tracks finite writes per variable rather than assuming a shared year coordinate
means every variable has data. Zarr writes are serialized per resolution within
one store instance to avoid shared-chunk races. Separate processes/store instances
must not write the same output concurrently.

## Conditional uncertainty ensembles

```python
from alsdb.processing import CHMConfig, CHMErrorModel, compute_chm_uncertainty

config = CHMConfig(
    method="spikefree", freeze_distance=1.5, height_buffer=0.5,
    max_triangle_edge=10.0, max_ground_distance=20.0,
)
# Illustrative only: substitute independently estimated survey error parameters.
errors = CHMErrorModel(
    strip_z_sigma=0.10, strip_xy_sigma=0.15,
    pulse_z_sigma=0.05, terrain_sigma=0.20,
    terrain_correlation_length=20.0,
    calibration="Example parameters; replace with survey validation reference",
)
compute_chm_uncertainty(
    provider, store, year=2021, resolution=1.0, config=config,
    error_model=errors, n_realizations=100, seed=42,
    interval=0.95, min_valid_fraction=0.8,
    tile_size=500.0, tile_buffer=50.0, n_workers=2,
)
```

Use the same configuration, footprint and input revision as the base CHM. This
function writes separate layers and does not overwrite CHM:

- `chm_uncertainty_lower`, `chm_uncertainty_upper`: ensemble quantiles.
- `chm_uncertainty_std`: sample standard deviation.
- `chm_uncertainty_n_valid`: finite realizations per pixel.

Bounds and SD remain missing when fewer than the required fraction (or fewer than
two realizations) is valid. All model parameters and the seed are persisted.
`estimate_chm_uncertainty` offers the same calculation on a buffered structured
point array without database I/O.

Strip offsets affect canopy and ground jointly, so shared vertical error can
cancel in relative height. Pulse errors preserve within-pulse dependence.
`PointSourceId` must identify flight strips and `GpsTime` pulses within each strip;
the software cannot infer whether a survey used those fields correctly. Optional
pulse thinning (`pulse_keep_probability < 1`) measures sampling sensitivity.
The terrain residual uses a smooth random Fourier field approximating a Gaussian
spatial covariance. Random draws are keyed by strip/pulse identity and global
coordinates, so overlapping tiles share the same perturbations.

Every realization reruns ground cleaning, normalization and CHM reconstruction.
Horizontal-error queries include an additional six-sigma margin; Gaussian errors
are not strictly bounded. Finite-buffer and filtering sensitivity still need
checking. Ensemble storage requires about `4 * realizations * tile_pixels` bytes
per worker, plus temporary arrays and triangulation; start with small tiles.

These are **conditional error-model intervals**, not automatically calibrated
95% confidence intervals. Validate empirical coverage against independent,
spatially matched references (the `interval_coverage` helper is provided).
Missed treetops, incorrect classification, temporal mismatch and unknown sensor
bias are not repaired by resampling observed returns. Algorithm differences should
be reported as method sensitivity separately from these intervals.

## Benchmarking and LAStools reference adapter

Prepare one buffered, normalized, eligible point set as a structured `.npy` file
with X/Y/Z/HeightAboveGround. All benchmark methods receive exactly that set:

```bash
pixi run -e spikefree python -m alsdb.processing.benchmark \
  --input normalized.npy --bbox 0 0 100 100 --resolution 1 \
  --tile-size 50 --tile-buffer 20 --output comparison.json
```

Add `--lastools /path/to/las2dem64` for a reference run, or `--reference truth.npy`
for an independent raster on the identical grid. The adapter never downloads an
executable. `--demo` explicitly enables LAStools demo mode for fewer than 1.5M
points; otherwise the adapter requests a valid licence via `-fail`.
`rasterize_lastools` is also callable directly on eligible normalized points.
The database pipeline accepts `method="lastools", lastools_executable="/path/to/las2dem64"`
and records the executable version. `lastools_demo=True` must be explicit; the
external method defaults to a 100 m final triangle limit if `max_triangle_edge`
is omitted, matching the executable default.
The adapter checks output shape and geotransform rather than silently resampling.

The report records input hash, versions, parameters, raster differences,
coverage, a local-pit indicator, one-to-one peak matching, whole-scene versus tiled
differences, runtime and process peak RSS. Methods run in separate processes;
RSS includes the interpreter/libraries and both full/tiled passes. Pit counts and
peak matching are diagnostics rather than independent truth. Choose the same
triangle edge limit, normalization and class/return selection for comparisons.
Native continuous freezing and LAStools interval freezing remain documented
algorithmic differences. The adjacent `.npz` contains each method's raster.

The reproducible synthetic example is `doc/benchmarks/synthetic_chm.py`.
Real-survey comparisons with a LAStools binary and independently calibrated
uncertainty coverage remain necessary before claiming numerical equivalence or
operational accuracy.

## Migration

- `spikefree=True` still selects the old approximation with a deprecation warning.
  Use `method="highest_subcell_tin"` to name it explicitly or `method="spikefree"`
  for the new native implementation.
- Pit filling and ground extrapolation are opt-in. New support masks can reduce
  coverage compared with previous plausible-looking but unsupported values.
- The CHM ground model now explicitly interpolates the buffered ground TIN rather
  than relying on PDAL's local-neighbour HAG defaults. Results can change.
- Use a new store or `overwrite=True` when migrating legacy products; incompatible
  spatial grids require a new store.

References: [Khosravipour et al. (2016)](https://doi.org/10.1016/j.jag.2016.06.005),
[thesis algorithm description](https://research.utwente.nl/files/278716183/khosravipour.pdf),
[LAStools las2dem parameters](https://downloads.rapidlasso.de/html/las2dem_README.html),
[CGAL constrained triangulation](https://doc.cgal.org/latest/Triangulation_2/classCGAL_1_1Constrained__Delaunay__triangulation__2.html).
