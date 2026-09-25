# Forest processor updates

Use `compute_forest_products` when you need several products at the same
resolution. It queries each buffered tile once and shares terrain normalization:

```python
from alsdb.processing import CHMConfig, compute_forest_products

compute_forest_products(
    provider, store, resolution=10, year=2014,
    metrics=("h95", "cc", "density"),
    chm=True, chm_config=CHMConfig(method="max"),
    gap=True, lai=True,
    tile_size=500, tile_buffer=50, n_workers=4,
)
```

The existing `compute_metrics`, `compute_biomass` and `compute_gap_fraction`
functions use the same implementation. `metrics=None` means all structural
metrics; `metrics=()` requests none. `compute_metrics(metrics=("h95", "cc"))`
avoids computing unused entropy, strata and height metrics. Circular BABA
neighborhoods also support selection.

Ground normalization allocates the output record array once. Point queries
select the required attributes, and pending work is bounded to twice the worker
count. `compute_all` also shares terrain between DTM and CHM. Raster writes
remain locked per resolution for consistency; increasing workers indefinitely
will not increase write throughput. Choose worker count with point density and
buffer size in mind.

## Biomass and resuming

Supply the inputs and identity of a fitted model:

```python
from alsdb.processing.biomass import compute_biomass, naesset_model

compute_biomass(
    provider, store, resolution=10, year=2014,
    model_fn=lambda m: naesset_model(m, a=a, b=b, c=c),
    model_features=("h95", "cc"), model_id="local-inventory-fit-v2",
)
```

Matching completed metric slices are read from Zarr in tile-sized windows, so
changing the biomass model does not require rereading and normalizing the point
cloud. Reuse requires identical input fragments, extent, buffers, ground policy,
metric settings and software provenance. `wrap_sklearn_model` records its feature
list and hashes the fitted estimator automatically. For other callables, give a
model ID that changes when the fitted model changes. Unidentified callables are
recomputed rather than incorrectly reused.

Raster runs record complete/failed state. Failed runs restart by clearing the
old slice. Different settings or legacy data without provenance require
`overwrite=True`; all-NaN replacements also clear previous values. No per-tile
checkpoint recovery is claimed.

`validate_naesset(h95, cc, agb_field, spatial_groups)` holds out complete spatial
blocks and returns predictions, fold assignments, RMSE, bias and empirical
residual quantiles. Choose blocks appropriate to spatial dependence. This checks
prediction error, whereas `calibrate_naesset(..., return_cov=True)` describes
coefficient uncertainty. Neither alone supplies calibrated pixel uncertainty.
Field calibration remains necessary; default allometric coefficients are still
placeholders.

## Grid ownership and gap support

All raster products use fixed-size pixels anchored at the requested northwest
corner; partial extents expand east and south. Tile size must be a whole multiple
of resolution. Ownership is `xmin <= X < xmax` and `ymin < Y <= ymax`. Buffer
points support terrain/circular neighborhoods but are not clipped into edge
cells. LAS coordinates are quantized, so points exactly on pixel boundaries
occur in real data: adopting consistent ownership can change individual cell
values even after correcting the old buffer contamination.

Unsupported heights are excluded from canopy-cover denominators. Gap uses
classified first returns directly; enabling LAI never triggers normalization or
changes classification. `compute_gap_fraction` defaults to
`ground_outlier_removal=False`; combined forest processing defaults to `True`.
Set that option explicitly when comparing the two workflows.

Gap quality outputs are `gap_n_ground`, `gap_n_classified`, `gap_n_first` and
`gap_saturated`. The last is 1 for supported cells with no observed ground first
returns, 0 otherwise, and NaN for unsupported cells. Zero observed gap does not
identify a finite LAI: LAI remains NaN there. Positive-gap LAI retains the existing
15 ceiling. These are support diagnostics, not statistical confidence intervals.

## PAVD and waveforms

`compute_als_pavd_profile` now interpolates ground height at each first return
using **all** classified ground returns in a buffered footprint. The default
`terrain_buffer=30` metres prevents relying solely on sparse ground first
returns. Ground first returns contribute at height zero; unsupported canopy
heights are excluded and counted in `n_unsupported`. Optional extrapolation is
recorded in `n_extrapolated`. The profile is a return-count estimate affected by
sampling, classification and extinction assumptions, not an independent truth
measurement.

```python
import numpy as np
from alsdb.processing.pavd import compute_als_pavd_batch
from alsdb.processing.waveform import simulate_batch

profiles = compute_als_pavd_batch(provider, shots, year=2014, n_workers=4)
waveform_metrics = simulate_batch(
    provider, shots, year=2014, n_workers=4, batch_tile_size=100,
    rng=np.random.default_rng(42),
)
```

Both batch APIs query nearby shots together and return results in input row
order. Duplicate DataFrame index labels are supported. Independent positional
random seeds make waveform noise reproducible across worker counts. The batch
waveform API retains scalar metrics, not every waveform array; use
`simulate_waveform` or `waveform_from_points` for the arrays.

Vertical bins now have exactly `z_step` spacing. Measured pulse kernels are
resampled using their physical positions, centered on the peak, normalized, and
convolved without trimming tails. Gaussian pulses also retain their tails.
Slope-plane fits use centered coordinates and require full rank. Output reports
`ground_supported`, `ground_offset` (detected minus median classified-ground
height), and `slope_corrected`. The ground flag requires three ground returns and
agreement within `max(2*sigma, z_step)`; it is a diagnostic heuristic.

Optional `return_weighting="fractional"` gives each return weight
`1/NumberOfReturns`. Optional `density_radius` weights by inverse first-return
sampling density measured within that radius; the query includes an extra halo
for this calculation. The default remains count weighting. These options do not
provide radiometric calibration or establish equivalence with the GEDI simulator.

## Trees

PDAL's current `ClusterID` output is now mapped to the public `TreeID` field;
previously this mismatch silently produced empty tree tables. Points are sorted
by descending normalized height before segmentation. The default `radius` is
now PDAL's 100 metres: it controls non-tree seed insertion, not crown search.
The old height-based `adaptive_radius` heuristic is deprecated. See the
[PDAL litree documentation](https://pdal.io/en/stable/stages/filters.litree.html).

Tree metrics are calculated from complete buffered crowns before ownership is
assigned by their highest point; XY breaks height ties. Buffered crown points
are retained, including points outside the output extent. Exact duplicate apex
coordinates are suppressed. Finite buffers can still change segmentation: use a
buffer wider than the crowns and validate boundary behavior for the survey.

`iter_segment_trees(...)` yields `(points, trees)` per tile with compact global
uint64 IDs. Consume/write each result before requesting the next to keep memory
bounded with respect to point data (apex deduplication retains a set of tree
coordinates). `segment_trees(...)` still collects and returns the full result for
compatibility. Point attributes are preserved by default; pass
`point_attributes=()` for minimal fields, or list additional attributes to retain.
Trees are grouped once for metrics; no allocation depends on the
largest raw TreeID. `veg_classes` is configurable. Optional `voxel_size` remains
Poisson sampling before segmentation, and changes the point support of crown
metrics; no automatic decimation is applied.

## Validation and performance

The regression tests cover buffer contamination, exact boundary ownership,
selective metrics, shared normalization, provenance/restarts, full crowns,
sparse tree IDs, slope-normalized PAVD, fixed pulse spacing, batch/single
agreement, duplicate row labels, and deterministic noise.

See [the benchmark instructions](benchmarks/README.md) for a reproducible real
sample benchmark and measured timings. Benchmarks are local observations, not
promises for full surveys or different hardware.
