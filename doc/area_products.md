# Area-based products

`compute_metrics`, `compute_biomass` and `compute_gap_fraction` share one
implementation: each buffered tile is queried once and normalised to height
above ground once for everything a call computes.

```python
from alsdb.products.biomass import compute_metrics
from alsdb.products.gap import compute_gap_fraction

compute_metrics(provider, store, resolution=10, year=2014, metrics=("h95", "cc"))
compute_gap_fraction(provider, store, resolution=10, year=2014, lai=True)
```

`metrics=None` computes all structural metrics; `metrics=()` none. Selecting
metrics avoids computing unused entropy, strata and height metrics. Circular
BABA neighbourhoods (`baba_radius`) also support selection. Raster writes are
locked per resolution, so more workers do not raise write throughput
indefinitely: choose the worker count with point density and buffer size in
mind.

## Biomass and resuming

Supply the inputs and identity of a fitted model:

```python
from alsdb.products.biomass import compute_biomass, naesset_model

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
from alsdb.products.pavd import compute_als_pavd_batch
from alsdb.products.waveform import simulate_batch

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

Defaults follow the GEDI simulator (Hancock et al. 2019, `gediRat`): a
Gaussian footprint of sigma `0.44 * footprint_radius` (5.5 m for GEDI), a
Gaussian pulse of 15 ns FWHM (sigma 0.955 m) unless a measured GEDI pulse is
chosen with `beam_id`, and each point divided by the number of pulses (last
returns) in its 1.5 m cell (`density_cell`; `None` turns it off), so flight-line
overlaps do not count twice. The pulse is applied with its later parts below
each return, as in `gediRat` (measured GEDI pulses have a long trailing tail).
RH follows libclidar's `findRH`: energy accumulated from the lowest bin, with RH0
the bottom of the signal (usually negative), as GEDI's `rh_000`.

Optional `return_weighting="fractional"` gives each return weight
`1/NumberOfReturns`. The default remains count weighting. These options do not
provide radiometric calibration.

`simulate_batch` groups shots into spatial blocks (`batch_tile_size`), reads
each chunk once and, with `n_workers > 1`, simulates chunks in separate
processes, each with its own TileDB context (cores / workers threads, small
cache); results do not depend on the number of workers. Scripts using workers
need an `if __name__ == "__main__":` guard (processes start with `spawn`).
On 6,241 footprints over 2 x 2 km of 30 pulses/m2 ALS (Chapitre Petit Buech)
it took 41 s on one worker and 19 s on four (250 m blocks, laptop
i7-12800H), against 565 s for gediRat; waveforms agree with gediRat to within
0.6% of their energy and RH10-RH98 to one 0.15 m bin. Larger blocks (250 m)
help dense footprint grids; sparse tracks keep the 100 m default.

## Validation and performance

The regression tests cover buffer contamination, exact boundary ownership,
selective metrics, shared normalization, provenance/restarts,
slope-normalized PAVD, fixed pulse spacing, batch/single
agreement, duplicate row labels, and deterministic noise.

See [the benchmark instructions](benchmarks/README.md) for a reproducible real
sample benchmark and measured timings. Benchmarks are local observations, not
promises for full surveys or different hardware.
