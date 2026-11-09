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
BABA neighbourhoods (`baba_radius`) also support selection; each neighbourhood
is found with a cell grid and its heights sorted once (on 2.1 M points at 2 m
resolution and 12.5 m radius, 21 s instead of 94 s, identical values).

Height metrics use vegetation returns above ground. Canopy cover (`cc`) is the
share of classified (ground and vegetation) first returns that are vegetation
above `cc_threshold`: buildings, wires and unclassified points neither add
cover nor dilute it, as for the gap fraction. `compute_metrics` and
`compute_biomass` give returns outside the ground TIN the height above the
nearest ground (`ground_extrapolation=True`, as the vertical profiles); off,
those returns are left out. Raster writes are
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

The Næsset-style model (`naesset_model`, Næsset 2002) gives 0 where a cell has
returns but no canopy cover, including cells without vegetation returns, and
NaN only where it has no first returns, so area means count open land.
`calibrate_naesset` and `validate_naesset` keep plots without cover or biomass.

Any model with `predict(X)` (scikit-learn, XGBoost, LightGBM, Keras, or a
small wrapper around a PyTorch network) goes through `wrap_sklearn_model`.
Features default to the estimator's `feature_names_in_` and must match them if
given; a feature count that disagrees with `n_features_in_` raises; models
fitted on a DataFrame receive one with the same columns; predictions of shape
(cells, 1) are accepted. Cells without cover whose height metrics are missing
get 0 (`zero_without_cover`), as with the Næsset model. Models that cannot be
pickled get no automatic `model_id`: pass one to `compute_biomass` to reuse
results. Tiles run in threads, so `predict` may run concurrently. Any other
callable taking the metric dict and returning the grid also works as
`model_fn`, with `model_features` and `model_id`.

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

## Vertical profiles and waveforms

`alsdb.products.profiles` holds every vertical canopy profile. They share one
chain: a gap probability over height above ground, from ALS returns
(MacArthur & Horn 1969, `gap_from_returns`) or from a waveform split into
canopy and ground (GEDI L2B, `gap_from_waveform`), then the plant area
traversed `PAI = -ln(gap) / k` (`pai_from_gap`) and the plant area volume
density `PAVD = -dPAI/dh` (`pavd_from_pai`).

`compute_als_pavd_profile` interpolates ground height at each first return
using **all** classified ground returns in a buffered footprint. The default
`terrain_buffer=30` metres prevents relying solely on sparse ground first
returns. Ground first returns contribute at height zero. First returns outside
the ground TIN take the height above the nearest ground (`ground_extrapolation`,
default on, counted in `n_extrapolated`); switched off, they are excluded and
counted in `n_unsupported`, which raises the gap and lowers the PAI because the
ground returns stay. The profile is a return-count estimate affected by
sampling, classification and extinction assumptions, not an independent truth
measurement.

`compute_als_pavd_batch` reads nearby shots together, with one ground TIN and one
normalisation per chunk, in worker processes when `n_workers > 1` (as
`simulate_batch`). On 1,600 footprints of 30 pulses/m2 ALS on a 30 degree slope
it took 9 ms per footprint on one worker and 4 ms on four, against about 100 ms
for single profiles, with identical results. `fit_transmittance_model` accepts
the cumulative PAI per bin or at the bin edges (`PAVDProfile.pai_cumulative`)
and, with `free_amplitude=True`, fits `T = a exp(-beta C)` instead of forcing
`T = 1` at the canopy top.

```python
import numpy as np
from alsdb.products.profiles import compute_als_pavd_batch
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

## Simulated GEDI canopy profiles (attenuation)

`alsdb.products.profiles.simulate_gedi_profiles(provider, shots, out_dir)`
simulates, for every shot, the canopy and ground parts of the waveform (split
by ALS class, as the GEDI simulator's `-ground` option) and three gap
probability profiles on one height-above-ground grid (default -10 to 80 m in
0.15 m bins):

- `pgap_waveform`: the GEDI L2B estimate, `1 - Rv(h) / (Rv + rv_rg * Rg)`,
  with `rv_rg` the canopy-to-ground reflectance ratio (`rv_rg`, or
  `rv_rg_col` for per-shot values such as `rhov / rhog` of matched L2B shots);
- `pgap_als_first`, `pgap_als_all`: the MacArthur-Horn estimate from the same
  points with the same footprint and density weights, from first returns and
  from all returns weighted by `1/NumberOfReturns`.

It writes `profiles.zarr` (arrays of shape shots x heights, plus `height`) and
`footprints.parquet` (shot columns, `shot_index`, waveform metrics,
`signal_top`, `z_ground_true`, `canopy_energy`, `ground_energy`, `rv_rg`).
The gap profiles are NaN above `signal_top`, as GEDI's `pgap_theta_z` covers
only the signal. `gap_from_waveform(canopy, ground, rv_rg, totals=...)` recomputes
`pgap_waveform` for another ratio from the stored arrays, without simulating
again.

All three profiles go through the same retrieval, e.g. gedidb's
`GEDIVerticalProfiler.compute_profiles(pgap, signal_heights(height,
signal_top), np.pi / 2, 0.5, 1.0)` for `cp_pavd`; `fit_transmittance_model`
then relates the ratio of simulated to ALS PAVD to the plant area above.
Simulated waveforms inherit the ALS occlusion: simulated against ALS isolates
what the waveform and the L2B retrieval add, real GEDI against simulated what
the simulation cannot represent. Heights above ground come from one TIN per
chunk of shots; on the 64 Petit Buech test footprints the canopy/ground split
agreed with gediRat's to 0.3% of the energy and its ground height to 0.21 m.

## Validation and performance

The regression tests cover buffer contamination, exact boundary ownership,
selective metrics, shared normalization, provenance/restarts,
slope-normalized PAVD, fixed pulse spacing, batch/single
agreement, duplicate row labels, and deterministic noise.

See [the benchmark instructions](benchmarks/README.md) for a reproducible real
sample benchmark and measured timings. Benchmarks are local observations, not
promises for full surveys or different hardware.
