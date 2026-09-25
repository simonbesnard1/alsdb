# Synthetic CHM benchmark

Run from the repository root after building the optional native backend:

```bash
pixi run -e spikefree python doc/benchmarks/synthetic_chm.py /tmp/chm-benchmark
```

This generates a known sloping canopy surface and 1,200 lower returns, then runs
max binning, threshold pit-free, highest-subcell TIN and native spike-free in
separate processes. The JSON report includes versions, parameters, input hash,
memory/time diagnostics, comparisons to the known plane, and tiled comparisons.

A local validation run produced 100% coverage and zero full-versus-tiled
height differences for all four methods. Native spike-free reproduced the known
plane to float32 precision (RMSE approximately 5.2e-7 m). Maximum binning had
0.075 m positive bias against cell-centre truth, illustrating the distinction
between a per-cell maximum and a surface evaluated at the cell centre.

This small regular scene checks geometry and the executable benchmark workflow.
It does not establish performance on national surveys, recovery of unsampled
canopy tops, or equivalence with a LAStools binary. Use irregular/sparse canopy,
steep terrain, gaps and actual survey strips for those comparisons; the benchmark
CLI accepts an installed LAStools executable and independent reference rasters.

# Forest processor benchmark

```bash
pixi run python doc/benchmarks/forest_processors.py \
  data/example_als/brazil_als/RIB_A01_2014_laz_0.laz \
  /tmp/forest-results.json --database
```

The optional database section writes temporary TileDB/Zarr stores and measures
actual reads, nearby footprints, combined raster products and tree segmentation.
The checked-in [report](brazil_forest_results.json) uses the first 500,000 records
of the Brazil file (125,787 points in the 100 × 100 m output square). This bounded
sample is not spatially representative of the whole survey. The report records
the input hash, baseline commit, platform and individual timings.

| Operation | Reference | Updated | Ratio |
|---|---:|---:|---:|
| Normalize 500,000 points | 1.649 s, four field appends | 0.612 s, one allocation | 2.7× |
| Read attributes for output square | 0.0454 s, all attributes | 0.0152 s, selected | 3.0× |
| Metrics with buffered input | 0.202 s, previous implementation | 0.0565 s | 3.6× |
| 30 waveform footprints | 0.407 s, separate queries | 0.218 s, spatial batches | 1.9× |
| 30 PAVD footprints | 1.206 s, separate queries | 1.032 s, spatial batches | 1.2× |
| CHM + h95/cc + gap/LAI | 2.456 s, separate calls | 1.774 s, combined | 1.4× |

CPU and attribute-read timings are medians of three runs. Footprints and raster
workflows are single local runs with warm filesystem caches; their ratios vary
between runs. Footprint comparisons use the corrected algorithms on both sides,
isolating query batching. Combined/separate raster arrays agree, and the
single-allocation normalization matches the four-append reference field by field.

The metrics speedup also includes a correctness change: the previous code
incorrectly assigned buffer points to edge pixels. On input already cropped by
the caller, the old metric implementation took 0.0434 s. The new full-buffer
measurement includes ownership filtering, so the 3.6× ratio should not be
interpreted as a general speedup on already-cropped arrays. Exact pixel-edge
ownership and exclusion of unsupported heights can legitimately change results;
the JSON records differences against the legacy implementation.

The corrected tree processor returned **7 owned crowns / 1,972 sampled points**
in **0.963 s**, using a 15 × 15 m crop, 10 m buffer, `voxel_size=0.5`, and minimal
point attributes. This verifies a nonempty real-data execution, not crown accuracy
against a field inventory. PDAL input sorting, its `ClusterID` output and its
non-tree seed radius are now handled correctly. Different point ordering can
change Poisson sampling and the resulting crowns.

The local LAS reader emitted PROJ lookup warnings; these benchmarks use the
native input XY coordinates without reprojection. CRS/reprojection accuracy is
not evaluated by this benchmark.
