# alsDB

**alsDB** is a Python package for ingesting, storing, and processing Airborne Laser Scanning (ALS/LiDAR) point clouds at scale. It reads LAZ/LAS files via [PDAL](https://pdal.io), stores them in a [TileDB](https://tiledb.com) sparse array (locally or on S3-compatible object storage), and provides a processing pipeline for canopy height models (CHM), digital terrain/surface models (DTM/DSM), above-ground biomass (AGB) estimation, and GEDI waveform simulation.

The package is dataset-agnostic: CRS, bounding box, and acquisition year are read directly from the LAZ file header, so any national or global ALS dataset works without custom filename parsers.

---

## Features

- **Scalable ingestion** — append LAZ tiles one at a time or in parallel batches; each tile becomes a new TileDB fragment
- **Multi-temporal** — X / Y / Year as TileDB dimensions; repeated surveys of the same area are stored and queryable independently
- **Ingestion manifest** — tracks which files have been ingested, their CRS, bounding box, point count, and status; re-ingestion is a no-op by default
- **CRS-aware schema** — TileDB domain bounds are selected automatically from the tile CRS (EPSG:25830, 28992, 2154, 27700, or generic UTM fallback)
- **Local and S3 storage** — identical API for filesystem paths and `s3://` URIs (tested on GFZ Ceph / RadosGW)
- **Processing pipelines** — CHM, DTM, DSM (via PDAL + GDAL), biomass estimation (Næsset power-law model), GEDI full-waveform simulation
- **CLI** — `alsdb ingest` and `alsdb info` commands

---

## Installation

The recommended way is [pixi](https://pixi.sh), which resolves the full conda dependency stack and creates an isolated environment:

```bash
git clone https://github.com/simonbesnard1/alsdb.git
cd alsdb
pixi install          # installs all conda-forge dependencies
pixi shell            # activates the environment
pip install -e .      # installs the alsdb package itself (editable)
```
### Dependencies

| Package | Purpose |
|---------|---------|
| `pdal` / `python-pdal` | LAZ/LAS reading, HAG filter, raster writing |
| `tiledb` | Sparse array storage (local + S3) |
| `numpy` / `scipy` | Array operations, peak detection |
| `pandas` / `xarray` | Query result formats |
| `rasterio` | Reading CHM/DTM/DSM GeoTIFFs |
| `pyproj` | CRS parsing from WKT |
| `matplotlib` / `plotly` | Visualisation |
| `click` | CLI |

---

## Quick start

### Ingest a single tile

```python
import alsdb
from alsdb import ALSDatabase

alsdb.setup_logging()   # INFO-level logging to stderr

db = ALSDatabase(storage_type="local", uri="my_array")
db.ingest("path/to/tile.laz")
```

Year, CRS, and bounding box are read from the LAZ header automatically. The array is created on first write with domain bounds sized for the detected CRS.

### Ingest a directory in parallel

```python
from pathlib import Path

paths = sorted(Path("/data/als/").glob("*.laz"))

db.ingest_many(
    paths,
    max_workers=8,          # parallel LAZ readers / TileDB writers
    consolidate_every=50,   # consolidate fragments every 50 tiles
)
```

Already-ingested files (tracked in the manifest) are skipped automatically. Pass `overwrite=True` to start from scratch.

### Inspect the manifest

```python
for entry in db.list_ingested():
    print(entry["filename"], entry["year"], entry["n_points"], entry["status"])
```

### Query points

```python
from alsdb import ALSProvider

reader = ALSProvider(storage_type="local", uri="my_array")

# All points in a bounding box
df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000)

# Restrict to a single survey year
df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

# Which years are stored?
print(reader.available_years())   # [2019, 2021, 2023]

# As xarray Dataset
ds = reader.to_xarray(308_000, 4_688_000, 310_000, 4_690_000, year=2021)
```

---

## Storage on S3

```python
import boto3

session = boto3.Session(profile_name="my-profile")
creds = session.get_credentials().get_frozen_credentials()

db = ALSDatabase(
    storage_type="s3",
    uri="s3://owner.bucket-name/als_array",
    url="https://s3.example.com",          # endpoint URL for non-AWS S3
    region="eu-central-1",
    credentials={
        "AccessKeyId": creds.access_key,
        "SecretAccessKey": creds.secret_key,
    },
)

db.ingest("path/to/tile.laz")
```

Credentials can also be supplied via environment variables (`ALSDB_S3_ACCESS_KEY`, `ALSDB_S3_SECRET_KEY`, `ALSDB_S3_URL`).

---

## Processing

All processing functions accept a `year` parameter to restrict the query to a single survey.

### Canopy Height Model / DTM / DSM

```python
from alsdb.processing.chm import compute_chm, compute_all

# Single area (fits in memory)
compute_chm(
    provider=reader,
    output_path="outputs/chm.tif",
    bbox=(308_000, 4_688_000, 310_000, 4_690_000),
    resolution=1.0,
    year=2021,
)

# Large area — tiled processing (500 m sub-tiles, 50 m buffer, 4 workers)
compute_chm(
    provider=reader,
    output_path="outputs/chm.tif",
    bbox=(308_000, 4_688_000, 318_000, 4_698_000),
    resolution=1.0,
    year=2021,
    tile_size=500.0,    # sub-tile size in metres
    tile_buffer=50.0,   # overlap buffer for accurate TIN at tile edges
    n_workers=4,
)

# All three products (DTM + DSM + CHM) in one call
compute_all(
    provider=reader,
    output_dir="outputs/",
    bbox=(308_000, 4_688_000, 318_000, 4_698_000),
    resolution=1.0,
    year=2021,
    tile_size=500.0, tile_buffer=50.0, n_workers=4,
)
# writes outputs/chm.tif, outputs/dtm.tif, outputs/dsm.tif
```

The pipeline queries the array, injects the point cloud into a PDAL pipeline (`filters.hag_delaunay` for height-above-ground), and writes GeoTIFFs via `writers.gdal`.  Sub-tiles with no data (outside the flight swath) are silently skipped and appear as nodata in the mosaic.

**Tiling parameters** apply to all three products — CHM uses a 50 m buffer to ensure accurate TIN values at tile edges; DTM and DSM require no buffer.

### Biomass estimation

```python
from alsdb.processing.biomass import compute_biomass, compute_metrics

# Single area
compute_biomass(
    provider=reader,
    output_path="output/agb.tif",
    bbox=(308_000, 4_688_000, 310_000, 4_690_000),
    resolution=25.0,
    year=2021,
)

# Large area — tiled
compute_biomass(
    provider=reader,
    output_path="output/agb.tif",
    bbox=(308_000, 4_688_000, 318_000, 4_698_000),
    resolution=25.0,
    year=2021,
    tile_size=500.0, tile_buffer=50.0, n_workers=4,
)
```

Uses the Næsset (2002) power-law model: `AGB = a × h95^b × cc^c`, where `h95` is the 95th-percentile height, `cc` is canopy cover, and `a / b / c` are configurable coefficients.

Intermediate metrics (h50, h75, h95, hmean, canopy cover, point density) are also available via `compute_metrics()`, which accepts the same tiling parameters.

### Gap fraction and effective LAI

```python
from alsdb.processing.gap import compute_gap_fraction

# Gap fraction only — no assumptions
compute_gap_fraction(
    provider=reader,
    output_path="output/gap.tif",
    bbox=(308_000, 4_688_000, 310_000, 4_690_000),
    resolution=10.0,
    year=2021,
)

# Gap fraction + effective LAI (Beer-Lambert, must supply k explicitly)
compute_gap_fraction(
    provider=reader,
    output_path="output/gap.tif",
    resolution=10.0,
    year=2021,
    lai=True,
    k=0.5,                      # spherical leaf angle distribution
    lai_path="output/lai.tif",
)

# Tiled for large areas
compute_gap_fraction(
    provider=reader,
    output_path="output/gap.tif",
    resolution=10.0,
    year=2021,
    tile_size=500.0, tile_buffer=50.0, n_workers=4,
)
```

Gap fraction is the MacArthur-Wilson return-count estimator — a direct observable with no canopy-structure assumptions:

    P_gap = N_ground_first / (N_ground_first + N_veg_first)

Effective LAI is opt-in via `lai=True` and requires an explicit extinction coefficient `k` (Beer-Lambert: `L_e = -ln(P_gap) / k`).  The result is effective LAI, not true LAI — ALS cannot separate leaves from woody material.  LAI is capped at 10 m²/m² to avoid `ln(0)` artefacts in fully closed canopy cells.

### GEDI waveform simulation

```python
from alsdb.processing.waveform import simulate_waveform, simulate_batch

# Single footprint (25 m diameter, like GEDI)
result = simulate_waveform(
    provider=reader,
    center_x=308_500.0, center_y=4_689_000.0,
    footprint_radius=12.5,
    year=2021,
)
print(result.rh[50])     # RH50 height above ground (m) — dict has integer keys RH0–RH100
print(result.rh[98])     # RH98, equivalent to GEDI L2A rh98
print(result.cover)      # canopy cover fraction
print(result.z_ground)   # estimated ground elevation (m)

# Batch — shots must be a DataFrame with center_x / center_y columns (UTM)
import numpy as np
import pandas as pd

# Synthetic 60 m spaced GEDI-like shot grid over the tile
xs, ys = np.meshgrid(
    np.arange(308_100, 309_900, 60),
    np.arange(4_688_500, 4_689_900, 60),
)
shots = pd.DataFrame({"center_x": xs.ravel(), "center_y": ys.ravel()})

results = simulate_batch(
    provider=reader,
    shots=shots,
    year=2021,
    n_workers=4,
    footprint_radius=12.5,
)
# results is a DataFrame: original columns + z_ground, home, cover, rh0…rh100
print(results[["center_x", "center_y", "rh50", "rh98", "cover"]].head())
```

The simulator builds a vertical return histogram (0.15 m bins), convolves it with a Gaussian pulse (σ = 0.64 m full-waveform, 0.93 m cover), detects the ground return, and computes cumulative RH metrics from ground up.

---

## Visualisation

All 2-D plot functions accept a point-cloud DataFrame from `query_bbox()` and rasterize it to a regular grid before rendering, so they stay fast even for multi-million-point tiles.

### 4-panel overview

```python
from alsdb.utils.viz import plot_overview

df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

fig = plot_overview(df, resolution=1.0)
fig.savefig("tile_308_4690.png", dpi=150, bbox_inches="tight")
```

Produces a 2×2 figure with DSM+hillshade, RGB orthoimage, intensity, and classification map.

### Individual 2-D panels

```python
from alsdb.utils.viz import plot_dsm, plot_rgb, plot_intensity, plot_classification

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_dsm(df, resolution=1.0, hillshade=True, ax=axes[0])
plot_rgb(df, resolution=1.0, ax=axes[1])
```

| Function | What it shows |
|----------|--------------|
| `plot_dsm()` | Max-Z raster with optional hillshade |
| `plot_rgb()` | RGB orthoimage (percentile contrast stretch) |
| `plot_intensity()` | Mean return intensity (greyscale) |
| `plot_classification()` | LAS class codes with standard colour palette |

### GEDI waveform

```python
from alsdb.utils.viz import plot_waveform, plot_rh_profile

result = simulate_waveform(reader, center_x=308_500, center_y=4_689_000, year=2021)

# Raw waveform vs elevation with annotated RH levels
fig = plot_waveform(result)
fig.savefig("waveform_308500_4689000.png", dpi=150, bbox_inches="tight")

# GEDI L2A style: RH(p) curve + W(h) waveform with layer detection
fig = plot_rh_profile(result)
fig.savefig("rh_profile_308500_4689000.png", dpi=150, bbox_inches="tight")
```

`plot_waveform` has two panels:

- **Left** — normalised waveform energy vs elevation, with the ground return shaded brown, the canopy layer shaded green, and RH25/50/75/95/100 annotated as horizontal lines. An info box shows canopy cover, HOME (RH50), and point count.
- **Right** — horizontal bar chart of RH heights above ground for quick comparison across footprints.

`plot_rh_profile` matches the GEDI L2A canonical representation:

- **(a)** RH(p) curve — height above ground vs percent cumulative energy, with understory (green) and overstory (orange) layer shading
- **(b)** W(h) = dE/dh — normalised waveform energy density vs height, with auto-detected layer peaks annotated, layer boundary, Δh inter-layer distance, and the fraction of energy below the split printed

### 3-D waveform waterfall

Visualise all simulated shots as a waterfall of RH(p) curves — each shot is a vertical ribbon at its X position, with cumulative energy (0–100 %) on the Y axis and height above ground on Z, coloured by a summary metric.

```python
from alsdb.utils.viz import plot_waveforms_3d

# Static matplotlib figure, coloured by RH98
fig = plot_waveforms_3d(results, color_by="rh98", backend="matplotlib")
fig.savefig("waveforms_3d.png", dpi=150, bbox_inches="tight")

# Interactive plotly version (better for dense grids)
fig = plot_waveforms_3d(results, color_by="cover", backend="plotly")
fig.show()
```

`color_by` accepts any column in the `simulate_batch` output: `"rh50"`, `"rh98"`, `"cover"`, `"z_ground"`, etc.

### Raster products (from GeoTIFF)

```python
from alsdb.utils.viz_raster import plot_chm, plot_agb, plot_products, plot_products_agb

# Individual panels
plot_chm("outputs/chm.tif")
plot_agb("outputs/agb.tif")

# Three-panel overview: DTM | DSM | CHM
plot_products("outputs/dtm.tif", "outputs/dsm.tif", "outputs/chm.tif")

# Four-panel overview: DTM | DSM | CHM | AGB
plot_products_agb("outputs/dtm.tif", "outputs/dsm.tif", "outputs/chm.tif", "outputs/agb.tif")
```

### 3-D point cloud

```python
from alsdb.utils.viz import plot_pointcloud_3d

plot_pointcloud_3d(df, color_by="Z",              backend="matplotlib")
plot_pointcloud_3d(df, color_by="RGB",            backend="plotly")
plot_pointcloud_3d(df, color_by="Classification", max_points=100_000)
```

---

## CLI

```bash
# Ingest a tile
alsdb ingest tile.laz my_array

# Ingest to S3
alsdb ingest tile.laz s3://owner.bucket/als_array \
    --storage-type s3 \
    --s3-url https://s3.example.com \
    --s3-access-key KEY --s3-secret-key SECRET

# Show file metadata
alsdb info tile.laz

# Filter to ground + vegetation classes only
alsdb ingest tile.laz my_array -c 2 -c 3 -c 4 -c 5

# Verbose logging
alsdb -v ingest tile.laz my_array
```

---

## Architecture

```
LAZ file
   │
   ▼
PDAL (readers.las)
   │  year / bbox / CRS ← LAZ header
   ▼
ALSTile.iter_chunks()       ← optional classification filter
   │  X, Y, attrs numpy arrays (1 M pts/chunk)
   ▼
ALSDatabase.write()
   │  TileDB sparse array  (X × Y × Year)
   │  ZSTD-9 compression on all attributes + coordinates
   │  allows_duplicates=True  (multiple returns per XY)
   ▼
TileDB array (local  /  s3://)
   │
   ├── ALSProvider.query_bbox()     → pandas / xarray
   ├── processing.chm               → GeoTIFF (CHM / DTM / DSM)
   ├── processing.biomass           → AGB raster
   └── processing.waveform          → GEDI-like RH metrics
```

**TileDB schema**

| Dimension | Type | Role |
|-----------|------|------|
| `X` | float64 | UTM easting (m) |
| `Y` | float64 | UTM northing (m) |
| `Year` | int16 | Survey year |

18 LAS attributes (Z, Intensity, ReturnNumber, Classification, RGB, …) are stored as TileDB attributes with ZSTD-9 compression. Spatial tile size defaults to 500 × 500 m; domain bounds are selected automatically per CRS.

Each ingested tile becomes a new TileDB **fragment**. Fragments are consolidated every `consolidate_every` tiles (default 50) to keep read performance healthy as the array grows.

---

## License

EUPL-1.2 — see [LICENSE](LICENSE).

© 2026 Simon Besnard, Helmholtz Centre Potsdam – GFZ German Research Centre for Geosciences.
