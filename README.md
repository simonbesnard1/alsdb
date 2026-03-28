# alsdb

**alsdb** is a Python package for ingesting, storing, and processing Airborne Laser Scanning (ALS/LiDAR) point clouds at scale. It reads LAZ/LAS files via [PDAL](https://pdal.io), stores them in a [TileDB](https://tiledb.com) sparse array (locally or on S3-compatible object storage), and provides a processing pipeline for canopy height models (CHM), digital terrain/surface models (DTM/DSM), above-ground biomass (AGB) estimation, and GEDI waveform simulation.

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

The recommended way is [pixi](https://pixi.sh), which resolves the full conda + PyPI dependency stack in one step:

```bash
git clone https://github.com/your-org/alsdb.git
cd alsdb
pixi install
pixi shell          # activates the environment
```

### Manual (conda + pip)

```bash
conda create -n alsdb python=3.12
conda activate alsdb
conda install -c conda-forge pdal python-pdal tiledb numpy scipy rasterio matplotlib click xarray pandas pyproj
pip install -e .
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
from alsdb.processing.chm import compute_chm

compute_chm(
    provider=reader,
    bbox=(308_000, 4_688_000, 310_000, 4_690_000),
    resolution=1.0,           # metres per pixel
    output_dir="outputs/",
    year=2021,
)
# writes outputs/chm.tif, outputs/dtm.tif, outputs/dsm.tif
```

The pipeline queries the array, injects the point cloud into a PDAL pipeline (`filters.hag_delaunay` for height-above-ground), and writes GeoTIFFs via `writers.gdal`.

### Biomass estimation

```python
from alsdb.processing.biomass import compute_biomass

agb = compute_biomass(
    provider=reader,
    bbox=(308_000, 4_688_000, 310_000, 4_690_000),
    resolution=25.0,          # metres per pixel for the output raster
    year=2021,
)
# agb is a numpy array of AGB (Mg/ha) per pixel
```

Uses the Næsset (2002) power-law model: `AGB = a × h95^b × cc^c`, where `h95` is the 95th-percentile height, `cc` is canopy cover, and `a / b / c` are configurable coefficients.

Intermediate metrics (h50, h75, h95, hmean, canopy cover, point density) are also available via `compute_metrics()`.

### GEDI waveform simulation

```python
from alsdb.processing.waveform import simulate_waveform, simulate_batch

# Single footprint (25 m diameter, like GEDI)
result = simulate_waveform(
    provider=reader,
    lon=308_500.0, lat=4_689_000.0,
    footprint_radius=12.5,
    year=2021,
)
print(result.rh)         # {"rh25": 8.1, "rh50": 14.3, "rh75": 19.7, "rh95": 23.1, ...}
print(result.cover)      # canopy cover fraction
print(result.z_ground)   # estimated ground elevation (m)

# Batch over a list of (lon, lat) footprint centres
results = simulate_batch(
    provider=reader,
    centres=[(308_500, 4_689_000), (309_000, 4_689_500)],
    footprint_radius=12.5,
    year=2021,
    max_workers=4,
)
```

The simulator builds a vertical return histogram (0.15 m bins), convolves it with a Gaussian pulse (σ = 0.64 m full-waveform, 0.93 m cover), detects the ground return, and computes cumulative RH metrics from ground up.

---

## Visualisation

### Raster products

```python
from alsdb.utils.viz_raster import plot_products

plot_products("outputs/chm.tif", "outputs/dtm.tif", "outputs/dsm.tif")
```

### 3-D point cloud

```python
from alsdb.utils.viz import plot_pointcloud_3d

df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

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
