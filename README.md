<p align="center">
  <a href="https://github.com/simonbesnard1/alsdb">
    <img src="https://github.com/simonbesnard1/alsdb/blob/main/doc/_static/logos/alsdb_logo.png"
         alt="alsdb Logo" height="160px">
  </a>
</p>

<h3 align="center">Airborne Laser Scanning point clouds - from LAZ to TileDB at scale</h3>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="https://tiledb.com"><img src="https://img.shields.io/badge/storage-TileDB-orange.svg" alt="TileDB"></a>
  <a href="https://pdal.io"><img src="https://img.shields.io/badge/processing-PDAL-green.svg" alt="PDAL"></a>
  <a href="https://opensource.org/licenses/EUPL-1.2"><img src="https://img.shields.io/badge/license-EUPL--1.2-blue.svg" alt="License"></a>
  <a href="https://github.com/simonbesnard1/alsdb/actions?query=workflow%3ACI"><img src="https://github.com/simonbesnard1/alsdb/actions/workflows/ci.yaml/badge.svg" alt="Pipelines"></a>
  <a href="https://codecov.io/gh/simonbesnard1/alsdb"><img src="https://codecov.io/gh/simonbesnard1/alsdb/branch/main/graph/badge.svg?flag=unittests" alt="Code coverage"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://alsdb.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/alsdb/badge/?version=latest" alt="Docs"></a>
</p>

---

**alsDB** reads LAZ/LAS files via [PDAL](https://pdal.io), stores them in a [TileDB](https://tiledb.com) sparse array (locally or on S3-compatible object storage), and provides a full processing pipeline for forest structure and biomass products.

The package is dataset-agnostic — CRS, bounding box, and acquisition year are read directly from the LAZ file header, so any national or global ALS dataset works without custom parsers.

## Features

| | |
|---|---|
| **Scalable ingestion** | Parallel batch ingest; each tile becomes a TileDB fragment |
| **Multi-temporal** | X / Y / Year dimensions; repeated surveys stored and queryable independently |
| **Ingestion manifest** | Tracks CRS, bbox, point count and status; re-ingestion is a no-op by default |
| **CRS-aware schema** | Domain bounds selected automatically from tile CRS or global fallback |
| **Local + S3 storage** | Identical API for filesystem paths and `s3://` URIs (tested on Ceph / RadosGW) |
| **Zarr gridded output** | CHM, DTM, DSM, AGB, gap fraction, LAI, and LiDAR metrics written directly to Zarr v3; no GeoTIFF intermediates, no mosaic step |
| **Tiled processing** | All products support `tile_size` / `n_workers` for large areas; parallel writes go to non-overlapping Zarr chunks |
| **GEDI simulation** | Full-waveform simulation and batch RH metric extraction at GEDI footprint scale |
| **CLI** | `alsdb ingest` and `alsdb info` commands |

---

## Installation

[pixi](https://pixi.sh) is the **only supported install method** — `pdal` and `python-pdal` are conda-only packages and cannot be installed via pip alone.

```bash
git clone https://github.com/simonbesnard1/alsdb.git
cd alsdb
pixi install          # resolves all conda-forge + pip dependencies in one step
pixi shell            # activate the environment
```

> **Note:** `pip install alsdb` will install the pure-Python dependencies but will fail to import without `pdal` present in your environment.

### Dependencies

| Package | Purpose |
|---------|---------|
| `pdal` / `python-pdal` | LAZ/LAS reading, HAG filter |
| `tiledb` | Sparse point-cloud storage (local + S3) |
| `zarr` | Zarr v3 store for gridded products |
| `numpy` / `scipy` | Array operations, rasterisation, peak detection |
| `pandas` / `xarray` | Query result formats |
| `pyarrow` | Parquet I/O for waveform batch results |
| `pyproj` | CRS parsing from WKT |
| `rioxarray` | CRS attachment on `to_dataset()` output (optional) |
| `matplotlib` / `plotly` | Visualisation |
| `click` | CLI |

---

## Quick start

```python
import alsdb
from alsdb import ALSDatabase, ALSProvider

alsdb.setup_logging()

# Ingest
db = ALSDatabase(storage_type="local", uri="my_array")
db.ingest("path/to/tile.laz")  # year, CRS, bbox from LAZ header
db.ingest_many(paths, max_workers=8)  # batch — already-ingested files skipped

# Query
reader = ALSProvider(storage_type="local", uri="my_array")
df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)
print(reader.available_years())  # [2019, 2021, 2023]
```

---

## Examples

Runnable Jupyter notebooks are in the [`examples/`](examples/) folder:

| Notebook | What it covers |
|---|---|
| [`01_ingest.ipynb`](examples/01_ingest.ipynb) | Single tile, batch ingest, manifest, S3 |
| [`02_processing.ipynb`](examples/02_processing.ipynb) | CHM / DTM / DSM, gap / LAI, metrics, biomass (Næsset + sklearn), xarray, change detection |
| [`03_waveform.ipynb`](examples/03_waveform.ipynb) | Single footprint, batch simulation, 3-D waterfall, GEDI comparison |
| [`04_visualisation.ipynb`](examples/04_visualisation.ipynb) | Point cloud plots, waveform plots, gridded product plots |

Full API documentation and a conceptual guide are on [Read the Docs](https://alsdb.readthedocs.io).

---

## CHM methods and uncertainty

See [the CHM guide](doc/chm_methods.md) for native constrained-Delaunay spike-free
processing, terrain support and quality flags, reproducible benchmarks, and
conditional uncertainty ensembles. The native backend is optional:

```bash
pixi run -e spikefree build-spikefree
```

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
ALSTile.iter_chunks()           ← optional classification filter
   │  X, Y, attrs numpy arrays (1 M pts/chunk)
   ▼
ALSDatabase.write()
   │  TileDB sparse array  (X × Y × Year)
   │  ByteShuffle+ZSTD for float64 X/Y; DoubleDelta+ZSTD for int16 Year
   │  allows_duplicates=True  (multiple returns per XY)
   ▼
TileDB array (local  /  s3://)
   │
   ├── ALSProvider.query_bbox()          → pandas / xarray
   │
   ├── processing.chm / gap / biomass
   │      │  ground TIN normalization + CHM reconstruction
   │      ▼
   │   ALSZarrStore  (Zarr v3, local / s3://)
   │      ├── 1m/   chm, dtm, dsm          (T × ny × nx) float32
   │      └── 10m/  gap, lai, biomass,
   │                h50…density            (T × ny × nx) float32
   │      store.to_dataset(resolution)     → xarray.Dataset (CRS-aware)
   │
   └── processing.waveform               → GEDI-like RH metrics
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

Forest processing now supports shared CHM/metrics/gap computation, selective and
reusable biomass metrics, batched footprints and streamed tree segmentation.
See [the forest processing guide](doc/forest_processing.md).
