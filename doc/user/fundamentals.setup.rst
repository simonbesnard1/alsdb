.. _fundamentals-setup:

***********************
Storage Architecture
***********************

alsdb uses two complementary storage layers. Understanding how they fit together is the key to understanding the rest of the package.

Overview
--------

.. code-block:: text

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
      │
      ▼
   TileDB sparse array (local / s3://)
      │  dimensions: X (float64) × Y (float64) × Year (int16)
      │  allows_duplicates=True (multiple returns per XY)
      │
      ├── ALSProvider.query_bbox()  → pandas DataFrame / xarray Dataset
      │
      └── processing.*
             │  PDAL hag_delaunay + scipy binned_statistic_2d
             ▼
          ALSZarrStore  (Zarr v3, local / s3://)
             ├── 1m/   chm, dtm, dsm          (T × ny × nx) float32
             └── 10m/  gap, lai, biomass,
                       h50…density            (T × ny × nx) float32

Layer 1 — TileDB sparse array (point clouds)
---------------------------------------------

Each ingested LAZ file writes its points into a single shared **TileDB sparse array**. The array has three dimensions:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Dimension
     - Type
     - Role
   * - ``X``
     - float64
     - UTM easting (m)
   * - ``Y``
     - float64
     - UTM northing (m)
   * - ``Year``
     - int16
     - Survey acquisition year

All 18 standard LAS point attributes (``Z``, ``Intensity``, ``ReturnNumber``, ``Classification``, ``R``, ``G``, ``B``, …) are stored as TileDB attributes with ZSTD-9 compression. ``X`` and ``Y`` use ByteShuffle + ZSTD. ``Year`` uses DoubleDelta + ZSTD.

**Why TileDB?**

- **Sparse by design** — point density varies enormously; TileDB stores only occupied cells with no wasted space.
- **Fragment-based writes** — each ingested tile appends a new *fragment*. No locking, no global index rebuild. Multiple workers can write simultaneously.
- **Multi-temporal in one array** — surveys from 2019, 2021, and 2023 coexist in the same array. ``query_bbox(..., year=2021)`` reads only the 2021 fragments.
- **S3-native** — the array URI can be ``s3://bucket/als_array``; TileDB handles multipart I/O transparently.

**Fragment consolidation**

Each ingest call creates a new fragment. After many ingestions, fragment count grows and query overhead increases. ``ALSDatabase.consolidate()`` merges fragments to restore performance. ``ingest_many()`` calls consolidation automatically every ``consolidate_every`` tiles (default 50).

**Domain bounds**

The spatial domain (``X`` and ``Y`` extents of the TileDB array) is chosen automatically from the CRS of the first ingested tile. For UTM Zone 30N (EPSG:25830) the domain spans approximately [100 000, 1 000 000] × [0, 10 000 000]. For unknown CRS a global ±10⁷ fallback is used. The domain only needs to contain all future ingested data; it is set at array creation time and cannot be changed.

Layer 2 — Zarr v3 store (gridded products)
-------------------------------------------

All processing outputs (CHM, DTM, DSM, gap fraction, LAI, structural metrics, biomass) are written to an ``ALSZarrStore`` — a Zarr v3 hierarchy on disk or S3:

.. code-block:: text

   forest.zarr/
   ├── 1m/
   │   ├── chm     (T, ny, nx)  float32   ← Canopy Height Model
   │   ├── dtm     (T, ny, nx)  float32   ← Digital Terrain Model
   │   └── dsm     (T, ny, nx)  float32   ← Digital Surface Model
   └── 10m/
       ├── gap     (T, ny, nx)  float32   ← gap fraction
       ├── lai     (T, ny, nx)  float32   ← effective LAI
       ├── biomass (T, ny, nx)  float32   ← aboveground biomass (Mg/ha)
       ├── h50     (T, ny, nx)  float32   ← height percentile 50
       ├── h75     (T, ny, nx)  float32   ← height percentile 75
       ├── h95     (T, ny, nx)  float32   ← height percentile 95
       ├── hmean   (T, ny, nx)  float32   ← mean vegetation height
       ├── cc      (T, ny, nx)  float32   ← canopy cover fraction
       └── density (T, ny, nx)  float32   ← point density (pts/m²)

The ``T`` (time) axis stores survey years as integers (e.g. 2019, 2021, 2023). Each variable has an associated ``x`` and ``y`` coordinate array. Opening the store with ``to_dataset(resolution=1.0)`` returns an ``xarray.Dataset`` with CRS attached via ``rioxarray``.

**Why Zarr instead of GeoTIFF?**

- **No mosaic step** — each tile writes directly to its spatial slice. When all tiles are done the product is already a single, complete array.
- **Parallel writes** — non-overlapping crop bboxes guarantee no two workers write to the same chunk simultaneously.
- **Multi-resolution** — 1 m and 10 m products live in the same store file. ``to_dataset(resolution=1.0)`` or ``to_dataset(resolution=10.0)`` opens either.
- **Idempotent** — ``has_data(variable, resolution, year)`` checks whether data is already present; all compute functions skip already-written data unless ``overwrite=True``.

Tiling strategy
----------------

For areas larger than a single tile, processing is split into **500 m sub-tiles** (configurable via ``tile_size``). Each sub-tile has two bounding boxes:

``query_bbox``
  Inflated by ``tile_buffer`` (default 50 m) on all sides. Used for the TileDB query and for HAG computation via ``filters.hag_delaunay``. The larger neighbourhood ensures the Delaunay TIN is well-constrained near tile edges.

``crop_bbox``
  The original, non-overlapping tile extent. Only pixels inside this bbox are written to the Zarr store. This prevents double-writing at tile boundaries.

.. note::

   The 50 m buffer is important for **CHM accuracy** near tile edges. Without it, the ground TIN is under-constrained at the boundary and HAG values for vegetation near the edge can be wrong. DTM and DSM do not require a buffer because they do not depend on the HAG computation.

The combination of buffered query + non-overlapping write is the key design decision that allows correct, parallel, direct-to-Zarr tiled processing.
