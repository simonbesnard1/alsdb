"""
Ingesting LAZ files into a TileDB array
========================================

This example shows how to ingest a single LAZ tile and a full directory of tiles
into a TileDB sparse array using :py:class:`alsdb.ALSDatabase`.

The TileDB array is created automatically on first write. CRS, bounding box, and
acquisition year are read from the LAZ header — no configuration file required.
"""

# %%
# Single tile ingest
# ------------------
#
# :py:meth:`alsdb.ALSDatabase.ingest` reads the LAZ header, writes all points to
# the TileDB array, and records the tile in the manifest.

import alsdb
from alsdb import ALSDatabase

alsdb.setup_logging()  # INFO-level logging to stderr

db = ALSDatabase(storage_type="local", uri="/path/to/my_array")

# Ingest a single tile — year, CRS, and bbox are read from the LAZ header
db.ingest("/path/to/tile.laz")

# Already-ingested tiles are skipped automatically (manifest tracking)
db.ingest("/path/to/tile.laz")   # no-op

# %%
# Inspect the manifest
# ---------------------
#
# The manifest tracks the status, point count, CRS, and bounding box of every
# ingested file.

for entry in db.list_ingested():
    print(f"{entry['filename']:40s}  year={entry['year']}  "
          f"n_points={entry['n_points']:>10,}  status={entry['status']}")

print("\nStored CRS:", db.stored_crs())

# %%
# Batch ingest — parallel, with consolidation
# --------------------------------------------
#
# :py:meth:`alsdb.ALSDatabase.ingest_many` processes a list of LAZ files in parallel
# using a ``ThreadPoolExecutor``. Fragment consolidation is triggered every
# ``consolidate_every`` tiles.

from pathlib import Path  # noqa: E402

paths = sorted(Path("/path/to/als_tiles/").glob("*.laz"))

db.ingest_many(
    paths,
    max_workers=8,          # number of parallel LAZ readers / TileDB writers
    consolidate_every=50,   # merge fragments every 50 tiles for better query performance
)

# %%
# Filter to specific LAS classes
# --------------------------------
#
# Storing only ground (2) and vegetation (3–5) points reduces array size and is
# sufficient for all forest structure products.

db.ingest(
    "/path/to/tile.laz",
    classes=[2, 3, 4, 5],   # ASPRS classification codes
)

# %%
# Ingest to S3
# -------------
#
# The API is identical for S3-backed arrays; only the constructor arguments change.

db_s3 = ALSDatabase(
    storage_type="s3",
    uri="s3://my-bucket/als_array",
    url="https://s3.example.com",     # endpoint URL (omit for AWS)
    region="eu-central-1",
    credentials={
        "AccessKeyId": "ACCESS_KEY",
        "SecretAccessKey": "SECRET_KEY",
    },
)

db_s3.ingest("/path/to/tile.laz")
