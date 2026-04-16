.. _fundamentals-processor:

################
Data Ingestion
################

:py:class:`alsdb.ALSDatabase` manages the entire ingestion workflow: reading LAZ files via PDAL, writing point data to the TileDB array, tracking ingestion status in a manifest, and consolidating fragments for query performance.

Overview of the ingestion workflow
------------------------------------

1. **Read the LAZ header**: year, CRS, bounding box, and tile name are extracted from the file using PDAL metadata and the ``PNOATileName`` / ``GenericTileName`` parser.

2. **Check the manifest**: if the file has already been ingested (same filename, status ``"ingested"``), the tile is skipped. Pass ``overwrite=True`` to force re-ingestion.

3. **Read points in chunks**: ``ALSTile.iter_chunks()`` yields ~1 M points at a time as structured NumPy arrays. An optional classification filter can restrict ingestion to specific LAS class codes (e.g. ground + vegetation only).

4. **Write to TileDB**: ``ALSDatabase.write()`` appends points to the sparse array. Each chunk creates or extends a TileDB fragment. Coordinates are clipped to the array domain bounds.

5. **Update the manifest**: on success the manifest entry is marked ``"ingested"`` with point count, CRS, and bbox recorded.

6. **Consolidate (optional)**: after every ``consolidate_every`` tiles ``ingest_many()`` triggers ``ALSDatabase.consolidate()`` to merge fragments.

Example usage
-------------

**Single tile**

.. code-block:: python

    import alsdb
    from alsdb import ALSDatabase

    alsdb.setup_logging()

    db = ALSDatabase(storage_type="local", uri="my_array")
    db.ingest("path/to/tile.laz")

**Batch ingest (parallel, with consolidation)**

.. code-block:: python

    from pathlib import Path

    paths = sorted(Path("/data/als/").glob("*.laz"))

    db.ingest_many(
        paths,
        max_workers=8,          # parallel LAZ readers / TileDB writers
        consolidate_every=50,   # consolidate fragments every 50 tiles
    )

**Filter to specific LAS classes**

Pass a list of LAS class codes to ``ingest()`` or ``ingest_many()`` to restrict which points are stored. This reduces array size when only ground (2) and first vegetation returns (3–5) are needed:

.. code-block:: python

    db.ingest("tile.laz", classes=[2, 3, 4, 5])

**Re-ingest from scratch**

.. code-block:: python

    db.ingest_many(paths, overwrite=True)

Inspecting the manifest
------------------------

The manifest records the ingestion status of every file. It is stored as JSON alongside the TileDB array.

.. code-block:: python

    for entry in db.list_ingested():
        print(entry["filename"], entry["year"], entry["n_points"], entry["status"])

    # Check stored CRS
    print(db.stored_crs())   # e.g. "EPSG:25830"

Fragment consolidation
-----------------------

Every ingested tile creates a new TileDB *fragment*. After many ingestions, fragment count grows and the overhead of opening/scanning fragments during queries increases. Consolidation merges fragments into larger ones:

.. code-block:: python

    db.consolidate()

By default ``ingest_many()`` consolidates every 50 tiles automatically. For very large campaigns you may want to run a final consolidation after all tiles are ingested:

.. code-block:: python

    db.ingest_many(paths, max_workers=8, consolidate_every=500)
    db.consolidate()   # final consolidation

Performance considerations
--------------------------

- Use ``max_workers=8`` or more for large campaigns. Each worker reads a LAZ file and writes to TileDB independently.
- Keep ``consolidate_every`` between 50 and 200. Too frequent consolidation adds overhead; too infrequent degrades query performance.
- On S3, set ``multipart_part_size`` in the TileDB S3 config to at least 50 MB for efficient multipart uploads.
- The TileDB domain bounds are set once at array creation. If you later ingest data outside the initial domain, points will be silently clipped. Choose domain bounds conservatively (see :ref:`fundamentals-setup`).
