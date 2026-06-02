.. _overview:

##############
Quick Overview
##############

This section provides a brief end-to-end example of the alsDB workflow: ingesting LAZ files, querying points, and computing a Canopy Height Model. For detailed explanations refer to :ref:`fundamentals`.

Setup
-----

.. code-block:: python

    import alsdb
    from alsdb import ALSDatabase, ALSProvider
    from alsdb.storage import ALSZarrStore

    alsdb.setup_logging()   # INFO-level logging to stderr

Ingest a LAZ tile
-----------------

:py:class:`alsdb.ALSDatabase` reads the LAZ header automatically to extract the acquisition year, CRS, and bounding box. The TileDB array is created on first write.

.. code-block:: python

    db = ALSDatabase(storage_type="local", uri="my_array")
    db.ingest("path/to/tile.laz")

To ingest a full directory in parallel:

.. code-block:: python

    from pathlib import Path

    paths = sorted(Path("/data/als/").glob("*.laz"))
    db.ingest_many(paths, max_workers=8, consolidate_every=50)

Already-ingested files (tracked in the manifest) are skipped automatically.

Query points
------------

:py:class:`alsdb.ALSProvider` queries the TileDB array by bounding box and year.

.. code-block:: python

    reader = ALSProvider(storage_type="local", uri="my_array")

    # All points in a bounding box (returns pandas DataFrame)
    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000)

    # Restrict to a single survey year
    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

    # Which survey years are stored?
    print(reader.available_years())   # e.g. [2019, 2021, 2023]

    # As xarray Dataset
    ds = reader.to_xarray(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

Compute a Canopy Height Model
-----------------------------

All processing functions write directly to an ``ALSZarrStore``: a Zarr v3 hierarchy that holds multiple resolutions and survey years.

.. code-block:: python

    from alsdb.processing.chm import compute_chm

    store = ALSZarrStore("output/forest.zarr")

    compute_chm(
        provider=reader,
        store=store,
        resolution=1.0,
        bbox=(308_000, 4_688_000, 310_000, 4_690_000),
        year=2021,
    )

For a large area, split into sub-tiles with a parallel worker pool:

.. code-block:: python

    compute_chm(
        provider=reader,
        store=store,
        resolution=1.0,
        year=2021,
        tile_size=500.0,    # 500 m sub-tiles
        tile_buffer=50.0,   # 50 m buffer for accurate HAG at tile edges
        n_workers=4,
    )

Read the result as xarray
--------------------------

.. code-block:: python

    ds = store.to_dataset(resolution=1.0)     # coords: time, y, x
    chm = ds["chm"].sel(time=2021)            # DataArray (ny, nx)

    # CRS is attached via rioxarray if available
    print(ds.rio.crs)

Compute biomass and structural metrics
---------------------------------------

.. code-block:: python

    from alsdb.processing.biomass import compute_biomass, compute_metrics

    # 16 LiDAR structural metrics at 10 m:
    # height percentiles (h50, h75, h95, hmax, hmean), canopy cover (cc),
    # point density, FHD, VCI, CRR, and 6 height-stratum proportions (pv_*)
    compute_metrics(
        provider=reader, store=store, resolution=10.0, year=2021,
        min_density=1.0,   # mask cells below 1 pt m⁻² (optional guard)
    )

    # Aboveground biomass using the Næsset (2002) model
    # Always calibrate coefficients against field plots before scientific use
    compute_biomass(provider=reader, store=store, resolution=10.0, year=2021)

    # Read results
    ds10 = store.to_dataset(resolution=10.0)
    agb = ds10["biomass"].sel(time=2021)

Multi-temporal change detection
---------------------------------

:py:func:`alsdb.processing.change.compute_change` computes pixel-wise absolute change,
relative change, and a gain/loss flag between two survey years:

.. code-block:: python

    from alsdb.processing.change import compute_change

    # Compute CHM change 2017 → 2021, ignoring sub-0.5 m differences
    compute_change(store, "chm", year_from=2017, year_to=2021,
                   resolution=1.0, min_delta=0.5, pct_min_abs=0.5)

    # Three products are written: chm_delta, chm_delta_pct, chm_change_flag
    ds = store.to_dataset(resolution=1.0)
    gain_loss = ds["chm_change_flag"].sel(time=2021)  # +1 gain, −1 loss, 0 stable

Simulate large-footprint waveforms
-----------------------------------

.. code-block:: python

    from alsdb.processing.waveform import simulate_waveform

    result = simulate_waveform(
        provider=reader,
        center_x=308_500.0, center_y=4_689_000.0,
        footprint_radius=12.5,
        year=2021,
    )
    print(result.rh[98])    # RH98 height above ground (m)
    print(result.cover)     # canopy cover fraction

---

For a complete explanation of these concepts, including the two-layer storage architecture, tiling strategy, and overwrite semantics, continue to :ref:`fundamentals`.
