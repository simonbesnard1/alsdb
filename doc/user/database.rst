.. _database:

################################
Visualisation and Advanced Usage
################################

Visualisation
=============

alsdb includes a set of plot functions for both point-cloud data and gridded products from the ``ALSZarrStore``.

Point-cloud visualisation
--------------------------

All 2-D plot functions accept a point-cloud DataFrame from ``ALSProvider.query_bbox()`` and rasterise it to a regular grid before rendering.

.. code-block:: python

    from alsdb import ALSProvider
    from alsdb.utils.viz import plot_overview, plot_dsm, plot_rgb, plot_intensity, plot_classification

    reader = ALSProvider(storage_type="local", uri="my_array")
    df = reader.query_bbox(308_000, 4_688_000, 310_000, 4_690_000, year=2021)

    # 4-panel overview: DSM+hillshade, RGB, intensity, classification
    fig = plot_overview(df, resolution=1.0)
    fig.savefig("overview.png", dpi=150, bbox_inches="tight")

    # Individual panels
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_dsm(df, resolution=1.0, hillshade=True, ax=axes[0])
    plot_rgb(df, resolution=1.0, ax=axes[1])

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - What it shows
   * - ``plot_dsm()``
     - Max-Z raster with optional hillshade
   * - ``plot_rgb()``
     - RGB orthoimage (percentile contrast stretch)
   * - ``plot_intensity()``
     - Mean return intensity (greyscale)
   * - ``plot_classification()``
     - LAS class codes with standard colour palette

GEDI waveform visualisation
-----------------------------

.. code-block:: python

    from alsdb.processing.waveform import simulate_waveform
    from alsdb.utils.viz import plot_waveform, plot_rh_profile

    result = simulate_waveform(reader, center_x=308_500, center_y=4_689_000, year=2021)

    # Waveform energy vs elevation with annotated RH levels
    fig = plot_waveform(result)
    fig.savefig("waveform.png", dpi=150, bbox_inches="tight")

    # GEDI L2A style: RH(p) curve + W(h) energy density
    fig = plot_rh_profile(result)
    fig.savefig("rh_profile.png", dpi=150, bbox_inches="tight")

``plot_waveform`` produces a two-panel figure:

- **Left** — normalised waveform energy vs elevation, ground return shaded brown, canopy shaded green, RH25/50/75/95/100 annotated as horizontal lines.
- **Right** — horizontal bar chart of RH heights above ground.

``plot_rh_profile`` matches the GEDI L2A canonical representation:

- **(a)** RH(p) curve — height above ground vs cumulative energy percent, with understory and overstory layer shading.
- **(b)** W(h) = dE/dh — normalised energy density vs height with auto-detected layer peaks.

3-D waveform waterfall
~~~~~~~~~~~~~~~~~~~~~~~

Visualise all simulated shots as a waterfall of RH(p) curves:

.. code-block:: python

    from alsdb.utils.viz import plot_waveforms_3d

    fig = plot_waveforms_3d(results, color_by="rh98", backend="matplotlib")
    fig.savefig("waveforms_3d.png", dpi=150, bbox_inches="tight")

    # Interactive plotly version
    fig = plot_waveforms_3d(results, color_by="cover", backend="plotly")
    fig.show()

3-D point cloud
~~~~~~~~~~~~~~~~

.. code-block:: python

    from alsdb.utils.viz import plot_pointcloud_3d

    plot_pointcloud_3d(df, color_by="Z",              backend="matplotlib")
    plot_pointcloud_3d(df, color_by="RGB",            backend="plotly")
    plot_pointcloud_3d(df, color_by="Classification", max_points=100_000)

Gridded product visualisation
-------------------------------

All raster plot functions read from an ``ALSZarrStore`` at a given resolution and year:

.. code-block:: python

    from alsdb.storage import ALSZarrStore
    from alsdb.utils.viz_raster import (
        plot_chm, plot_dtm, plot_dsm,
        plot_agb, plot_gap, plot_lai,
        plot_metrics,
        plot_products, plot_products_agb,
    )

    store = ALSZarrStore("output/forest.zarr")

    # Individual product panels
    plot_chm(store, resolution=1.0,  year=2021)
    plot_dtm(store, resolution=1.0,  year=2021, hillshade=True)
    plot_dsm(store, resolution=1.0,  year=2021)
    plot_agb(store, resolution=10.0, year=2021)
    plot_gap(store, resolution=10.0, year=2021)
    plot_lai(store, resolution=10.0, year=2021)

    # Six structural metrics in one figure
    plot_metrics(store, resolution=10.0, year=2021)

    # Three-panel: DTM | DSM | CHM
    plot_products(store, resolution=1.0, year=2021)

    # Four-panel: DTM | DSM | CHM | AGB
    plot_products_agb(store, resolution=10.0, year=2021)

If the store contains only one survey year the ``year=`` argument can be omitted.

---

Advanced Usage
==============

ALSZarrStore internals
-----------------------

The ``ALSZarrStore`` is a thin wrapper around a Zarr v3 group hierarchy. For advanced use cases you can access the underlying Zarr groups directly:

.. code-block:: python

    from alsdb.storage import ALSZarrStore

    store = ALSZarrStore("output/forest.zarr")

    # List what is in the store
    print(store.resolutions())            # [1.0, 10.0]
    print(store.variables(resolution=1.0))   # ['chm', 'dsm', 'dtm']
    print(store.has_data("chm", 1.0, 2021))  # True / False

    # Open as xarray (CRS-aware via rioxarray)
    ds = store.to_dataset(resolution=1.0)
    print(ds)

    # Write a custom tile (advanced)
    import numpy as np
    data = np.zeros((100, 100), dtype=np.float32)
    bbox = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
    store.write_tile("my_variable", 10.0, 2021, data, bbox, crs="EPSG:25830")

The CLI
--------

.. code-block:: bash

    # Ingest a single tile (local)
    alsdb ingest tile.laz my_array

    # Ingest to S3
    alsdb ingest tile.laz s3://bucket/als_array \
        --storage-type s3 \
        --s3-url https://s3.example.com \
        --s3-access-key KEY --s3-secret-key SECRET

    # Filter to specific classes
    alsdb ingest tile.laz my_array -c 2 -c 3 -c 4 -c 5

    # Show file metadata (year, CRS, bbox, point count)
    alsdb info tile.laz

    # Verbose logging
    alsdb -v ingest tile.laz my_array

Multi-temporal workflows
-------------------------

Because the TileDB array stores all survey years together, comparing two surveys requires only a change in the ``year`` argument:

.. code-block:: python

    from alsdb.processing.chm import compute_chm

    for year in [2017, 2021]:
        compute_chm(provider=reader, store=store, resolution=1.0,
                    year=year, tile_size=500.0, n_workers=4)

    ds = store.to_dataset(resolution=1.0)
    chm_2017 = ds["chm"].sel(time=2017)
    chm_2021 = ds["chm"].sel(time=2021)
    delta_chm = chm_2021 - chm_2017   # canopy height change (m)

The ``overwrite=False`` default means re-running the pipeline for a new year never re-computes existing years.
