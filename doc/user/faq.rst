.. _faq:

################################
Frequently Asked Questions (FAQ)
################################

How should I cite alsDB?
-------------------------

Please use the following citation when referencing alsDB in your work:

  Besnard, S. alsDB [Computer software]. https://github.com/simonbesnard1/alsdb

Why can't I install alsDB with pip alone?
------------------------------------------

``pdal`` and ``python-pdal`` are only available through conda-forge and cannot be installed via pip. Use `pixi <https://pixi.sh>`_ which resolves both conda-forge and PyPI dependencies automatically:

.. code-block:: bash

    git clone https://github.com/simonbesnard1/alsdb.git
    cd alsdb && pixi install

See :ref:`installing` for full instructions.

Why does ingestion say "already ingested, skipping"?
------------------------------------------------------

The manifest records every file that has been successfully ingested. Re-running ``ingest()`` or ``ingest_many()`` on the same files is safe (i.e., they are skipped by default). To force re-ingestion, pass ``overwrite=True``:

.. code-block:: python

    db.ingest("tile.laz", overwrite=True)
    db.ingest_many(paths, overwrite=True)

What datasets does alsDB support?
-----------------------------------

alsDB is dataset-agnostic. CRS, bounding box, and acquisition year are read from the LAZ/LAS header automatically via PDAL. Any dataset that follows the LAS specification works, including:

- Spanish PNOA (Plan Nacional de Ortofotografía Aérea)
- German ATKIS DGM
- French RGE ALTI
- UK Environment Agency National LiDAR Programme
- US 3DEP (USGS 3D Elevation Program)
- Any custom or research ALS campaign

For PNOA tiles, the tile name parser (``PNOATileName``) extracts year, CRS, and bounding box from the filename as a fallback if the LAZ header is incomplete.

What happens to cells with no data?
-------------------------------------

Cells in the Zarr store that have no contributing points remain ``NaN``. This happens for:

- Sub-tiles outside the flight swath.
- Cells where all points were filtered out by the classification filter.
- Cells in raster products where insufficient points exist to compute a statistic.

``NaN`` cells are preserved faithfully through all processing steps and are visible in ``to_dataset()`` output.

How do I process multiple survey years?
-----------------------------------------

Run the same processing function for each year. The ``overwrite=False`` default ensures existing years are never recomputed:

.. code-block:: python

    for year in [2017, 2021, 2023]:
        compute_chm(provider=reader, store=store, resolution=1.0,
                    year=year, tile_size=500.0, n_workers=4)

Results are stored in the time axis of the Zarr arrays and can be accessed with ``.sel(time=year)``.

My query returns empty results, what is wrong?
-------------------------------------------------

Check the following:

1. **Year**: call ``reader.available_years()`` to confirm the year is stored.
2. **Bounding box**: the coordinates must be in the same CRS as the array (typically UTM). Passing geographic coordinates (lon/lat) to a UTM array returns no data.
3. **Fragment consolidation**: if you just ingested data, try ``db.consolidate()`` first.
4. **Domain bounds**: data ingested outside the TileDB domain bounds is silently clipped. Check ``db.stored_crs()`` and verify the coordinates.

Why is my CHM noisy at tile edges?
------------------------------------

This is a HAG artefact. The Delaunay TIN built from ground points becomes unreliable near tile edges where only one side of the boundary has data. Increase ``tile_buffer`` from 50 m to 100 m:

.. code-block:: python

    compute_chm(provider=reader, store=store, resolution=1.0,
                year=2021, tile_size=500.0, tile_buffer=100.0)

How do I use a scikit-learn biomass model?
-------------------------------------------

Use :py:func:`alsdb.processing.biomass.wrap_sklearn_model` to wrap any sklearn-compatible estimator. The features must be in the same order as the training data. The default feature set is all 16 metrics produced by ``compute_metrics()`` (``_METRIC_NAMES``): h50, h75, h95, hmax, hmean, cc, density, fhd, vci, crr, pv_0_2, pv_2_5, pv_5_10, pv_10_20, pv_20_40, pv_above40. Pass an explicit ``features`` list when the model was trained on a subset:

.. code-block:: python

    from alsdb.processing.biomass import compute_biomass, wrap_sklearn_model
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor()
    model.fit(X_train, y_agb)   # X columns must match _METRIC_NAMES order

    compute_biomass(
        provider=reader, store=store, resolution=10.0, year=2021,
        model_fn=wrap_sklearn_model(model),
    )

    # If the model was trained on a subset, declare the feature order explicitly:
    model_subset = GradientBoostingRegressor()
    model_subset.fit(X_train[["h95", "cc", "density"]], y_agb)
    compute_biomass(
        provider=reader, store=store, resolution=10.0, year=2021,
        model_fn=wrap_sklearn_model(model_subset, features=["h95", "cc", "density"]),
    )

How do I detect forest change between two surveys?
----------------------------------------------------

Use :py:func:`alsdb.processing.change.compute_change` after computing the same variable for both years.  Three derived products are written: absolute change, relative change (%), and a gain/loss flag:

.. code-block:: python

    from alsdb.processing.change import compute_change

    # CHM must already be present for both years in the store
    compute_change(
        store, "chm",
        year_from=2017, year_to=2021,
        resolution=1.0,
        min_delta=0.5,    # suppress sub-0.5 m noise in the flag layer
        pct_min_abs=0.5,  # avoid extreme % changes where base value < 0.5 m
    )

    ds = store.to_dataset(resolution=1.0)
    flag = ds["chm_change_flag"].sel(time=2021)   # +1 gain, −1 loss, 0 stable

How do I calibrate the Næsset biomass model?
----------------------------------------------

Use :py:func:`alsdb.processing.biomass.calibrate_naesset` with co-located field inventory plots.  At least 20 valid plots are required; fewer than 50 emits a quality warning:

.. code-block:: python

    from alsdb.processing.biomass import calibrate_naesset, naesset_model

    # h95_plots, cc_plots, agb_plots — 1-D arrays, one value per field plot
    (a, b, c), pcov = calibrate_naesset(
        h95_plots, cc_plots, agb_plots, return_cov=True
    )
    import numpy as np
    print(f"a={a:.3f} ± {np.sqrt(pcov[0,0]):.3f}")

    compute_biomass(
        provider=reader, store=store, resolution=10.0, year=2021,
        model_fn=lambda m: naesset_model(m, a=a, b=b, c=c),
    )

Can I store the Zarr output on S3?
------------------------------------

Yes. Pass an ``s3://`` URI and ``storage_options`` with your credentials:

.. code-block:: python

    store = ALSZarrStore(
        "s3://my-bucket/forest.zarr",
        storage_options={"key": "...", "secret": "...", "endpoint_url": "..."},
    )

See :ref:`fundamentals-s3` for a complete S3 setup guide.

How do I contribute to alsDB?
-------------------------------

Contributions are welcome! See :ref:`devindex` for guidelines on submitting bug reports, feature requests, and pull requests.
