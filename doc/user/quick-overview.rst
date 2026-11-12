.. _overview:

##############
Quick Overview
##############

This page runs the whole alsDB workflow on a sample tile that ships with the
package: ingest, query, terrain and canopy height, structural metrics and a
simulated GEDI waveform. Copy the blocks in order into a script or notebook;
they run as written (a test executes this page). It takes about ten seconds.

The sample is 100 m × 100 m of tropical forest in Acre, Brazil, from the 2014
survey of the Sustainable Landscapes Brazil project (dos-Santos, Keller and
Morton 2019, ORNL DAAC, https://doi.org/10.3334/ORNLDAAC/1644): about 66 points
per m², classified ground and vegetation, canopy up to ~45 m.

Setup
-----

.. code-block:: python

    import alsdb
    from alsdb import ALSDatabase, ALSProvider
    from alsdb.datasets import sample_info, sample_laz
    from alsdb.storage import ALSZarrStore

    alsdb.setup_logging()  # INFO-level progress messages

    info = sample_info()
    print(info["crs"], info["year"], info["bbox"])  # EPSG:32719 2014 (658180.0, ...)

Ingest a LAZ tile
-----------------

:py:class:`alsdb.ALSDatabase` stores the points in a TileDB array, under the
survey year and CRS read from the LAZ header. The array is created on first
write; ingesting the same file again is skipped (it is recorded in a manifest).

.. code-block:: python

    db = ALSDatabase(storage_type="local", uri="quickstart_array")
    db.ingest(sample_laz())

    print(db.stored_crs())          # EPSG:32719
    print(db.list_ingested()[0])    # file, year, CRS, bounds, number of points

For your own survey, ingest a whole folder in parallel:

.. code-block:: python
   :class: norun

    from pathlib import Path

    paths = sorted(Path("/data/als/").glob("*.laz"))
    db.ingest_many(paths, n_workers=8)

Query points
------------

:py:class:`alsdb.ALSProvider` reads points by bounding box and year.

.. code-block:: python

    reader = ALSProvider(storage_type="local", uri="quickstart_array")
    print(reader.available_years())  # [2014]

    df = reader.query_bbox(*info["bbox"], year=2014)  # pandas DataFrame
    print(len(df), "points")

    ds_points = reader.to_xarray(*info["bbox"], year=2014)  # the same as xarray

Terrain and canopy height
-------------------------

All products are written to an :py:class:`~alsdb.storage.ALSZarrStore`, a Zarr
store holding several resolutions and survey years. ``compute_all`` makes the
DTM, DSM and CHM in one pass; ``method`` chooses the CHM algorithm (see
:ref:`algorithms`).

.. code-block:: python

    from alsdb.products.chm import compute_all

    store = ALSZarrStore("quickstart.zarr")
    compute_all(reader, store, resolution=1.0, year=2014, method="pitfree")

    ds = store.to_dataset(resolution=1.0)  # xarray: dtm, dsm, chm and quality layers
    chm = ds["chm"].sel(time=2014)
    print(f"CHM: mean {float(chm.mean()):.1f} m, max {float(chm.max()):.1f} m")

    from alsdb.utils.viz_raster import plot_chm

    plot_chm(store, resolution=1.0, year=2014)

Large areas are processed in buffered tiles, in parallel:

.. code-block:: python
   :class: norun

    compute_all(reader, store, resolution=1.0, year=2021, method="pitfree",
                tile_size=500.0, tile_buffer=50.0, n_workers=8)

Structural metrics and biomass
------------------------------

``compute_metrics`` writes height percentiles, canopy cover, foliage height
diversity and height strata per cell.

.. code-block:: python

    from alsdb.products.biomass import compute_metrics

    compute_metrics(reader, store, resolution=10.0, year=2014)

    ds10 = store.to_dataset(resolution=10.0)
    print(f"h95 {float(ds10['h95'].mean()):.1f} m, cover {float(ds10['cc'].mean()):.2f}")

``compute_biomass`` applies a model to those metrics: the Næsset power law,
the GEDI L4A models, your own fitted model, or a scikit-learn or PyTorch
model. Its default coefficients are placeholders: calibrate them against field
plots first (``plot_metrics`` and ``calibrate_naesset``; see :ref:`algorithms`).

Simulate a GEDI waveform
------------------------

.. code-block:: python

    from alsdb.products.waveform import simulate_waveform

    result = simulate_waveform(reader, center_x=658_230.0, center_y=8_901_910.0, year=2014)
    print(f"RH98 {result.rh[98]:.1f} m, cover {result.cover:.2f}")

    from alsdb.utils.viz import plot_waveform

    plot_waveform(result)

Change between surveys
----------------------

With two survey years in the same array and store, ``compute_change`` maps the
difference of any product:

.. code-block:: python
   :class: norun

    from alsdb.products.change import compute_change

    compute_change(store, "chm", year_from=2017, year_to=2021, resolution=1.0, min_delta=0.5)
    # writes chm_delta, chm_delta_pct and chm_change_flag (+1 gain, -1 loss, 0 stable)

---

Next: :ref:`algorithms` for every method and its defaults, and
:ref:`fundamentals` for the storage design, tiling and overwrite rules.
