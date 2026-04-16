.. _fundamentals-tiledb-processing:

Processing Pipeline
===================

All processing functions follow the same pattern: query a TileDB bounding box, compute the product tile by tile, write results directly to an ``ALSZarrStore``. This page documents each product and its parameters.

.. important::

   All processing functions are **idempotent by default** (``overwrite=False``). If data already exists for the requested variable, resolution, and year the function returns immediately without recomputing. Pass ``overwrite=True`` to force recomputation.

Store setup
-----------

Create a single ``ALSZarrStore`` to hold all products for a study area:

.. code-block:: python

    from alsdb import ALSProvider
    from alsdb.storage import ALSZarrStore

    reader = ALSProvider(storage_type="local", uri="my_array")
    store  = ALSZarrStore("output/forest.zarr")

All processing functions accept ``provider``, ``store``, ``resolution``, ``bbox``, and ``year`` as their first arguments. If ``year`` or ``bbox`` does not overlap any stored data a ``WARNING`` is logged and the function returns immediately.

Canopy Height Model / DTM / DSM
---------------------------------

.. code-block:: python

    from alsdb.processing.chm import compute_chm, compute_dtm, compute_dsm, compute_all

    # CHM at 1 m resolution
    compute_chm(
        provider=reader,
        store=store,
        resolution=1.0,
        bbox=(308_000, 4_688_000, 310_000, 4_690_000),
        year=2021,
    )

    # Large area: tiled (500 m sub-tiles, 50 m HAG buffer, 4 workers)
    compute_chm(
        provider=reader,
        store=store,
        resolution=1.0,
        year=2021,
        tile_size=500.0,
        tile_buffer=50.0,
        n_workers=4,
    )

    # Compute DTM + DSM + CHM in one call
    compute_all(
        provider=reader,
        store=store,
        resolution=1.0,
        year=2021,
        tile_size=500.0,
        tile_buffer=50.0,
        n_workers=4,
    )

**How it works:**

The pipeline queries the TileDB array, runs PDAL's ``filters.hag_delaunay`` to compute height-above-ground, then rasterises with ``scipy.stats.binned_statistic_2d``:

- **DTM**: minimum ``Z`` of ground points (``Classification == 2``) per cell.
- **DSM**: maximum ``Z`` of first returns per cell.
- **CHM**: ``HeightAboveGround`` 95th-percentile of vegetation returns per cell (equivalent to maximum canopy surface height; more robust to outliers than DSM − DTM).

Sub-tiles with no returns remain ``NaN`` in the store. The 50 m buffer is used for CHM only; DTM and DSM do not require it.

Gap fraction and effective LAI
--------------------------------

.. code-block:: python

    from alsdb.processing.gap import compute_gap_fraction

    # Gap fraction only
    compute_gap_fraction(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
    )

    # Gap fraction + effective LAI via Beer–Lambert
    compute_gap_fraction(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        lai=True,
        k=0.5,       # extinction coefficient (spherical leaf angle distribution)
    )

    # Large area: tiled
    compute_gap_fraction(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        tile_size=500.0,
        tile_buffer=50.0,
        n_workers=4,
    )

**Gap fraction estimator (MacArthur–Wilson):**

.. math::

   P_\text{gap} = \frac{N_\text{ground, first}}{N_\text{ground, first} + N_\text{veg, first}}

where ground = ``Classification == 2``, vegetation = ``Classification`` ∈ {3, 4, 5}, and only first returns (``ReturnNumber == 1``) are used. Cells with no first returns are ``NaN``.

**Effective LAI (Beer–Lambert):**

.. math::

   L_e = -\ln(P_\text{gap}) \, / \, k

capped at 10 m² m⁻² to suppress noise in near-zero gap-fraction cells. Set ``lai=False`` (default) to skip LAI computation and store only gap fraction.

LiDAR structural metrics
--------------------------

.. code-block:: python

    from alsdb.processing.biomass import compute_metrics

    compute_metrics(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
    )

Writes six variables to the store:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Variable
     - Description
   * - ``h50``
     - 50th percentile of vegetation HeightAboveGround (m)
   * - ``h75``
     - 75th percentile (m)
   * - ``h95``
     - 95th percentile (m) (commonly used as a proxy for top-of-canopy height)
   * - ``hmean``
     - Mean vegetation HeightAboveGround (m)
   * - ``cc``
     - Canopy cover fraction (vegetation first returns / all first returns)
   * - ``density``
     - Total point density (all returns, pts/m²)

Aboveground biomass
---------------------

.. code-block:: python

    from alsdb.processing.biomass import compute_biomass

    # Default Næsset (2002) allometric model
    compute_biomass(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
    )

    # Large area: tiled
    compute_biomass(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        tile_size=500.0,
        tile_buffer=50.0,
        n_workers=4,
    )

**Default model (i.e., Næsset (2002) power law):**

.. math::

   \text{AGB} = a \cdot h_{95}^{b} \cdot cc^{c}

Default parameters: ``a=0.8``, ``b=1.8``, ``c=0.5``. Output in Mg ha⁻¹.

.. warning::

   The default parameters are generic and **must be calibrated** against field inventory plots before using the results scientifically. Use ``model_fn`` to supply a calibrated model (see below).

**Custom allometric model:**

.. code-block:: python

    # Simple custom model
    def my_model(metrics):
        return 1.2 * metrics["h95"] ** 2.1 * metrics["cc"] ** 0.6

    compute_biomass(provider=reader, store=store, resolution=10.0,
                    year=2021, model_fn=my_model)

**scikit-learn model via wrap_sklearn_model:**

Any scikit-learn-compatible estimator can be used. ``wrap_sklearn_model`` handles the ``(ny, nx)`` → ``(n_pixels, n_features)`` reshape and NaN masking automatically:

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    from alsdb.processing.biomass import compute_biomass, wrap_sklearn_model

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_agb)   # columns: h50, h75, h95, hmean, cc, density

    compute_biomass(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        model_fn=wrap_sklearn_model(rf),
    )

    # Custom feature subset (must match training order)
    compute_biomass(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        model_fn=wrap_sklearn_model(rf, features=["h95", "cc", "density"]),
    )

GEDI waveform simulation
-------------------------

.. code-block:: python

    from alsdb.processing.waveform import simulate_waveform, simulate_batch

    # Single footprint (25 m diameter, GEDI-like)
    result = simulate_waveform(
        provider=reader,
        center_x=308_500.0,
        center_y=4_689_000.0,
        footprint_radius=12.5,
        year=2021,
    )

    print(result.rh[50])    # RH50 (height above ground at 50 % cumulative energy (m))
    print(result.rh[98])    # RH98 (equivalent to GEDI L2A rh98)
    print(result.cover)     # canopy cover fraction
    print(result.z_ground)  # estimated ground elevation (m)

    # Batch: shots must be a DataFrame with center_x / center_y columns (UTM)
    import pandas as pd
    import numpy as np

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
        output_path="shots_2021.parquet",   # optional Parquet output
    )
    print(results[["center_x", "center_y", "rh50", "rh98", "cover"]].head())

**Simulation algorithm:**

1. Query all returns within the footprint circle.
2. Build a vertical histogram of returns in 0.15 m bins.
3. Convolve with a Gaussian pulse (σ = 0.64 m for full-waveform energy, 0.93 m for canopy cover).
4. Detect the ground return as the lowest prominent peak.
5. Compute cumulative relative height (RH) metrics from ground up: RH0, RH10, …, RH100.

The output ``results`` DataFrame has one row per shot and columns: ``z_ground``, ``home`` (height of median energy, equivalent to RH50), ``cover``, ``rh0`` … ``rh100``.

Reading results
---------------

All products are accessible from the same store:

.. code-block:: python

    ds1m  = store.to_dataset(resolution=1.0)
    ds10m = store.to_dataset(resolution=10.0)

    chm  = ds1m["chm"].sel(time=2021)       # (ny, nx) DataArray
    agb  = ds10m["biomass"].sel(time=2021)
    h95  = ds10m["h95"].sel(time=2021)
    gap  = ds10m["gap"].sel(time=2021)

    # CRS is attached via rioxarray
    print(ds1m.rio.crs)    # e.g. "EPSG:25830"

    # List what is in the store
    print(store.resolutions())         # [1.0, 10.0]
    print(store.variables(10.0))       # ['biomass', 'cc', 'density', 'gap', ...]
