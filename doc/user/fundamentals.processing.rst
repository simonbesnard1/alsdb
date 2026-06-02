.. _fundamentals-processing:

Processing Pipeline
===================

All processing functions share a common pattern: query a TileDB bounding box,
optionally reject ground point outliers, compute the product tile by tile, and
write results directly to an :class:`~alsdb.storage.ALSZarrStore`.

.. important::

   All processing functions are **idempotent by default** (``overwrite=False``).
   If data already exists for the requested variable, resolution, and year the
   function returns immediately without recomputing. Pass ``overwrite=True`` to
   force recomputation.

Common preprocessing
--------------------

Before any height-above-ground computation, ground points (``Classification == 2``)
are screened for outliers using two PDAL filters applied to the ground-point
subset only (vegetation classification is never changed):

- ``filters.elm`` (Extended Local Minimum) — detects below-ground misclassifications
  (multipath returns, water-surface reflections).
- ``filters.outlier`` (statistical mode) — detects above-ground spikes in the ground
  class (misclassified buildings or vegetation returns).

Height-above-ground is then computed with ``filters.hag_delaunay`` (Delaunay TIN,
the same method used by the DTM pipeline) with an automatic ``filters.hag_nn``
fallback for tiles that contain fewer than 3 ground points.

Store setup
-----------

Create a single ``ALSZarrStore`` to hold all products for a study area:

.. code-block:: python

    from alsdb import ALSProvider
    from alsdb.storage import ALSZarrStore

    reader = ALSProvider(storage_type="local", uri="my_array")
    store  = ALSZarrStore("output/forest.zarr")

All processing functions accept ``provider``, ``store``, ``resolution``, ``bbox``,
and ``year`` as their first arguments. If ``year`` or ``bbox`` does not overlap any
stored data a ``WARNING`` is logged and the function returns immediately.

Canopy Height Model / DTM / DSM
---------------------------------

.. code-block:: python

    from alsdb.processing.chm import compute_chm, compute_dtm, compute_dsm, compute_all

    # CHM at 1 m resolution (first returns only, max HAG per cell)
    compute_chm(
        provider=reader,
        store=store,
        resolution=1.0,
        bbox=(308_000, 4_688_000, 310_000, 4_690_000),
        year=2021,
    )

    # Large area: tiled (500 m sub-tiles, 50 m buffer, 4 workers)
    compute_chm(
        provider=reader,
        store=store,
        resolution=1.0,
        year=2021,
        tile_size=500.0,
        tile_buffer=50.0,  # buffer for accurate HAG at tile edges
        n_workers=4,
    )

    # DTM only — choose interpolation method
    compute_dtm(
        provider=reader,
        store=store,
        resolution=1.0,
        year=2021,
        dtm_method="tin",  # "tin" (default), "idw", or "min"
    )

    # Compute DTM + DSM + CHM in one tiled pass (recommended for large areas)
    compute_all(
        provider=reader,
        store=store,
        resolution=1.0,
        year=2021,
        tile_size=500.0,
        tile_buffer=50.0,
        n_workers=4,
        dtm_method="tin",
    )

**Product descriptions:**

- **DTM**: Delaunay TIN interpolation of ground points (``Classification == 2``)
  by default (``dtm_method="tin"``). IDW and minimum-Z binning are available as
  alternatives via ``dtm_method="idw"`` and ``dtm_method="min"``.
- **DSM**: maximum ``Z`` of first returns per cell (no HAG normalisation).
- **CHM**: maximum ``HeightAboveGround`` of vegetation first returns per cell
  (``Classification`` ∈ {3, 4, 5}, ``ReturnNumber == 1``).

Sub-tiles with no returns remain ``NaN`` in the store.

Gap fraction and effective LAI
--------------------------------

.. code-block:: python

    from alsdb.processing.gap import LAI_K_PRESETS, compute_gap_fraction

    # Gap fraction only
    compute_gap_fraction(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
    )

    # Gap fraction + effective LAI with clumping correction
    compute_gap_fraction(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        lai=True,
        k=LAI_K_PRESETS["spherical"],  # 0.5 — random leaf angles (default)
        clumping_index=0.7,            # Ω < 1 for clumped canopies (conifers)
        min_density=0.5,               # mask cells below 0.5 first returns m⁻²
        tile_size=500.0,
        n_workers=4,
    )

Preset extinction coefficients for common forest types:

.. code-block:: python

    print(LAI_K_PRESETS)
    # {"spherical": 0.5, "planophile": 0.8, "erectophile": 0.35, "conifer": 0.45}

**Gap fraction estimator (MacArthur–Wilson):**

.. math::

   P_\text{gap} = \frac{N_\text{ground, first}}{N_\text{ground, first} + N_\text{veg, first}}

where ground = ``Classification == 2``, vegetation = ``Classification`` ∈ {3, 4, 5},
and only first returns (``ReturnNumber == 1``) are counted. Unclassified, building,
and noise returns do not contribute to the denominator. Cells with no classified
first returns are ``NaN``.

**Effective LAI (Beer–Lambert with clumping correction):**

.. math::

   L_\text{true} = -\frac{\ln(P_\text{gap})}{k \cdot \Omega}

where *k* is the extinction coefficient and Ω is the element clumping index
(default 1.0 = no correction). Output is capped at 15 m² m⁻² to suppress noise
in near-zero gap-fraction cells.

LiDAR structural metrics
--------------------------

.. code-block:: python

    from alsdb.processing.biomass import compute_metrics

    compute_metrics(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        min_density=1.0,  # cells below 1 pt m⁻² receive NaN for all metrics
    )

Writes **16 variables** to the store:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Variable
     - Description
   * - ``h50``
     - 50th percentile of vegetation HeightAboveGround (m)
   * - ``h75``
     - 75th percentile (m)
   * - ``h95``
     - 95th percentile (m) — commonly used as a proxy for top-of-canopy height
   * - ``hmax``
     - Maximum HeightAboveGround of vegetation returns per cell (m)
   * - ``hmean``
     - Mean vegetation HeightAboveGround (m)
   * - ``cc``
     - Canopy cover fraction (vegetation first returns above ``cc_threshold``) / all first returns)
   * - ``density``
     - Total point density (all returns, pts m⁻²)
   * - ``fhd``
     - Foliage Height Diversity — Shannon entropy of the vertical HAG distribution
   * - ``vci``
     - Vegetation Complexity Index — FHD normalised to [0, 1]
   * - ``crr``
     - Canopy Relief Ratio = (hmean − hmin) / (hmax − hmin)
   * - ``pv_0_2``
     - Fraction of vegetation returns with 0 < HAG ≤ 2 m
   * - ``pv_2_5``
     - Fraction with 2 < HAG ≤ 5 m
   * - ``pv_5_10``
     - Fraction with 5 < HAG ≤ 10 m
   * - ``pv_10_20``
     - Fraction with 10 < HAG ≤ 20 m
   * - ``pv_20_40``
     - Fraction with 20 < HAG ≤ 40 m
   * - ``pv_above40``
     - Fraction with HAG > 40 m

The ``min_density`` guard sets all 16 metrics to ``NaN`` for cells below the
threshold, suppressing unreliable estimates in data-sparse areas.

Aboveground biomass
---------------------

.. code-block:: python

    from alsdb.processing.biomass import compute_biomass

    # Default Næsset (2002) allometric model — calibrate before scientific use
    compute_biomass(
        provider=reader,
        store=store,
        resolution=10.0,
        year=2021,
        tile_size=500.0,
        n_workers=4,
    )

**Calibrating the Næsset model:**

Use :func:`alsdb.processing.biomass.calibrate_naesset` with field inventory plots
co-located with ALS acquisitions (≥ 20 plots required; ≥ 50 recommended):

.. code-block:: python

    from alsdb.processing.biomass import calibrate_naesset, naesset_model

    (a, b, c), pcov = calibrate_naesset(
        h95_plots, cc_plots, agb_plots, return_cov=True
    )
    # pcov is the 3×3 parameter covariance matrix
    import numpy as np
    print(f"a = {a:.3f} ± {np.sqrt(pcov[0,0]):.3f}")

    compute_biomass(
        provider=reader, store=store, resolution=10.0, year=2021,
        model_fn=lambda m: naesset_model(m, a=a, b=b, c=c),
    )

**Default model (Næsset 2002 power law):**

.. math::

   \text{AGB} = a \cdot h_{95}^{b} \cdot cc^{c}

Default parameters: ``a=0.8``, ``b=1.8``, ``c=0.5``. Output in Mg ha⁻¹.

.. warning::

   The default parameters are generic placeholder values and **must be calibrated**
   against field inventory plots before using the results scientifically.

**scikit-learn model via** ``wrap_sklearn_model``:

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    from alsdb.processing.biomass import compute_biomass, wrap_sklearn_model

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    # X_train columns must match _METRIC_NAMES order (16 features by default)
    rf.fit(X_train, y_agb)

    compute_biomass(
        provider=reader, store=store, resolution=10.0, year=2021,
        model_fn=wrap_sklearn_model(rf),
    )

    # Explicit feature subset (must match the training column order)
    compute_biomass(
        provider=reader, store=store, resolution=10.0, year=2021,
        model_fn=wrap_sklearn_model(rf, features=["h95", "cc", "density"]),
    )

Multi-temporal change detection
----------------------------------

:func:`alsdb.processing.change.compute_change` computes pixel-wise change between
two survey years and writes three derived products:

.. code-block:: python

    from alsdb.processing.change import compute_change

    # CHM change 2017 → 2021; suppress sub-0.5 m noise
    compute_change(
        store, "chm",
        year_from=2017, year_to=2021,
        resolution=1.0,
        min_delta=0.5,    # absolute threshold — smaller changes flagged 0
        pct_min_abs=0.5,  # suppress % change where |year_from| < 0.5 m
    )

    ds = store.to_dataset(resolution=1.0)
    flag  = ds["chm_change_flag"].sel(time=2021)   # +1 gain / −1 loss / 0 stable
    delta = ds["chm_delta"].sel(time=2021)          # absolute change (m)

**Derived variables written per call:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``{variable}_delta``
     - Absolute change (``year_to − year_from``), same units as source
   * - ``{variable}_delta_pct``
     - Relative change (%), NaN where ``|year_from| < pct_min_abs``
   * - ``{variable}_change_flag``
     - +1 gain, −1 loss, 0 no significant change (controlled by ``min_delta``)

Large-footprint waveform simulation
-------------------------------------

.. code-block:: python

    from alsdb.processing.waveform import simulate_waveform, simulate_batch

    # Single footprint (25 m diameter, GEDI default)
    result = simulate_waveform(
        provider=reader,
        center_x=308_500.0,
        center_y=4_689_000.0,
        footprint_radius=12.5,
        year=2021,
        beam_id="BEAM0000",   # use empirical GEDI TX pulse kernel
    )

    print(result.rh[50])    # RH50 — height of median energy (m above ground)
    print(result.rh[98])    # RH98 — equivalent to GEDI L2A rh98
    print(result.cover)     # canopy cover fraction
    print(result.z_ground)  # estimated ground elevation (m)

    # Batch simulation — shots must be a DataFrame with center_x / center_y
    import numpy as np, pandas as pd

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
        output_path="shots_2021.parquet",  # optional Parquet output
        # rng=np.random.default_rng(42),  # for reproducible noise (noise_std > 0)
    )

**Simulation algorithm:**

1. Query all returns within the footprint circle (bounding box + circular mask).
2. Weight returns by the Gaussian beam profile (enabled by default).
3. Build a vertical histogram in 0.15 m bins.
4. Convolve with the empirical GEDI TX pulse kernel (per ``beam_id``) or a
   Gaussian fallback.
5. Detect the ground return as the lowest prominent peak.
6. Compute cumulative relative height (RH) metrics from ground up: RH0 … RH100.

Reading results
---------------

All products are accessible from the same store:

.. code-block:: python

    ds1m  = store.to_dataset(resolution=1.0)
    ds10m = store.to_dataset(resolution=10.0)

    chm      = ds1m["chm"].sel(time=2021)
    agb      = ds10m["biomass"].sel(time=2021)
    h95      = ds10m["h95"].sel(time=2021)
    crr      = ds10m["crr"].sel(time=2021)
    gap      = ds10m["gap"].sel(time=2021)
    pv_5_10  = ds10m["pv_5_10"].sel(time=2021)

    # CRS is attached via rioxarray
    print(ds1m.rio.crs)            # e.g. "EPSG:25830"
    print(store.resolutions())     # [1.0, 10.0]
    print(store.variables(10.0))   # all 16 metrics + biomass + gap + lai …
