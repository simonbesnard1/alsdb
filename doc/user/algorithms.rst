.. _algorithms:

##########
Algorithms
##########

This page lists every algorithm alsDB offers, the defaults it uses and where it
comes from, so you can see exactly what a product contains and choose for your
own data. Each product records the method and every setting it was made with
in the store's provenance (see *Provenance and reruns* in :doc:`../chm_methods`). For the detailed behaviour,
see :doc:`../chm_methods`, :doc:`../area_products` and :doc:`../benchmarking`.

The processing chain is:

.. code-block:: text

   LAZ ─► ingest ─► [optional] ground classification ─► TileDB
   per tile: query ─► clean ─► ground model ─► heights above ground
     ├─► DTM, DSM, CHM           compute_dtm, compute_dsm, compute_chm, compute_all
     ├─► metrics, biomass        compute_metrics, compute_biomass
     ├─► gap fraction, LAI       compute_gap_fraction
     └─► waveforms, profiles     simulate_waveform, simulate_batch, simulate_gedi_profiles
   Zarr store ─► change between years   compute_change

Ground classification
=====================

By default alsDB keeps the **provider's ground classes** (2 ground, 9 water).
To classify at ingest, pass a ``ClassificationConfig`` to
``ALSDatabase.ingest`` / ``ingest_many``:

.. code-block:: python

   from alsdb.methods.classification import ClassificationConfig

   db.ingest_many(files, classification_config=ClassificationConfig(ground_method="ptd"))

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - ``ground_method``
     - Algorithm
     - Main settings (defaults)
   * - ``"csf"`` (default)
     - Cloth simulation: a cloth dropped on the inverted cloud (Zhang W. et al. 2016; PDAL ``filters.csf``)
     - ``csf_resolution`` 0.5 m, ``csf_threshold`` 0.5 m, ``csf_rigidness`` 1
   * - ``"pmf"``
     - Progressive morphological filter: openings with growing windows (Zhang K. et al. 2003; PDAL ``filters.pmf``)
     - ``pmf_max_window_size`` 33 m, ``pmf_slope`` 1, ``pmf_initial_distance`` 0.15 m, ``pmf_max_distance`` 2.5 m
   * - ``"smrf"``
     - Simple morphological filter with a slope rule (Pingel et al. 2013; PDAL ``filters.smrf``)
     - ``smrf_cell`` 1 m, ``smrf_slope`` 0.15, ``smrf_window`` 18 m, ``smrf_threshold`` 0.5 m, ``smrf_scalar`` 1.25
   * - ``"ptd"``
     - Progressive TIN densification (Axelsson 2000; r-lidar/PTD by J.-R. Roussel, vendored), run as a
       **tilt ensemble**: the cloud and copies tilted towards evenly spaced azimuths are classified, a point
       is ground when enough runs agree (idea of F. Fischer)
     - ``ptd_seed_resolution`` 10 m, ``ptd_angle`` 45°, ``ptd_distance`` 1 m, ``ptd_spacing`` 0.25 m,
       ``ptd_tilt_runs`` 8, ``ptd_tilt`` 10°, ``ptd_votes`` 3; ``ptd_tilt_runs=1`` is a single run
       (lasR's ``classify_with_ptd`` uses angle 30°, distance 2 m)

Options for all methods: optional outlier removal (``denoise``), and points of
``source_classes`` (default 1) sorted into low, medium and high vegetation
(classes 3, 4, 5) by height above ground (thresholds 0.5, 2 and 5 m);
existing classes are kept (``preserve_existing``).

*Which to use* (alsDB benchmark, 2,579 scenes): PTD and its tilt ensemble give
the lowest DTM errors; on hand-corrected IGN Lidar HD at 1 pulse/m² the DTM
MAE is 0.24 m (PTD), 0.42 (PMF), 0.59 (SMRF), 0.64 (CSF). The tilt ensemble
helps most on sparse data. Above 45° of slope every filter is off by 0.5 m or
more.

Terrain model and height above ground
=====================================

Every product that needs heights builds the same ground model per tile.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Step
     - What happens (defaults)
   * - Cleaning
     - Noise classes 7 and 18 and withheld points dropped; low ground outliers (multipath, water reflections)
       reclassified as noise with PDAL ``filters.elm`` (``ground_outlier_removal=True``)
   * - Ground model
     - Delaunay TIN of the ground classes (``ground_classes=(2, 9)``); in ``compute_dtm``,
       ``dtm_method="idw"`` (inverse distance weighting) or ``"min"`` (lowest ground return per cell)
   * - Beyond the ground
     - ``ground_extrapolation=True`` extends the model past the outermost ground points;
       ``max_ground_distance`` leaves points too far from any ground point without a height
   * - Tiles
     - Computed in buffered tiles (``tile_size``, ``tile_buffer``) so edges are seamless; threads or worker
       processes (``n_workers``, ``parallel="threads"|"processes"``)

Surfaces: DTM, DSM and CHM
==========================

``compute_dtm`` rasterises the ground model; ``compute_dsm`` takes the highest
first return per cell (noise excluded); ``compute_chm`` builds the canopy
height model from normalised vegetation **and** ground returns, so crown gaps
stay at ground level. ``compute_all`` makes all three at once.

.. list-table::
   :header-rows: 1
   :widths: 14 50 36

   * - ``method``
     - Algorithm
     - Main settings (defaults)
   * - ``"max"`` (default)
     - Highest return per cell (any statistic with ``height_statistic``)
     - ``first_returns_only=True``
   * - ``"tin"``
     - Delaunay TIN of all first returns, no pit protection (as Fischer et al. 2024 ``CHMtin``)
     - optional ``max_triangle_edge``
   * - ``"pitfree"``
     - Highest return per cell, then the maximum of TIN layers above height thresholds
       (Khosravipour et al. 2014, as lidR ``pitfree()``)
     - ``pitfree_thresholds`` 0, 2, 5, 10, 15, 20 m; ``pitfree_max_edge="auto"`` (3 × pulse spacing)
   * - ``"spikefree"``
     - Incremental constrained Delaunay from the top down; short triangles freeze so lower returns cannot
       cut pits (Khosravipour et al. 2016). Native C++ (CGAL) backend
     - ``freeze_distance="auto"`` (3 × first-return spacing), ``"local"`` (per-cell spacing, Fischer et al.
       2024) or metres; ``height_buffer`` 0.5 m; ``spikefree_first_returns``
   * - ``"lastools"``
     - The same algorithm through LAStools ``las2dem -spike_free`` (licensed, external)
     - ``lastools_executable``, ``freeze_interval``

Options: ``pit_fill`` fills small holes (up to ``fill_max_cells`` 4 cells);
``remove_outliers`` drops statistical outliers; ``max_height`` caps heights.
With ``quality=True`` every product gets quality layers: pulse density
(below ``min_pulse_density`` 2 pulses/m² flagged), scan angle (above
``max_scan_angle`` 20° flagged), distance to ground and slope.

*Which to use* (alsDB benchmark, 2,813 scenes, compared with lasR and LAStools):

- alsDB's pit-free equals lasR's (median difference +0.002 m); its spike-free
  equals LAStools' (0.03 m apart; lasR's spike-free freezes triangles later and
  sits ~0.15 m lower).
- **Pit-free** estimates the highest point of each cell (MAE 0.92 m on real
  ALS at 8 pulses/m²); **spike-free** estimates the canopy surface at the cell
  centre (0.87 m against that truth). Choose by what the product is for: tree
  heights and tops → pit-free; cover, gaps and surface roughness → spike-free.
- **Local spike-free** (``freeze_distance="local"``) varies least with pulse
  density, for surveys of uneven density.
- The plain TIN keeps every pit and is the least accurate.

Structural metrics and biomass
==============================

``compute_metrics`` writes, per cell, ``h50``, ``h75``, ``h95``, ``hmax``,
``hmean`` (vegetation heights), ``cc`` (canopy cover: share of classified first
returns above ``cc_threshold`` 2 m), ``density``, ``fhd`` (foliage height
diversity, Shannon entropy in 1 m bins), ``vci`` (FHD over its maximum),
``crr`` (canopy relief ratio) and the shares of returns in height strata
``pv_0_2`` … ``pv_above40``. With ``baba_radius`` the metrics are taken in a
circle around each cell centre (buffered area-based approach), e.g. a 25 m
circle matching field plots.

``compute_biomass`` applies a model to those metrics:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``model_fn``
     - Model
   * - ``naesset_model`` (default)
     - AGB = a · h95\ :sup:`b` · cc\ :sup:`c` (Næsset 2002). Placeholder coefficients warn: fit them with
       ``calibrate_naesset`` and check them with ``validate_naesset`` (spatial cross-validation)
   * - ``gedi_l4a_model(stratum)``
     - The GEDI L4A models (Kellner et al. 2023): 15 models for 33 plant functional type × region strata,
       checked against the GEDI04_A product (within 1 % on every shot of 15 strata). They take RH metrics
       of simulated 25 m waveforms, not point percentiles
   * - ``LinearAGBModel`` / ``fit_linear_agb_model``
     - Your own linear model on transformed metrics (log or square root, bias correction), with
       ``cross_validate``
   * - ``wrap_sklearn_model`` / ``wrap_torch_model``
     - Any trained scikit-learn model or PyTorch network (CPU or GPU)

For calibration, ``plot_metrics`` computes the same metrics over circular field
plots with exactly the code of ``compute_metrics(..., baba_radius=radius)``.
``python -m alsdb.benchmark calibrate-biomass`` runs the whole calibration
against the Spanish forest inventory: on 1,629 Barcelona plots the calibrated
Næsset model reaches an out-of-fold RMSE of 35 Mg/ha (R² 0.59).

Gap fraction and LAI
====================

``compute_gap_fraction`` writes the gap fraction: the share of classified first
returns that reach the ground, per cell or in a ``baba_radius`` circle. With
``lai=True`` it adds the leaf area index by Beer–Lambert:
LAI = −ln(gap) / (k · Ω), with extinction ``k`` (presets: spherical 0.5,
planophile 0.8, erectophile 0.35, conifer 0.45) and clumping index ``Ω``
(``clumping_index``, default 1; Chen and Black 1992, Jonckheere et al. 2004).

Waveforms and vertical profiles
===============================

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Function
     - Algorithm
   * - ``simulate_waveform`` / ``simulate_batch``
     - Large-footprint waveforms from the points, as the GEDI simulator (Hancock et al. 2019): Gaussian beam
       over a 25 m footprint, 15 ns pulse (or a measured GEDI pulse with ``beam_id``), returns weighted per
       1.5 m cell; RH metrics, cover, canopy and ground parts. Batches in worker processes
   * - ``compute_als_pavd_profile``
     - Plant area volume density profile of a footprint from the ALS returns (MacArthur and Horn 1969)
   * - ``simulate_gedi_profiles``
     - Simulated GEDI gap profiles (L2B formula) next to the ALS gap profiles of the same footprints
   * - ``run_gedi_attenuation``
     - Real GEDI shots from gedidb, the same footprints simulated from ALS, and the three canopy profiles
       compared (``alsdb.products.gedi_attenuation``)

Change between surveys
======================

``compute_change`` compares two years of any gridded variable in a store and
writes the change (``<variable>_delta``), the relative change in %
(``<variable>_delta_pct``) and a flag (+1 gain, −1 loss, 0 below
``min_delta``).

References
==========

- Axelsson, P. (2000). DEM generation from laser scanner data using adaptive TIN models. *IAPRS* 33(B4), 110–117.
- Chen, J. M., & Black, T. A. (1992). Defining leaf area index for non-flat leaves. *Plant, Cell & Environment* 15, 421–429.
- Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. (2024). Robust characterisation of forest structure from airborne laser scanning. *Methods Ecol. Evol.* 15, 1873–1888.
- Hancock, S., et al. (2019). The GEDI simulator. *Earth and Space Science* 6, 294–310.
- Jonckheere, I., et al. (2004). Review of methods for in situ leaf area index determination. *Agric. For. Meteorol.* 121, 19–35.
- Kellner, J. R., Armston, J., & Duncanson, L. (2023). Algorithm theoretical basis document for GEDI footprint aboveground biomass density. *Earth and Space Science* 10, e2022EA002516.
- Khosravipour, A., et al. (2014). Generating pit-free canopy height models from airborne LiDAR. *PE&RS* 80, 863–872.
- Khosravipour, A., et al. (2016). Generating spike-free digital surface models using LiDAR raw point clouds. *JAG* 52, 104–114.
- MacArthur, R. H., & Horn, H. S. (1969). Foliage profile by vertical measurements. *Ecology* 50, 802–804.
- Næsset, E. (2002). Predicting forest stand characteristics with airborne scanning laser. *Remote Sens. Environ.* 80, 88–99.
- Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). An improved simple morphological filter. *ISPRS J.* 77, 21–30.
- Roussel, J.-R., et al. (2020). lidR. *Remote Sens. Environ.* 251, 112061.
- Zhang, K., et al. (2003). A progressive morphological filter for removing nonground measurements. *IEEE TGRS* 41, 872–882.
- Zhang, W., et al. (2016). An easy-to-use airborne LiDAR data filtering method based on cloth simulation. *Remote Sens.* 8, 501.
