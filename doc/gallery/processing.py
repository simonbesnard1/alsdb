"""
Computing forest structure products
=====================================

This example shows how to use :py:class:`alsdb.ALSProvider` and
``ALSZarrStore`` to compute CHM, DTM, DSM, gap fraction, LAI, structural
metrics, aboveground biomass, and multi-temporal change from an ingested
TileDB point-cloud array.

All products are written directly to a Zarr v3 store — no GeoTIFF
intermediates and no mosaic step.

Key improvements in the processing pipeline
-------------------------------------------

- **Height normalisation**: ``filters.hag_delaunay`` (Delaunay TIN) is used
  by default for height-above-ground computation, matching the DTM pipeline for
  internal consistency.  A ``filters.hag_nn`` fallback activates automatically
  for tiles with fewer than 3 ground points.
- **Ground outlier rejection**: PDAL ``filters.elm`` + ``filters.outlier``
  remove misclassified building/vegetation returns from Class 2 before the TIN
  is built, preventing HAG bias from corrupt ground classifications.
- **Minimum density guard**: ``min_density`` (pts m⁻²) masks metric cells that
  are statistically unreliable due to low point coverage.
"""

# %%
# Setup
# -----

from alsdb import ALSProvider
from alsdb.storage import ALSZarrStore

reader = ALSProvider(storage_type="local", uri="/path/to/my_array")
store = ALSZarrStore("/path/to/forest.zarr")

BBOX = (308_000, 4_688_000, 310_000, 4_690_000)
YEAR = 2021

# %%
# Canopy Height Model / DTM / DSM
# ---------------------------------
#
# :py:func:`alsdb.processing.chm.compute_all` computes DTM, DSM, and CHM in a
# single tiled pass.  The default DTM uses a Delaunay TIN (``dtm_method="tin"``);
# IDW and min-Z binning are available as alternatives.

from alsdb.processing.chm import compute_all  # noqa: E402

compute_all(
    provider=reader,
    store=store,
    resolution=1.0,
    bbox=BBOX,
    year=YEAR,
    tile_size=500.0,  # 500 m sub-tiles
    tile_buffer=50.0,  # 50 m buffer for accurate HAG at tile edges
    n_workers=4,
    dtm_method="tin",  # "tin" (default), "idw", or "min"
)

# %%
# Gap fraction and effective LAI
# --------------------------------
#
# Gap fraction uses the MacArthur–Wilson first-return estimator over classified
# (ground + vegetation) returns only.  Effective LAI is computed via
# Beer–Lambert inversion; use ``clumping_index`` to correct for foliage clumping
# in conifers or dense broadleaf stands.

from alsdb.processing.gap import LAI_K_PRESETS, compute_gap_fraction  # noqa: E402

compute_gap_fraction(
    provider=reader,
    store=store,
    resolution=10.0,
    bbox=BBOX,
    year=YEAR,
    lai=True,
    k=LAI_K_PRESETS["spherical"],  # 0.5 — random leaf angles (default)
    clumping_index=0.7,  # Ω < 1 for clumped conifer canopies
    min_density=0.5,  # mask cells below 0.5 first returns m⁻²
)

# %%
# LiDAR structural metrics
# -------------------------
#
# :py:func:`alsdb.processing.biomass.compute_metrics` writes 16 structural
# metrics to the store:
#
# * **Height percentiles**: h50, h75, h95, hmax, hmean
# * **Canopy structure**: cc (cover), density (pts m⁻²), FHD, VCI, CRR
# * **Height stratum proportions**: pv_0_2, pv_2_5, pv_5_10, pv_10_20,
#   pv_20_40, pv_above40

from alsdb.processing.biomass import compute_metrics  # noqa: E402

compute_metrics(
    provider=reader,
    store=store,
    resolution=10.0,
    bbox=BBOX,
    year=YEAR,
    min_density=1.0,  # cells below 1 pt m⁻² receive NaN for all metrics
)

# %%
# Aboveground biomass — calibrated Næsset model
# -----------------------------------------------
#
# The Næsset (2002) power-law model ``AGB = a × h95^b × cc^c`` requires
# region- and species-specific calibration against field inventory plots.
# Use :py:func:`alsdb.processing.biomass.calibrate_naesset` to fit coefficients;
# pass ``return_cov=True`` to also retrieve the parameter covariance matrix.

from alsdb.processing.biomass import (  # noqa: E402
    compute_biomass,
)

# Example calibration (requires at least 20 field plots):
# h95_plots, cc_plots, agb_plots are 1-D arrays — one value per plot
#
# (a, b, c), pcov = calibrate_naesset(
#     h95_plots, cc_plots, agb_plots, return_cov=True
# )
# a_std, b_std, c_std = np.sqrt(np.diag(pcov))  # parameter uncertainties
#
# compute_biomass(
#     provider=reader, store=store, resolution=10.0, year=YEAR, bbox=BBOX,
#     model_fn=lambda m: naesset_model(m, a=a, b=b, c=c),
# )

# Using the uncalibrated placeholder coefficients (not for scientific use):
compute_biomass(
    provider=reader,
    store=store,
    resolution=10.0,
    bbox=BBOX,
    year=YEAR,
)

# %%
# Aboveground biomass — scikit-learn model
# -----------------------------------------
#
# :py:func:`alsdb.processing.biomass.wrap_sklearn_model` wraps any fitted
# sklearn-compatible estimator as a ``model_fn``.  The default feature set is
# all 16 metrics from ``compute_metrics()``; pass ``features=[...]`` to match
# the column order used during training.

# Example with a pre-trained Random Forest (not executed here):
# from sklearn.ensemble import RandomForestRegressor
# from alsdb.processing.biomass import wrap_sklearn_model
#
# rf = RandomForestRegressor(n_estimators=200)
# rf.fit(X_train, y_agb)   # X columns must match _METRIC_NAMES order
#
# compute_biomass(provider=reader, store=store, resolution=10.0, year=YEAR,
#                 bbox=BBOX, model_fn=wrap_sklearn_model(rf))

# %%
# Read results as xarray
# -----------------------
#
# ``to_dataset()`` opens a resolution group as a CRS-aware
# :py:class:`xarray.Dataset`.

ds1m = store.to_dataset(resolution=1.0)
ds10m = store.to_dataset(resolution=10.0)

chm = ds1m["chm"].sel(time=YEAR)  # CHM at 1 m (ny, nx)
dtm = ds1m["dtm"].sel(time=YEAR)  # DTM at 1 m
agb = ds10m["biomass"].sel(time=YEAR)
h95 = ds10m["h95"].sel(time=YEAR)
gap = ds10m["gap"].sel(time=YEAR)
crr = ds10m["crr"].sel(time=YEAR)  # Canopy Relief Ratio
pv_5_10 = ds10m["pv_5_10"].sel(time=YEAR)  # fraction of veg returns 5–10 m

print(ds1m)

# CRS is attached via rioxarray if available
# print(ds1m.rio.crs)

# %%
# Multi-temporal change detection
# ---------------------------------
#
# Store all survey years in the same Zarr store, then use
# :py:func:`alsdb.processing.change.compute_change` to derive pixel-wise
# absolute change, relative change, and a gain/loss flag.
#
# Three derived variables are written per call:
# ``{variable}_delta``, ``{variable}_delta_pct``, ``{variable}_change_flag``

from alsdb.processing.change import compute_change  # noqa: E402

for year in [2017, 2021, 2023]:
    compute_all(
        provider=reader,
        store=store,
        resolution=1.0,
        year=year,
        tile_size=500.0,
        n_workers=4,
    )

# CHM change 2017 → 2021; suppress sub-0.5 m noise
compute_change(
    store,
    "chm",
    year_from=2017,
    year_to=2021,
    resolution=1.0,
    min_delta=0.5,  # absolute threshold in metres
    pct_min_abs=0.5,  # suppress % change where base value < 0.5 m
)

ds = store.to_dataset(resolution=1.0)
flag = ds["chm_change_flag"].sel(time=2021)  # +1 gain, −1 loss, 0 stable
delta = ds["chm_delta"].sel(time=2021)  # absolute change (m)

print("Gain pixels:", int((flag == 1).sum()))
print("Loss pixels:", int((flag == -1).sum()))
