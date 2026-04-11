"""
Computing forest structure products
=====================================

This example shows how to use :py:class:`alsdb.ALSProvider` and
``ALSZarrStore`` to compute CHM, DTM, DSM, gap fraction, LAI, structural
metrics, and aboveground biomass from an ingested TileDB point-cloud array.

All products are written directly to a Zarr v3 store — no GeoTIFF
intermediates and no mosaic step.
"""

# %%
# Setup
# -----

from alsdb import ALSProvider
from alsdb.storage import ALSZarrStore

reader = ALSProvider(storage_type="local", uri="/path/to/my_array")
store  = ALSZarrStore("/path/to/forest.zarr")

BBOX = (308_000, 4_688_000, 310_000, 4_690_000)
YEAR = 2021

# %%
# Canopy Height Model / DTM / DSM
# ---------------------------------
#
# :py:func:`alsdb.processing.chm.compute_all` computes DTM, DSM, and CHM in a
# single tiled pass. Use ``tile_size`` and ``n_workers`` to parallelise over
# large areas.

from alsdb.processing.chm import compute_all  # noqa: E402

compute_all(
    provider=reader,
    store=store,
    resolution=1.0,
    bbox=BBOX,
    year=YEAR,
    tile_size=500.0,    # 500 m sub-tiles
    tile_buffer=50.0,   # 50 m buffer for accurate HAG at tile edges
    n_workers=4,
)

# %%
# Gap fraction and effective LAI
# --------------------------------
#
# Gap fraction uses the MacArthur–Wilson first-return estimator.
# Effective LAI is computed via Beer–Lambert inversion with ``lai=True``.

from alsdb.processing.gap import compute_gap_fraction  # noqa: E402

compute_gap_fraction(
    provider=reader,
    store=store,
    resolution=10.0,
    bbox=BBOX,
    year=YEAR,
    lai=True,
    k=0.5,   # extinction coefficient (spherical leaf angle distribution)
)

# %%
# LiDAR structural metrics
# -------------------------
#
# :py:func:`alsdb.processing.biomass.compute_metrics` writes h50, h75, h95,
# hmean, canopy cover, and point density to the store.

from alsdb.processing.biomass import compute_metrics  # noqa: E402

compute_metrics(
    provider=reader,
    store=store,
    resolution=10.0,
    bbox=BBOX,
    year=YEAR,
)

# %%
# Aboveground biomass — default Næsset model
# -------------------------------------------
#
# The default model is ``AGB = a × h95^b × cc^c`` (Næsset 2002).
# Calibrate the parameters against field inventory plots before
# using the output scientifically.

from alsdb.processing.biomass import compute_biomass  # noqa: E402

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
# Use :py:func:`alsdb.processing.biomass.wrap_sklearn_model` to plug in any
# trained sklearn-compatible estimator.


# Example with a pre-trained Random Forest (not executed here)
# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor().fit(X_train, y_agb)
# compute_biomass(provider=reader, store=store, resolution=10.0, year=YEAR,
#                 bbox=BBOX, model_fn=wrap_sklearn_model(rf))

# %%
# Read results as xarray
# -----------------------
#
# ``to_dataset()`` opens a resolution group as a CRS-aware
# :py:class:`xarray.Dataset`.

ds1m  = store.to_dataset(resolution=1.0)
ds10m = store.to_dataset(resolution=10.0)

chm    = ds1m["chm"].sel(time=YEAR)    # (ny, nx)
dtm    = ds1m["dtm"].sel(time=YEAR)
agb    = ds10m["biomass"].sel(time=YEAR)
h95    = ds10m["h95"].sel(time=YEAR)
gap    = ds10m["gap"].sel(time=YEAR)

print(ds1m)

# CRS is attached via rioxarray if available
# print(ds1m.rio.crs)

# %%
# Multi-temporal change detection
# ---------------------------------
#
# Store all survey years in the same Zarr store; the time axis handles them.

for year in [2017, 2021, 2023]:
    compute_all(
        provider=reader,
        store=store,
        resolution=1.0,
        year=year,
        tile_size=500.0,
        n_workers=4,
    )

ds = store.to_dataset(resolution=1.0)
delta_chm = ds["chm"].sel(time=2021) - ds["chm"].sel(time=2017)
print("Max canopy height gain 2017→2021:", float(delta_chm.max()), "m")
