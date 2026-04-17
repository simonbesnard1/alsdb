#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:32:51 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.chm import compute_chm, compute_all
from alsdb.storage import ALSZarrStore
import alsdb

alsdb.setup_logging()  # INFO by default

provider = ALSProvider(storage_type="local", uri="array_")
store = ALSZarrStore("output/brazil.zarr")

compute_all(provider, store, resolution=1.0, year=2016, tile_size=500.0, tile_buffer=50.0)

ds = store.to_dataset(1.0)  # → xarray.Dataset


store = ALSZarrStore.create(
    "output/spain.zarr",
    crs_wkt=provider.crs_wkt,
    variables={"1m": ["chm", "dtm", "dsm"], "10m": ["gap", "lai", "biomass"]},
)
compute_chm(provider, store, resolution=1.0, year=2021, n_workers=8)
ds = store.to_dataset(1.0)  # → xarray.Dataset
