#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:31:34 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.biomass import compute_biomass, compute_metrics
from alsdb.storage import ALSZarrStore
import alsdb
alsdb.setup_logging()

provider = ALSProvider(storage_type="local", uri="array_")
store = ALSZarrStore("output/brazil.zarr")

bbox = (655000.0, 8901000.0, 656000.0, 8902000.0)

# All structural metrics at 10 m
compute_metrics(provider, store, resolution=10.0, year=2019)

# AGB at 10 m and 100 m — both go into the same store, different groups
compute_biomass(provider, store, resolution=10.0, year=2014, n_workers=6)
compute_biomass(provider, store, resolution=100.0, year=2021, n_workers=6)


