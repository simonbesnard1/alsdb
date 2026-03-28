#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:46:37 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.waveform import simulate_waveform, simulate_batch, _SIGMA_COV

provider = ALSProvider(storage_type="local", uri="array_")

# Single footprint (UTM coords)
result = simulate_waveform(provider, center_x=309000, center_y=4689000, year=2021)
print(result.rh)        # {10: 1.2, 25: 3.4, 50: 8.1, 75: 14.3, ...}
print(result.home)      # 8.1 m
print(result.cover)     # 0.72

# Compare with real GEDI — convert lat/lon first
from pyproj import Transformer
t = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
cx, cy = t.transform(gedi_shot.lon, gedi_shot.lat)
result = simulate_waveform(provider, cx, cy, sigma=_SIGMA_COV)  # coverage beam

# Batch over all GEDI shots in the tile
# gedi_df has columns lon, lat from gedidb
gedi_df["center_x"], gedi_df["center_y"] = t.transform(gedi_df.lon, gedi_df.lat)
sim = simulate_batch(provider, gedi_df, x_col="center_x", y_col="center_y",
                     year=2021, n_workers=8)
# sim has all original gedidb columns + rh25, rh50, rh75, rh95, home, cover
