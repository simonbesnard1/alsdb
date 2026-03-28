#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:46:37 2026

@author: simon
"""

from alsdb import ALSProvider
from alsdb.processing.waveform import simulate_waveform, simulate_batch

reader = ALSProvider(storage_type="local", uri="array_")

# Single footprint (25 m diameter, like GEDI)
result = simulate_waveform(
    provider=reader,
    center_x=309720.0, center_y=4689880.0,
    footprint_radius=12.5,
    year=2021
)
print(result.rh)         # {"rh25": 8.1, "rh50": 14.3, "rh75": 19.7, "rh95": 23.1, ...}
print(result.cover)      # canopy cover fraction
print(result.z_ground)   # estimated ground elevation (m)

# Batch over a list of (lon, lat) footprint centres
import numpy as np
import pandas as pd

provider = ALSProvider(storage_type="local", uri="array_")

xs, ys = np.meshgrid(
    np.arange(658000.0, 658442.25, 60),
    np.arange(8901518.0, 8902000.0, 60),
)
shots = pd.DataFrame({"center_x": xs.ravel(), "center_y": ys.ravel()})

results = simulate_batch(
    provider=provider,
    shots=shots,
    year=2014,
    n_workers=4,
    footprint_radius=12.5,
)
print(f"{len(results)} shots simulated")
print(results[["center_x", "center_y", "rh50", "rh98", "cover"]].head())

from alsdb.utils.viz import plot_waveforms_3d

# Static matplotlib figure
fig = plot_waveforms_3d(results, color_by="rh98", backend="matplotlib")
fig.savefig("waveforms_3d.png", dpi=150, bbox_inches="tight")

