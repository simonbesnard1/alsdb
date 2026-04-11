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
    center_x=309720.0,
    center_y=4689880.0,
    footprint_radius=12.5,
    year=2021,
)
print(result.rh)  # {"rh25": 8.1, "rh50": 14.3, "rh75": 19.7, "rh95": 23.1, ...}
print(result.cover)  # canopy cover fraction
print(result.z_ground)  # estimated ground elevation (m)

# Batch over a list of (lon, lat) footprint centres
import numpy as np
import pandas as pd
from alsdb import ALSProvider

_BEAM_IDS = [
    "BEAM0000",
    "BEAM0001",
    "BEAM0010",
    "BEAM0011",
    "BEAM0101",
    "BEAM0110",
    "BEAM1000",
    "BEAM1011",
]

xs, ys = np.meshgrid(
    np.arange(655025.0, 656000.0, 25),  # 25 m spacing ≈ GEDI along-track
    np.arange(8901025.0, 8901025.0, 25),
)
n = xs.size
shots = pd.DataFrame(
    {
        "center_x": xs.ravel(),
        "center_y": ys.ravel(),
        "beam": np.array(_BEAM_IDS)[np.arange(n) % len(_BEAM_IDS)],
    }
)

provider = ALSProvider(storage_type="local", uri="array_")

results = simulate_batch(
    provider=provider,
    shots=shots,
    year=2014,
    n_workers=4,
    footprint_radius=12.5,
)

print(results[["center_x", "center_y", "rh50", "rh98", "cover", "n_points"]].head(10))
print(f"\n{results['n_points'].gt(0).sum()} / {len(results)} footprints with data")


print(f"{len(results)} shots simulated")
print(results[["center_x", "center_y", "rh50", "rh98", "cover"]].head())

from alsdb.utils.viz import plot_waveforms_3d

# Static matplotlib figure
fig = plot_waveforms_3d(results.head(100), color_by="rh98", backend="matplotlib")
fig.savefig("waveforms_3d_gaussian.png", dpi=150, bbox_inches="tight")
