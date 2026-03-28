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
    center_x=308_500.0, center_y=4_689_000.0,
    footprint_radius=12.5,
    year=2021,
)
print(result.rh)         # {"rh25": 8.1, "rh50": 14.3, "rh75": 19.7, "rh95": 23.1, ...}
print(result.cover)      # canopy cover fraction
print(result.z_ground)   # estimated ground elevation (m)

# Batch over a list of (lon, lat) footprint centres
results = simulate_batch(
    provider=reader,
    centres=[(308_500, 4_689_000), (309_000, 4_689_500)],
    footprint_radius=12.5,
    year=2021,
    max_workers=4,
)
