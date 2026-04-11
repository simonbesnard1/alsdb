"""
GEDI waveform simulation
=========================

This example shows how to simulate GEDI-like full waveforms from an ALS
point cloud and extract relative height (RH) metrics for direct comparison
with space-borne GEDI L2A retrievals.

The simulator builds a vertical return histogram, convolves it with a
Gaussian pulse, detects the ground return, and computes cumulative RH
metrics from the ground up.
"""

# %%
# Single footprint simulation
# ----------------------------
#
# :py:func:`alsdb.processing.waveform.simulate_waveform` simulates a
# single GEDI-like footprint (25 m diameter by default).

from alsdb import ALSProvider
from alsdb.processing.waveform import simulate_waveform

reader = ALSProvider(storage_type="local", uri="/path/to/my_array")

result = simulate_waveform(
    provider=reader,
    center_x=308_500.0,
    center_y=4_689_000.0,
    footprint_radius=12.5,   # 25 m diameter, matching GEDI
    year=2021,
)

print(f"RH50  = {result.rh[50]:.2f} m")    # height of median energy
print(f"RH98  = {result.rh[98]:.2f} m")    # equivalent to GEDI rh98
print(f"Cover = {result.cover:.3f}")        # canopy cover fraction
print(f"Ground elevation = {result.z_ground:.2f} m")

# %%
# Visualise the waveform
# -----------------------

from alsdb.utils.viz import plot_waveform, plot_rh_profile

# Energy vs elevation, annotated with RH levels
fig = plot_waveform(result)
fig.savefig("waveform.png", dpi=150, bbox_inches="tight")

# GEDI L2A style: RH(p) curve + W(h) energy density
fig = plot_rh_profile(result)
fig.savefig("rh_profile.png", dpi=150, bbox_inches="tight")

# %%
# Batch simulation
# -----------------
#
# :py:func:`alsdb.processing.waveform.simulate_batch` processes a DataFrame
# of shot centres in parallel. Results can be saved to Parquet for later
# co-location with real GEDI observations.

import numpy as np
import pandas as pd
from alsdb.processing.waveform import simulate_batch

# Build a grid of GEDI-like shot centres (60 m spacing)
xs, ys = np.meshgrid(
    np.arange(308_100, 309_900, 60),
    np.arange(4_688_500, 4_689_900, 60),
)
shots = pd.DataFrame({"center_x": xs.ravel(), "center_y": ys.ravel()})
print(f"Simulating {len(shots)} footprints...")

results = simulate_batch(
    provider=reader,
    shots=shots,
    year=2021,
    n_workers=4,
    footprint_radius=12.5,
    output_path="shots_2021.parquet",   # optional; omit for in-memory only
)

print(results[["center_x", "center_y", "rh50", "rh98", "cover"]].head())

# %%
# 3-D waveform waterfall
# -----------------------
#
# :py:func:`alsdb.utils.viz.plot_waveforms_3d` shows all shots as a waterfall
# of RH(p) curves coloured by a summary metric.

from alsdb.utils.viz import plot_waveforms_3d

# Static matplotlib figure
fig = plot_waveforms_3d(results, color_by="rh98", backend="matplotlib")
fig.savefig("waveforms_3d.png", dpi=150, bbox_inches="tight")

# Interactive plotly version (better for dense grids)
# fig = plot_waveforms_3d(results, color_by="cover", backend="plotly")
# fig.show()

# %%
# Compare with GEDI L2A
# ----------------------
#
# Co-locate simulated shots with real GEDI observations by matching shot
# coordinates. Both datasets use RH metrics at the same footprint scale.

# Example (requires a GEDI L2A DataFrame from gedidb or similar):
#
# gedi = pd.read_parquet("gedi_l2a_region.parquet")
#
# # Nearest-neighbour match
# from scipy.spatial import cKDTree
# tree = cKDTree(gedi[["longitude_lastbin", "latitude_lastbin"]].values)
# _, idx = tree.query(results[["center_x", "center_y"]].values)
# comparison = results.copy()
# comparison["gedi_rh98"] = gedi.iloc[idx]["rh98"].values
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(comparison["gedi_rh98"], comparison["rh98"], s=5, alpha=0.4)
# ax.plot([0, 50], [0, 50], "k--", lw=1)
# ax.set_xlabel("GEDI RH98 (m)")
# ax.set_ylabel("Simulated RH98 (m)")
# fig.savefig("gedi_vs_simulated.png", dpi=150)
