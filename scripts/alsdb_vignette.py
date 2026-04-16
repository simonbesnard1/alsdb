"""
From point cloud to biomass: a Brazil ALS vignette
====================================================

End-to-end showcase of the ``alsdb`` pipeline applied to a 1 km² tropical
forest tile from Brazil (2014 survey).  The notebook covers:

* raw point cloud visualisation
* 1 m Canopy Height Model (CHM) and Digital Terrain Model (DTM)
* Aboveground Biomass (AGB) at 10 m and 100 m resolution
* GEDI-like full-waveform simulation and RH-metric extraction
"""

# %%
# 0 · Paths and constants
# ------------------------
#
# Adjust ``ARRAY_URI`` to point at your TileDB array and ``ZARR_PATH`` to
# the ``ALSZarrStore`` that already holds the 1 m and 100 m products from
# your earlier processing run.

from alsdb import ALSProvider
from alsdb.storage import ALSZarrStore

ARRAY_URI = "/home/simon/array_"  # ← your TileDB array
ZARR_PATH = "/home/simon/output/brazil.zarr"  # ← your Zarr store
YEAR = 2014

# Bounding box derived from the 1 m dataset (1 000 × 1 000 cells, 1 m each)
BBOX = (656_000, 8_902_000, 657_000, 8_903_000)

reader = ALSProvider(storage_type="local", uri=ARRAY_URI)
store = ALSZarrStore(ZARR_PATH)

# %%
# 2 · Load all products as xarray Datasets
# ----------------------------------------

ds1m = store.to_dataset(resolution=1.0)
ds10m = store.to_dataset(resolution=10.0)
ds100m = store.to_dataset(resolution=100.0)

chm = ds1m["chm"].sel(time=YEAR)  # (1000, 1000)
dtm = ds1m["dtm"].sel(time=YEAR)
agb10 = ds10m["biomass"].sel(time=YEAR)  # (100, 100)
agb100 = ds100m["biomass"].sel(time=YEAR)  # (10, 10)

# %%
# 3 · Query raw point cloud for a 200 × 200 m sub-area
# -----------------------------------------------------
#
# Keep a small area for the scatter plot so the figure stays readable.

MINI_BBOX = (
    BBOX[0],
    BBOX[1],
    BBOX[0] + 200,
    BBOX[1] + 200,
)

df_pts = reader.query_bbox(
    min_x=MINI_BBOX[0],
    min_y=MINI_BBOX[1],
    max_x=MINI_BBOX[2],
    max_y=MINI_BBOX[3],
    attributes=["Z", "Intensity", "ReturnNumber", "Classification"],
    year=YEAR,
)

print(f"Points in sub-area: {len(df_pts):,}")

# %%
# 4 · Simulate a GEDI waveform at scene centre
# ---------------------------------------------

from alsdb.processing.waveform import simulate_waveform, simulate_batch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Single shot at scene centre
cx = (BBOX[0] + BBOX[2]) / 2.0  # 656 500 m
cy = (BBOX[1] + BBOX[3]) / 2.0  # 8 902 500 m

result = simulate_waveform(
    provider=reader,
    center_x=cx,
    center_y=cy,
    footprint_radius=12.5,  # 25 m diameter — matches GEDI
    year=YEAR,
)

print(f"RH50  = {result.rh[50]:.1f} m")
print(f"RH98  = {result.rh[98]:.1f} m")
print(f"Cover = {result.cover:.2f}")
print(f"n pts = {result.n_points}")

# %%
# Batch simulation — regular 60 m grid across the 1 km² tile
# -----------------------------------------------------------
xs, ys = np.meshgrid(
    np.arange(BBOX[0] + 30, BBOX[2], 60),
    np.arange(BBOX[1] + 30, BBOX[3], 60),
)
shots = pd.DataFrame({"center_x": xs.ravel(), "center_y": ys.ravel()})
print(f"Simulating {len(shots)} footprints …")

results_batch = simulate_batch(
    provider=reader,
    shots=shots,
    year=YEAR,
    n_workers=4,
    footprint_radius=12.5,
)
print(results_batch[["center_x", "center_y", "rh50", "rh98", "cover"]].head())

# %%
# 5 · Vignette figure — the full pipeline in one view
# ----------------------------------------------------
#
# Six-panel figure:
#   top row    — point cloud (classification) · CHM 1 m · DTM 1 m (hill-shaded)
#   bottom row — AGB 10 m · AGB 100 m · simulated waveform

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from alsdb.utils.viz import plot_intensity  # noqa: E402

fig = plt.figure(figsize=(18, 11), constrained_layout=True)
fig.suptitle(
    "alsDB · Brazil ALS 2014 · 1 km² tropical forest tile",
    fontsize=22,
    fontweight="bold",
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.05)

ax_pts = fig.add_subplot(gs[0, 0])
ax_chm = fig.add_subplot(gs[0, 1])
ax_dtm = fig.add_subplot(gs[0, 2])
ax_agb10 = fig.add_subplot(gs[1, 0])
ax_agb100 = fig.add_subplot(gs[1, 1])
ax_wave = fig.add_subplot(gs[1, 2])

# ── panel A: point cloud (classification colours) ──────────────────────────
plot_intensity(df_pts, resolution=1.0, ax=ax_pts)
ax_pts.set_title("a · Point intensity (200 x 200m)", loc="left", fontsize=18)
ax_pts.set_xlabel("Easting (m)")
ax_pts.set_ylabel("Northing (m)")

# ── panel B: CHM at 1 m ────────────────────────────────────────────────────
chm_arr = chm.values
im_chm = ax_chm.imshow(
    chm_arr,
    origin="upper",
    extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]],
    cmap="Greens",
    vmin=0,
    vmax=float(np.nanpercentile(chm_arr, 99)),
)
cb = plt.colorbar(im_chm, ax=ax_chm, label="CHM (m)", fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=12)  # tick labels
cb.set_label("CHM (m)", fontsize=16)  # axis label
ax_chm.set_title("b · Canopy Height Model · 1 m", loc="left", fontsize=18)
ax_chm.set_xlabel("Easting (m)")
ax_chm.tick_params(labelleft=False)

# ── panel C: DTM with hill-shading at 1 m ─────────────────────────────────
dtm_arr = dtm.values
ls = LightSource(azdeg=315, altdeg=45)
dtm_filled = np.where(np.isnan(dtm_arr), np.nanmean(dtm_arr), dtm_arr)
rgb = ls.shade(dtm_filled, cmap=plt.cm.terrain, blend_mode="soft", vert_exag=3.0)
ax_dtm.imshow(
    rgb,
    origin="upper",
    extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]],
)
ax_dtm.set_title("c · Digital Terrain Model · 1 m (hill-shaded)", loc="left", fontsize=18)
ax_dtm.set_xlabel("Easting (m)")
ax_dtm.tick_params(labelleft=False)

# ── panel D: AGB at 10 m ──────────────────────────────────────────────────
agb10_arr = agb10.values
x10 = ds10m.x.values
y10 = ds10m.y.values
im_agb10 = ax_agb10.pcolormesh(
    x10,
    y10,
    agb10_arr,
    cmap="YlGn",
    vmin=0,
    vmax=float(np.nanpercentile(agb10_arr, 99)),
    shading="nearest",
)
cb = plt.colorbar(im_agb10, ax=ax_agb10, label="AGB (Mg ha⁻¹)", fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=12)  # tick labels
cb.set_label("AGB (Mg ha⁻¹)", fontsize=16)  # axis label
ax_agb10.set_title("d · Aboveground Biomass · 10 m", loc="left", fontsize=18)
ax_agb10.set_xlabel("Easting (m)")
ax_agb10.set_ylabel("Northing (m)")

# ── panel E: AGB at 100 m ─────────────────────────────────────────────────
agb100_arr = agb100.values
x100 = ds100m.x.values
y100 = ds100m.y.values
im_agb100 = ax_agb100.pcolormesh(
    x100,
    y100,
    agb100_arr,
    cmap="YlGn",
    vmin=0,
    vmax=float(np.nanpercentile(agb100_arr, 99)),
    shading="nearest",
)
cb = plt.colorbar(im_agb100, ax=ax_agb100, label="AGB (Mg ha⁻¹)", fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=12)  # tick labels
cb.set_label("AGB (Mg ha⁻¹)", fontsize=16)  # axis label

ax_agb100.set_title("e · Aboveground Biomass · 100 m", loc="left", fontsize=18)
ax_agb100.set_xlabel("Easting (m)")
ax_agb100.tick_params(labelleft=False)

# ── panel F: simulated waveform ───────────────────────────────────────────
rh_levels = [10, 25, 50, 75, 98]
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(rh_levels)))

ax_wave.fill_betweenx(
    result.z_bins - result.z_ground,
    0,
    result.waveform / result.waveform.max(),
    alpha=0.3,
    color="steelblue",
    label="Waveform",
)
ax_wave.plot(
    result.waveform / result.waveform.max(),
    result.z_bins - result.z_ground,
    color="steelblue",
    lw=1.2,
)
for rh, c in zip(rh_levels, colors):
    h = result.rh[rh]
    ax_wave.axhline(h, color=c, ls="--", lw=0.9, alpha=0.85)
    ax_wave.text(
        1.03,
        h,
        f"RH{rh}\n{h:.1f} m",
        va="center",
        ha="left",
        fontsize=14,
        color=c,
        transform=ax_wave.get_yaxis_transform(),
    )

ax_wave.axhline(0, color="saddlebrown", lw=1.2, ls="-", label="Ground")
ax_wave.set_xlabel("Normalised energy", fontsize=16)
ax_wave.set_ylabel("Height above ground (m)", fontsize=16)
ax_wave.set_xlim(0, 1.35)
ax_wave.set_ylim(-3, max(result.rh[98] * 1.15, 5))
ax_wave.set_title(
    f"f · GEDI-like waveform · cover={result.cover:.2f}",
    loc="left",
    fontsize=18,
)
ax_wave.legend(fontsize=14, loc="upper right", frameon=False)
ax_wave.spines["top"].set_visible(False)
ax_wave.spines["right"].set_visible(False)

fig.savefig("/home/simon/brazil_vignette.png", dpi=180, bbox_inches="tight")
plt.show()
