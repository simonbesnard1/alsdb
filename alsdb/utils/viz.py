# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Visualisation helpers for ALS point-cloud data.

All functions accept a :class:`pandas.DataFrame` as returned by
:class:`~alsdb.core.alsprovider.ALSProvider` and rasterize it to a regular
grid before plotting.  This keeps rendering fast even for millions of points.

Typical usage::

    from alsdb import ALSProvider
    from alsdb.utils.viz import plot_overview

    provider = ALSProvider(storage_type="local", uri="./my_array")
    df = provider.query_tile(308, 4690)
    fig = plot_overview(df)
    fig.savefig("tile_308_4690.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# LAS classification code → (label, hex colour)
_LAS_CLASSES: dict[int, tuple[str, str]] = {
    0:  ("Never classified", "#aaaaaa"),
    1:  ("Unclassified",      "#cccccc"),
    2:  ("Ground",            "#8b5e3c"),
    3:  ("Low vegetation",    "#a8d08d"),
    4:  ("Medium vegetation", "#538135"),
    5:  ("High vegetation",   "#1f5c00"),
    6:  ("Building",          "#c00000"),
    7:  ("Low noise",         "#ff66cc"),
    9:  ("Water",             "#4472c4"),
    11: ("Road surface",      "#f4b942"),
    17: ("Bridge deck",       "#e06c00"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extent(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return float(df.X.min()), float(df.X.max()), float(df.Y.min()), float(df.Y.max())


def _bin_counts(
    df: pd.DataFrame,
    resolution: float,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return (nx, ny, x_edges, y_edges) for a given resolution."""


    x_min, x_max, y_min, y_max = _extent(df)
    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    return nx, ny, x_edges, y_edges


def rasterize(
    df: pd.DataFrame,
    field: str,
    resolution: float = 1.0,
    statistic: str = "mean",
) -> tuple[np.ndarray, list[float]]:
    """
    Rasterize one DataFrame column onto a regular grid.

    Parameters
    ----------
    df:
        Point-cloud DataFrame with columns ``X``, ``Y``, and *field*.
    field:
        Column name to aggregate (e.g. ``"Z"``, ``"Intensity"``).
    resolution:
        Grid cell size in the same units as ``X``/``Y`` (metres).
    statistic:
        Aggregation function passed to :func:`scipy.stats.binned_statistic_2d`
        (``"mean"``, ``"max"``, ``"min"``, ``"median"``, ``"count"``).

    Returns
    -------
    grid : np.ndarray, shape (ny, nx)
        2-D array; ``np.nan`` where no points fall.
    extent : [x_min, x_max, y_min, y_max]
        Spatial extent for use with ``imshow(extent=...)``.
    """
    from scipy.stats import binned_statistic_2d

    x_min, x_max, y_min, y_max = _extent(df)
    nx, ny, x_edges, y_edges = _bin_counts(df, resolution)

    result, _, _, _ = binned_statistic_2d(
        df.X.to_numpy(), df.Y.to_numpy(), df[field].to_numpy(),
        statistic=statistic,
        bins=[x_edges, y_edges],
    )
    # binned_statistic_2d returns (nx, ny); transpose and flip Y for imshow
    grid = np.where(np.isnan(result.T), np.nan, result.T)
    grid = np.flipud(grid)
    return grid, [x_min, x_max, y_min, y_max]


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_dsm(
    df: pd.DataFrame,
    resolution: float = 1.0,
    hillshade: bool = True,
    cmap: str = "terrain",
    vert_exag: float = 3.0,
    ax=None,
):
    """
    Plot a Digital Surface Model (max Z per cell), optionally with hillshade.

    Parameters
    ----------
    df:
        Point-cloud DataFrame.
    resolution:
        Grid resolution in metres.
    hillshade:
        Overlay a hillshade derived from the DSM.
    cmap:
        Matplotlib colormap name for elevation.
    vert_exag:
        Vertical exaggeration for the hillshade.
    ax:
        Matplotlib axes.  A new figure + axes is created if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    dsm, extent = rasterize(df, "Z", resolution=resolution, statistic="max")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(np.where(np.isnan(dsm), np.nanmin(dsm), dsm),
                          vert_exag=vert_exag)
        # Blend elevation colour with hillshade
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=np.nanmin(dsm), vmax=np.nanmax(dsm))
        rgb = ls.shade(
            np.where(np.isnan(dsm), np.nanmin(dsm), dsm),
            cmap=cmap_obj, norm=norm, vert_exag=vert_exag, blend_mode="soft",
        )
        ax.imshow(rgb, extent=extent, aspect="equal", origin="upper")
    else:
        im = ax.imshow(dsm, extent=extent, aspect="equal", origin="upper",
                       cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"DSM  —  {resolution} m resolution")
    return ax


def plot_rgb(
    df: pd.DataFrame,
    resolution: float = 1.0,
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    ax=None,
):
    """
    Plot an RGB orthoimage from the colourised point cloud.

    The 16-bit RGB channels are rasterized and contrast-stretched using
    percentile clipping before display.

    Parameters
    ----------
    df:
        Point-cloud DataFrame with columns ``Red``, ``Green``, ``Blue``.
    resolution:
        Grid resolution in metres.
    percentile_clip:
        Low and high percentile for contrast stretching.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    def _stretch(arr: np.ndarray) -> np.ndarray:
        valid = arr[~np.isnan(arr)]
        lo, hi = np.percentile(valid, percentile_clip)
        stretched = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
        stretched[np.isnan(arr)] = 0.0
        return stretched

    r, extent = rasterize(df, "Red",   resolution=resolution, statistic="mean")
    g, _      = rasterize(df, "Green", resolution=resolution, statistic="mean")
    b, _      = rasterize(df, "Blue",  resolution=resolution, statistic="mean")

    rgb = np.dstack([_stretch(r), _stretch(g), _stretch(b)])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(rgb, extent=extent, aspect="equal", origin="upper",
              interpolation="nearest")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"RGB orthoimage  —  {resolution} m resolution")
    return ax


def plot_intensity(
    df: pd.DataFrame,
    resolution: float = 1.0,
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    ax=None,
):
    """
    Plot a mean-intensity raster (greyscale).

    Parameters
    ----------
    df:
        Point-cloud DataFrame with column ``Intensity``.
    resolution:
        Grid resolution in metres.
    percentile_clip:
        Low/high percentile for contrast stretching.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = rasterize(df, "Intensity", resolution=resolution, statistic="mean")

    valid = grid[~np.isnan(grid)]
    lo, hi = np.percentile(valid, percentile_clip)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(grid, extent=extent, aspect="equal", origin="upper",
              cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Intensity  —  {resolution} m resolution")
    return ax


def plot_classification(
    df: pd.DataFrame,
    resolution: float = 1.0,
    ax=None,
):
    """
    Plot a classification map with standard LAS colour coding.

    Parameters
    ----------
    df:
        Point-cloud DataFrame with column ``Classification``.
    resolution:
        Grid resolution in metres.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    grid, extent = rasterize(df, "Classification", resolution=resolution,
                             statistic="mean")
    grid_int = np.round(grid).astype("float")
    grid_int[np.isnan(grid)] = np.nan

    classes = sorted({int(c) for c in df.Classification.unique()
                      if not np.isnan(c)})

    colours = [_LAS_CLASSES.get(c, (str(c), "#999999"))[1] for c in classes]
    cmap = ListedColormap(colours)

    # Map class codes to 0-based indices for imshow
    lut = {cls: i for i, cls in enumerate(classes)}
    mapped = np.full_like(grid_int, np.nan)
    for cls, idx in lut.items():
        mapped[grid_int == cls] = idx

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(mapped, extent=extent, aspect="equal", origin="upper",
              cmap=cmap, vmin=0, vmax=len(classes) - 1,
              interpolation="nearest")

    patches = [
        mpatches.Patch(
            color=_LAS_CLASSES.get(c, (str(c), "#999999"))[1],
            label=f"{c} – {_LAS_CLASSES.get(c, (str(c), '#999999'))[0]}",
        )
        for c in classes
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.8, title="Classification")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Classification  —  {resolution} m resolution")
    return ax


# ---------------------------------------------------------------------------
# GEDI waveform
# ---------------------------------------------------------------------------

def plot_waveform(
    result,
    ax_wave=None,
    ax_rh=None,
    rh_levels: tuple[int, ...] = (25, 50, 75, 95, 100),
    figsize: tuple[float, float] = (11, 6),
    title: Optional[str] = None,
):
    """
    Two-panel waveform plot for a :class:`~alsdb.processing.waveform.WaveformResult`.

    Left panel
        Normalised waveform energy vs elevation.  Ground peak, canopy top,
        and each requested RH level are annotated.  The canopy layer is
        shaded in green; the ground return in brown.

    Right panel
        Horizontal bar chart of RH heights above ground.

    Parameters
    ----------
    result:
        A :class:`~alsdb.processing.waveform.WaveformResult` instance.
    ax_wave, ax_rh:
        Pre-existing Matplotlib axes.  A new figure is created when both
        are ``None``.
    rh_levels:
        RH percentile levels to annotate and plot.
    figsize:
        Figure size in inches (used only when axes are not supplied).
    title:
        Figure suptitle.  Auto-generated if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator

    if ax_wave is None or ax_rh is None:
        fig, (ax_wave, ax_rh) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [2, 1]},
        )
    else:
        fig = ax_wave.get_figure()

    z      = result.z_bins
    wave   = result.waveform
    z_gnd  = result.z_ground
    rh     = result.rh

    # Height above ground for canopy top (RH100)
    z_top  = z_gnd + rh.get(100, float(z.max() - z_gnd))

    # Colour palette for RH levels
    _rh_colours = {
        25:  "#2196F3",
        50:  "#4CAF50",
        75:  "#FF9800",
        95:  "#F44336",
        100: "#9C27B0",
    }

    # ------------------------------------------------------------------
    # Left panel — waveform profile
    # ------------------------------------------------------------------
    # Filled waveform
    ax_wave.fill_betweenx(z, 0, wave, alpha=0.25, color="#78909C", label="Waveform")
    ax_wave.plot(wave, z, color="#37474F", linewidth=1.2)

    # Ground layer shading
    ground_mask = z <= z_gnd
    ax_wave.fill_betweenx(
        z[ground_mask], 0, wave[ground_mask],
        alpha=0.55, color="#8D6E63", label="Ground return",
    )

    # Canopy layer shading
    canopy_mask = z >= z_gnd
    ax_wave.fill_betweenx(
        z[canopy_mask], 0, wave[canopy_mask],
        alpha=0.20, color="#66BB6A", label="Canopy return",
    )

    # Ground and canopy top lines
    ax_wave.axhline(z_gnd, color="#6D4C41", linewidth=1.2,
                    linestyle="--", label=f"Ground  {z_gnd:.1f} m")
    ax_wave.axhline(z_top, color="#7B1FA2", linewidth=1.0,
                    linestyle=":", label=f"Canopy top  {z_top:.1f} m")

    # RH level lines
    for lvl in rh_levels:
        z_rh = z_gnd + rh.get(lvl, np.nan)
        if not np.isfinite(z_rh):
            continue
        colour = _rh_colours.get(lvl, "#555555")
        ax_wave.axhline(z_rh, color=colour, linewidth=1.0,
                        linestyle="-.", alpha=0.85,
                        label=f"RH{lvl:d}  {rh[lvl]:.1f} m")

    ax_wave.set_xlabel("Normalised energy", fontsize=10)
    ax_wave.set_ylabel("Elevation (m)", fontsize=10)
    ax_wave.set_xlim(left=0)
    ax_wave.xaxis.set_major_locator(MaxNLocator(4))
    ax_wave.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    # Annotation box
    info = (
        f"Cover: {result.cover:.2f}\n"
        f"HOME: {result.home:.1f} m\n"
        f"N pts: {result.n_points:,}"
    )
    ax_wave.text(
        0.03, 0.04, info, transform=ax_wave.transAxes,
        fontsize=8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
    )

    # ------------------------------------------------------------------
    # Right panel — RH bar chart
    # ------------------------------------------------------------------
    levels_present = [lvl for lvl in rh_levels if lvl in rh and np.isfinite(rh[lvl])]
    values  = [rh[lvl] for lvl in levels_present]
    labels  = [f"RH{lvl}" for lvl in levels_present]
    colours = [_rh_colours.get(lvl, "#555555") for lvl in levels_present]

    bars = ax_rh.barh(labels, values, color=colours, alpha=0.85, edgecolor="white")
    ax_rh.bar_label(bars, fmt="%.1f m", padding=4, fontsize=8)
    ax_rh.set_xlabel("Height above ground (m)", fontsize=10)
    ax_rh.set_xlim(0, max(values) * 1.25 if values else 1)
    ax_rh.invert_yaxis()
    ax_rh.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Shared title
    # ------------------------------------------------------------------
    if title is None:
        title = (
            f"Simulated GEDI waveform  |  "
            f"({result.center_x:.0f}, {result.center_y:.0f}) UTM"
        )
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RH profile + waveform (GEDI L2A style)
# ---------------------------------------------------------------------------

def plot_rh_profile(
    result,
    ax_rh: Optional = None,
    ax_wave: Optional = None,
    figsize: tuple[float, float] = (9, 4.6),
    savgol_window: Optional[int] = None,
    peak_prominence: float = 0.12,
    title: Optional[str] = None,
):
    """
    Two-panel figure matching the GEDI L2A canonical waveform representation.

    Panel (a) — RH curve
        Height above ground (m) as a function of cumulative energy percentile
        (0–100 %).  Understory and overstory layers are shaded.

    Panel (b) — Waveform W(h)
        Normalised energy density vs height above ground.
        ``result.waveform`` is already W(h) = dE/dh (Gaussian-convolved ALS
        histogram), so no reconstruction from the RH profile is needed.
        Detected layer peaks, the layer boundary, and the Δh inter-layer
        distance are annotated.

    Parameters
    ----------
    result:
        :class:`~alsdb.processing.waveform.WaveformResult` with RH0–RH100
        computed (default when using :func:`~alsdb.processing.waveform.simulate_waveform`).
    ax_rh, ax_wave:
        Pre-existing Matplotlib axes.  A new figure is created when both are ``None``.
    figsize:
        Figure size in inches.
    savgol_window:
        Savitzky–Golay smoothing window for peak detection.  Auto-sized if ``None``.
    peak_prominence:
        Minimum prominence for layer peak detection (normalised units, 0–1).
    title:
        Figure suptitle.  Auto-generated if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter, find_peaks

    if ax_rh is None or ax_wave is None:
        fig, (ax_rh, ax_wave) = plt.subplots(1, 2, figsize=figsize)
        fig.subplots_adjust(wspace=0.35)
    else:
        fig = ax_rh.get_figure()

    # ------------------------------------------------------------------
    # RH profile: p (0–100) → height above ground
    # ------------------------------------------------------------------
    levels = sorted(result.rh.keys())
    p_arr  = np.array(levels, dtype=float)
    h_arr  = np.array([result.rh[l] for l in levels])

    valid  = np.isfinite(h_arr) & (h_arr >= 0)
    p_arr, h_arr = p_arr[valid], h_arr[valid]
    h_mono = np.maximum.accumulate(h_arr)   # enforce monotonicity

    hmin = float(h_mono.min())
    hmax = float(h_mono.max())
    hpad = max(0.5, (hmax - hmin) * 0.1)   # at least 0.5 m padding for readability

    # ------------------------------------------------------------------
    # Waveform: W(h) above ground — result.waveform IS dE/dh already
    # ------------------------------------------------------------------
    h_bins = result.z_bins - result.z_ground
    above  = h_bins >= 0
    h_w    = h_bins[above]
    W_raw  = result.waveform[above]
    Wn     = W_raw / (np.max(np.abs(W_raw)) + 1e-12)

    # Savitzky-Golay: window must be odd and < len(Wn)
    n = len(Wn)
    if savgol_window is not None:
        win = savgol_window
    else:
        win = max(7, (n // 40) * 2 + 1)
    win = min(win, n if n % 2 == 1 else n - 1)   # must be odd and <= n
    win = win if win % 2 == 1 else win - 1
    Wn_s = savgol_filter(Wn, window_length=win, polyorder=min(3, win - 1)) if n >= win else Wn.copy()

    peaks,   _ = find_peaks( Wn_s, prominence=peak_prominence)
    valleys, _ = find_peaks(-Wn_s, prominence=peak_prominence / 2)

    if len(peaks) >= 2:
        p_sorted  = peaks[np.argsort(h_w[peaks])]
        low_peak  = p_sorted[0]
        high_peak = p_sorted[-1]
    elif len(peaks) == 1:
        low_peak = high_peak = peaks[0]
    else:
        low_peak  = int(np.argmin(np.abs(h_w - np.percentile(h_w, 10))))
        high_peak = int(np.argmin(np.abs(h_w - np.percentile(h_w, 75))))

    mid_h   = 0.5 * (h_w[low_peak] + h_w[high_peak])
    h_split = (
        h_w[valleys[np.argmin(np.abs(h_w[valleys] - mid_h))]]
        if len(valleys) > 0 else mid_h
    )
    U_strength = float(np.interp(h_split, h_mono, p_arr / 100.0))

    # ------------------------------------------------------------------
    # Panel (a) — RH curve
    # ------------------------------------------------------------------
    ax_rh.plot(p_arr, h_mono, lw=1.6, color="#37474F")
    ax_rh.axhspan(hmin,    h_split, alpha=0.20, color="#1b9e77", label="Understorey")
    ax_rh.axhspan(h_split, hmax,    alpha=0.20, color="#d95f02", label="Overstorey")
    ax_rh.set_xlim(0, 100)
    ax_rh.set_ylim(hmin - hpad, hmax + hpad)
    ax_rh.set_xlabel("Percent energy returned [%]")
    ax_rh.set_ylabel("Height above ground [m]")
    ax_rh.set_title(r"Relative height: $h = \mathrm{RH}(p)$", fontsize=12)
    ax_rh.spines["top"].set_visible(False)
    ax_rh.spines["right"].set_visible(False)
    ax_rh.text(0.02, 0.98, "(a)", transform=ax_rh.transAxes,
               fontsize=16, fontweight="bold", va="top")
    ax_rh.legend(frameon=True, fontsize=10, loc="lower right")

    # ------------------------------------------------------------------
    # Panel (b) — Waveform W(h)
    # ------------------------------------------------------------------
    ax_wave.plot(Wn, h_w, lw=1.4, color="#37474F", label=r"$W(h)=dE/dh$")
    ax_wave.set_xlim(-0.05, 1.35)           # fixed x range — annotations stay inside
    ax_wave.set_ylim(hmin - hpad, hmax + hpad)

    single_layer = (low_peak == high_peak)
    if not single_layer:
        ax_wave.scatter(Wn_s[low_peak],  h_w[low_peak],  s=50, zorder=3, color="#1b9e77")
        ax_wave.scatter(Wn_s[high_peak], h_w[high_peak], s=50, zorder=3, color="#d95f02")

    ax_wave.axhline(h_split, ls="--", lw=1.0, alpha=0.7, label="Layer boundary")
    ax_wave.set_xlabel("Waveform intensity (normalized)")
    ax_wave.set_ylabel("Height above ground [m]")
    ax_wave.set_title(r"$W(h)=\frac{dE}{dh}$", fontsize=12)
    ax_wave.spines["top"].set_visible(False)
    ax_wave.spines["right"].set_visible(False)
    ax_wave.text(0.02, 0.98, "(b)", transform=ax_wave.transAxes,
                 fontsize=16, fontweight="bold", va="top")

    if not single_layer:
        # Use axes-fraction x so text never extends beyond the fixed xlim
        ax_wave.annotate(
            "Understorey",
            xy=(Wn_s[low_peak], h_w[low_peak]),
            xytext=(1.05, (h_w[low_peak] - (hmin - hpad)) / (hmax + hpad - (hmin - hpad))),
            textcoords=("axes fraction" if False else "data",
                        "axes fraction"),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=10,
            annotation_clip=False,
        )
        ax_wave.annotate(
            "Overstorey",
            xy=(Wn_s[high_peak], h_w[high_peak]),
            xytext=(1.05, (h_w[high_peak] - (hmin - hpad)) / (hmax + hpad - (hmin - hpad))),
            textcoords=("axes fraction" if False else "data",
                        "axes fraction"),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=10,
            annotation_clip=False,
        )
        # Δh bracket between the two peaks — placed at x=1.25 (within xlim)
        xb = 1.20
        ax_wave.annotate(
            "", xy=(xb, h_w[high_peak]), xytext=(xb, h_w[low_peak]),
            arrowprops=dict(arrowstyle="<->", lw=1.0),
        )
        ax_wave.text(
            xb + 0.04, 0.5 * (h_w[high_peak] + h_w[low_peak]),
            r"$\Delta h$", va="center", fontsize=11,
        )

    ax_wave.text(
        0.05, 0.12,
        rf"$E(h_{{\mathrm{{split}}}}) = {U_strength:.2f}$",
        transform=ax_wave.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
    )
    ax_wave.legend(frameon=False, fontsize=10, loc="upper right")

    if title is None:
        title = (
            f"Simulated waveform  |  "
            f"({result.center_x:.0f}, {result.center_y:.0f}) UTM  |  "
            f"cover={result.cover:.2f}   HOME={result.home:.1f} m"
        )
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3-D point cloud
# ---------------------------------------------------------------------------

def plot_pointcloud_3d(
    df: pd.DataFrame,
    color_by: str = "Z",
    max_points: int = 50_000,
    point_size: float = 1.0,
    cmap: str = "terrain",
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    elev: float = 25.0,
    azim: float = -60.0,
    backend: str = "matplotlib",
    figsize: tuple[float, float] = (10, 8),
):
    """
    3-D scatter plot of the point cloud.

    Always subsamples to *max_points* for performance.  With a full 11 M-point
    PNOA tile, use ``max_points=50_000`` (matplotlib) or up to ``200_000``
    (plotly/WebGL).

    Parameters
    ----------
    df:
        Point-cloud DataFrame with columns ``X``, ``Y``, ``Z``.
    color_by:
        Attribute used for colour:

        * ``"Z"`` — elevation (default)
        * ``"RGB"`` — true colour (requires ``Red``, ``Green``, ``Blue`` columns)
        * ``"Intensity"`` — return intensity
        * ``"Classification"`` — LAS class codes (uses :data:`_LAS_CLASSES` palette)

    max_points:
        Maximum number of points to render.  Random subsample if exceeded.
    point_size:
        Marker size in points (matplotlib) or pixels (plotly).
    cmap:
        Matplotlib colormap name (ignored when *color_by* is ``"RGB"`` or
        ``"Classification"``).
    percentile_clip:
        Low / high percentile for colour-scale clipping.
    elev, azim:
        Initial viewing elevation and azimuth angles (matplotlib only).
    backend:
        ``"matplotlib"`` (default, static) or ``"plotly"`` (interactive,
        requires ``plotly`` to be installed).
    figsize:
        Figure size in inches (matplotlib only).

    Returns
    -------
    matplotlib.figure.Figure  or  plotly.graph_objects.Figure
    """
    # --- subsample ---
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()

    # --- colour array ---
    def _scalar_colour(values):
        valid = values[np.isfinite(values)]
        lo = np.percentile(valid, percentile_clip[0])
        hi = np.percentile(valid, percentile_clip[1])
        return np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)

    if color_by == "RGB":
        def _ch(col):
            v = df[col].to_numpy().astype(np.float32)
            valid = v[np.isfinite(v)]
            lo, hi = np.percentile(valid, percentile_clip)
            return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
        colours_rgb = np.stack([_ch("Red"), _ch("Green"), _ch("Blue")], axis=1)
        colours_scalar = None
    elif color_by == "Classification":
        codes = df["Classification"].to_numpy(dtype=int)
        hex_colours = [_LAS_CLASSES.get(int(c), (None, "#999999"))[1] for c in codes]
        colours_rgb = np.array([
            [int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255]
            for h in hex_colours
        ])
        colours_scalar = None
    else:
        field = "Intensity" if color_by == "Intensity" else "Z"
        colours_scalar = _scalar_colour(df[field].to_numpy().astype(np.float64))
        colours_rgb = None

    # -------------------------------------------------------------------
    if backend == "plotly":
        import plotly.graph_objects as go

        if colours_rgb is not None:
            colour_arg = [
                f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                for r, g, b in colours_rgb
            ]
            marker = dict(size=point_size, color=colour_arg, opacity=0.8)
        else:
            import matplotlib.pyplot as plt
            cmap_obj = plt.get_cmap(cmap)
            rgba = cmap_obj(colours_scalar)
            colour_arg = [
                f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                for r, g, b, _ in rgba
            ]
            marker = dict(size=point_size, color=colour_arg, opacity=0.8)

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=marker,
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title="Easting (m)",
                yaxis_title="Northing (m)",
                zaxis_title="Elevation (m)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title=f"Point cloud — {len(df):,} pts  |  colour: {color_by}",
        )
        return fig

    # -------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if colours_rgb is not None:
        ax.scatter(x, y, z, c=colours_rgb, s=point_size, linewidths=0,
                   depthshade=True, rasterized=True)
    else:
        import matplotlib.cm as cm
        cmap_obj = cm.get_cmap(cmap)
        ax.scatter(x, y, z, c=colours_scalar, cmap=cmap_obj,
                   s=point_size, linewidths=0, depthshade=True, rasterized=True)

    ax.ticklabel_format(useOffset=False)   # show full UTM coords, not offset notation
    ax.set_xlabel("Easting (m)", labelpad=8)
    ax.set_ylabel("Northing (m)", labelpad=8)
    ax.set_zlabel("Elevation (m)", labelpad=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Point cloud — {len(df):,} pts  |  colour: {color_by}", pad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3-D waveform waterfall
# ---------------------------------------------------------------------------

def plot_waveforms_3d(
    results: pd.DataFrame,
    color_by: str = "rh98",
    cmap: str = "YlGn",
    alpha: float = 0.7,
    line_width: float = 1.5,
    backend: str = "matplotlib",
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (12, 8),
    title: Optional[str] = None,
):
    """
    3-D waterfall plot of simulated GEDI-like waveforms.

    Each shot is drawn as a vertical RH(p) curve at its (X, Y) position —
    height above ground on the Z axis, cumulative energy percentile (0–100)
    on the Y offset.  Lines are coloured by a summary metric.

    Requires ``rh0``–``rh100`` columns in *results* (as returned by
    :func:`~alsdb.processing.waveform.simulate_batch`).

    Parameters
    ----------
    results:
        DataFrame from ``simulate_batch()``, must contain ``center_x``,
        ``center_y``, and ``rh0``–``rh100`` columns.
    color_by:
        Column used to colour the lines (default ``"rh98"``).
        Other useful choices: ``"cover"``, ``"rh50"``, ``"z_ground"``.
    cmap:
        Matplotlib colormap name.
    alpha:
        Line opacity.
    line_width:
        Line width in points.
    backend:
        ``"matplotlib"`` (static) or ``"plotly"`` (interactive).
    elev / azim:
        Matplotlib 3-D view angles (ignored for plotly).
    figsize:
        Figure size in inches (matplotlib only).
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure  or  plotly.graph_objects.Figure
    """
    rh_cols = [f"rh{p}" for p in range(101)]
    missing = [c for c in rh_cols if c not in results.columns]
    if missing:
        raise ValueError(f"results is missing RH columns: {missing[:5]} …")
    if color_by not in results.columns:
        raise ValueError(f"color_by column {color_by!r} not found in results")

    percentiles = np.arange(101, dtype=float)
    color_vals = results[color_by].to_numpy(dtype=float)
    vmin, vmax = np.nanmin(color_vals), np.nanmax(color_vals)

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    if backend == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for backend='plotly'") from exc

        fig = go.Figure()
        for _, row in results.iterrows():
            heights = row[rh_cols].to_numpy(dtype=float)
            rgba = cmap_obj(norm(row[color_by]))
            hex_col = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            fig.add_trace(go.Scatter3d(
                x=[row["center_x"]] * 101,
                y=percentiles,
                z=heights,
                mode="lines",
                line=dict(color=hex_col, width=line_width * 2),
                showlegend=False,
                opacity=alpha,
            ))
        fig.update_layout(
            scene=dict(
                xaxis_title="Easting (m)",
                yaxis_title="Cumulative energy (%)",
                zaxis_title="Height above ground (m)",
            ),
            title=title or "Simulated waveforms — RH profiles",
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig

    # --- matplotlib ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for _, row in results.iterrows():
        heights = row[rh_cols].to_numpy(dtype=float)
        color = cmap_obj(norm(row[color_by]))
        ax.plot(
            [row["center_x"]] * 101,
            percentiles,
            heights,
            color=color,
            alpha=alpha,
            linewidth=line_width,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=color_by, shrink=0.6, pad=0.1)

    ax.set_xlabel("Easting (m)", labelpad=8)
    ax.set_ylabel("Cumulative energy (%)", labelpad=8)
    ax.set_zlabel("Height above ground (m)", labelpad=8)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Overview figure (all four panels)
# ---------------------------------------------------------------------------

def plot_overview(
    df: pd.DataFrame,
    resolution: float = 1.0,
    figsize: tuple[float, float] = (16, 14),
    title: Optional[str] = None,
):
    """
    Four-panel overview figure: DSM with hillshade, RGB, intensity, classification.

    Parameters
    ----------
    df:
        Point-cloud DataFrame as returned by :class:`~alsdb.core.alsprovider.ALSProvider`.
    resolution:
        Grid resolution in metres.
    figsize:
        Figure size in inches.
    title:
        Optional suptitle.  Auto-generated from the coordinate range if None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_dsm(df, resolution=resolution, hillshade=True, ax=axes[0, 0])
    plot_rgb(df, resolution=resolution, ax=axes[0, 1])
    plot_intensity(df, resolution=resolution, ax=axes[1, 0])
    plot_classification(df, resolution=resolution, ax=axes[1, 1])

    if title is None:
        x_min, x_max, y_min, y_max = _extent(df)
        title = (
            f"ALS tile  |  X [{x_min:.0f} – {x_max:.0f}]  "
            f"Y [{y_min:.0f} – {y_max:.0f}]  |  {len(df):,} points"
        )
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
