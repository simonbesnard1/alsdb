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
    from scipy.stats import binned_statistic_2d  # noqa: F401 — checked at call site

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
