# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Visualisation helpers for gridded ALS products stored in an
:class:`~alsdb.storage.ALSZarrStore`.

Typical usage::

    from alsdb.storage import ALSZarrStore
    from alsdb.utils.viz_raster import (
        plot_chm, plot_dtm, plot_dsm, plot_agb,
        plot_gap, plot_lai, plot_metrics,
        plot_products, plot_products_agb,
    )

    store = ALSZarrStore("output/spain.zarr")

    plot_chm(store, resolution=1.0, year=2021)
    plot_agb(store, resolution=10.0, year=2021)
    plot_products(store, resolution=1.0, year=2021)
    plot_products_agb(store, resolution=10.0, year=2021)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from alsdb.storage.zarr_store import ALSZarrStore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_from_store(
    store: "ALSZarrStore",
    variable: str,
    resolution: float,
    year: Optional[int] = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Extract a 2-D float32 grid and imshow extent from the store.

    Parameters
    ----------
    store:
        Open :class:`~alsdb.storage.ALSZarrStore`.
    variable:
        Variable name, e.g. ``"chm"``.
    resolution:
        Resolution group in metres.
    year:
        Survey year to select.  If ``None`` and the store contains exactly
        one time step, that step is used automatically.

    Returns
    -------
    grid : np.ndarray (ny, nx) float32 — NaN for missing cells
    extent : [x_min, x_max, y_min, y_max]  for ``imshow(extent=...)``
    """
    ds = store.to_dataset(resolution)

    if variable not in ds:
        available = list(ds.data_vars)
        raise KeyError(
            f"Variable '{variable}' not found in store at {resolution} m. "
            f"Available: {available}"
        )

    da = ds[variable]

    if year is not None:
        if year not in da.time.values:
            raise ValueError(
                f"Year {year} not in store (available: {da.time.values.tolist()})"
            )
        da = da.sel(time=year)
    elif da.sizes["time"] == 1:
        da = da.isel(time=0)
    else:
        raise ValueError(
            f"Store has multiple years {da.time.values.tolist()} — "
            "specify year= to select one."
        )

    grid = da.values.astype(np.float32)  # (ny, nx)

    x = ds.x.values          # ascending cell centres
    y = ds.y.values           # descending cell centres (north-up)
    half = resolution / 2.0
    extent = [
        float(x[0]  - half),   # x_min
        float(x[-1] + half),   # x_max
        float(y[-1] - half),   # y_min (southernmost edge)
        float(y[0]  + half),   # y_max (northernmost edge)
    ]
    return grid, extent


def _hillshade_blend(grid: np.ndarray, cmap, vert_exag: float = 3.0):
    """Return an RGBA array blending elevation colour with hillshade."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=315, altdeg=45)
    filled = np.where(np.isnan(grid), np.nanmin(grid), grid)
    norm = plt.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return ls.shade(filled, cmap=cmap_obj, norm=norm,
                    vert_exag=vert_exag, blend_mode="soft")


def _label(store: "ALSZarrStore", variable: str, year: Optional[int]) -> str:
    year_str = f" ({year})" if year is not None else ""
    return f"{variable.upper()}{year_str} — {store.path.name}"


# ---------------------------------------------------------------------------
# Individual product plots
# ---------------------------------------------------------------------------

def plot_chm(
    store: "ALSZarrStore",
    resolution: float = 1.0,
    year: Optional[int] = None,
    cmap: str = "Greens",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    ax=None,
):
    """
    Plot the Canopy Height Model from *store*.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing a ``"chm"`` variable.
    resolution:
        Resolution group in metres (default 1 m).
    year:
        Survey year.  Required when the store has more than one time step.
    cmap:
        Matplotlib colormap (default ``"Greens"``).
    vmin / vmax:
        Colour scale limits.  ``vmax`` defaults to the 98th percentile of
        vegetation pixels (HAG > 0.5 m).
    ax:
        Matplotlib axes.  A new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_from_store(store, "chm", resolution, year)
    valid = grid[~np.isnan(grid)]
    if vmax is None:
        veg = valid[valid > 0.5]
        vmax = float(np.percentile(veg, 98)) if veg.size else float(np.nanmax(valid)) if valid.size else 30.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Height above ground (m)", shrink=0.7)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(_label(store, "chm", year))
    return ax


def plot_dtm(
    store: "ALSZarrStore",
    resolution: float = 1.0,
    year: Optional[int] = None,
    cmap: str = "terrain",
    hillshade: bool = True,
    vert_exag: float = 3.0,
    ax=None,
):
    """
    Plot the Digital Terrain Model from *store*, optionally with hillshade.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing a ``"dtm"`` variable.
    resolution:
        Resolution group in metres.
    year:
        Survey year.
    cmap:
        Matplotlib colormap (default ``"terrain"``).
    hillshade:
        Blend a hillshade overlay (default ``True``).
    vert_exag:
        Vertical exaggeration for the hillshade (default 3).
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_from_store(store, "dtm", resolution, year)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if hillshade:
        rgb = _hillshade_blend(grid, cmap, vert_exag=vert_exag)
        ax.imshow(rgb, extent=extent, origin="upper", aspect="equal")
    else:
        im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                       cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(_label(store, "dtm", year))
    return ax


def plot_dsm(
    store: "ALSZarrStore",
    resolution: float = 1.0,
    year: Optional[int] = None,
    cmap: str = "terrain",
    hillshade: bool = True,
    vert_exag: float = 3.0,
    ax=None,
):
    """
    Plot the Digital Surface Model from *store*, optionally with hillshade.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing a ``"dsm"`` variable.
    resolution:
        Resolution group in metres.
    year:
        Survey year.
    cmap:
        Matplotlib colormap (default ``"terrain"``).
    hillshade:
        Blend a hillshade overlay (default ``True``).
    vert_exag:
        Vertical exaggeration for the hillshade (default 3).
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_from_store(store, "dsm", resolution, year)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if hillshade:
        rgb = _hillshade_blend(grid, cmap, vert_exag=vert_exag)
        ax.imshow(rgb, extent=extent, origin="upper", aspect="equal")
    else:
        im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                       cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(_label(store, "dsm", year))
    return ax


def plot_agb(
    store: "ALSZarrStore",
    resolution: float = 10.0,
    year: Optional[int] = None,
    cmap: str = "YlGn",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    ax=None,
):
    """
    Plot Above-Ground Biomass (AGB) from *store*.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing a ``"biomass"`` variable.
    resolution:
        Resolution group in metres (default 10 m).
    year:
        Survey year.
    cmap:
        Matplotlib colormap (default ``"YlGn"``).
    vmin / vmax:
        Colour scale limits.  ``vmax`` defaults to the 98th percentile.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_from_store(store, "biomass", resolution, year)
    valid = grid[~np.isnan(grid)]
    if vmax is None:
        vmax = float(np.percentile(valid, 98)) if valid.size else 500.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="AGB (Mg ha⁻¹)", shrink=0.7)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(_label(store, "biomass", year))
    return ax


def plot_gap(
    store: "ALSZarrStore",
    resolution: float = 10.0,
    year: Optional[int] = None,
    cmap: str = "RdYlGn_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax=None,
):
    """
    Plot gap fraction from *store*.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing a ``"gap"`` variable.
    resolution:
        Resolution group in metres.
    year:
        Survey year.
    cmap:
        Matplotlib colormap (default ``"RdYlGn_r"``).
    vmin / vmax:
        Colour scale limits (default 0–1).
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_from_store(store, "gap", resolution, year)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Gap fraction", shrink=0.7)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(_label(store, "gap", year))
    return ax


def plot_lai(
    store: "ALSZarrStore",
    resolution: float = 10.0,
    year: Optional[int] = None,
    cmap: str = "YlGn",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    ax=None,
):
    """
    Plot effective LAI from *store*.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing a ``"lai"`` variable.
    resolution:
        Resolution group in metres.
    year:
        Survey year.
    cmap:
        Matplotlib colormap (default ``"YlGn"``).
    vmin / vmax:
        Colour scale limits.  ``vmax`` defaults to the 98th percentile.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_from_store(store, "lai", resolution, year)
    valid = grid[~np.isnan(grid)]
    if vmax is None:
        vmax = float(np.percentile(valid, 98)) if valid.size else 8.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Effective LAI (m² m⁻²)", shrink=0.7)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(_label(store, "lai", year))
    return ax


def plot_metrics(
    store: "ALSZarrStore",
    resolution: float = 10.0,
    year: Optional[int] = None,
    variables: Optional[list[str]] = None,
    figsize: tuple[float, float] = (20, 10),
    title: Optional[str] = None,
):
    """
    Multi-panel overview of LiDAR structural metrics.

    Plots up to six panels: ``h50``, ``h75``, ``h95``, ``hmean``, ``cc``,
    ``density``.  Variables missing from the store are silently skipped.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore`.
    resolution:
        Resolution group in metres (default 10 m).
    year:
        Survey year.
    variables:
        Subset of metric names to plot.  Defaults to all six standard metrics.
    figsize:
        Figure size in inches.
    title:
        Optional suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    _default_metrics = ["h50", "h75", "h95", "hmean", "cc", "density"]
    _cmaps = {
        "h50": "viridis", "h75": "viridis", "h95": "viridis",
        "hmean": "viridis", "cc": "YlGn", "density": "plasma",
    }
    _labels = {
        "h50": "h50 (m)", "h75": "h75 (m)", "h95": "h95 (m)",
        "hmean": "Mean height (m)", "cc": "Canopy cover", "density": "Density (pts m⁻²)",
    }

    vars_to_plot = variables or _default_metrics
    available = store.variables(resolution)
    vars_to_plot = [v for v in vars_to_plot if v in available]

    if not vars_to_plot:
        raise ValueError(
            f"None of the requested variables are in the store at {resolution} m. "
            f"Available: {available}"
        )

    ncols = min(3, len(vars_to_plot))
    nrows = (len(vars_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, var in enumerate(vars_to_plot):
        ax = axes[idx // ncols][idx % ncols]
        grid, extent = _read_from_store(store, var, resolution, year)
        valid = grid[~np.isnan(grid)]
        vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
        cmap = _cmaps.get(var, "viridis")
        im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                       cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, label=_labels.get(var, var), shrink=0.7)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(var.upper())

    # Hide unused subplots
    for idx in range(len(vars_to_plot), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    year_str = f" ({year})" if year is not None else ""
    fig.suptitle(title or f"LiDAR metrics — {store.path.name}{year_str}", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-panel overviews
# ---------------------------------------------------------------------------

def plot_products(
    store: "ALSZarrStore",
    resolution: float = 1.0,
    year: Optional[int] = None,
    figsize: tuple[float, float] = (18, 6),
    hillshade: bool = True,
    title: Optional[str] = None,
):
    """
    Three-panel overview: DTM | DSM | CHM.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing ``"dtm"``, ``"dsm"``,
        and ``"chm"`` at *resolution*.
    resolution:
        Resolution group in metres (default 1 m).
    year:
        Survey year.
    figsize:
        Figure size in inches.
    hillshade:
        Apply hillshade to DTM and DSM panels (default ``True``).
    title:
        Optional suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_dtm(store, resolution=resolution, year=year, hillshade=hillshade, ax=axes[0])
    plot_dsm(store, resolution=resolution, year=year, hillshade=hillshade, ax=axes[1])
    plot_chm(store, resolution=resolution, year=year, ax=axes[2])

    year_str = f" ({year})" if year is not None else ""
    fig.suptitle(title or f"DTM / DSM / CHM — {store.path.name}{year_str}", fontsize=13)
    fig.tight_layout()
    return fig


def plot_products_agb(
    store: "ALSZarrStore",
    resolution: float = 10.0,
    year: Optional[int] = None,
    figsize: tuple[float, float] = (22, 6),
    hillshade: bool = True,
    title: Optional[str] = None,
):
    """
    Four-panel overview: DTM | DSM | CHM | AGB.

    All panels read from *store* at *resolution*.  The CHM and terrain models
    are typically computed at 1 m while AGB is at 10 m, so this composite is
    most useful when a single resolution holds all variables (or when the user
    passes a coarser resolution for all).

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing ``"dtm"``, ``"dsm"``,
        ``"chm"``, and ``"biomass"`` at *resolution*.
    resolution:
        Resolution group in metres (default 10 m).
    year:
        Survey year.
    figsize:
        Figure size in inches.
    hillshade:
        Apply hillshade to DTM and DSM panels.
    title:
        Optional suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    plot_dtm(store, resolution=resolution, year=year, hillshade=hillshade, ax=axes[0])
    plot_dsm(store, resolution=resolution, year=year, hillshade=hillshade, ax=axes[1])
    plot_chm(store, resolution=resolution, year=year, ax=axes[2])
    plot_agb(store, resolution=resolution, year=year, ax=axes[3])

    year_str = f" ({year})" if year is not None else ""
    fig.suptitle(title or f"DTM / DSM / CHM / AGB — {store.path.name}{year_str}", fontsize=13)
    fig.tight_layout()
    return fig
