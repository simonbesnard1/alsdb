# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Visualisation helpers for raster products (CHM, DTM, DSM GeoTIFFs).

Typical usage::

    from alsdb.utils.viz_raster import plot_chm, plot_dtm, plot_dsm, plot_products

    plot_chm("output/chm.tif")
    plot_products("output/dtm.tif", "output/dsm.tif", "output/chm.tif")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _read_raster(path: str | Path) -> tuple[np.ndarray, list[float]]:
    """
    Read a single-band GeoTIFF.

    Returns
    -------
    grid : np.ndarray (2-D, float32), nodata → np.nan
    extent : [x_min, x_max, y_min, y_max]  for imshow(extent=...)
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        bounds = src.bounds  # left, bottom, right, top

    if nodata is not None:
        data[data == nodata] = np.nan

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    return data, extent


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


# ---------------------------------------------------------------------------
# Individual product plots
# ---------------------------------------------------------------------------

def plot_chm(
    path: str | Path,
    cmap: str = "Greens",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    ax=None,
):
    """
    Plot a Canopy Height Model GeoTIFF.

    Parameters
    ----------
    path:
        Path to the CHM GeoTIFF.
    cmap:
        Matplotlib colormap (default ``"Greens"``).
    vmin / vmax:
        Colour scale limits.  ``vmax`` defaults to the 98th percentile.
    ax:
        Matplotlib axes.  A new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_raster(path)
    valid = grid[~np.isnan(grid)]
    if vmax is None:
        # Use the 98th percentile of vegetated pixels only (>0.5 m) so that
        # sparse tall trees are not swamped by a majority of bare-ground zeros.
        veg = valid[valid > 0.5]
        vmax = float(np.percentile(veg, 98)) if veg.size else float(np.nanmax(valid)) if valid.size else 30.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(grid, extent=extent, origin="upper", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Height above ground (m)", shrink=0.7)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"CHM — {Path(path).name}")
    return ax


def plot_dtm(
    path: str | Path,
    cmap: str = "terrain",
    hillshade: bool = True,
    vert_exag: float = 3.0,
    ax=None,
):
    """
    Plot a Digital Terrain Model GeoTIFF, optionally with hillshade.

    Parameters
    ----------
    path:
        Path to the DTM GeoTIFF.
    cmap:
        Matplotlib colormap (default ``"terrain"``).
    hillshade:
        Blend a hillshade overlay.
    vert_exag:
        Vertical exaggeration for the hillshade.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_raster(path)

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
    ax.set_title(f"DTM — {Path(path).name}")
    return ax


def plot_dsm(
    path: str | Path,
    cmap: str = "terrain",
    hillshade: bool = True,
    vert_exag: float = 3.0,
    ax=None,
):
    """
    Plot a Digital Surface Model GeoTIFF, optionally with hillshade.

    Parameters
    ----------
    path:
        Path to the DSM GeoTIFF.
    cmap:
        Matplotlib colormap (default ``"terrain"``).
    hillshade:
        Blend a hillshade overlay.
    vert_exag:
        Vertical exaggeration for the hillshade.
    ax:
        Matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    grid, extent = _read_raster(path)

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
    ax.set_title(f"DSM — {Path(path).name}")
    return ax


# ---------------------------------------------------------------------------
# Three-panel overview
# ---------------------------------------------------------------------------

def plot_products(
    dtm_path: str | Path,
    dsm_path: str | Path,
    chm_path: str | Path,
    figsize: tuple[float, float] = (18, 6),
    hillshade: bool = True,
    title: Optional[str] = None,
):
    """
    Three-panel overview: DTM | DSM | CHM.

    Parameters
    ----------
    dtm_path / dsm_path / chm_path:
        Paths to the respective GeoTIFFs.
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

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_dtm(dtm_path, hillshade=hillshade, ax=axes[0])
    plot_dsm(dsm_path, hillshade=hillshade, ax=axes[1])
    plot_chm(chm_path, ax=axes[2])

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
