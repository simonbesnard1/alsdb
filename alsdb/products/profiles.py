# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Vertical canopy profiles: gap probability, plant area and attenuation.

Every profile here follows one chain: a gap probability ``Pgap(h)`` over
height above ground, plant area traversed down to ``h`` from Beer-Lambert,
``PAI(h) = -ln Pgap(h) / k`` (``k = G Omega / mu``, 0.5 for a spherical
leaf angle distribution at nadir), and plant area volume density
``PAVD = -dPAI/dh``. The gap probability comes from:

- ALS returns (MacArthur & Horn 1969): the share of returns not intercepted
  above ``h``, from first returns or from all returns weighted by
  ``1 / NumberOfReturns``, counted or weighted by a footprint
  (:func:`gap_from_returns`, :func:`matched_als_gaps`);
- a simulated waveform split into canopy and ground (GEDI L2B):
  ``Pgap(h) = 1 - Rv(h) / (Rv + rv_rg * Rg)``, with ``Rv(h)`` the canopy
  energy above ``h``, ``Rv`` and ``Rg`` the canopy and ground energy and
  ``rv_rg`` the canopy-to-ground reflectance ratio (L2B ``rhov / rhog``)
  (:func:`gap_from_waveform`).

Two kinds of footprint profile use it:

- the ALS profile of a circular footprint, counted, with layers from the
  ground to the canopy height (:class:`PAVDProfile`,
  :func:`compute_als_pavd_profile`, :func:`compute_als_pavd_batch`);
- the GEDI-matched set for attenuation studies: the canopy and ground parts
  of the waveform of :mod:`alsdb.products.waveform` (the GEDI simulator's
  ``-ground`` split by ALS class) and the waveform and ALS gap profiles from
  the same points with the same footprint and density weights, on one
  height grid (:func:`gedi_footprint_profiles`, :func:`simulate_gedi_profiles`).
  All three are gap probabilities on the same heights, so one retrieval turns
  them into plant area, e.g. gedidb's ``GEDIVerticalProfiler.compute_profiles``
  (use :func:`signal_heights` for its height layout).

:func:`fit_transmittance_model` relates an observed to an ALS PAVD profile
through the plant area above. ALS profiles depend on sampling, occlusion,
classification, terrain and extinction assumptions: a reference for waveform
retrievals, not an independent measurement. Simulated waveforms are built
from the ALS returns and inherit their occlusion: simulated against ALS
isolates what the waveform and the L2B retrieval add, real GEDI against
simulated what the simulation cannot represent.

References
----------
Hancock, S., et al. (2019). The GEDI simulator: A large-footprint waveform
    lidar simulator for calibration and validation of spaceborne missions.
    Remote Sensing of Environment, 220, 309-323.
MacArthur, R. H., & Horn, H. S. (1969). Foliage profile by vertical
    measurements. Ecology, 50(5), 802-804.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alsdb.methods._terrain import VEG_CLASSES as _VEG_CLASSES
from alsdb.products import waveform as _wf
from alsdb.products._tiling import PointGrid
from alsdb.products.gap import _GROUND_CLASS, _LAI_K_DEFAULT, _LAI_MAX

logger = logging.getLogger(__name__)

_Z_STEP_DEFAULT = 0.15  # m, GEDI native vertical resolution, as waveform.py
_MIN_POINTS_DEFAULT = 50
_MIN_GROUND_POINTS_DEFAULT = 5  # too few ground returns -> Pgap(0) near 0 -> ln blows up

HEIGHT_RANGE = (-10.0, 80.0)  # m above ground, default grid of the GEDI-matched profiles
# GEDI-matched profiles stop at the signal top: the height above which the canopy
# holds less than this share of the waveform energy (a noise-threshold stand-in).
# GEDI's pgap_theta_z spans only the detected signal, and gedidb's profiler takes
# the highest valid height as the canopy height.
TOP_THRESHOLD = 1e-3
PROFILE_ARRAYS = ("canopy", "ground", "pgap_waveform", "pgap_als_first", "pgap_als_all")
GEDI_FILL = -9999.0  # GEDI's fill value; gedidb treats heights below -1000 as fill


# ---------------------------------------------------------------------------
# Gap probability
# ---------------------------------------------------------------------------


def gap_from_returns(heights, weights, at):
    """Return the MacArthur-Horn gap probability at the heights *at*.

    The gap at height ``h`` is the share of the (weighted) returns at or below
    ``h``: the returns not intercepted above it (MacArthur & Horn 1969). Heights
    below 0 are evaluated at 0, the gap at the ground (1 - cover).

    Parameters
    ----------
    heights : np.ndarray
        Heights above ground of the returns (ground returns at 0).
    weights : np.ndarray
        Weight of each return (1 for counts).
    at : np.ndarray
        Heights to evaluate.

    Returns
    -------
    np.ndarray
        Gap probability at *at*; NaN without weight.
    """
    heights, weights, at = (np.asarray(a, dtype=float) for a in (heights, weights, at))
    total = weights.sum()
    if total <= 0:
        return np.full(len(at), np.nan)
    order = np.argsort(heights, kind="stable")
    cumulative = np.r_[0.0, np.cumsum(weights[order])]
    index = np.searchsorted(heights[order], np.maximum(at, 0.0), side="right")
    return cumulative[index] / total


def gap_from_waveform(canopy, ground, rv_rg=1.0, totals=None):
    """Return the waveform gap probability at the bin centres, as GEDI L2B.

    ``Pgap(h) = 1 - Rv(h) / (Rv + rv_rg * Rg)``: ``Rv(h)`` is the canopy energy
    above the centre of each bin, ``Rv`` and ``Rg`` the canopy and ground
    totals, ``rv_rg`` the canopy-to-ground reflectance ratio.

    Parameters
    ----------
    canopy, ground : np.ndarray
        Canopy and ground energy per bin, ascending heights.
    rv_rg : float
        Canopy-to-ground reflectance ratio.
    totals : tuple of float, optional
        Canopy and ground energy of the whole waveform (``canopy_energy``,
        ``ground_energy`` of footprints.parquet); default the sums of the
        profiles, which miss energy outside the height grid (e.g. a ground
        return spread below it on steep slopes).

    Returns
    -------
    np.ndarray
        Gap probability per bin: 1 above the canopy, ``1 - cover`` at the ground.
    """
    canopy, ground = np.asarray(canopy, float), np.asarray(ground, float)
    if not np.isfinite(rv_rg) or rv_rg <= 0:
        raise ValueError("rv_rg must be finite and positive")
    rv, rg = (canopy.sum(), ground.sum()) if totals is None else totals
    denominator = rv + rv_rg * rg
    if denominator <= 0:
        return np.full(len(canopy), np.nan)
    above = np.cumsum(canopy[::-1])[::-1] - canopy / 2  # energy above each centre
    return 1.0 - above / denominator


def matched_als_gaps(
    points, center_x, center_y, edges, *, beam_sigma, density_cell, terrain=None, hag=None
):
    """Return ALS gap probabilities with the footprint and density weights of the waveform.

    Parameters
    ----------
    points : np.ndarray
        Structured point array of the footprint (X, Y, Z, ReturnNumber,
        NumberOfReturns, Classification).
    center_x, center_y : float
        Footprint centre.
    edges : np.ndarray
        Height-above-ground edges of the profile grid.
    beam_sigma : float or None
        Gaussian footprint sigma (m); None weights all points equally.
    density_cell : float or None
        Cell (m) for the inverse pulse-density weight; None turns it off.
    terrain : TerrainModel, optional
        Ground surface for heights above ground, when ``hag`` is not given.
    hag : np.ndarray, optional
        Heights above ground of *points*, computed once for many footprints.

    Returns
    -------
    first, all_returns : np.ndarray
        Gap probability at the bin centres from first returns, and from all
        returns weighted by ``1 / NumberOfReturns``. Only ground (2) and
        vegetation (3-5) returns count, as in :func:`compute_als_pavd_profile`.
    """
    centres = (edges[:-1] + edges[1:]) / 2
    cls = points["Classification"]
    keep = (cls == 2) | np.isin(cls, _VEG_CLASSES)
    used = points[keep]
    nan = np.full(len(centres), np.nan)
    if not len(used):
        return nan, nan
    if hag is None:
        hag = heights_above_ground(used, terrain)
    else:
        hag = np.asarray(hag, float)[keep].copy()
    hag[used["Classification"] == 2] = 0.0
    weights = np.ones(len(used))
    if beam_sigma is not None:
        r2 = (used["X"] - center_x) ** 2 + (used["Y"] - center_y) ** 2
        weights = np.exp(-r2 / (2.0 * beam_sigma**2))
    if density_cell is not None:
        ix = np.floor(used["X"] / density_cell).astype(np.int64)
        iy = np.floor(used["Y"] / density_cell).astype(np.int64)
        ix -= ix.min()
        iy -= iy.min()
        index = ix * (int(iy.max()) + 1) + iy
        first = used["ReturnNumber"] == 1
        pulses = np.bincount(index, weights=first)[index]
        weights = np.divide(weights, pulses, out=np.zeros_like(weights), where=pulses > 0)
    ok = np.isfinite(hag)
    first = ok & (used["ReturnNumber"] == 1)
    gap_first = gap_from_returns(hag[first], weights[first], centres)
    nret = np.maximum(used["NumberOfReturns"].astype(float), 1.0)
    gap_all = gap_from_returns(hag[ok], (weights / nret)[ok], centres)
    return gap_first, gap_all


# ---------------------------------------------------------------------------
# Gap probability to plant area
# ---------------------------------------------------------------------------


def pai_from_gap(gap, k=_LAI_K_DEFAULT, pai_max=_LAI_MAX):
    """Return the plant area traversed, ``-ln(gap) / k``, clipped to [0, pai_max].

    Parameters
    ----------
    gap : np.ndarray
        Gap probability.
    k : float
        Extinction coefficient ``G Omega / mu`` (see ``gap.LAI_K_PRESETS``).
    pai_max : float
        Physical ceiling of the plant area index.

    Returns
    -------
    np.ndarray
        Cumulative plant area from the canopy top down to each height.
    """
    return np.clip(-np.log(gap) / k, 0.0, pai_max)


def pavd_from_pai(pai, edges):
    """Return the plant area volume density of each layer, ``-dPAI/dh``.

    Parameters
    ----------
    pai : np.ndarray
        Cumulative plant area at the layer edges (:func:`pai_from_gap`).
    edges : np.ndarray
        Ascending layer edges (m above ground).

    Returns
    -------
    np.ndarray
        PAVD (m2/m3) per layer; positive, as less plant area lies above higher edges.
    """
    return -np.diff(pai) / np.diff(edges)


# ---------------------------------------------------------------------------
# ALS profile of a circular footprint
# ---------------------------------------------------------------------------


@dataclass
class PAVDProfile:
    """Counted ALS PAVD profile of one footprint, layers from the ground to the canopy height."""

    z: np.ndarray  # layer edges, height above ground, ascending (m), ground=0
    z_rel: np.ndarray  # z / canopy_height, clipped to [0, 1]
    pavd: np.ndarray  # m^2/m^3, one value per LAYER (len = len(z) - 1)
    pavd_z: np.ndarray  # layer-centre heights matching pavd, ascending (m)
    pavd_z_rel: np.ndarray  # pavd_z / canopy_height
    pai_cumulative: np.ndarray  # cumulative PAI traversed from the canopy top
    # down to each edge in z; the predictor of fit_transmittance_model().
    total_pai: float
    canopy_height: float
    n_points: int
    n_ground_points: int
    n_unsupported: int = 0
    n_extrapolated: int = 0


def _pavd_profile_from_hag(
    hag: np.ndarray,
    classification: np.ndarray,
    z_step: float = _Z_STEP_DEFAULT,
    k: float = _LAI_K_DEFAULT,
    min_ground_points: int = _MIN_GROUND_POINTS_DEFAULT,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
) -> PAVDProfile | None:
    """Compute the PAVD profile of one footprint from heights above ground.

    Returns None with too few ground returns or no vegetation above ground.

    veg_classes:
        LAS classes treated as vegetation (default ``(3, 4, 5)``). Some
        surveys use other codes (e.g. class 20 for all above-ground returns):
        check the classes present.
    """
    if not np.isfinite([z_step, k]).all() or z_step <= 0 or k <= 0 or min_ground_points < 1:
        raise ValueError("z_step, k and min_ground_points must be positive")
    hag = np.asarray(hag, dtype=float).copy()
    classification = np.asarray(classification)
    if hag.shape != classification.shape or hag.ndim != 1:
        raise ValueError("Heights and classification must be matching one-dimensional arrays")
    hag[classification == _GROUND_CLASS] = 0
    valid = np.isfinite(hag)
    hag, classification = hag[valid], classification[valid]
    gnd_mask = classification == _GROUND_CLASS
    n_gnd = int(gnd_mask.sum())
    if n_gnd < min_ground_points:
        return None

    veg_mask = np.isin(classification, veg_classes)
    if not veg_mask.any():
        logger.warning(
            "_pavd_profile_from_hag: veg_classes=%s matched 0 of %d points -- "
            "this survey likely uses different classification codes than the "
            "ASPRS default; the resulting profile (if any) is almost certainly "
            "wrong (ground-only). Check np.unique(classification) for this "
            "dataset's real codes and pass the correct veg_classes explicitly.",
            veg_classes,
            len(classification),
        )
    classified = gnd_mask | veg_mask  # as gap.py: unclassified, noise, buildings left out
    hag_cls = hag[classified]
    n_tot = int(classified.sum())
    if n_tot == 0:
        return None

    above_ground = hag[veg_mask & (hag > 0)]
    canopy_height = float(np.percentile(above_ground, 98)) if len(above_ground) else 0.0
    if canopy_height <= 0:
        return None

    n_bins = max(1, int(np.ceil(canopy_height / z_step)))
    z_edges = np.linspace(0.0, n_bins * z_step, n_bins + 1)  # ascending: ground -> top

    gap = gap_from_returns(hag_cls, np.ones(n_tot), z_edges)
    # One unseen return is a finite lower bound for poorly sampled ground.
    pai_traversed = pai_from_gap(np.clip(gap, 1.0 / n_tot, 1.0), k)
    pavd = pavd_from_pai(pai_traversed, z_edges)
    pavd_z = (z_edges[:-1] + z_edges[1:]) / 2.0

    return PAVDProfile(
        z=z_edges,
        z_rel=np.clip(z_edges / canopy_height, 0.0, 1.0),
        pavd=pavd,
        pavd_z=pavd_z,
        pavd_z_rel=np.clip(pavd_z / canopy_height, 0.0, 1.0),
        pai_cumulative=pai_traversed,
        total_pai=float(pai_traversed[0]),
        canopy_height=canopy_height,
        n_points=n_tot,
        n_ground_points=n_gnd,
    )


def compute_als_pavd_profile(
    provider,
    center_x: float,
    center_y: float,
    footprint_radius: float,
    z_step: float = _Z_STEP_DEFAULT,
    k: float = _LAI_K_DEFAULT,
    year: int | None = None,
    min_points: int = _MIN_POINTS_DEFAULT,
    min_ground_points: int = _MIN_GROUND_POINTS_DEFAULT,
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    terrain_buffer: float = 30.0,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = True,
) -> PAVDProfile | None:
    """Return the counted PAVD profile of a circular footprint.

    Parameters
    ----------
    provider : TileDBProvider
        Point cloud provider.
    center_x, center_y : float
        Footprint centre, in the array's CRS.
    footprint_radius : float
        Footprint radius (m), e.g. 12.5 for GEDI.
    z_step : float
        Vertical bin size (m).
    k : float
        Beer-Lambert extinction coefficient (see ``gap.LAI_K_PRESETS``).
    year : int, optional
        Survey year; ``None`` uses all years.
    min_points : int
        Return None with fewer points.
    min_ground_points : int
        Return None with fewer ground first returns.
    veg_classes : tuple[int, ...]
        LAS classes treated as vegetation (default ``(3, 4, 5)``). Some
        surveys use other codes (e.g. class 20 for all above-ground returns):
        check the classes present.
    terrain_buffer : float
        Ground returns this far (m) beyond the footprint shape the terrain.
    max_ground_distance : float, optional
        Returns farther (m) from ground support get no height.
    ground_extrapolation : bool
        Give returns outside the ground TIN the height above the nearest
        ground (default). Off, they are left out (``n_unsupported``) while
        the ground returns stay: the gap rises and the PAI falls.

    Returns
    -------
    PAVDProfile or None
    """
    options = dict(locals())
    options.pop("provider")
    if not np.isfinite(terrain_buffer) or terrain_buffer < 0:
        raise ValueError("terrain_buffer must be finite and nonnegative")
    data = _wf._query_footprint(
        provider, center_x, center_y, footprint_radius + terrain_buffer, year
    )
    return pavd_from_points(data, **options)


def pavd_from_points(
    data,
    center_x,
    center_y,
    footprint_radius,
    z_step=_Z_STEP_DEFAULT,
    k=_LAI_K_DEFAULT,
    year=None,
    min_points=_MIN_POINTS_DEFAULT,
    min_ground_points=_MIN_GROUND_POINTS_DEFAULT,
    veg_classes=_VEG_CLASSES,
    terrain_buffer=30.0,
    max_ground_distance=None,
    ground_extrapolation=True,
):
    """Compute a footprint profile from points covering the footprint and terrain buffer.

    Parameters as :func:`compute_als_pavd_profile`, with *data* the point
    arrays instead of the provider. First returns without terrain support are
    counted in ``n_unsupported``.
    """
    from alsdb.methods._surface import clean_points
    from alsdb.methods._terrain import TerrainModel, normalize_points
    from alsdb.products._footprints import point_array

    if (
        not np.isfinite([footprint_radius, terrain_buffer, z_step, k]).all()
        or footprint_radius <= 0
        or terrain_buffer < 0
        or z_step <= 0
        or k <= 0
    ):
        raise ValueError("Invalid footprint, terrain buffer or profile parameters")
    if data is None:
        return None
    points = clean_points(point_array(data))
    distance = (points["X"] - center_x) ** 2 + (points["Y"] - center_y) ** 2
    terrain = TerrainModel(
        points[
            (points["Classification"] == 2) & (distance <= (footprint_radius + terrain_buffer) ** 2)
        ]
    )
    inside = distance <= footprint_radius**2
    if inside.sum() < min_points:
        return None
    first = points[inside & (points["ReturnNumber"] == 1)]
    normalized, _ = normalize_points(
        first, model=terrain, max_distance=max_ground_distance, extrapolate=ground_extrapolation
    )
    heights = normalized["HeightAboveGround"]
    canopy = first["Classification"] != 2
    result = _pavd_profile_from_hag(
        heights, first["Classification"], z_step, k, min_ground_points, veg_classes
    )
    if result is not None:
        result.n_unsupported = int((~np.isfinite(heights) & canopy).sum())
        result.n_extrapolated = int(
            (normalized["GroundExtrapolated"].astype(bool) & np.isfinite(heights) & canopy).sum()
        )
    return result


# ---------------------------------------------------------------------------
# GEDI-matched profiles of one footprint
# ---------------------------------------------------------------------------


def height_edges(height_range=HEIGHT_RANGE, z_step=0.15):
    """Return bin edges (m above ground) of the GEDI-matched profile grid.

    Parameters
    ----------
    height_range : tuple of float
        Lowest and highest height (m above ground).
    z_step : float
        Bin size (m), the waveform's.

    Returns
    -------
    np.ndarray
        Ascending edges, ``z_step`` apart, covering ``height_range``.
    """
    low, high = height_range
    if not np.isfinite([low, high, z_step]).all() or high <= low or z_step <= 0:
        raise ValueError("Invalid height range or bin size")
    n = int(np.ceil((high - low) / z_step))
    return low + np.arange(n + 1) * z_step


def rebin(z_bins, values, z_ref, edges):
    """Rebin a waveform onto height-above-ground edges, keeping its energy.

    Parameters
    ----------
    z_bins : np.ndarray
        Ascending, equally spaced bin centres (m, elevation).
    values : np.ndarray
        Energy per bin.
    z_ref : float
        Ground elevation (m) the heights are measured from.
    edges : np.ndarray
        Target edges (m above ground).

    Returns
    -------
    np.ndarray
        Energy per target bin; energy outside the edges is dropped.
    """
    step = z_bins[1] - z_bins[0] if len(z_bins) > 1 else 1.0
    source = np.r_[z_bins - step / 2, z_bins[-1] + step / 2] - z_ref
    cumulative = np.r_[0.0, np.cumsum(values)]
    return np.diff(np.interp(edges, source, cumulative))


def signal_heights(height, signal_top):
    """Return per-footprint heights in GEDI's layout: fill above each signal top.

    GEDI's ``pgap_theta_z`` heights span only the detected signal, and
    gedidb's profiler takes the highest valid height as the canopy height.
    Pass the result with a gap profile to gedidb, e.g.::

        from gedidb.utils.profiles import GEDIVerticalProfiler

        heights = signal_heights(root["height"][:], table["signal_top"])
        cp_pavd, cp_height, H = GEDIVerticalProfiler().compute_profiles(
            root["pgap_waveform"][:], heights, np.pi / 2, 0.5, 1.0
        )  # nadir beam, G = 0.5, Omega = 1

    Parameters
    ----------
    height : np.ndarray
        Bin centres (m above ground), ``profiles.zarr/height``.
    signal_top : array-like
        ``signal_top`` of each footprint (footprints.parquet), in row order.

    Returns
    -------
    np.ndarray
        (footprints, heights) with :data:`GEDI_FILL` above each signal top
        and for footprints without a profile.
    """
    height = np.asarray(height, dtype=float)
    top = np.asarray(signal_top, dtype=float)[:, None]
    out = np.broadcast_to(height, (len(top), len(height))).copy()
    out[~(out < top)] = GEDI_FILL  # NaN tops (no profile) are fill too
    return out


def heights_above_ground(points, terrain):
    """Return heights above a ground TIN; nearest ground outside it, so no point is lost."""
    from alsdb.methods._terrain import normalize_points

    normalized, _ = normalize_points(points, model=terrain, distances=False, extrapolate=True)
    return np.asarray(normalized["HeightAboveGround"], float)


def gedi_footprint_profiles(
    data, center_x, center_y, edges, rv_rg=1.0, top_threshold=TOP_THRESHOLD, hag=None, **options
):
    """Return the waveform, canopy/ground profiles and gap profiles of one GEDI footprint.

    Parameters
    ----------
    data
        Point arrays of the footprint and its surroundings (as for
        :func:`alsdb.products.waveform.waveform_from_points`).
    center_x, center_y : float
        Footprint centre.
    edges : np.ndarray
        Height-above-ground edges (:func:`height_edges`).
    rv_rg : float
        Canopy-to-ground reflectance ratio for the waveform gap probability.
    top_threshold : float
        Share of the waveform energy above the signal top; the gap profiles
        are NaN above it (see :data:`TOP_THRESHOLD`).
    hag : np.ndarray, optional
        Heights above ground of *data*, already cleaned (as
        :func:`simulate_gedi_profiles` computes them once per chunk); default:
        a TIN of the footprint's own ground returns.
    **options
        Passed to :func:`~alsdb.products.waveform.waveform_from_points`.

    Returns
    -------
    tuple of (WaveformResult, dict, float) or None
        The waveform result, the arrays of :data:`PROFILE_ARRAYS` and the
        signal top (m above ground); None without enough points or without
        ground returns.
    """
    from alsdb.methods._surface import clean_points
    from alsdb.methods._terrain import TerrainModel
    from alsdb.products._footprints import point_array

    result = _wf.waveform_from_points(data, center_x, center_y, **options)
    if result is None or not np.isfinite(result.z_ground_true):
        return None
    canopy = rebin(result.z_bins, result.canopy_waveform, result.z_ground_true, edges)
    ground = rebin(result.z_bins, result.ground_waveform, result.z_ground_true, edges)
    points = point_array(data) if hag is not None else clean_points(point_array(data))
    ground_points = points[points["Classification"] == 2]
    if hag is None and len(ground_points) < 3:
        return None
    radius = options.get("footprint_radius", _wf._FOOTPRINT_RADIUS)
    sigma = options.get("sigma_beam") or _wf._SIGMA_BEAM_FRACTION * radius
    weighted = options.get("gaussian_beam_weighting", True)
    # the waveform's points: those within its query radius (ground beyond it
    # still shapes the terrain model)
    reach = _wf._effective_query_radius(
        radius, weighted, options.get("sigma_beam"),
        options.get("beam_extent_sigma", _wf._BEAM_EXTENT_SIGMA),
    )  # fmt: skip
    inside = (points["X"] - center_x) ** 2 + (points["Y"] - center_y) ** 2 <= reach**2
    gap_first, gap_all = matched_als_gaps(
        points[inside], center_x, center_y, edges,
        beam_sigma=sigma if weighted else None,
        density_cell=options.get("density_cell", _wf._DENSITY_CELL),
        terrain=TerrainModel(ground_points) if hag is None else None,
        hag=None if hag is None else np.asarray(hag)[inside],
    )  # fmt: skip
    totals = (result.canopy_waveform.sum(), result.ground_waveform.sum())
    arrays = {
        "canopy": canopy,
        "ground": ground,
        "pgap_waveform": gap_from_waveform(canopy, ground, rv_rg, totals=totals),
        "pgap_als_first": gap_first,
        "pgap_als_all": gap_all,
    }
    # the signal top: highest bin with more than top_threshold of the energy above it
    above = np.cumsum(canopy[::-1])[::-1]
    reaching = np.flatnonzero(above > top_threshold * (canopy.sum() + ground.sum()))
    top = float(edges[reaching[-1] + 1]) if len(reaching) else float(edges[0])
    beyond = edges[:-1] >= top
    for name in ("pgap_waveform", "pgap_als_first", "pgap_als_all"):
        arrays[name][beyond] = np.nan
    return result, arrays, top


# ---------------------------------------------------------------------------
# Batches: chunks of nearby shots, one query and one TIN each, worker processes
# ---------------------------------------------------------------------------


def _read_chunk(xy, reach, attributes):
    """Return the cleaned points around a chunk of shots, their ground and the chunk origin."""
    from alsdb.methods._surface import clean_points
    from alsdb.products._footprints import point_array
    from alsdb.products._tiling import query_to_array

    w = _wf._WORKER
    low, high = xy.min(axis=0) - reach, xy.max(axis=0) + reach
    points = query_to_array(w["provider"], (low[0], low[1], high[0], high[1]), year=w["year"],
                            attributes=attributes)  # fmt: skip
    points = clean_points(point_array(points)) if len(points) else points
    ground = points[points["Classification"] == 2] if len(points) else points
    return points, ground, low


def _pavd_chunk(task):
    """Return the counted PAVD profiles of one chunk of nearby shots."""
    from alsdb.methods._terrain import TerrainModel, normalize_points

    indices, xy = task
    o = dict(_wf._WORKER["options"])
    radius = o.pop("footprint_radius")
    points, ground, low = _read_chunk(
        xy, _wf._WORKER["radius"], ("Z", "ReturnNumber", "Classification", "Withheld")
    )
    if len(ground) < 3:
        return [(int(i), None) for i in indices]
    # Heights of every first return once per chunk, from one TIN of all its ground.
    first = points["ReturnNumber"] == 1
    normalized, _ = normalize_points(
        points[first], model=TerrainModel(ground), max_distance=o["max_ground_distance"],
        extrapolate=o["ground_extrapolation"],
    )  # fmt: skip
    hag = np.full(len(points), np.nan)
    hag[first] = normalized["HeightAboveGround"]
    extrapolated = np.zeros(len(points), bool)
    extrapolated[first] = normalized["GroundExtrapolated"].astype(bool)
    grid = PointGrid(points["X"], points["Y"], low, radius)
    cls = points["Classification"]
    output = []
    for k in range(len(indices)):
        inside = grid.disc(xy[k, 0], xy[k, 1], radius)
        if len(inside) < o["min_points"]:
            output.append((int(indices[k]), None))
            continue
        firsts = inside[first[inside]]
        heights, classes = hag[firsts], cls[firsts]
        result = _pavd_profile_from_hag(heights, classes, o["z_step"], o["k"],
                                        o["min_ground_points"], o["veg_classes"])  # fmt: skip
        if result is not None:
            canopy = classes != 2
            result.n_unsupported = int((~np.isfinite(heights) & canopy).sum())
            result.n_extrapolated = int(
                (extrapolated[firsts] & np.isfinite(heights) & canopy).sum()
            )
        output.append((int(indices[k]), result))
    return output


def compute_als_pavd_batch(
    provider,
    shots,
    x_col="center_x",
    y_col="center_y",
    *,
    footprint_radius=12.5,
    terrain_buffer=30.0,
    year=None,
    n_workers=1,
    batch_tile_size=100.0,
    max_shots_per_batch=256,
    z_step=_Z_STEP_DEFAULT,
    k=_LAI_K_DEFAULT,
    min_points=_MIN_POINTS_DEFAULT,
    min_ground_points=_MIN_GROUND_POINTS_DEFAULT,
    veg_classes=_VEG_CLASSES,
    max_ground_distance=None,
    ground_extrapolation=True,
):
    """Return the counted PAVD profile of every shot, in input row order.

    Nearby shots are read together: chunks of up to ``max_shots_per_batch``
    within ``batch_tile_size`` blocks, one query, one ground TIN and one
    normalisation per chunk (the TIN of a chunk has at least the ground a
    single :func:`compute_als_pavd_profile` uses, so on rough terrain
    heights near a footprint's edge can differ slightly). With
    ``n_workers > 1`` chunks run in worker processes, as
    :func:`alsdb.products.waveform.simulate_batch` (scripts need an
    ``if __name__ == "__main__":`` guard).

    Parameters
    ----------
    provider : TileDBProvider
    shots : pd.DataFrame
        Shot table with ``x_col`` and ``y_col``.
    x_col, y_col : str
        Easting and northing columns.
    footprint_radius, terrain_buffer, z_step, k, min_points, min_ground_points,
    veg_classes, max_ground_distance, ground_extrapolation
        As :func:`compute_als_pavd_profile`.
    year : int, optional
        Survey year.
    n_workers : int
        Worker processes (1: run in this process).
    batch_tile_size, max_shots_per_batch
        Spatial chunking, as :func:`alsdb.products.waveform.simulate_batch`.

    Returns
    -------
    list of PAVDProfile or None
        One entry per shot (None without enough points or ground returns).
    """
    if (
        not np.isfinite([footprint_radius, terrain_buffer, z_step, k]).all()
        or footprint_radius <= 0
        or terrain_buffer < 0
        or z_step <= 0
        or k <= 0
    ):
        raise ValueError("Invalid footprint, terrain buffer or profile parameters")
    centers = shots[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(centers).all():
        raise ValueError("Shot coordinates must be finite")
    options = {
        "footprint_radius": footprint_radius, "z_step": z_step, "k": k, "min_points": min_points,
        "min_ground_points": min_ground_points, "veg_classes": tuple(veg_classes),
        "max_ground_distance": max_ground_distance, "ground_extrapolation": ground_extrapolation,
    }  # fmt: skip
    tasks = (
        (idx, centers[idx])
        for idx in _wf._shot_chunks(centers, batch_tile_size, max_shots_per_batch)
    )
    results = [None] * len(shots)
    reach = footprint_radius + terrain_buffer
    for output in _wf._run_chunks(_pavd_chunk, tasks, n_workers, provider, options, reach, year):
        for index, result in output:
            results[index] = result
    return results


def _gedi_chunk(task):
    """Return the GEDI-matched profiles of one chunk of nearby shots."""
    from alsdb.methods._terrain import TerrainModel

    indices, xy, seeds, rv_rg = task
    w = _wf._WORKER
    options = dict(w["options"])
    edges = options.pop("edges")
    radius = w["radius"]
    points, ground, low = _read_chunk(xy, radius, _wf._ATTRIBUTES)
    # Normalise once per chunk: footprints share most of their points, and one
    # TIN of all the chunk's ground serves every footprint.
    hag = heights_above_ground(points, TerrainModel(ground)) if len(ground) >= 3 else None
    grid = PointGrid(points["X"], points["Y"], low, radius) if hag is not None else None
    output = []
    for k in range(len(indices)):
        neighborhood = grid.disc(xy[k, 0], xy[k, 1], radius) if grid is not None else ()
        found = None
        if len(neighborhood):
            found = gedi_footprint_profiles(
                points[neighborhood], *xy[k], edges, rv_rg=float(rv_rg[k]), hag=hag[neighborhood],
                year=w["year"], rng=np.random.default_rng(seeds[k]), **options,
            )  # fmt: skip
        if found is None:
            output.append((int(indices[k]), None, None))
            continue
        result, arrays, top = found
        metrics = result.to_dict()
        metrics["signal_top"] = top
        metrics["z_ground_true"] = result.z_ground_true
        metrics["canopy_energy"] = float(result.canopy_waveform.sum())
        metrics["ground_energy"] = float(result.ground_waveform.sum())
        output.append(
            (int(indices[k]), metrics, {k_: v.astype(np.float32) for k_, v in arrays.items()})
        )
    return output


def simulate_gedi_profiles(
    provider,
    shots: pd.DataFrame,
    out_dir,
    *,
    x_col: str = "center_x",
    y_col: str = "center_y",
    rv_rg: float = 1.0,
    rv_rg_col: str | None = None,
    year: int | None = None,
    height_range: tuple[float, float] = HEIGHT_RANGE,
    n_workers: int = 1,
    rng: np.random.Generator | None = None,
    batch_tile_size: float = 100.0,
    max_shots_per_batch: int = 256,
    **kwargs,
) -> Path:
    """Simulate GEDI canopy profiles and their ALS reference for many shots.

    Writes ``<out_dir>/profiles.zarr`` with arrays of shape (shots, heights):
    ``canopy``, ``ground``, ``pgap_waveform``, ``pgap_als_first``,
    ``pgap_als_all`` (float32, NaN for shots without enough points or ground)
    and ``height`` (bin centres, m above ground); and
    ``<out_dir>/footprints.parquet`` with one row per array row: the shot
    columns, ``shot_index`` (its position in *shots*), the waveform metrics
    (:func:`~alsdb.products.waveform.simulate_batch`), ``signal_top`` (m
    above ground; the gap profiles are NaN above it, as GEDI's
    ``pgap_theta_z`` spans only the signal), ``z_ground_true``,
    ``canopy_energy`` and ``ground_energy`` (shares of the whole waveform) and
    ``rv_rg``. Rows follow the processing order of spatial chunks.

    Pgap for other reflectance ratios follows from the stored canopy and
    ground profiles with :func:`gap_from_waveform` (pass ``totals`` from the
    parquet), without simulating again.

    Parameters
    ----------
    provider : TileDBProvider
    shots : pd.DataFrame
        Shot table with ``x_col`` and ``y_col``; other columns are kept.
    out_dir : str or Path
        Output folder (created).
    x_col, y_col : str
        Easting and northing columns.
    rv_rg : float
        Canopy-to-ground reflectance ratio, when ``rv_rg_col`` is None.
    rv_rg_col : str, optional
        Column with each shot's ratio, e.g. ``rhov / rhog`` of matched GEDI L2B shots.
    year : int, optional
        Survey year.
    height_range : tuple of float
        Heights (m above ground) of the profile grid.
    n_workers : int
        Worker processes (see :func:`~alsdb.products.waveform.simulate_batch`;
        scripts need an ``if __name__ == "__main__":`` guard).
    rng : np.random.Generator, optional
        Generator for reproducible noise.
    batch_tile_size, max_shots_per_batch
        Spatial chunking, as :func:`~alsdb.products.waveform.simulate_batch`.
    **kwargs
        Passed to :func:`~alsdb.products.waveform.waveform_from_points`.

    Returns
    -------
    Path
        ``out_dir``.
    """
    import zarr

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers = shots[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(centers).all():
        raise ValueError("Shot coordinates must be finite")
    ratios = (shots[rv_rg_col].to_numpy(dtype=float) if rv_rg_col is not None
              else np.full(len(shots), float(rv_rg)))  # fmt: skip
    if not (np.isfinite(ratios) & (ratios > 0)).all():
        raise ValueError("Reflectance ratios must be finite and positive")
    edges = height_edges(height_range, kwargs.get("z_step", _wf._Z_STEP))
    radius = _wf._effective_query_radius(
        kwargs.get("footprint_radius", _wf._FOOTPRINT_RADIUS),
        kwargs.get("gaussian_beam_weighting", True),
        kwargs.get("sigma_beam"),
        kwargs.get("beam_extent_sigma", _wf._BEAM_EXTENT_SIGMA),
    )
    random = rng if rng is not None else np.random.default_rng()
    seeds = random.integers(0, np.iinfo(np.uint64).max, size=len(shots), dtype=np.uint64)
    chunks = list(_wf._shot_chunks(centers, batch_tile_size, max_shots_per_batch))
    order = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
    row_of = np.empty(len(shots), np.int64)
    row_of[order] = np.arange(len(order))

    n, nz = len(shots), len(edges) - 1
    root = zarr.open_group(str(out / "profiles.zarr"), mode="w")
    root.create_array("height", data=((edges[:-1] + edges[1:]) / 2).astype(np.float32))
    stores = {
        name: root.create_array(name, shape=(n, nz), chunks=(min(max(n, 1), 512), nz),
                                dtype="float32", fill_value=np.nan)
        for name in PROFILE_ARRAYS
    }  # fmt: skip
    root.attrs.update({"rv_rg_note": "pgap_waveform uses footprints.parquet rv_rg per row",
                       "z_step": float(edges[1] - edges[0])})  # fmt: skip
    metrics = [None] * n
    tasks = ((idx, centers[idx], seeds[idx], ratios[idx]) for idx in chunks)
    options = {**kwargs, "edges": edges}
    for output in _wf._run_chunks(_gedi_chunk, tasks, n_workers, provider, options, radius, year):
        rows = np.array([row_of[i] for i, _, _ in output])
        for name in PROFILE_ARRAYS:
            block = np.full((len(output), nz), np.nan, np.float32)
            for k, (_, _, arrays) in enumerate(output):
                if arrays is not None:
                    block[k] = arrays[name]
            start, stop = rows.min(), rows.max() + 1
            if stop - start == len(rows):  # chunks are contiguous rows by construction
                stores[name][start:stop] = block[np.argsort(rows)]
            else:
                for k, row in enumerate(rows):
                    stores[name][row] = block[k]
        for index, m, _ in output:
            metrics[index] = m
    records = []
    for index in order:
        record = shots.iloc[index].to_dict()
        record["shot_index"] = int(index)
        record["rv_rg"] = float(ratios[index])
        record.update(metrics[index] or {"n_points": 0})
        records.append(record)
    pd.DataFrame(records).to_parquet(out / "footprints.parquet", engine="pyarrow", index=False)
    return out


# ---------------------------------------------------------------------------
# Attenuation
# ---------------------------------------------------------------------------


def fit_transmittance_model(
    cumulative_pai: np.ndarray,
    als_pavd: np.ndarray,
    observed_pavd: np.ndarray,
    free_amplitude: bool = False,
) -> dict:
    """Return the transmittance of an observed PAVD profile relative to the ALS profile.

    ``T = observed_pavd / als_pavd`` per bin, fitted as
    ``T(C) = a * exp(-beta * C)`` with C the cumulative PAI from the canopy
    top (a first-order model). ``a`` is 1 unless ``free_amplitude``: a fixed
    amplitude assumes the two profiles agree at the top, which a waveform
    spread by its pulse may not (the ratio there can be well above 1).

    Parameters
    ----------
    cumulative_pai : np.ndarray
        Cumulative ALS PAI from the canopy top, per bin, or at the bin edges
        (one value more than the profiles, as ``PAVDProfile.pai_cumulative``;
        the midpoints are then used).
    als_pavd, observed_pavd : np.ndarray
        PAVD profiles on the same grid.
    free_amplitude : bool
        Fit ``a`` as well as ``beta``.

    Returns
    -------
    dict
        ``transmittance`` (per bin, NaN where als_pavd <= 0), ``beta`` and
        ``amplitude`` (NaN if the fit fails) and ``fitted``.
    """
    from scipy.optimize import curve_fit

    cumulative_pai = np.asarray(cumulative_pai, dtype=np.float64)
    als_pavd = np.asarray(als_pavd, dtype=np.float64)
    observed_pavd = np.asarray(observed_pavd, dtype=np.float64)
    if als_pavd.shape != observed_pavd.shape:
        raise ValueError("als_pavd and observed_pavd must have the same shape")
    if cumulative_pai.shape == (len(als_pavd) + 1,):  # bin edges -> bin midpoints
        cumulative_pai = (cumulative_pai[:-1] + cumulative_pai[1:]) / 2
    if cumulative_pai.shape != als_pavd.shape:
        raise ValueError("cumulative_pai must give one value per bin or per bin edge")

    with np.errstate(invalid="ignore", divide="ignore"):
        transmittance = np.where(als_pavd > 0, observed_pavd / als_pavd, np.nan)

    valid = np.isfinite(transmittance) & np.isfinite(cumulative_pai)
    beta = amplitude = np.nan
    needed = 4 if free_amplitude else 3
    if valid.sum() >= needed:
        if free_amplitude:

            def _model(c, a, beta):
                return a * np.exp(-beta * c)

            p0, bounds = [1.0, 0.1], ([0.0, 0.0], [np.inf, np.inf])
        else:

            def _model(c, beta):
                return np.exp(-beta * c)

            p0, bounds = [0.1], (0.0, np.inf)
        try:
            params, _ = curve_fit(
                _model, cumulative_pai[valid], transmittance[valid], p0=p0, bounds=bounds
            )
            amplitude, beta = (
                (float(params[0]), float(params[1])) if free_amplitude else (1.0, float(params[0]))
            )
        except Exception as exc:  # noqa: BLE001 - curve_fit's failure modes vary, fall back to NaN
            logger.warning("fit_transmittance_model: curve_fit failed (%s) -- beta=NaN", exc)
            beta = amplitude = np.nan
    else:
        logger.warning(
            "fit_transmittance_model: only %d valid (finite, als_pavd>0) points -- "
            "beta=NaN, need >= %d",
            int(valid.sum()),
            needed,
        )

    fitted = (
        amplitude * np.exp(-beta * cumulative_pai)
        if np.isfinite(beta)
        else np.full_like(cumulative_pai, np.nan)
    )

    return {"transmittance": transmittance, "beta": beta, "amplitude": amplitude, "fitted": fitted}
