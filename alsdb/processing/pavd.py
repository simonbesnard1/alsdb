# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Vertical PAVD (Plant Area Volume Density) profiles from ALS point clouds, and
calibration of a spaceborne full-waveform lidar's PAVD retrieval against them.

Theory
------
Extends gap.py's horizontal MacArthur-Wilson gap-fraction estimator
(``P_gap = N_gnd / (N_gnd + N_veg)``) to the vertical dimension. For a
circular footprint, define the gap probability at height z above local
ground as the fraction of first-return pulses that reach z without having
been intercepted by canopy above it:

    G(z) = 1 - N_above(z) / N_classified

where ``N_above(z)`` is the count of classified (ground + vegetation)
first returns with height-above-ground > z, and ``N_classified`` is the
total classified first-return count in the footprint. At z=0 (ground),
this reduces exactly to gap.py's own ``P_gap``.

Beer-Lambert then gives the cumulative plant area index a downward pulse
has already traversed by the time it reaches height z:

    PAI_traversed(z) = -ln(G(z)) / k

(the same conversion as ``gap.py``'s ``_gap_to_lai``, evaluated at every
height level instead of once at the ground.) This is maximal at the ground
(= total canopy PAI) and ~0 at the canopy top.

Plant area volume density is its vertical derivative:

    PAVD(z) = -d(PAI_traversed)/dz

This is the classic MacArthur & Horn (1969) canopy profile. Because it is
built from the ALS point cloud's own many, largely-independent pulses
(rather than one energy-limited pulse), it does not suffer the depth-
dependent energy attenuation a real single-pulse full-waveform sensor
does, so it serves as the physical "ground truth" PAVD profile a real GEDI
L2A/L2B ``cp_pavd`` retrieval at the same footprint can be validated
against -- see :func:`fit_transmittance_model`.

Ground is the median height of ``Classification == 2`` (ground) first
returns in the footprint -- matching ``gap.py``'s own ground convention --
rather than the peak-detected ground ``waveform.py`` uses (there is no
waveform here to peak-detect on).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from alsdb.processing._tiling import VEG_CLASSES as _VEG_CLASSES
from alsdb.processing.gap import _GROUND_CLASS, _LAI_K_DEFAULT, _LAI_MAX
from alsdb.processing.waveform import _query_footprint

logger = logging.getLogger(__name__)

_Z_STEP_DEFAULT = 0.15  # m, GEDI native vertical resolution -- match waveform.py
_MIN_POINTS_DEFAULT = 50
_MIN_GROUND_POINTS_DEFAULT = 5  # guard against the ground-return blow-up noted in
# figure5_sitestack.py's load_als() docstring: too
# few ground returns -> G(0) near 0 -> ln blows up


@dataclass
class PAVDProfile:
    """Vertical PAVD profile for one footprint. See module docstring."""

    z: np.ndarray  # bin edges, height above ground, ascending (m), ground=0
    z_rel: np.ndarray  # z / canopy_height, clipped to [0, 1]
    pavd: np.ndarray  # m^2/m^3, one value per LAYER (len = len(z) - 1)
    pavd_z: np.ndarray  # layer-centre heights matching pavd, ascending (m)
    pavd_z_rel: np.ndarray  # pavd_z / canopy_height
    pai_cumulative: np.ndarray  # cumulative PAI traversed from the canopy top
    # down to each edge in z (same length/order as z) -- maximal at the
    # ground edge (= total_pai), ~0 at the canopy-top edge. This is the
    # "cumulative canopy traversed" quantity fit_transmittance_model() fits
    # transmittance against.
    total_pai: float
    canopy_height: float
    n_points: int
    n_ground_points: int


def _pavd_profile_from_hag(
    hag: np.ndarray,
    classification: np.ndarray,
    z_step: float = _Z_STEP_DEFAULT,
    k: float = _LAI_K_DEFAULT,
    min_ground_points: int = _MIN_GROUND_POINTS_DEFAULT,
) -> PAVDProfile | None:
    """
    Pure-math core: first-return height-above-ground values and LAS
    classification codes for ONE footprint (already ground-referenced,
    ``hag = Z - z_ground``) -> :class:`PAVDProfile`.

    Kept separate from the TileDB query (see :func:`compute_als_pavd_profile`)
    so it is unit-testable against synthetic point sets, mirroring
    ``gap.py``'s ``_compute_gap_grid`` / ``compute_gap_fraction`` split.

    Returns ``None`` if there are too few ground returns to estimate
    ``G(0)`` reliably, or no vegetation signal above ground at all.
    """
    gnd_mask = classification == _GROUND_CLASS
    n_gnd = int(gnd_mask.sum())
    if n_gnd < min_ground_points:
        return None

    veg_mask = np.isin(classification, _VEG_CLASSES)
    classified = gnd_mask | veg_mask  # matches gap.py's denominator convention:
    # unclassified/noise/building returns excluded, not just left in as dead weight
    hag_cls = hag[classified]
    n_tot = int(classified.sum())
    if n_tot == 0:
        return None

    above_ground = hag_cls[hag_cls > 0]
    canopy_height = float(np.percentile(above_ground, 98)) if len(above_ground) else 0.0
    if canopy_height <= 0:
        return None

    n_bins = max(1, int(np.ceil(canopy_height / z_step)))
    z_edges = np.linspace(0.0, n_bins * z_step, n_bins + 1)  # ascending: ground -> top

    # N_above(e) = count of classified points with hag > e, computed via a single
    # sort + vectorised searchsorted rather than one boolean sum per edge.
    hag_sorted = np.sort(hag_cls)
    n_le = np.searchsorted(hag_sorted, z_edges, side="right")
    n_above = (n_tot - n_le).astype(np.float64)

    gap = 1.0 - n_above / n_tot
    # Floor at "one unseen return" rather than 0, to avoid -inf when a footprint's
    # ground is barely sampled (dense overstory) -- same boundary-blow-up risk
    # figure5_sitestack.py's load_als() docstring already flagged for a from-scratch
    # MacArthur-Horn attempt on a single small footprint.
    gap_safe = np.clip(gap, 1.0 / n_tot, 1.0)
    pai_traversed = np.clip(-np.log(gap_safe) / k, 0.0, _LAI_MAX)

    # pai_traversed decreases as z increases (less canopy still to traverse near the
    # top), so negate the derivative to get a positive density.
    pavd = -np.diff(pai_traversed) / np.diff(z_edges)
    pavd_z = (z_edges[:-1] + z_edges[1:]) / 2.0
    z_rel = np.clip(z_edges / canopy_height, 0.0, 1.0)
    pavd_z_rel = np.clip(pavd_z / canopy_height, 0.0, 1.0)

    return PAVDProfile(
        z=z_edges,
        z_rel=z_rel,
        pavd=pavd,
        pavd_z=pavd_z,
        pavd_z_rel=pavd_z_rel,
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
) -> PAVDProfile | None:
    """
    Vertical PAVD profile from real ALS returns in a circular footprint --
    the "ALS truth" reference a real GEDI L2A/L2B PAVD retrieval at the same
    footprint should be compared against (see :func:`fit_transmittance_model`).

    Parameters
    ----------
    provider : TileDBProvider
        Local or S3 ALS array.
    center_x, center_y : float
        Footprint centre in UTM metres (same CRS as the array).
    footprint_radius : float
        Footprint radius in metres. Use the same radius as the sensor being
        validated against (e.g. 12.5 m for GEDI's 25 m-diameter footprint).
    z_step : float
        Vertical bin size in metres (default 0.15 m, GEDI native).
    k : float
        Beer-Lambert extinction coefficient (default 0.5, spherical leaf
        angle distribution -- see ``gap.LAI_K_PRESETS`` for biome presets).
    year : int, optional
        Survey year filter. ``None`` merges all years.
    min_points : int
        Minimum total ALS points in the footprint; returns ``None`` below this.
    min_ground_points : int
        Minimum classified-ground first returns; returns ``None`` below this
        (see module docstring on the ground-return floor).

    Returns
    -------
    PAVDProfile or None
    """
    data = _query_footprint(provider, center_x, center_y, footprint_radius, year)
    n_pts = 0 if data is None else len(data["Z"])
    if data is None or n_pts < min_points:
        logger.debug(
            "compute_als_pavd_profile: only %d points at (%.0f, %.0f) year=%s "
            "(min_points=%d) -- skipped",
            n_pts,
            center_x,
            center_y,
            year,
            min_points,
        )
        return None

    fr = data["ReturnNumber"] == 1
    z_fr = data["Z"][fr].astype(np.float64)
    cls_fr = data["Classification"][fr]

    gnd_mask = cls_fr == _GROUND_CLASS
    if int(gnd_mask.sum()) < min_ground_points:
        logger.debug(
            "compute_als_pavd_profile: only %d ground returns at (%.0f, %.0f) "
            "(need >= %d) -- skipped",
            int(gnd_mask.sum()),
            center_x,
            center_y,
            min_ground_points,
        )
        return None

    z_ground = float(np.median(z_fr[gnd_mask]))
    hag = z_fr - z_ground

    return _pavd_profile_from_hag(
        hag, cls_fr, z_step=z_step, k=k, min_ground_points=min_ground_points
    )


def fit_transmittance_model(
    cumulative_pai: np.ndarray,
    als_pavd: np.ndarray,
    observed_pavd: np.ndarray,
) -> dict:
    """
    Empirical transmittance of an observed (e.g. real GEDI) PAVD profile
    relative to the ALS-true PAVD profile, plus a first-order parametric fit
    against cumulative canopy traversed from the top.

    Empirical transmittance is simply ``T = observed_pavd / als_pavd`` per
    bin. The fitted model is a single-parameter exponential decay in
    cumulative PAI traversed:

        T(C) = exp(-beta * C)

    the simplest form consistent with a signal deficit that grows
    monotonically with how much canopy the pulse has already passed through
    -- an illustrative first-order model, not a claimed final calibration;
    refine once validated against more sites/instruments.

    Parameters
    ----------
    cumulative_pai : np.ndarray
        Cumulative ALS-true PAI traversed from the canopy top down to each
        profile point (:attr:`PAVDProfile.pai_cumulative`, or the equivalent
        for a stratum-mean profile).
    als_pavd, observed_pavd : np.ndarray
        ALS-true and observed PAVD on the same grid as ``cumulative_pai``.

    Returns
    -------
    dict with keys:
        ``transmittance`` -- raw per-bin ratio (NaN where als_pavd <= 0)
        ``beta`` -- fitted decay rate (NaN if the fit failed or too few
            valid points)
        ``fitted`` -- ``T(C)`` evaluated at each input ``cumulative_pai``
    """
    from scipy.optimize import curve_fit

    cumulative_pai = np.asarray(cumulative_pai, dtype=np.float64)
    als_pavd = np.asarray(als_pavd, dtype=np.float64)
    observed_pavd = np.asarray(observed_pavd, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        transmittance = np.where(als_pavd > 0, observed_pavd / als_pavd, np.nan)

    valid = np.isfinite(transmittance) & np.isfinite(cumulative_pai)
    beta = np.nan
    if valid.sum() >= 3:

        def _model(c, beta):
            return np.exp(-beta * c)

        try:
            (beta,), _ = curve_fit(
                _model, cumulative_pai[valid], transmittance[valid], p0=[0.1], bounds=(0.0, np.inf)
            )
            beta = float(beta)
        except Exception as exc:
            logger.warning("fit_transmittance_model: curve_fit failed (%s) -- beta=NaN", exc)
            beta = np.nan
    else:
        logger.warning(
            "fit_transmittance_model: only %d valid (finite, als_pavd>0) points -- "
            "beta=NaN, need >= 3",
            int(valid.sum()),
        )

    fitted = (
        np.exp(-beta * cumulative_pai)
        if np.isfinite(beta)
        else np.full_like(cumulative_pai, np.nan)
    )

    return dict(transmittance=transmittance, beta=beta, fitted=fitted)
