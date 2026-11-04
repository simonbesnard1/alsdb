# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Vertical plant area volume density (PAVD) profiles from ALS

Extends the MacArthur-Horn gap-fraction estimator of :mod:`gap` to height:
the gap probability at height z is the share of first returns that reach z,

    G(z) = 1 - N_above(z) / N_classified

Beer-Lambert gives the plant area traversed down to z,
``PAI(z) = -ln(G(z)) / k``, and PAVD is its vertical derivative. The profile
depends on sampling, occlusion, classification, terrain and extinction
assumptions: it is a reference for waveform retrievals, not an independent
measurement. Heights are normalised against a buffered ground TIN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from alsdb.methods._terrain import VEG_CLASSES as _VEG_CLASSES
from alsdb.products.gap import _GROUND_CLASS, _LAI_K_DEFAULT, _LAI_MAX
from alsdb.products.waveform import _query_footprint

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
    """Compute the PAVD profile of one footprint from heights above ground

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
    classified = gnd_mask | veg_mask  # matches gap.py's denominator convention:
    # unclassified/noise/building returns excluded, not just left in as dead weight
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
    veg_classes: tuple[int, ...] = _VEG_CLASSES,
    terrain_buffer: float = 30.0,
    max_ground_distance: float | None = None,
    ground_extrapolation: bool = False,
) -> PAVDProfile | None:
    """Return-count PAVD profile in a circular footprint

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

    Returns
    -------
    PAVDProfile or None
    """
    options = dict(locals())
    options.pop("provider")
    if not np.isfinite(terrain_buffer) or terrain_buffer < 0:
        raise ValueError("terrain_buffer must be finite and nonnegative")
    data = _query_footprint(provider, center_x, center_y, footprint_radius + terrain_buffer, year)
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
    ground_extrapolation=False,
):
    """Compute a footprint profile from points covering the footprint and terrain buffer

    First returns without terrain support are counted in ``n_unsupported``.
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
    unsupported = int((~np.isfinite(heights) & (first["Classification"] != 2)).sum())
    result = _pavd_profile_from_hag(
        heights, first["Classification"], z_step, k, min_ground_points, veg_classes
    )
    if result is not None:
        result.n_unsupported = unsupported
        result.n_extrapolated = int(
            (
                normalized["GroundExtrapolated"].astype(bool)
                & np.isfinite(heights)
                & (first["Classification"] != 2)
            ).sum()
        )
    return result


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
    **kwargs,
):
    """Return profiles in input row order, sharing queries for nearby footprints."""
    from alsdb.products._footprints import footprint_batches

    centers = shots[[x_col, y_col]].to_numpy(dtype=float)
    results = [None] * len(shots)

    def process(indices, points, neighborhoods):
        output = []
        for index, neighborhood in zip(indices, neighborhoods):
            result = pavd_from_points(
                points[neighborhood],
                *centers[index],
                footprint_radius,
                terrain_buffer=terrain_buffer,
                year=year,
                **kwargs,
            )
            output.append((index, result))
        return output

    for batch in footprint_batches(
        provider,
        centers,
        footprint_radius + terrain_buffer,
        year=year,
        n_workers=n_workers,
        batch_tile_size=batch_tile_size,
        attributes=("Z", "ReturnNumber", "Classification", "Withheld"),
        process=process,
    ):
        for index, result in batch:
            results[index] = result
    return results


def fit_transmittance_model(
    cumulative_pai: np.ndarray,
    als_pavd: np.ndarray,
    observed_pavd: np.ndarray,
) -> dict:
    """Transmittance of an observed PAVD profile relative to the ALS profile

    ``T = observed_pavd / als_pavd`` per bin, fitted as
    ``T(C) = exp(-beta * C)`` with C the cumulative PAI from the canopy top (a
    first-order model).

    Parameters
    ----------
    cumulative_pai : np.ndarray
        Cumulative ALS PAI from the canopy top at each bin.
    als_pavd, observed_pavd : np.ndarray
        PAVD profiles on the same grid.

    Returns
    -------
    dict
        ``transmittance`` (per bin, NaN where als_pavd <= 0), ``beta`` (NaN if
        the fit fails) and ``fitted``.
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
        except Exception as exc:  # noqa: BLE001 - curve_fit's failure modes vary, fall back to NaN
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

    return {"transmittance": transmittance, "beta": beta, "fitted": fitted}
