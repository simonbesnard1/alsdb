# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Large-footprint waveform simulation from ALS point clouds.

Defaults follow the GEDI simulator (Hancock et al. 2019; ``gediRat``):
Gaussian footprint of sigma 5.5 m (0.44 x the 12.5 m radius), Gaussian pulse
of 15 ns FWHM (sigma 0.955 m) or a measured GEDI pulse (``beam_id``), 0.15 m
bins, and each point weighted by the inverse number of pulses in its 1.5 m
cell (``density_cell``), which keeps flight-line overlaps from counting
twice. Other instruments are simulated by changing ``footprint_radius``,
``sigma`` and ``beam_id`` (e.g. LVIS: radius 10 m, sigma 0.7 m; ICESat/GLAS:
radius 32.5 m, sigma 2 m).

Steps: query the points of a circular footprint (beyond its radius when the
Gaussian beam weighting is on), build a vertical histogram, convolve it with
the transmitted pulse (later parts of the pulse land lower, as in
``gediRat``), detect the ground peak (the lowest maximum, as gediMetric's
``maxGround``), and derive relative heights (RH, as libclidar's ``findRH``)
and canopy cover.

References
----------
Hancock, S., et al. (2019). The GEDI simulator: A large-footprint waveform
    lidar simulator for calibration and validation of spaceborne missions.
    Remote Sensing of Environment, 220, 309-323.
GEDI Algorithm Theoretical Basis Document (ATBD), NASA (2019).
Blair, J. B., et al. (1999). The Laser Vegetation Imaging Sensor (LVIS).
    IGARSS, Hamburg.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

# Default constants — tuned for GEDI; override via simulate_waveform() parameters
_FOOTPRINT_RADIUS: float = 12.5  # m  (25 m diameter)
_Z_STEP: float = 0.15  # m  native vertical resolution
# gediRat's default pulse: 15 ns FWHM outgoing, two-way range 15 * 0.2998 / 2 m,
# sigma = FWHM / 2.35482 (gediIO.c, setGediPulse).
_SIGMA_PULSE: float = 15.0 * 0.2998 / 2.0 / 2.35482  # m, 0.955
# gediRat's footprint sigma, 5.5 m for GEDI's 12.5 m radius ("86% of energy
# within a diameter of 20-25 m"), as a fraction of the radius.
_SIGMA_BEAM_FRACTION: float = 5.5 / 12.5
# gediRat's sampling-density grid (normCover): pulses per 1.5 m cell.
_DENSITY_CELL: float = 1.5  # m
_MIN_POINTS: int = 25
_RH_LEVELS: tuple[int, ...] = tuple(range(101))  # RH0–RH100, matches GEDI L2A
_COVER_THRESHOLD: float = 2.0  # m  above ground
_BEAM_EXTENT_SIGMA: float = 3.0  # query out to this many sigma_beam (see _effective_query_radius)

# Empirical GEDI L1B transmit kernels, from Hancock et al.'s (2019) gediSimulator;
# other sensors use a Gaussian kernel.
_BEAM_IDS: frozenset[str] = frozenset(
    {
        "BEAM0000",
        "BEAM0001",
        "BEAM0010",
        "BEAM0011",
        "BEAM0101",
        "BEAM0110",
        "BEAM1000",
        "BEAM1011",
    }
)


@functools.lru_cache(maxsize=8)
def _load_pulse(beam_id: str, z_step: float = _Z_STEP) -> np.ndarray:
    """Return the bundled TX pulse kernel of *beam_id*, centred and normalised to unit sum."""
    path = files("alsdb") / "data" / "pulse_shapes" / f"meanPulse.{beam_id}.txt"
    data = np.loadtxt(str(path))  # (N, 2): col0 = position (m), col1 = amplitude
    if not np.isfinite(z_step) or z_step <= 0:
        raise ValueError("z_step must be finite and positive")
    position = data[:, 0].astype(float)
    amp = np.maximum(data[:, 1].astype(float), 0)
    position -= position[np.argmax(amp)]
    spacing = np.median(np.diff(position))
    edges = np.r_[position - spacing / 2, position[-1] + spacing / 2]
    cumulative = np.r_[0.0, np.cumsum(amp)]
    radius = int(np.ceil(max(abs(edges[0]), abs(edges[-1])) / z_step))
    target_edges = (np.arange(-radius, radius + 2) - 0.5) * z_step
    kernel = np.diff(np.interp(target_edges, edges, cumulative, left=0, right=cumulative[-1]))
    return kernel / kernel.sum()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class WaveformResult:
    """Simulated waveform and metrics of one footprint.

    Attributes
    ----------
    z_bins : np.ndarray
        Bin centre elevations (m, ascending).
    waveform : np.ndarray
        Normalised energy per bin (sums to 1).
    z_ground : float
        Ground peak elevation (m).
    rh : dict[int, float]
        Relative heights (m above ground) keyed by percentile.
    home : float
        Height of median energy (m above ground).
    cover : float
        Canopy cover fraction.
    n_points : int
        Number of points used.
    center_x, center_y : float
        Footprint centre (m).
    """

    z_bins: np.ndarray
    waveform: np.ndarray
    z_ground: float
    rh: dict[int, float]
    home: float
    cover: float
    n_points: int
    center_x: float
    center_y: float
    ground_supported: bool = False
    ground_offset: float = np.nan
    slope_corrected: bool = False
    # Canopy (non-ground) and ground (class 2) parts of the noise-free waveform,
    # on z_bins, scaled so together they sum to 1 (gediRat's -ground option).
    canopy_waveform: np.ndarray | None = None
    ground_waveform: np.ndarray | None = None
    # Footprint-weighted mean elevation of the ground returns (gediRat's ground).
    z_ground_true: float = np.nan

    def rh_array(self, levels: tuple[int, ...] = _RH_LEVELS) -> np.ndarray:
        """Return RH values as a 1-D array for the given levels."""
        return np.array([self.rh.get(lv, np.nan) for lv in levels])

    def to_dict(self) -> dict:
        """Return the scalar metrics as a flat dict."""
        d = {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "z_ground": self.z_ground,
            "home": self.home,
            "cover": self.cover,
            "n_points": self.n_points,
            "ground_supported": self.ground_supported,
            "ground_offset": self.ground_offset,
            "slope_corrected": self.slope_corrected,
        }
        d.update({f"rh{lv}": v for lv, v in sorted(self.rh.items())})
        return d


# ---------------------------------------------------------------------------
# Step 1 — footprint query
# ---------------------------------------------------------------------------


def _effective_query_radius(
    footprint_radius: float,
    gaussian_beam_weighting: bool,
    sigma_beam: float | None,
    beam_extent_sigma: float = _BEAM_EXTENT_SIGMA,
) -> float:
    """Return the query radius, including the Gaussian beam tail."""
    if not np.isfinite(footprint_radius) or footprint_radius <= 0:
        raise ValueError("footprint_radius must be finite and positive")
    if sigma_beam is not None and (not np.isfinite(sigma_beam) or sigma_beam <= 0):
        raise ValueError("sigma_beam must be finite and positive")
    if not np.isfinite(beam_extent_sigma) or beam_extent_sigma <= 0:
        raise ValueError("beam_extent_sigma must be finite and positive")
    if not gaussian_beam_weighting:
        return footprint_radius
    sb = sigma_beam if sigma_beam is not None else _SIGMA_BEAM_FRACTION * footprint_radius
    return max(footprint_radius, beam_extent_sigma * sb)


def _query_footprint(
    provider: TileDBProvider,
    center_x: float,
    center_y: float,
    radius: float,
    year: int | None,
) -> dict[str, np.ndarray] | None:
    """Return the points inside a circular footprint as a dict of arrays, or None if empty."""
    yr_dim = provider.schema.domain.dim("Year")
    y0 = year if year is not None else int(yr_dim.domain[0])
    # +1: TileDB-Py int-dimension slices are exclusive-end (like Python slices)
    y1 = (year + 1) if year is not None else int(yr_dim.domain[1]) + 1

    attrs = ["Z", "Intensity", "ReturnNumber", "NumberOfReturns", "Classification", "Withheld"]
    with provider.open("r") as arr:
        data = arr.query(attrs=attrs)[
            center_x - radius : center_x + radius,
            center_y - radius : center_y + radius,
            y0:y1,
        ]

    if len(data["X"]) == 0:
        return None

    mask = (data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2 <= radius**2
    if not mask.any():
        return None

    return {k: v[mask] for k, v in data.items()}


# ---------------------------------------------------------------------------
# Step 2 — vertical histogram
# ---------------------------------------------------------------------------


def _build_histogram(
    z: np.ndarray,
    weights: np.ndarray,
    z_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin Z values into a vertical histogram.

    Returns
    -------
    z_bins : np.ndarray
        Bin centre elevations (m, ascending).
    hist : np.ndarray
        Weighted count per bin.
    """
    if not np.isfinite(z_step) or z_step <= 0:
        raise ValueError("z_step must be finite and positive")
    z_min = float(z.min())
    z_max = float(z.max())
    n_bins = max(1, int(np.ceil((z_max - z_min) / z_step)))
    edges = z_min + np.arange(n_bins + 1) * z_step
    # Equal-width bins: index each point directly (np.histogram sorts the points);
    # the top edge belongs to the last bin, as in np.histogram.
    index = np.clip(np.floor((z - z_min) / z_step).astype(np.int64), 0, n_bins - 1)
    hist = np.bincount(index, weights=weights, minlength=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, hist.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 4 — ground detection
# ---------------------------------------------------------------------------


def _detect_ground(
    waveform: np.ndarray,
    z_bins: np.ndarray,
    noise_threshold: float = 0.005,
) -> float:
    """Return the elevation of the lowest significant peak (the ground return)."""
    smoothed = gaussian_filter1d(waveform, sigma=2)
    threshold = noise_threshold * smoothed.max()

    peaks, _ = find_peaks(smoothed, height=threshold, prominence=threshold * 0.5)

    if len(peaks):
        return float(z_bins[peaks[0]])

    # Fallback: lowest bin with meaningful energy
    nonzero = np.where(waveform > threshold)[0]
    return float(z_bins[nonzero[0]]) if len(nonzero) else float(z_bins[0])


# ---------------------------------------------------------------------------
# Step 5 — metric extraction
# ---------------------------------------------------------------------------


def _rh_metrics(
    waveform: np.ndarray,
    z_bins: np.ndarray,
    z_ground: float,
    levels: tuple[int, ...] = _RH_LEVELS,
) -> dict[int, float]:
    """Relative heights above the ground peak.

    RH(x) is the height at which x% of the total waveform energy is reached,
    counted from the lowest bin, as libclidar's ``findRH`` (GEDI simulator).
    RH0 is the lowest bin with energy, usually below the ground peak (pulse
    and slope broadening), as GEDI's ``rh_000``.
    """
    ground_idx = int(np.searchsorted(z_bins, z_ground))
    total = float(waveform.sum())

    if total == 0.0 or ground_idx >= len(z_bins):
        return {lv: np.nan for lv in levels}

    cumulative = np.cumsum(waveform) / total
    z_above = z_bins - z_ground  # height relative to ground peak; can be negative

    first = int(np.flatnonzero(waveform > 0)[0])  # bottom of the signal
    rh = {}
    for level in levels:
        idx = int(np.searchsorted(cumulative, level / 100.0)) if level > 0 else first
        rh[level] = float(z_above[min(idx, len(z_above) - 1)])
    return rh


def _canopy_cover(
    waveform: np.ndarray,
    z_bins: np.ndarray,
    z_ground: float,
    threshold: float = _COVER_THRESHOLD,
    rv_rg: float = 1.0,
) -> float:
    """Canopy cover from the gap probability.

    ``cover = veg_energy / (veg_energy + rv_rg * ground_energy)``, with
    ``rv_rg`` the vegetation-to-ground reflectance ratio. ``rv_rg=1`` gives the
    plain energy ratio, not GEDI's calibrated L2B cover.
    """
    total = float(waveform.sum())
    if total == 0.0:
        return 0.0
    veg_energy = float(waveform[z_bins >= (z_ground + threshold)].sum())
    ground_energy = total - veg_energy
    denom = veg_energy + rv_rg * ground_energy
    if denom <= 0.0:
        return 0.0
    return float(veg_energy / denom)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_waveform(
    provider: TileDBProvider,
    center_x: float,
    center_y: float,
    year: int | None = None,
    footprint_radius: float = _FOOTPRINT_RADIUS,
    z_step: float = _Z_STEP,
    sigma: float = _SIGMA_PULSE,
    beam_id: str | None = None,
    noise_std: float = 0.0,
    intensity_weighted: bool = False,
    gaussian_beam_weighting: bool = True,
    sigma_beam: float | None = None,
    beam_extent_sigma: float = _BEAM_EXTENT_SIGMA,
    slope_correction: bool = False,
    min_points: int = _MIN_POINTS,
    cover_threshold: float = _COVER_THRESHOLD,
    rv_rg: float = 1.0,
    rh_levels: tuple[int, ...] = _RH_LEVELS,
    rng: np.random.Generator | None = None,
    return_weighting: str = "count",
    density_cell: float | None = _DENSITY_CELL,
) -> WaveformResult | None:
    """Query a footprint and simulate its waveform; see :func:`waveform_from_points`."""
    options = dict(locals())
    options.pop("provider")
    radius = _effective_query_radius(
        footprint_radius, gaussian_beam_weighting, sigma_beam, beam_extent_sigma
    )
    data = _query_footprint(provider, center_x, center_y, radius, year)
    return waveform_from_points(data, **options)


def waveform_from_points(
    data,
    center_x: float,
    center_y: float,
    year: int | None = None,
    footprint_radius: float = _FOOTPRINT_RADIUS,
    z_step: float = _Z_STEP,
    sigma: float = _SIGMA_PULSE,
    beam_id: str | None = None,
    noise_std: float = 0.0,
    intensity_weighted: bool = False,
    gaussian_beam_weighting: bool = True,
    sigma_beam: float | None = None,
    beam_extent_sigma: float = _BEAM_EXTENT_SIGMA,
    slope_correction: bool = False,
    min_points: int = _MIN_POINTS,
    cover_threshold: float = _COVER_THRESHOLD,
    rv_rg: float = 1.0,
    rh_levels: tuple[int, ...] = _RH_LEVELS,
    rng: np.random.Generator | None = None,
    return_weighting: str = "count",
    density_cell: float | None = _DENSITY_CELL,
) -> WaveformResult | None:
    """Simulate a large-footprint waveform from points.

    Parameters
    ----------
    data
        Point arrays, or the provider when called through :func:`simulate_waveform`.
    center_x, center_y : float
        Footprint centre, in the CRS of the points.
    year : int, optional
        Survey year; ``None`` uses all years.
    footprint_radius : float
        Footprint radius (m).
    z_step : float
        Vertical bin size (m).
    sigma : float
        Gaussian pulse sigma (m), used when ``beam_id`` is None; default 0.955 m,
        the GEDI simulator's 15 ns FWHM.
    beam_id : str, optional
        GEDI beam (e.g. ``"BEAM0000"``) whose measured pulse is used.
    noise_std : float
        Standard deviation of additive noise (0: none).
    intensity_weighted : bool
        Weight returns by intensity instead of counting them.
    gaussian_beam_weighting : bool
        Weight points by the Gaussian beam profile.
    sigma_beam : float, optional
        Beam sigma (m); default ``0.44 * footprint_radius`` (5.5 m for GEDI, the
        GEDI simulator's value).
    beam_extent_sigma : float
        Query out to this many ``sigma_beam`` when weighting.
    slope_correction : bool
        Remove a plane fitted to the ground points before binning.
    min_points : int
        Return None with fewer points.
    cover_threshold : float
        Height (m) separating canopy from ground for cover.
    rv_rg : float
        Vegetation-to-ground reflectance ratio; set from a calibration only.
    rh_levels : tuple of int
        RH percentiles to compute.
    rng : np.random.Generator, optional
        Generator for the noise.
    return_weighting : {"count", "fractional"}
        ``"fractional"`` weights each return by ``1 / NumberOfReturns``.
    density_cell : float, optional
        Divide each point's weight by the number of pulses (last returns) in its
        cell of this size (m), as the GEDI simulator's ``normCover``; None
        turns the correction off.

    Returns
    -------
    WaveformResult or None
    """
    from alsdb.methods._surface import clean_points
    from alsdb.products._footprints import point_array

    if (
        not np.isfinite([z_step, sigma, noise_std, rv_rg]).all()
        or z_step <= 0
        or sigma <= 0
        or noise_std < 0
        or rv_rg <= 0
    ):
        raise ValueError("Invalid waveform bin, pulse, noise or reflectance parameters")
    if return_weighting not in ("count", "fractional"):
        raise ValueError("return_weighting must be count or fractional")
    if density_cell is not None and (not np.isfinite(density_cell) or density_cell <= 0):
        raise ValueError("density_cell must be finite and positive")
    radius = _effective_query_radius(
        footprint_radius, gaussian_beam_weighting, sigma_beam, beam_extent_sigma
    )
    density_weight = None
    if data is not None:
        data = clean_points(point_array(data))
        inside = (data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2 <= radius**2
        data = data[inside]
        if density_cell is not None and len(data):
            # pulses per cell, counted once per pulse at its last return (gediRat)
            ix = np.floor(data["X"] / density_cell).astype(np.int64)
            iy = np.floor(data["Y"] / density_cell).astype(np.int64)
            ix -= ix.min()
            iy -= iy.min()
            index = ix * (int(iy.max()) + 1) + iy  # flat cell index; counting needs no sort
            last = data["ReturnNumber"] >= data["NumberOfReturns"]
            pulses = np.bincount(index, weights=last)[index]
            density_weight = np.divide(1.0, pulses, out=np.ones(len(pulses)), where=pulses > 0)
    n_pts = 0 if data is None else len(data["Z"])
    if data is None or n_pts < min_points:
        logger.debug(
            "simulate_waveform: footprint at (%.0f, %.0f) year=%s has only %d points "
            "(min_points=%d) — skipped.",
            center_x,
            center_y,
            year,
            n_pts,
            min_points,
        )
        return None

    z = data["Z"].astype(np.float64)

    slope_corrected = False
    # Slope correction: fit plane to ground points, subtract from all Z
    if slope_correction:
        gnd = data["Classification"] == 2
        if gnd.sum() >= 3:
            x_gnd = data["X"][gnd].astype(np.float64) - center_x
            y_gnd = data["Y"][gnd].astype(np.float64) - center_y
            z_gnd = z[gnd]
            A = np.column_stack([x_gnd, y_gnd, np.ones(gnd.sum())])
            coeffs, _, rank, _ = np.linalg.lstsq(A, z_gnd, rcond=None)
            a, b, c = coeffs
            slope_deg = float(np.degrees(np.arctan(np.sqrt(a**2 + b**2))))
            plane_z = (
                a * (data["X"].astype(float) - center_x)
                + b * (data["Y"].astype(float) - center_y)
                + c
            )
            if rank == 3:
                z = z - plane_z + float(z_gnd.mean())
                slope_corrected = True
            logger.debug("Slope correction applied: θ=%.1f°", slope_deg)
        else:
            logger.debug("Slope correction skipped: only %d ground points (need ≥ 3)", gnd.sum())

    # Build point weights: beam profile × optional intensity
    if gaussian_beam_weighting:
        sb = sigma_beam if sigma_beam is not None else _SIGMA_BEAM_FRACTION * footprint_radius
        r2 = (data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2
        beam_w = np.exp(-r2 / (2.0 * sb**2))
    else:
        beam_w = np.ones(len(z))

    if intensity_weighted:
        weights = data["Intensity"].astype(np.float64) * beam_w
    else:
        weights = beam_w

    if return_weighting == "fractional":
        returns = data["NumberOfReturns"]
        weights = np.divide(weights, returns, out=np.zeros_like(weights), where=returns > 0)
    if density_weight is not None:
        weights *= density_weight
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Waveform weights must be finite and nonnegative")
    if weights.sum() <= 0:
        return None

    # 1. Vertical histogram, and its ground part (class 2) on the same bins
    is_ground = data["Classification"] == 2
    z_bins, hist = _build_histogram(z, weights, z_step)
    _, hist_ground = _build_histogram(z, np.where(is_ground, weights, 0.0), z_step)

    # Retain convolution tails: trimming here biases ground and canopy energy.
    if beam_id is not None and beam_id in _BEAM_IDS:
        kernel = _load_pulse(beam_id, z_step)
    else:
        radius_bins = max(1, int(np.ceil(4 * sigma / z_step)))
        offsets = np.arange(-radius_bins, radius_bins + 1) * z_step
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
    pad = (len(kernel) - 1) // 2
    # Reversed: kernel positions run with time, so later (positive) offsets
    # land lower, as in gediRat; measured GEDI pulses have a long trailing tail.
    waveform = np.convolve(hist, kernel[::-1], mode="full")
    ground_waveform = np.convolve(hist_ground, kernel[::-1], mode="full")
    canopy_waveform = np.clip(waveform - ground_waveform, 0.0, None)
    parts = canopy_waveform.sum() + ground_waveform.sum()
    if parts > 0:
        canopy_waveform, ground_waveform = canopy_waveform / parts, ground_waveform / parts
    z_bins = z_bins[0] + (np.arange(len(waveform)) - pad) * z_step
    w_ground = weights[is_ground]
    z_ground_true = (
        float(np.sum(w_ground * z[is_ground]) / w_ground.sum()) if w_ground.sum() > 0 else np.nan
    )

    # 3. Add noise
    if noise_std > 0.0:
        _rng = rng if rng is not None else np.random.default_rng()
        waveform = np.maximum(0.0, waveform + _rng.normal(0.0, noise_std, len(waveform)))

    # 4. Normalise to unit energy
    total = waveform.sum()
    if total > 0.0:
        waveform /= total

    # 5. Ground detection + metrics
    z_ground = _detect_ground(waveform, z_bins)
    ground = z[data["Classification"] == 2]
    ground_reference = float(np.median(ground)) if len(ground) else np.nan
    ground_offset = z_ground - ground_reference
    ground_supported = bool(len(ground) >= 3 and abs(ground_offset) <= max(2 * sigma, z_step))
    rh = _rh_metrics(waveform, z_bins, z_ground, levels=rh_levels)
    cover = _canopy_cover(waveform, z_bins, z_ground, cover_threshold, rv_rg=rv_rg)

    return WaveformResult(
        z_bins=z_bins,
        waveform=waveform,
        z_ground=z_ground,
        rh=rh,
        home=rh.get(50, np.nan),
        cover=cover,
        n_points=len(z),
        center_x=center_x,
        center_y=center_y,
        ground_supported=ground_supported,
        ground_offset=ground_offset,
        slope_corrected=slope_corrected,
        canopy_waveform=canopy_waveform,
        ground_waveform=ground_waveform,
        z_ground_true=z_ground_true,
    )


_ATTRIBUTES = ("Z", "Intensity", "ReturnNumber", "NumberOfReturns", "Classification", "Withheld")
_WORKER: dict = {}  # per-process state of simulate_batch workers (set by _init_worker)


def _init_worker(provider, options, radius, year, threads):
    """Set up one simulate_batch worker: its own TileDB context and the options."""
    if threads is not None:
        # Several processes share the machine: fewer TileDB threads and a small
        # tile cache each, instead of every process claiming all cores and 1 GiB.
        n = str(threads)
        provider = provider.with_config(**{
            "sm.num_reader_threads": n, "sm.num_tiledb_threads": n,
            "sm.compute_concurrency_level": n, "sm.io_concurrency_level": n,
            "sm.tile_cache_size": str(128 * 1024**2), "py.init_buffer_bytes": str(64 * 1024**2),
        })  # fmt: skip
    _WORKER.update(provider=provider, options=options, radius=radius, year=year)


def _simulate_chunk(task):
    """Waveform metrics of one chunk of nearby shots, read with one query."""
    from alsdb.products._tiling import PointGrid, query_to_array

    indices, xy, seeds, beams = task
    w = _WORKER
    radius = w["radius"]
    low, high = xy.min(axis=0) - radius, xy.max(axis=0) + radius
    points = query_to_array(w["provider"], (low[0], low[1], high[0], high[1]), year=w["year"],
                            attributes=_ATTRIBUTES)  # fmt: skip
    grid = PointGrid(points["X"], points["Y"], low, radius) if len(points) else None
    output = []
    for k in range(len(indices)):
        neighborhood = grid.disc(xy[k, 0], xy[k, 1], radius) if grid is not None else ()
        options = dict(w["options"])
        if beams is not None:
            options["beam_id"] = beams[k]
        result = waveform_from_points(points[neighborhood] if len(neighborhood) else None, *xy[k],
                                      year=w["year"], rng=np.random.default_rng(seeds[k]),
                                      **options)  # fmt: skip
        output.append((int(indices[k]), result.to_dict() if result is not None else None))
    return output


def _shot_chunks(centers, block_size, max_shots):
    """Yield indices of nearby shots: spatial blocks, sorted inside, cut into chunks.

    Sorting by row then column keeps each chunk of a crowded block compact, so
    chunks of the same block read little of each other's area.
    """
    blocks = {}
    cells = np.floor(centers / block_size).astype(np.int64)
    for index in np.lexsort((centers[:, 0], centers[:, 1])):
        blocks.setdefault((cells[index, 0], cells[index, 1]), []).append(index)
    for indices in blocks.values():
        for start in range(0, len(indices), max_shots):
            yield np.asarray(indices[start : start + max_shots])


def _run_chunks(fn, tasks, n_workers, provider, options, radius, year):
    """Yield ``fn(task)`` for every task, in this process or in worker processes.

    Workers start with ``spawn`` (a forked child would inherit TileDB's threads
    in whatever state they were), each with its own TileDB context of
    cores / n_workers threads; at most two tasks per worker are in flight.
    """
    import multiprocessing
    import os
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if n_workers == 1:
        _init_worker(provider, options, radius, year, threads=None)
        for task in tasks:
            yield fn(task)
        return
    threads = max(1, (os.cpu_count() or n_workers) // n_workers)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(n_workers, mp_context=context, initializer=_init_worker,
                             initargs=(provider, options, radius, year, threads)) as pool:  # fmt: skip
        pending = set()
        for task in tasks:
            pending.add(pool.submit(fn, task))
            if len(pending) >= 2 * n_workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
        for future in pending:
            yield future.result()


def simulate_batch(
    provider: TileDBProvider,
    shots: pd.DataFrame,
    x_col: str = "center_x",
    y_col: str = "center_y",
    beam_col: str = "beam",
    year: int | None = None,
    n_workers: int = 1,
    output_path: str | None = None,
    rng: np.random.Generator | None = None,
    batch_tile_size: float = 100.0,
    max_shots_per_batch: int = 256,
    **kwargs,
) -> pd.DataFrame:
    """Simulate waveforms for many shots, in parallel worker processes.

    Shots are grouped into spatial blocks of ``batch_tile_size`` (at most
    ``max_shots_per_batch`` per chunk); each chunk is read with one query.
    With ``n_workers > 1`` chunks run in separate processes, each with its own
    TileDB context (cores / n_workers threads, a small cache), so the work
    scales with cores; the per-footprint loop holds the GIL, so threads did
    not. Every shot has its own noise seed: results do not depend on
    ``n_workers``. Workers start with ``spawn`` (TileDB's threads make
    ``fork`` unsafe), so a script calling this with ``n_workers > 1`` must
    guard its entry point with ``if __name__ == "__main__":``; notebooks and
    the interactive interpreter need nothing.

    Parameters
    ----------
    provider : TileDBProvider
    shots : pd.DataFrame
        Shot table with ``x_col`` and ``y_col``; other columns are kept.
    x_col, y_col : str
        Easting and northing columns.
    beam_col : str
        Optional column with the GEDI beam of each shot.
    year : int, optional
        Survey year.
    n_workers : int
        Worker processes (1: run in this process).
    output_path : str, optional
        Also write the result to this Parquet file.
    rng : np.random.Generator, optional
        Generator for reproducible noise.
    batch_tile_size : float
        Side (m) of the spatial blocks shots are grouped by.
    max_shots_per_batch : int
        Largest number of shots read and simulated as one task.
    **kwargs
        Passed to :func:`simulate_waveform`.

    Returns
    -------
    pd.DataFrame
        Shot columns plus ground elevation, HOME, cover, point count and RH
        metrics (NaN without enough points).
    """
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if not np.isfinite(batch_tile_size) or batch_tile_size <= 0 or max_shots_per_batch < 1:
        raise ValueError("Batch size and maximum shot count must be positive")
    centers = shots[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(centers).all():
        raise ValueError("Shot coordinates must be finite")
    random = rng if rng is not None else np.random.default_rng()
    seeds = random.integers(0, np.iinfo(np.uint64).max, size=len(shots), dtype=np.uint64)
    radius = _effective_query_radius(
        kwargs.get("footprint_radius", _FOOTPRINT_RADIUS),
        kwargs.get("gaussian_beam_weighting", True),
        kwargs.get("sigma_beam"),
        kwargs.get("beam_extent_sigma", _BEAM_EXTENT_SIGMA),
    )
    beam_ids = shots[beam_col].astype(str).to_numpy() if beam_col in shots.columns else None
    tasks = (
        (idx, centers[idx], seeds[idx], None if beam_ids is None else beam_ids[idx])
        for idx in _shot_chunks(centers, batch_tile_size, max_shots_per_batch)
    )
    results = [None] * len(shots)
    for output in _run_chunks(_simulate_chunk, tasks, n_workers, provider, kwargs, radius, year):
        for index, result in output:
            results[index] = result
    nan_metrics = dict(
        z_ground=np.nan,
        home=np.nan,
        cover=np.nan,
        n_points=0,
        ground_supported=False,
        ground_offset=np.nan,
        slope_corrected=False,
        **{f"rh{lv}": np.nan for lv in kwargs.get("rh_levels", _RH_LEVELS)},
    )
    records = []
    for index, result in enumerate(results):
        record = shots.iloc[index].to_dict()
        record.update(result if result is not None else nan_metrics)
        records.append(record)
    result_df = pd.DataFrame(records, index=shots.index) if records else shots.copy()
    if output_path is not None:
        result_df.to_parquet(output_path, engine="pyarrow", index=False)
    return result_df
