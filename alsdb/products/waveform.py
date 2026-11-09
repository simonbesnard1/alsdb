# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Large-footprint waveform simulation from ALS point clouds.

Defaults follow GEDI (25 m footprint, 0.15 m bins, GEDI TX pulse kernels);
other instruments are simulated by changing ``footprint_radius``, ``sigma``
and ``beam_id`` (e.g. LVIS: radius 10 m, sigma 0.7 m; ICESat/GLAS: radius
32.5 m, sigma 2 m).

Steps: query the points of a circular footprint (beyond its radius when the
Gaussian beam weighting is on), build a vertical histogram, convolve it with
the transmitted pulse, detect the ground peak, and derive relative heights
(RH) and canopy cover.

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
_SIGMA_FULL: float = 0.64  # m  GEDI full-power beam pulse σ
_SIGMA_COV: float = 0.93  # m  GEDI coverage beam pulse σ
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
    sb = sigma_beam if sigma_beam is not None else footprint_radius / 2.0
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
    hist, _ = np.histogram(z, bins=edges, weights=weights)
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
    counted from the lowest bin. Low percentiles can be slightly negative where
    pulse or slope broadening puts ground energy below the peak.
    """
    ground_idx = int(np.searchsorted(z_bins, z_ground))
    total = float(waveform.sum())

    if total == 0.0 or ground_idx >= len(z_bins):
        return {lv: np.nan for lv in levels}

    cumulative = np.cumsum(waveform) / total
    z_above = z_bins - z_ground  # height relative to ground peak; can be negative

    rh = {}
    for level in levels:
        idx = int(np.searchsorted(cumulative, level / 100.0))
        rh[level] = float(z_above[min(idx, len(z_above) - 1)])
    # RH0 is defined as 0 m by construction (ground return height).
    # searchsorted may return a non-zero z_above[0] if the ground bin is
    # slightly offset, so we force it to the correct value.
    if 0 in levels:
        rh[0] = 0.0
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
    sigma: float = _SIGMA_FULL,
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
    density_radius: float | None = None,
) -> WaveformResult | None:
    """Query a footprint and simulate its waveform; see :func:`waveform_from_points`."""
    options = dict(locals())
    options.pop("provider")
    radius = _effective_query_radius(
        footprint_radius, gaussian_beam_weighting, sigma_beam, beam_extent_sigma
    )
    if density_radius is not None and (not np.isfinite(density_radius) or density_radius <= 0):
        raise ValueError("density_radius must be finite and positive")
    data = _query_footprint(provider, center_x, center_y, radius + (density_radius or 0), year)
    return waveform_from_points(data, **options)


def waveform_from_points(
    data,
    center_x: float,
    center_y: float,
    year: int | None = None,
    footprint_radius: float = _FOOTPRINT_RADIUS,
    z_step: float = _Z_STEP,
    sigma: float = _SIGMA_FULL,
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
    density_radius: float | None = None,
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
        Gaussian pulse width (m), used when ``beam_id`` is None.
    beam_id : str, optional
        GEDI beam (e.g. ``"BEAM0000"``) whose measured pulse is used.
    noise_std : float
        Standard deviation of additive noise (0: none).
    intensity_weighted : bool
        Weight returns by intensity instead of counting them.
    gaussian_beam_weighting : bool
        Weight points by the Gaussian beam profile.
    sigma_beam : float, optional
        Beam sigma (m); default ``footprint_radius / 2``.
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

    Returns
    -------
    WaveformResult or None
    """
    from scipy.spatial import cKDTree

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
    if density_radius is not None and (not np.isfinite(density_radius) or density_radius <= 0):
        raise ValueError("density_radius must be finite and positive")
    radius = _effective_query_radius(
        footprint_radius, gaussian_beam_weighting, sigma_beam, beam_extent_sigma
    )
    density_weight = None
    if data is not None:
        data = clean_points(point_array(data))
        inside = (data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2 <= radius**2
        if density_radius is not None:
            first = data[data["ReturnNumber"] == 1]
            tree = cKDTree(np.column_stack((first["X"], first["Y"])))
            support = tree.query_ball_point(
                np.column_stack((data["X"][inside], data["Y"][inside])),
                density_radius,
                return_length=True,
            )
            density_weight = np.divide(1.0, support, out=np.zeros(len(support)), where=support > 0)
        data = data[inside]
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
        sb = sigma_beam if sigma_beam is not None else footprint_radius / 2.0
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

    # 1. Vertical histogram
    z_bins, hist = _build_histogram(z, weights, z_step)

    # Retain convolution tails: trimming here biases ground and canopy energy.
    if beam_id is not None and beam_id in _BEAM_IDS:
        kernel = _load_pulse(beam_id, z_step)
    else:
        radius_bins = max(1, int(np.ceil(4 * sigma / z_step)))
        offsets = np.arange(-radius_bins, radius_bins + 1) * z_step
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
    pad = (len(kernel) - 1) // 2
    waveform = np.convolve(hist, kernel, mode="full")
    z_bins = z_bins[0] + (np.arange(len(waveform)) - pad) * z_step

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
    )


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
    **kwargs,
) -> pd.DataFrame:
    """Simulate waveforms for many shots in parallel.

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
        Threads.
    output_path : str, optional
        Also write the result to this Parquet file.
    rng : np.random.Generator, optional
        Generator for reproducible noise.
    **kwargs
        Passed to :func:`simulate_waveform`.

    Returns
    -------
    pd.DataFrame
        Shot columns plus ground elevation, HOME, cover, point count and RH
        metrics (NaN without enough points).
    """
    from alsdb.products._footprints import footprint_batches

    centers = shots[[x_col, y_col]].to_numpy(dtype=float)
    random = rng if rng is not None else np.random.default_rng()
    seeds = random.integers(0, np.iinfo(np.uint64).max, size=len(shots), dtype=np.uint64)
    radius = _effective_query_radius(
        kwargs.get("footprint_radius", _FOOTPRINT_RADIUS),
        kwargs.get("gaussian_beam_weighting", True),
        kwargs.get("sigma_beam"),
        kwargs.get("beam_extent_sigma", _BEAM_EXTENT_SIGMA),
    ) + (kwargs.get("density_radius") or 0)
    results = [None] * len(shots)
    attributes = ("Z", "Intensity", "ReturnNumber", "NumberOfReturns", "Classification", "Withheld")
    beam_ids = shots[beam_col].astype(str).to_numpy() if beam_col in shots.columns else None

    def process(indices, points, neighborhoods):
        output = []
        for index, neighborhood in zip(indices, neighborhoods):
            options = dict(kwargs)
            if beam_ids is not None:
                options["beam_id"] = beam_ids[index]
            result = waveform_from_points(
                points[neighborhood],
                *centers[index],
                year=year,
                rng=np.random.default_rng(seeds[index]),
                **options,
            )
            output.append((index, result.to_dict() if result is not None else None))
        return output

    for batch in footprint_batches(
        provider,
        centers,
        radius,
        year=year,
        attributes=attributes,
        batch_tile_size=batch_tile_size,
        n_workers=n_workers,
        process=process,
    ):
        for index, result in batch:
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
