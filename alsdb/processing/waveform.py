# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
GEDI large-footprint waveform simulator from ALS TileDB point clouds.

Pipeline
--------
1. Query TileDB for all ALS points within a 25 m diameter circular footprint.
2. Build a vertical histogram of Z values (0.15 m bins — GEDI native resolution),
   optionally weighted by return intensity.
3. Convolve with a Gaussian pulse kernel to simulate the GEDI instrument response.
4. Detect the ground peak (lowest significant peak in the waveform).
5. Extract GEDI-style metrics: RH10–RH100, HOME, canopy cover.

Key references
--------------
- Hancock et al. (2019). The GEDI simulator: A large-footprint waveform lidar
  simulator for calibration and validation of spaceborne missions.
  Remote Sensing of Environment, 220, 309–323.
- GEDI Algorithm Theoretical Basis Document (ATBD), NASA (2019).

Validation use case
-------------------
Simulate waveforms at actual GEDI shot locations, then compare simulated vs
observed waveforms to derive calibration offsets::

    from alsdb import ALSProvider
    from alsdb.processing.waveform import simulate_batch

    als = ALSProvider(storage_type="local", uri="array_")

    # gedi_shots is a DataFrame with center_x, center_y columns (UTM)
    results = simulate_batch(als, gedi_shots, year=2021)
    # results has columns: rh25, rh50, rh75, rh95, rh100, home, cover, ...
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from alsdb.providers.tiledb_provider import TileDBProvider

logger = logging.getLogger(__name__)

# GEDI instrument constants
_FOOTPRINT_RADIUS: float = 12.5     # m  (25 m diameter)
_Z_STEP: float = 0.15               # m  native vertical resolution
_SIGMA_FULL: float = 0.64           # m  full-power beam pulse σ
_SIGMA_COV: float = 0.93            # m  coverage beam pulse σ
_MIN_POINTS: int = 25
_RH_LEVELS: tuple[int, ...] = tuple(range(101))   # RH0–RH100, matches GEDI L2A
_COVER_THRESHOLD: float = 2.0       # m  above ground


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WaveformResult:
    """
    Simulated GEDI-style waveform and derived metrics for one footprint.

    Attributes
    ----------
    z_bins : np.ndarray
        Elevation of each bin centre (m, ascending).
    waveform : np.ndarray
        Normalised energy per bin (sums to 1.0).
    z_ground : float
        Detected ground peak elevation (m).
    rh : dict[int, float]
        Relative height metrics (m above ground) keyed by percentile level.
        Standard GEDI levels: 10, 25, 50, 75, 90, 95, 98, 100.
    home : float
        Height Of Median Energy = RH50 (m above ground).
    cover : float
        Canopy cover fraction (energy above ground + 2 m / total energy).
    n_points : int
        Number of ALS points used.
    center_x : float
        Footprint centre easting (m UTM).
    center_y : float
        Footprint centre northing (m UTM).
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

    def rh_array(self, levels: tuple[int, ...] = _RH_LEVELS) -> np.ndarray:
        """RH values as a 1-D array for the given levels."""
        return np.array([self.rh.get(l, np.nan) for l in levels])

    def to_dict(self) -> dict:
        """Flat dict of scalar metrics (no arrays) — useful for DataFrame rows."""
        d = {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "z_ground": self.z_ground,
            "home": self.home,
            "cover": self.cover,
            "n_points": self.n_points,
        }
        d.update({f"rh{l}": v for l, v in sorted(self.rh.items())})
        return d


# ---------------------------------------------------------------------------
# Step 1 — footprint query
# ---------------------------------------------------------------------------

def _query_footprint(
    provider: TileDBProvider,
    center_x: float,
    center_y: float,
    radius: float,
    year: Optional[int],
) -> Optional[dict[str, np.ndarray]]:
    """
    Return ALS points within a circular footprint as a dict of 1-D arrays.

    Queries a bounding box first (TileDB spatial index), then applies a
    circular mask in Python.  Returns ``None`` if the footprint is empty.
    """
    yr_dim = provider.schema.domain.dim("Year")
    y0 = year if year is not None else int(yr_dim.domain[0])
    # +1: TileDB-Py int-dimension slices are exclusive-end (like Python slices)
    y1 = (year + 1) if year is not None else int(yr_dim.domain[1]) + 1

    attrs = ["Z", "Intensity", "ReturnNumber", "Classification"]
    with provider.open("r") as arr:
        data = arr.query(attrs=attrs)[
            center_x - radius: center_x + radius,
            center_y - radius: center_y + radius,
            y0: y1,
        ]

    if len(data["X"]) == 0:
        return None

    mask = (data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2 <= radius ** 2
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
    """
    Bin Z values into a regular vertical histogram.

    Returns
    -------
    z_bins : np.ndarray
        Bin centre elevations (m, ascending).
    hist : np.ndarray
        Weighted point count per bin.
    """
    z_min = float(z.min())
    z_max = float(z.max())
    n_bins = max(1, int(np.ceil((z_max - z_min) / z_step)))
    edges = np.linspace(z_min, z_max, n_bins + 1)
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
    """
    Detect ground peak elevation from the waveform.

    Strategy: smooth to suppress multi-return noise, find all significant
    peaks, return the lowest-elevation one (= ground return).
    Falls back to the lowest non-negligible bin if no peak is found.
    """
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
    """
    Compute Relative Height (RH) metrics.

    RH(x) = height above ground at which x% of cumulative waveform energy
    (integrated upward from the ground peak) is reached.
    """
    ground_idx = int(np.searchsorted(z_bins, z_ground))
    above = waveform[ground_idx:]

    if above.sum() == 0:
        return {l: np.nan for l in levels}

    cumulative = np.cumsum(above) / above.sum()
    z_above = z_bins[ground_idx:] - z_ground  # heights above ground

    rh = {}
    for level in levels:
        idx = int(np.searchsorted(cumulative, level / 100.0))
        rh[level] = float(z_above[min(idx, len(z_above) - 1)])
    return rh


def _canopy_cover(
    waveform: np.ndarray,
    z_bins: np.ndarray,
    z_ground: float,
    threshold: float = _COVER_THRESHOLD,
) -> float:
    """Fraction of waveform energy above (z_ground + threshold)."""
    total = waveform.sum()
    if total == 0.0:
        return 0.0
    return float(waveform[z_bins >= (z_ground + threshold)].sum() / total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_waveform(
    provider: TileDBProvider,
    center_x: float,
    center_y: float,
    year: Optional[int] = None,
    footprint_radius: float = _FOOTPRINT_RADIUS,
    z_step: float = _Z_STEP,
    sigma: float = _SIGMA_FULL,
    noise_std: float = 0.0,
    intensity_weighted: bool = False,
    gaussian_beam_weighting: bool = True,
    sigma_beam: Optional[float] = None,
    min_points: int = _MIN_POINTS,
    cover_threshold: float = _COVER_THRESHOLD,
    rh_levels: tuple[int, ...] = _RH_LEVELS,
) -> Optional[WaveformResult]:
    """
    Simulate a GEDI large-footprint waveform at a given UTM location.

    Parameters
    ----------
    provider : TileDBProvider
        Local or S3 ALS array.
    center_x, center_y : float
        Footprint centre in UTM metres (same CRS as the array).
        Convert from GEDI lat/lon with e.g. ``pyproj.Transformer``.
    year : int, optional
        Survey year filter.  ``None`` merges all years.
    footprint_radius : float
        Footprint radius in metres (default 12.5 → 25 m diameter).
    z_step : float
        Vertical bin size in metres (GEDI native = 0.15 m).
    sigma : float
        Gaussian pulse width σ in metres.  Use ``_SIGMA_FULL`` (0.64 m) for
        full-power beams, ``_SIGMA_COV`` (0.93 m) for coverage beams.
    noise_std : float
        Standard deviation of additive Gaussian noise (0 = noise-free).
    intensity_weighted : bool
        Weight histogram by return intensity rather than point count.
    gaussian_beam_weighting : bool
        If ``True`` (default), weight each ALS point by the GEDI Gaussian beam
        profile: ``exp(-r² / (2 σ_beam²))``.  Points near the footprint edge
        contribute less than those at the centre, matching the real instrument
        response and reducing systematic bias in heterogeneous canopy.
    sigma_beam : float, optional
        Beam σ in metres for Gaussian weighting.  Defaults to
        ``footprint_radius / 2`` (so the 1/e² point is at the footprint edge).
    min_points : int
        Minimum ALS points in footprint; returns ``None`` below this.
    cover_threshold : float
        HAG threshold (m) for canopy cover calculation.
    rh_levels : tuple of int
        RH percentile levels to compute.  Default: RH0–RH100 (GEDI L2A).
        Result ``rh`` dict has integer keys, e.g. ``result.rh[50]`` → RH50.

    Returns
    -------
    WaveformResult or None
    """
    data = _query_footprint(provider, center_x, center_y, footprint_radius, year)
    n_pts = 0 if data is None else int(len(data["Z"]))
    if data is None or n_pts < min_points:
        logger.warning(
            "simulate_waveform: footprint at (%.0f, %.0f) year=%s has only %d points "
            "(min_points=%d) — returning None.  "
            "Hint: call provider.available_years() and provider.query_bbox() to verify "
            "data coverage.",
            center_x, center_y, year, n_pts, min_points,
        )
        return None

    z = data["Z"].astype(np.float64)

    # Build point weights: beam profile × optional intensity
    if gaussian_beam_weighting:
        sb = sigma_beam if sigma_beam is not None else footprint_radius / 2.0
        r2 = (data["X"] - center_x) ** 2 + (data["Y"] - center_y) ** 2
        beam_w = np.exp(-r2 / (2.0 * sb ** 2))
    else:
        beam_w = np.ones(len(z))

    if intensity_weighted:
        weights = data["Intensity"].astype(np.float64) * beam_w
    else:
        weights = beam_w

    # 1. Vertical histogram
    z_bins, hist = _build_histogram(z, weights, z_step)

    # 2. Convolve with Gaussian pulse
    waveform = gaussian_filter1d(hist, sigma=sigma / z_step)

    # 3. Add noise
    if noise_std > 0.0:
        waveform = np.maximum(0.0, waveform + np.random.normal(0.0, noise_std, len(waveform)))

    # 4. Normalise to unit energy
    total = waveform.sum()
    if total > 0.0:
        waveform /= total

    # 5. Ground detection + metrics
    z_ground = _detect_ground(waveform, z_bins)
    rh = _rh_metrics(waveform, z_bins, z_ground, levels=rh_levels)
    cover = _canopy_cover(waveform, z_bins, z_ground, cover_threshold)

    return WaveformResult(
        z_bins=z_bins,
        waveform=waveform,
        z_ground=z_ground,
        rh=rh,
        home=rh.get(50, np.nan),
        cover=cover,
        n_points=int(len(z)),
        center_x=center_x,
        center_y=center_y,
    )


def simulate_batch(
    provider: TileDBProvider,
    shots: pd.DataFrame,
    x_col: str = "center_x",
    y_col: str = "center_y",
    year: Optional[int] = None,
    n_workers: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """
    Simulate waveforms for a batch of shot locations in parallel.

    Parameters
    ----------
    provider : TileDBProvider
    shots : pd.DataFrame
        Must contain ``x_col`` and ``y_col`` columns (UTM metres).
        All other columns are preserved in the output.
    x_col, y_col : str
        Column names for easting and northing.
    year : int, optional
        Survey year filter applied to all shots.
    n_workers : int
        Thread pool size (TileDB reads are thread-safe).
    **kwargs
        Forwarded to :func:`simulate_waveform`.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``z_ground``, ``home``, ``cover``,
        ``n_points``, ``rh10`` … ``rh100``.
        Shots with insufficient ALS coverage have NaN metric values.
    """
    def _run(row):
        return row.name, simulate_waveform(
            provider,
            center_x=float(row[x_col]),
            center_y=float(row[y_col]),
            year=year,
            **kwargs,
        )

    results: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run, row): row.name for _, row in shots.iterrows()}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    _nan_metrics = {
        "z_ground": np.nan, "home": np.nan, "cover": np.nan, "n_points": 0,
        **{f"rh{l}": np.nan for l in _RH_LEVELS},
    }

    records = []
    for _, row in shots.iterrows():
        rec = row.to_dict()
        r = results.get(row.name)
        rec.update(r.to_dict() if r is not None else _nan_metrics)
        records.append(rec)

    return pd.DataFrame(records, index=shots.index)
