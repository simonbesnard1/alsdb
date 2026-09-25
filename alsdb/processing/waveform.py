# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Large-footprint waveform simulator from ALS TileDB point clouds.

The pipeline is sensor-agnostic: any ALS point cloud stored in TileDB can be
used as input.  Default parameters are tuned for **GEDI** (25 m footprint,
0.15 m bins, GEDI beam TX kernels); other spaceborne or airborne instruments
can be targeted by overriding the relevant parameters — see *Sensor presets*
below.

Pipeline
--------
1. Query TileDB for all ALS points within a circular footprint - when Gaussian
   beam weighting is on, the query itself extends past the nominal footprint
   radius (``beam_extent_sigma``) so the beam's Gaussian tail is actually
   captured before being weighted down, not truncated away (see
   :func:`_effective_query_radius`).
2. Build a vertical histogram of Z values, optionally weighted by return
   intensity and/or the Gaussian beam profile.
3. Convolve with a TX pulse kernel (per-beam empirical shape, or Gaussian).
4. Detect the ground peak (lowest significant peak in the waveform).
5. Extract relative-height metrics (RH0-RH100, HOME) as cumulative fractions
   of the *total* recorded waveform energy, and canopy cover as a
   gap-probability ratio with an optional reflectance-ratio (``rv_rg``)
   correction (default ``1.0`` reproduces the uncorrected energy-ratio
   proxy, not GEDI's true reflectance-corrected L2B cover - see
   :func:`_canopy_cover`).

Sensor presets
--------------
GEDI (default)::

    simulate_waveform(provider, x, y,
                      footprint_radius=12.5,   # 25 m diameter
                      z_step=0.15,             # GEDI native vertical resolution
                      beam_id="BEAM0000")      # uses bundled TX pulse kernel

LVIS (NASA airborne full-waveform)::

    simulate_waveform(provider, x, y,
                      footprint_radius=10.0,   # campaign-dependent (typically 10–25 m)
                      z_step=0.15,
                      beam_id=None,            # no bundled LVIS kernel; Gaussian used
                      sigma=0.7)               # approximate LVIS pulse width

ICESat / GLAS::

    simulate_waveform(provider, x, y,
                      footprint_radius=32.5,   # 65 m diameter
                      z_step=0.15,
                      beam_id=None,
                      sigma=2.0)               # GLAS pulse σ ≈ 2 m

Key references
--------------
- Hancock et al. (2019). The GEDI simulator: A large-footprint waveform lidar
  simulator for calibration and validation of spaceborne missions.
  Remote Sensing of Environment, 220, 309–323.
- GEDI Algorithm Theoretical Basis Document (ATBD), NASA (2019).
- Blair et al. (1999). The Laser Vegetation Imaging Sensor (LVIS).
  IGARSS, Hamburg.

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

# Empirical TX pulse kernels derived from GEDI L1B data (Hancock et al. gediSimulator).
# Used when beam_id is supplied; other sensors fall back to a Gaussian kernel (sigma=).
# Full-power beams: 0000, 0001, 0010, 0011, 1000, 1011
# Coverage beams:  0101, 0110
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
    """
    Load and return the normalised TX pulse kernel for *beam_id*.

    The kernel is read from the bundled mean-pulse ASCII file, centered on its
    peak bin, and normalised to unit sum so that convolution preserves total
    waveform energy.

    Returns a 1-D float64 array ready for ``np.convolve(..., mode="same")``.
    """
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
    ground_supported: bool = False
    ground_offset: float = np.nan
    slope_corrected: bool = False

    def rh_array(self, levels: tuple[int, ...] = _RH_LEVELS) -> np.ndarray:
        """RH values as a 1-D array for the given levels."""
        return np.array([self.rh.get(lv, np.nan) for lv in levels])

    def to_dict(self) -> dict:
        """Flat dict of scalar metrics (no arrays) — useful for DataFrame rows."""
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
    """
    Return the radius actually used to query/mask points for one footprint.

    When ``gaussian_beam_weighting`` is enabled, points are weighted by
    ``exp(-r^2 / (2 * sigma_beam^2))``.  A 2-D Gaussian's cumulative energy
    within a radius of exactly ``footprint_radius`` (= 2 * sigma_beam with
    the default ``sigma_beam = footprint_radius / 2``) is only
    ``1 - exp(-2) ≈ 86.5%`` - querying no farther than ``footprint_radius``
    silently discards the other ~13.5% of the beam's true energy before it
    ever reaches the weighting step. Querying out to
    ``beam_extent_sigma * sigma_beam`` instead (default 3σ ≈ 98.9% of total
    Gaussian energy) fixes this without changing the weighting formula
    itself - the tail is now present to be weighted down, rather than
    absent entirely.

    When ``gaussian_beam_weighting`` is ``False`` there's no Gaussian tail
    to account for - a flat circular footprint of exactly
    ``footprint_radius`` is already the correct, unmodified query.
    """
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
    """
    Return ALS points within a circular footprint as a dict of 1-D arrays.

    Queries a bounding box first (TileDB spatial index), then applies a
    circular mask in Python.  Returns ``None`` if the footprint is empty.
    """
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
    """
    Bin Z values into a regular vertical histogram.

    Returns
    -------
    z_bins : np.ndarray
        Bin centre elevations (m, ascending).
    hist : np.ndarray
        Weighted point count per bin.
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

    RH(x) = height above the detected ground peak at which x% of the
    **total recorded waveform energy** is reached, cumulative from the
    waveform's lowest bin upward - the same total :func:`_canopy_cover` uses
    for its own denominator (``waveform.sum()``), not a "from the ground
    peak upward" subset. The TX-pulse convolution and slope broadening both
    spread some genuine ground-return energy into bins *below* the detected
    peak; excluding that energy from the denominator (an earlier version of
    this function did exactly that, via ``waveform[ground_idx:].sum()``)
    understated the true total and made RH silently disagree with
    ``_canopy_cover`` computed on the same waveform. A consequence of using
    the full total: low RH percentiles can come out slightly *negative*
    when there's real energy below the peak - this is an expected, physical
    effect of pulse/slope broadening (also seen in real GEDI/LVIS RH
    products), not a bug, so it isn't clamped away.
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
    """
    Canopy cover from a gap-probability model, with an optional
    reflectance-ratio correction.

    Physical model: observed energy from a surface type is proportional to
    (true fractional footprint area of that type) x (its reflectance).
    Writing true vegetation cover as ``fv`` and ground exposure as
    ``1 - fv``: ``veg_energy ∝ fv * rho_v``, ``ground_energy ∝ (1 - fv) *
    rho_g``. Solving for ``fv`` given the observed energies and
    ``rv_rg = rho_v / rho_g``:

    ``cover = veg_energy / (veg_energy + rv_rg * ground_energy)``

    - the ratio corrects the **ground** term, not the vegetation term
    (scaling the vegetation term instead, as an earlier version of this
    function did, moves cover in the wrong direction - verified by direct
    derivation from the model above, not just recalled from memory, after a
    test caught the sign error). Not simply "energy above threshold / total
    energy", though that's exactly what this reduces to at the default
    ``rv_rg=1.0`` (equal-reflectance assumption): substituting
    ``ground_energy = total - veg_energy`` collapses the denominator back to
    ``total``, reproducing the naive ratio. This default is **not** GEDI's
    actual L2B cover, which applies a real, externally-calibrated rv/rg
    (commonly < 1 - vegetation typically backscatters less efficiently than
    bare ground at 1064 nm, so the uncorrected ratio systematically
    *underestimates* true cover: with a smaller ``rv_rg``, the same observed
    energy split implies a larger true ``fv``). Pass a calibrated ``rv_rg``
    for your study area to get the corrected metric; don't guess one.
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
    """Query a footprint and simulate it; see waveform_from_points for parameters."""
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
        Gaussian pulse width σ in metres.  Used only when ``beam_id`` is
        ``None``.  Use ``_SIGMA_FULL`` (0.64 m) for full-power beams or
        ``_SIGMA_COV`` (0.93 m) for coverage beams.
    beam_id : str, optional
        GEDI beam identifier (e.g. ``"BEAM0000"``).  When provided, the
        bundled mean TX pulse shape for that beam is used as the convolution
        kernel instead of a symmetric Gaussian, giving a more physically
        accurate waveform (the real GEDI pulse has an asymmetric tail).
        Must be one of ``_BEAM_IDS``; silently falls back to Gaussian if
        unrecognised.
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
    beam_extent_sigma : float
        How many ``sigma_beam`` to query out to when ``gaussian_beam_weighting``
        is ``True`` (default 3σ ≈ 98.9% of the beam's total Gaussian energy).
        Querying only to ``footprint_radius`` (= 2σ with the default
        ``sigma_beam``) would silently discard ~13.5% of the true beam
        energy before the weighting step ever sees it - see
        :func:`_effective_query_radius`. Ignored when
        ``gaussian_beam_weighting=False`` (a flat circular footprint of
        exactly ``footprint_radius`` has no tail to account for).
    slope_correction : bool
        If ``True``, fit a plane to the ALS ground points (Classification == 2)
        inside the footprint and subtract it from all Z values before building
        the histogram.  This collapses the slope-broadened ground return back
        to a sharp peak and retains vertical vegetation height above the fitted terrain.  Requires at least 3 ground
        points; silently skipped otherwise.  Default ``False`` — only enable
        for terrain with slopes > ~15°.
    min_points : int
        Minimum ALS points in footprint; returns ``None`` below this.
    cover_threshold : float
        HAG threshold (m) for canopy cover calculation.
    rv_rg : float
        Vegetation-to-ground reflectance ratio at the laser wavelength, used
        to correct canopy cover from a raw energy ratio to a true
        fractional-area gap probability: ``cover = veg_energy /
        (veg_energy + rv_rg * ground_energy)`` (see :func:`_canopy_cover`
        for the derivation). Default ``1.0`` assumes equal reflectance and
        reproduces the uncorrected energy-ratio proxy - **not** GEDI's
        actual L2B cover, which applies a real, externally-calibrated rv/rg
        (commonly < 1, since vegetation typically backscatters less
        efficiently than bare ground at 1064 nm - the uncorrected ratio then
        systematically *underestimates* true cover). Only meaningful when
        the histogram is genuinely energy-like, i.e. with
        ``intensity_weighted=True``; with plain point counts (the default),
        "energy" here just means point density; and ``rv_rg`` should still
        only be set from a real calibration for your study area, not
        guessed.
    rh_levels : tuple of int
        RH percentile levels to compute.  Default: RH0–RH100 (GEDI L2A).
        Result ``rh`` dict has integer keys, e.g. ``result.rh[50]`` → RH50.
    rng : np.random.Generator, optional
        Random number generator used for additive noise when ``noise_std > 0``.
        Pass ``np.random.default_rng(seed)`` for reproducible results.
        ``None`` (default) uses a fresh unseeded generator each call.

    Returns
    -------
    WaveformResult or None
    """
    from scipy.spatial import cKDTree

    from alsdb.processing._footprints import point_array
    from alsdb.processing._surface import clean_points

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
    """
    Simulate waveforms for a batch of shot locations in parallel.

    Parameters
    ----------
    provider : TileDBProvider
    shots : pd.DataFrame
        Must contain ``x_col`` and ``y_col`` columns (UTM metres).
        If a ``beam_col`` column is present (e.g. ``"BEAM0000"``), the
        per-beam mean TX pulse is used for each shot automatically.
        All other columns are preserved in the output.
    x_col, y_col : str
        Column names for easting and northing.
    beam_col : str
        Column name for the GEDI beam identifier.  Ignored if not present in
        ``shots``.  Per-row beam overrides any ``beam_id`` passed via
        ``**kwargs``.
    year : int, optional
        Survey year filter applied to all shots.
    n_workers : int
        Thread pool size (TileDB reads are thread-safe).
    output_path : str, optional
        If provided, the result DataFrame is written to this path as a
        Parquet file (``pyarrow`` engine).  The file is created or
        overwritten.  The DataFrame is still returned as usual.
    rng : np.random.Generator, optional
        Random number generator forwarded to each :func:`simulate_waveform`
        call for reproducible noise.  Pass ``np.random.default_rng(seed)``.
        Independent per-position seeds make results reproducible across worker counts,
        including when the input index contains duplicate labels.
    **kwargs
        Forwarded to :func:`simulate_waveform`.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``z_ground``, ``home``, ``cover``,
        ``n_points``, ``rh0`` … ``rh100``.
        Shots with insufficient ALS coverage have NaN metric values.
    """
    from alsdb.processing._footprints import footprint_batches

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

    def process(indices, points, neighborhoods):
        output = []
        for index, neighborhood in zip(indices, neighborhoods):
            options = dict(kwargs)
            if beam_col in shots.columns:
                options["beam_id"] = str(shots.iloc[index][beam_col])
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
