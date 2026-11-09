"""Progressive TIN densification (PTD) ground classification.

PTD (Axelsson 2000) as implemented in r-lidar/PTD by Jean-Romain Roussel
(MIT), vendored in ``alsdb/methods/native/ptd`` and compiled with
``pixi run -e spikefree build-spikefree``. Points are classified in a fixed
(X, Y, Z) order, so labels do not depend on how points were stored.

By default PTD runs as a tilt ensemble: once on the points and on copies
tilted towards evenly spaced azimuths; a point is ground when enough runs
agree. Classifying a rotated point cloud was suggested by Fabian Fischer
(University of Bristol, personal communication).

References
----------
Axelsson, P. (2000). DEM generation from laser scanner data using adaptive TIN
    models. International Archives of Photogrammetry and Remote Sensing,
    33(B4), 110-117.
Roussel, J.-R., et al. (2020). lidR: An R package for analysis of Airborne
    Laser Scanning (ALS) data. Remote Sensing of Environment, 251, 112061.
    https://doi.org/10.1016/j.rse.2020.112061 (lasR is its successor)
"""

from __future__ import annotations

import numpy as np

# lasR's classify_with_ptd defaults.
LASR_DEFAULTS = {"seed_resolution": 10.0, "angle": 30.0, "distance": 2.0, "spacing": 0.25}
# alsdb's defaults: tuned PTD settings and the tilt ensemble.
DEFAULTS = {"seed_resolution": 10.0, "angle": 45.0, "distance": 1.0, "spacing": 0.25,
            "tilt_runs": 8, "tilt": 10.0, "votes": 3}  # fmt: skip
ENSEMBLE = ("tilt_runs", "tilt", "votes")


def require_native():
    """Return the compiled PTD module, or raise an error explaining how to build it."""
    try:
        from alsdb.methods import _ptd_native
    except ImportError as exc:
        raise ImportError(
            "The native PTD module is not built: pixi run -e spikefree build-spikefree"
        ) from exc
    return _ptd_native


def ptd_ground(points: np.ndarray, *, workers: int = 1, **parameters) -> np.ndarray:
    """Classify ground points with the PTD tilt ensemble.

    Noise (classes 7 and 18) is never ground and is left out of the search.

    Parameters
    ----------
    points : numpy.ndarray
        Structured point array with X, Y, Z and Classification.
    workers : int
        Threads running the ensemble's runs in parallel; the result does not
        depend on it.
    **parameters
        Overrides of :data:`DEFAULTS`: seed_resolution, angle, distance,
        spacing, tilt_runs (1 is a single PTD run), tilt (degrees), votes.

    Returns
    -------
    numpy.ndarray
        Boolean ground mask, in the order of *points*.
    """
    unknown = set(parameters) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown PTD parameters: {sorted(unknown)}")
    settings = {**DEFAULTS, **parameters}
    runs, tilt, votes = (settings.pop(name) for name in ENSEMBLE)
    check_ensemble(runs, tilt, votes)
    if not len(points):
        return np.zeros(len(points), bool)
    if runs == 1:
        return _ptd_ground_native(points, settings)
    usable = np.flatnonzero(~np.isin(points["Classification"], (7, 18)))
    if not len(usable):
        return np.zeros(len(points), bool)
    xyz = tuple(np.ascontiguousarray(points[n][usable], dtype=np.float64) for n in ("X", "Y", "Z"))
    # Bounding-box centre, not the mean: the mean depends on summation order.
    centre = np.array([(c.min() + c.max()) / 2 for c in xyz])
    azimuths = [360.0 * i / (runs - 1) for i in range(runs - 1)]

    def run(azimuth):
        coords = xyz if azimuth is None else _tilt_xyz(*xyz, centre, tilt, azimuth)
        return _classify_xyz(*coords, settings)

    jobs = [None, *azimuths]
    votes_for = np.zeros(len(usable), np.uint8)
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(min(workers, runs)) as pool:  # the native PTD releases the GIL
            for mask in pool.map(run, jobs):
                votes_for += mask
    else:
        for job in jobs:
            votes_for += run(job)
    ground = np.zeros(len(points), bool)
    ground[usable] = votes_for >= votes
    return ground


def check_ensemble(runs, tilt, votes) -> None:
    """Raise ValueError for an invalid tilt ensemble."""
    if isinstance(runs, bool) or not isinstance(runs, (int, np.integer)) or runs < 1:
        raise ValueError("tilt_runs must be a positive integer")
    if (
        isinstance(votes, bool)
        or not isinstance(votes, (int, np.integer))
        or not 1 <= votes <= runs
    ):
        raise ValueError("votes must be an integer from 1 to tilt_runs")
    if not np.isfinite(tilt) or not 0 <= tilt < 45 or (runs > 1 and tilt == 0):
        raise ValueError("tilt must be in degrees, above 0 (with several runs) and below 45")


def _tilt_xyz(x, y, z, centre, tilt: float, azimuth: float):
    """Rotate coordinates by *tilt* degrees about the horizontal axis through *centre*, perpendicular to *azimuth*."""
    x, y, z = x - centre[0], y - centre[1], z - centre[2]
    t, a = np.radians(tilt), np.radians(azimuth)
    ca, sa, ct, st = np.cos(a), np.sin(a), np.cos(t), np.sin(t)
    u, v = x * ca + y * sa, -x * sa + y * ca
    u, z = u * ct - z * st, u * st + z * ct
    return u * ca - v * sa + centre[0], u * sa + v * ca + centre[1], z + centre[2]


def _classify_xyz(x, y, z, parameters: dict) -> np.ndarray:
    """Return the native PTD ground mask of non-noise points."""
    # PTD breaks ties by input order: sort so labels do not depend on it.
    order = np.lexsort((z, y, x))
    ground = np.zeros(len(x), bool)
    if len(x):
        ground[order] = require_native().classify(
            np.column_stack((x[order], y[order], z[order])), **parameters
        )
    return ground


def _ptd_ground_native(points: np.ndarray, parameters: dict) -> np.ndarray:
    keep = np.flatnonzero(~np.isin(points["Classification"], (7, 18)))
    ground = np.zeros(len(points), bool)
    ground[keep] = _classify_xyz(
        *(np.asarray(points[n][keep], dtype=np.float64) for n in ("X", "Y", "Z")), parameters
    )
    return ground
