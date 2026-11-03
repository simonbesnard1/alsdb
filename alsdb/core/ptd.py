"""Progressive TIN densification (PTD) ground classification.

PTD (Axelsson 2000) as implemented by Jean-Romain Roussel in r-lidar/PTD
(MIT), the library lasR and pyLASR run. alsdb vendors it unchanged
(``alsdb/processing/native/ptd``) and compiles it into
``alsdb.processing._ptd_native`` with ``pixi run -e spikefree build-spikefree``:
:func:`ptd_ground` classifies point arrays directly, without files.

``backend="pylasr"`` runs the same algorithm through pyLASR instead (optional,
``pixi run -e spikefree install-pylasr``; lasR is GPL-3 and only called as a
separately installed package), writing the points to a temporary LAS file.
It is kept to check the native module: given the same coordinates in the same
order, both give identical labels. PTD is sensitive to both: the 1 mm
rounding of that file changed ~1% of labels on a sparse tile, shuffling the
points ~0.5%, and DTM cells by up to metres. The native path therefore
classifies points in a fixed (X, Y, Z) order.

Tilt ensemble
-------------
By default :func:`ptd_ground` runs PTD ``tilt_runs`` times: once on the points
as they are, and on copies tilted by ``tilt`` degrees towards evenly spaced
azimuths. A point is ground when at least ``votes`` runs call it ground. A
rigid tilt leaves PTD's own tests (angles and distances to the TIN facets)
unchanged but moves the seed cells, the 0.25 m candidate cells and the
triangulation, so each run fails in different places on steep slopes; voting
keeps what the runs agree on. Classifying a rotated point cloud was suggested
by Fabian Fischer (University of Bristol, personal communication); the tilt version
and its settings were chosen for alsdb.

Selection (2026-11; settings fixed on 16 development blocks of the Chapitre
Petit Buëch data, then held out on 8 other Petit Buëch tiles and 4 Robson
Creek tiles, each at full density and thinned to 8, 4, 2, 1 and 0.5 pulses/m²;
error of a 1 m TIN DTM against the provider ground):

=====================  ===========  ============
method                 Petit Buëch  Robson Creek
=====================  ===========  ============
PTD, lasR defaults     0.192 m      0.937 m
PTD, tuned (one run)   0.180 m      0.843 m
tilt ensemble          0.160 m      0.694 m
=====================  ===========  ============

The ensemble was never worse than the single tuned run by more than 1 mm on
any of the 72 held-out cases; its gain grows as density falls (0.5 pulses/m²:
0.38 -> 0.32 m and 2.35 -> 1.76 m). It costs ``tilt_runs`` PTD runs, which
``workers`` threads can run in parallel. It is applied everywhere, never by a
density or slope rule, so that every product is made the same way.
``tilt_runs=1`` gives a single PTD run.

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

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pdal

NOISE_FILTER = "Classification %out% 7 18"  # lasR drop_noise()
# lasR's classify_with_ptd defaults (pyLASR's own differ: angle 20, distance 3).
LASR_DEFAULTS = {"seed_resolution": 10.0, "angle": 30.0, "distance": 2.0, "spacing": 0.25}
# alsdb's: PTD settings and tilt ensemble selected as described above.
DEFAULTS = {"seed_resolution": 10.0, "angle": 45.0, "distance": 1.0, "spacing": 0.25,
            "tilt_runs": 8, "tilt": 10.0, "votes": 3}  # fmt: skip
ENSEMBLE = ("tilt_runs", "tilt", "votes")


def require_native():
    """The compiled PTD module, or a clear error on how to build it."""
    try:
        from alsdb.processing import _ptd_native
    except ImportError as exc:
        raise ImportError(
            "The native PTD module is not built: pixi run -e spikefree build-spikefree"
        ) from exc
    return _ptd_native


def require_pylasr():
    """The pyLASR module, or a clear error on how to install it."""
    try:
        import pylasr
    except ImportError as exc:
        raise ImportError(
            "PTD needs pyLASR built from lasR source (the PyPI wheel lacks "
            "classify_with_ptd): pixi run -e spikefree install-pylasr"
        ) from exc
    if not hasattr(pylasr, "classify_with_ptd"):
        raise ImportError("Installed pyLASR does not expose classify_with_ptd; rebuild it")
    return pylasr


def run_ptd(input_path, output_path, *, seed_resolution, angle, distance, spacing) -> float:
    """Classify *input_path* with PTD into *output_path* (ground = 2); return seconds.

    Points of classes 7 and 18 (noise) are left out of the ground search.
    """
    pylasr = require_pylasr()
    pipeline = pylasr.classify_with_ptd(
        res=seed_resolution,
        angle=angle,
        distance=distance,
        spacing=spacing,
        classification=2,
        filter=[NOISE_FILTER],
    ) + pylasr.write_las(str(output_path))
    # pyLASR writes its pipeline to <temp dir>/pipeline.json, a fixed name:
    # parallel runs overwrote each other's (JSON parse errors, missing outputs).
    # std::filesystem::temp_directory_path follows TMPDIR, so give each call its own.
    with tempfile.TemporaryDirectory(prefix="alsdb-pylasr-") as private:
        previous = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = private
        try:
            start = time.perf_counter()
            result = pipeline.execute(str(input_path))
            elapsed = time.perf_counter() - start
        finally:
            if previous is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous
    if not result.get("success") or not Path(output_path).is_file():
        raise RuntimeError(f"pyLASR PTD failed: {result}")
    return elapsed


def _read(path) -> np.ndarray:
    pipeline = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(path)}]))
    pipeline.execute()
    return pipeline.arrays[0]


def ptd_ground(
    points: np.ndarray, *, backend: str = "native", workers: int = 1, **parameters
) -> np.ndarray:
    """Boolean ground mask of *points* (in their order) from the PTD tilt ensemble.

    *parameters* override :data:`DEFAULTS`: the PTD settings (*seed_resolution*,
    *angle*, *distance*, *spacing*) and the ensemble (*tilt_runs*, *tilt* in
    degrees, *votes*; ``tilt_runs=1`` is one plain PTD run). Noise (7, 18) is
    never ground and is left out of the ground search. *backend* is
    ``"native"`` (the vendored library) or ``"pylasr"``. *workers* threads run
    the ensemble's runs in parallel; the result does not depend on it.
    """
    unknown = set(parameters) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown PTD parameters: {sorted(unknown)}")
    if backend not in ("native", "pylasr"):
        raise ValueError("backend must be 'native' or 'pylasr'")
    settings = {**DEFAULTS, **parameters}
    runs, tilt, votes = (settings.pop(name) for name in ENSEMBLE)
    check_ensemble(runs, tilt, votes)
    if not len(points):
        return np.zeros(len(points), bool)
    single = _ptd_ground_native if backend == "native" else _ptd_ground_pylasr
    if runs == 1:
        return single(points, settings)
    usable = ~np.isin(points["Classification"], (7, 18))
    if not usable.any():
        return np.zeros(len(points), bool)
    # Tilt about the centre of the bounding box (not the mean, which depends on
    # the order the points are summed in).
    centre = np.array([(points[n][usable].min() + points[n][usable].max()) / 2
                       for n in ("X", "Y", "Z")])  # fmt: skip
    azimuths = [360.0 * i / (runs - 1) for i in range(runs - 1)]

    def run(azimuth):
        if azimuth is None:
            return single(points, settings)
        return single(_tilted(points, centre, tilt, azimuth), settings)

    jobs = [None, *azimuths]
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(min(workers, runs)) as pool:  # the native PTD releases the GIL
            masks = list(pool.map(run, jobs))
    else:
        masks = [run(job) for job in jobs]
    return np.sum(masks, axis=0) >= votes


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


def _tilted(points: np.ndarray, centre, tilt: float, azimuth: float) -> np.ndarray:
    """Copy of *points* rotated by *tilt* degrees about the horizontal axis
    through *centre* perpendicular to *azimuth* (degrees from +X towards +Y)."""
    out = points.copy()
    x, y, z = (points[n] - c for n, c in zip(("X", "Y", "Z"), centre))
    t, a = np.radians(tilt), np.radians(azimuth)
    u, v = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    u, z = u * np.cos(t) - z * np.sin(t), u * np.sin(t) + z * np.cos(t)
    out["X"] = u * np.cos(a) - v * np.sin(a) + centre[0]
    out["Y"] = u * np.sin(a) + v * np.cos(a) + centre[1]
    out["Z"] = z + centre[2]
    return out


def _ptd_ground_native(points: np.ndarray, parameters: dict) -> np.ndarray:
    noise = np.isin(points["Classification"], (7, 18))
    keep = np.flatnonzero(~noise)
    # PTD depends on input order (ties in its lowest point per cell and in
    # its height sort; shuffling a real tile changed 0.5% of labels):
    # classify in X, Y, Z order so the result does not depend on how the
    # points were stored or read.
    keep = keep[np.lexsort((points["Z"][keep], points["Y"][keep], points["X"][keep]))]
    xyz = np.column_stack((points["X"][keep], points["Y"][keep], points["Z"][keep]))
    ground = np.zeros(len(points), bool)
    if len(keep):
        ground[keep] = require_native().classify(xyz, **parameters)
    return ground


def _ptd_ground_pylasr(points: np.ndarray, parameters: dict) -> np.ndarray:
    fields = [
        n for n in ("X", "Y", "Z", "ReturnNumber", "NumberOfReturns") if n in points.dtype.names
    ]
    prepared = np.zeros(len(points), dtype=[*((n, points.dtype[n]) for n in fields),
                                             ("Classification", "u1")])  # fmt: skip
    for name in fields:
        prepared[name] = points[name]
    noise = np.isin(points["Classification"], (7, 18))
    prepared["Classification"] = np.where(noise, points["Classification"], 1)
    with tempfile.TemporaryDirectory(prefix="alsdb-ptd-") as tmp:
        source, classified = Path(tmp) / "in.las", Path(tmp) / "out.las"
        pdal.Pipeline(
            json.dumps([{"type": "writers.las", "filename": str(source), "scale_x": 0.001,
                         "scale_y": 0.001, "scale_z": 0.001, "offset_x": "auto",
                         "offset_y": "auto", "offset_z": "auto"}]),
            arrays=[prepared],
        ).execute()  # fmt: skip
        run_ptd(source, classified, **parameters)
        written, result = _read(source), _read(classified)
    if len(written) != len(points) or len(result) != len(points):
        raise RuntimeError("PTD changed the number of points")
    # The file was written in input order; lasR may reorder: match on rounded coordinates.
    keys = [n for n in ("ReturnNumber", "Z", "Y", "X") if n in written.dtype.names]
    order_in = np.lexsort(tuple(written[n] for n in keys))
    order_out = np.lexsort(tuple(result[n] for n in keys))
    for name in ("X", "Y", "Z"):
        if not np.allclose(written[name][order_in], result[name][order_out], rtol=0, atol=1e-6):
            raise RuntimeError("PTD output points do not match its input")
    ground = np.empty(len(points), bool)
    ground[order_in] = result["Classification"][order_out] == 2
    return ground & ~noise
