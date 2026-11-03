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
DEFAULTS = {"seed_resolution": 10.0, "angle": 30.0, "distance": 2.0, "spacing": 0.25}


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


def ptd_ground(points: np.ndarray, *, backend: str = "native", **parameters) -> np.ndarray:
    """Boolean ground mask of *points* (in their order) from PTD.

    *parameters* override :data:`DEFAULTS`. Noise (7, 18) is never ground and
    is left out of the ground search. *backend* is ``"native"`` (the vendored
    library) or ``"pylasr"``.
    """
    unknown = set(parameters) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown PTD parameters: {sorted(unknown)}")
    if backend not in ("native", "pylasr"):
        raise ValueError("backend must be 'native' or 'pylasr'")
    if not len(points):
        return np.zeros(0, bool)
    if backend == "native":
        noise = np.isin(points["Classification"], (7, 18))
        keep = np.flatnonzero(~noise)
        # PTD depends on input order (ties in its lowest point per cell and in
        # its height sort; shuffling a real tile changed 0.5% of labels):
        # classify in X, Y, Z order so the result does not depend on how the
        # points were stored or read.
        keep = keep[np.lexsort((points["Z"][keep], points["Y"][keep], points["X"][keep]))]
        xyz = np.column_stack((points["X"][keep], points["Y"][keep], points["Z"][keep]))
        ground = np.zeros(len(points), bool)
        ground[keep] = require_native().classify(xyz, **{**DEFAULTS, **parameters})
        return ground
    return _ptd_ground_pylasr(points, {**DEFAULTS, **parameters})


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
