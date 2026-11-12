"""Ground-classification adapters and accuracy metrics.

Classifiers compared (each cited where it is run):

- CSF, cloth simulation filter (Zhang et al. 2016), PMF, progressive
  morphological filter (Zhang et al. 2003) and SMRF, simple morphological
  filter (Pingel et al. 2013), as implemented in PDAL (Butler et al. 2021);
- PTD, progressive TIN densification (Axelsson 2000), lasR's implementation
  (Roussel, r-lidar/PTD) compiled into alsdb (:mod:`alsdb.methods.ptd`);
- LAStools ``lasground`` (rapidlasso GmbH), run as an external program.

Point labels are scored with the type I (ground rejected) and type II
(non-ground accepted) errors of Sithole & Vosselman (2004), and terrain
models from the predicted ground against the reference terrain model, overall
and by terrain slope and canopy cover.

References
----------
Axelsson, P. (2000). DEM generation from laser scanner data using adaptive TIN
    models. International Archives of Photogrammetry and Remote Sensing,
    33(B4), 110-117.
Butler, H., Chambers, B., Hartzell, P., & Glennie, C. (2021). PDAL: An open
    source library for the processing and analysis of point clouds. Computers
    & Geosciences, 148, 104680. https://doi.org/10.1016/j.cageo.2020.104680
Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). An improved simple
    morphological filter for the terrain classification of airborne LIDAR
    data. ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
    https://doi.org/10.1016/j.isprsjprs.2012.12.002
Sithole, G., & Vosselman, G. (2004). Experimental comparison of filter
    algorithms for bare-Earth extraction from airborne laser scanning point
    clouds. ISPRS Journal of Photogrammetry and Remote Sensing, 59(1-2),
    85-101. https://doi.org/10.1016/j.isprsjprs.2004.05.004
Zhang, K., Chen, S.-C., Whitman, D., Shyu, M.-L., Yan, J., & Zhang, C. (2003).
    A progressive morphological filter for removing nonground measurements
    from airborne LIDAR data. IEEE Transactions on Geoscience and Remote
    Sensing, 41(4), 872-882. https://doi.org/10.1109/TGRS.2003.810682
Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016). An
    easy-to-use airborne LiDAR data filtering method based on cloth
    simulation. Remote Sensing, 8(6), 501. https://doi.org/10.3390/rs8060501
lasR / r-lidar/PTD: Roussel, J.-R., https://github.com/r-lidar/lasR
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pdal

from alsdb.methods.classification import ClassificationConfig, ground_stage

NOISE_CLASSES = (7, 18)
# Reference points without a ground / non-ground decision are not scored:
# never classified (0), unclassified (1) and noise (7, 18).
UNSCORED_CLASSES = (0, 1, *NOISE_CLASSES)
STRATUM_CELL = 5.0  # m, cells for terrain slope and canopy cover
SLOPE_BREAKS = (0.0, 15.0, 30.0, 45.0, 90.0)  # degrees
COVER_BREAKS = (0.0, 0.5, 0.8, 0.95, 1.0)  # share of first returns above COVER_HEIGHT
COVER_HEIGHT = 2.0  # m above the reference terrain
DTM_RESOLUTION = 1.0  # m
# Full-density DTMs of different classifiers within this of each other count
# as agreeing; their median is then a reference (see build_dtm_references).
REFERENCE_TOLERANCE = 0.25  # m


@dataclass(frozen=True)
class ClassificationScore:
    """Point-level binary ground-classification scores."""

    true_ground: int
    true_non_ground: int
    predicted_ground: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    ground_fraction: float
    omission: float  # type I: ground rejected / true ground (Sithole & Vosselman 2004)
    commission: float  # type II: non-ground accepted / true non-ground
    total_error: float  # misclassified / all

    def to_dict(self) -> dict:
        """Return all score fields as a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ClassificationRun:
    """Ground mask and timing returned by one classifier."""

    method: str
    is_ground: np.ndarray
    wall_seconds: float


@dataclass(frozen=True)
class PTDParameters:
    """Progressive TIN densification parameters (Axelsson 2000).

    Defaults are alsdb's (:data:`alsdb.methods.ptd.DEFAULTS`): lowest point per
    *seed_resolution* cell as seeds, iteration *angle* (degrees) and
    *distance* (m), densification stops at triangles smaller than *spacing*
    (m); then the tilt ensemble, *tilt_runs* runs on copies tilted by *tilt*
    degrees, ground where at least *votes* runs agree. ``tilt_runs=1`` with
    angle 30 and distance 2 is lasR's ``classify_with_ptd`` default (pyLASR's
    own defaults differ: angle 20, distance 3).
    """

    seed_resolution: float = 10.0
    angle: float = 45.0
    distance: float = 1.0
    spacing: float = 0.25
    tilt_runs: int = 8
    tilt: float = 10.0
    votes: int = 3

    def __post_init__(self) -> None:
        """Validate the PTD and ensemble settings."""
        from alsdb.methods.ptd import check_ensemble

        check_ensemble(self.tilt_runs, self.tilt, self.votes)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else np.nan


def score_ground_classification(truth, prediction) -> ClassificationScore:
    """Compare Boolean ground masks with standard binary-classification metrics."""
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must be matching one-dimensional masks")
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    n = tp + fp + fn + tn
    return ClassificationScore(
        true_ground=int(truth.sum()),
        true_non_ground=int((~truth).sum()),
        predicted_ground=int(prediction.sum()),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        balanced_accuracy=(recall + specificity) / 2,
        ground_fraction=_ratio(tp + fn, n),
        omission=_ratio(fn, tp + fn),
        commission=_ratio(fp, fp + tn),
        total_error=_ratio(fp + fn, n),
    )


def classifier_input(points: np.ndarray) -> np.ndarray:
    """Copy of *points* with every class but noise (7, 18) reset to 1.

    Classifiers then start from scratch but still skip the reference's noise,
    as they would on a delivered survey.
    """
    prepared = points.copy()
    prepared["Classification"][~np.isin(prepared["Classification"], NOISE_CLASSES)] = 1
    # PDAL's ground filters refuse files where only some returns are numbered
    # 0: count those as single returns.
    if {"ReturnNumber", "NumberOfReturns"} <= set(prepared.dtype.names):
        rn, nr = prepared["ReturnNumber"], prepared["NumberOfReturns"]
        rn[rn == 0] = 1
        nr[:] = np.maximum(nr, rn)
    return prepared


def run_pdal_ground_classifier(
    points: np.ndarray,
    config: ClassificationConfig,
    *,
    reset_classes: bool = True,
) -> ClassificationRun:
    """Run CSF, PMF, or SMRF on an in-memory point array."""
    if config.ground_method not in ("csf", "pmf", "smrf"):
        raise ValueError(f"{config.ground_method!r} is not a PDAL ground method")
    prepared = classifier_input(points) if reset_classes else points.copy()
    start = time.perf_counter()
    pipeline = pdal.Pipeline(json.dumps([ground_stage(config)]), arrays=[prepared])
    pipeline.execute()
    result = pipeline.arrays[0]
    return ClassificationRun(
        method=config.ground_method,
        is_ground=np.asarray(result["Classification"] == 2),
        wall_seconds=time.perf_counter() - start,
    )


def _read_las(path: Path) -> np.ndarray:
    reader = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(path)}]))
    reader.execute()
    return reader.arrays[0]


def _ground_mask_from_las(path: Path, reference: np.ndarray | None = None) -> np.ndarray:
    classified = _read_las(path)
    mask = np.asarray(classified["Classification"] == 2)
    if reference is None:
        return mask
    if len(reference) != len(classified):
        raise ValueError(
            f"classifier changed the point count: {len(reference)} input, {len(classified)} output"
        )
    names = [
        name
        for name in ("X", "Y", "Z", "GpsTime", "ReturnNumber", "PointSourceId")
        if name in reference.dtype.names and name in classified.dtype.names
    ]
    if not {"X", "Y", "Z"}.issubset(names):
        raise ValueError("input and classified output must contain X, Y, and Z")

    def matches(name: str, left: np.ndarray, right: np.ndarray) -> bool:
        if name in ("X", "Y", "Z"):
            return bool(np.allclose(left, right, rtol=0, atol=1e-7))
        return bool(np.array_equal(left, right))

    if all(matches(name, reference[name], classified[name]) for name in names):
        return mask
    reference_order = np.lexsort(tuple(reference[name] for name in reversed(names)))
    classified_order = np.lexsort(tuple(classified[name] for name in reversed(names)))
    if any(
        not matches(name, reference[name][reference_order], classified[name][classified_order])
        for name in names
    ):
        raise ValueError("classifier output points do not match the input points")
    aligned = np.empty_like(mask)
    aligned[reference_order] = mask[classified_order]
    return aligned


def run_lastools_lasground(
    input_path: str | Path,
    output_path: str | Path,
    *,
    executable: str = "lasground64",
    options: tuple[str, ...] = (),
    timeout: float = 3600,
    reference: np.ndarray | None = None,
) -> ClassificationRun:
    """Run the external LAStools ground classifier."""
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"LAStools executable not found: {executable}")
    input_path, output_path = Path(input_path), Path(output_path)
    command = [resolved, "-i", str(input_path), "-o", str(output_path), *options, "-fail"]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    elapsed = time.perf_counter() - start
    text = result.stdout + result.stderr
    succeeded = result.returncode == 0 or (result.returncode == 1 and "ERROR" not in text)
    if not succeeded or not output_path.is_file():
        raise RuntimeError(f"lasground failed ({result.returncode}): {text[-4000:]}")
    return ClassificationRun(
        "lastools-lasground", _ground_mask_from_las(output_path, reference), elapsed
    )


FILE_RESULT_FIELDS = (
    "file",
    "method",
    "status",
    "error",
    "n_points",
    "wall_seconds",
    "true_ground",
    "true_non_ground",
    "predicted_ground",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "ground_fraction",
    "omission",
    "commission",
    "total_error",
    "boundary_f1",
    "interior_f1",
    "dtm_bias",
    "dtm_mae",
    "dtm_rmse",
    "dtm_nmad",
    "dtm_p95",
    "dtm_coverage",
    "n_scored",
    "parameters",
)

STRATA_FIELDS = (
    "file",
    "method",
    "parameters",
    "stratum",
    "label",
    "n_points",
    "true_ground",
    "ground_fraction",
    "omission",
    "commission",
    "total_error",
    "f1",
    "n_cells",
    "dtm_bias",
    "dtm_mae",
    "dtm_rmse",
    "dtm_p95",
    "dtm_coverage",
)


def _append_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _completed_file_runs(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["file"], row["method"], row.get("parameters", ""))
            for row in csv.DictReader(handle)
            if row["status"] == "ok"
        }


def _labels(values: np.ndarray, breaks: tuple[float, ...], scale: float = 1.0) -> np.ndarray:
    """Class label ("15-30") per value; "" where undefined. The last class includes its top."""
    index = np.clip(np.searchsorted(breaks, values, side="right") - 1, 0, len(breaks) - 2)
    names = np.array(
        [f"{breaks[i] * scale:g}-{breaks[i + 1] * scale:g}" for i in range(len(breaks) - 1)]
    )
    out = names[index].astype(object)
    out[~np.isfinite(values)] = ""
    return out


@dataclass
class ReferenceContext:
    """Per-file reference: scored points, terrain model, strata and DTM cells."""

    scored: np.ndarray  # points with a ground / non-ground reference label
    truth: np.ndarray  # reference ground (class 2)
    interior: np.ndarray  # points inside the boundary strip
    point_strata: dict[str, np.ndarray]  # stratum -> class label per point
    cells_xy: np.ndarray  # interior DTM cell centres
    cells_z: np.ndarray  # reference terrain there (NaN outside its TIN)
    cell_strata: dict[str, np.ndarray]


def reference_context(
    points: np.ndarray, bbox, boundary_width: float, reference: np.ndarray | None = None
) -> ReferenceContext:
    """Build reference labels, terrain, and slope and cover strata.

    Slope comes from the reference terrain TIN sampled every
    :data:`STRATUM_CELL` m; cover is the share of first returns more than
    :data:`COVER_HEIGHT` m above it in the same cells. Terrain and cover come
    from *reference* when given (e.g. the full-density original of a
    decimated copy), otherwise from *points*; point labels always come from
    *points*.
    """
    from alsdb.methods._terrain import TerrainModel

    classes = points["Classification"]
    scored = ~np.isin(classes, UNSCORED_CLASSES)
    if "Withheld" in points.dtype.names:
        scored &= ~points["Withheld"].astype(bool)
    truth = classes == 2
    x0, y0, x1, y1 = bbox
    width = boundary_width if 2 * boundary_width < min(x1 - x0, y1 - y0) else 0.0
    x, y = points["X"], points["Y"]
    interior = (x >= x0 + width) & (x < x1 - width) & (y >= y0 + width) & (y < y1 - width)
    surface = points if reference is None else reference
    terrain = TerrainModel(surface[surface["Classification"] == 2])

    c = STRATUM_CELL
    nx, ny = max(1, int(np.ceil((x1 - x0) / c))), max(1, int(np.ceil((y1 - y0) / c)))
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * c, y0 + (np.arange(ny) + 0.5) * c)
    z = terrain.evaluate(np.column_stack((gx.ravel(), gy.ravel())), distances=False)[0]
    z = z.reshape(ny, nx)
    if nx > 1 and ny > 1:
        dzdy, dzdx = np.gradient(z, c)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    else:
        slope = np.full_like(z, np.nan)

    def cell_of(px, py):
        col = np.clip(((px - x0) // c).astype(np.int64), 0, nx - 1)
        row = np.clip(((py - y0) // c).astype(np.int64), 0, ny - 1)
        return row * nx + col

    sx, sy = surface["X"], surface["Y"]
    first = np.flatnonzero(
        (surface["ReturnNumber"] == 1) & ~np.isin(surface["Classification"], NOISE_CLASSES)
    )
    ground_z = terrain.evaluate(np.column_stack((sx[first], sy[first])), distances=False)[0]
    canopy = surface["Z"][first] - ground_z > COVER_HEIGHT
    first_cell = cell_of(sx[first], sy[first])
    n_first = np.bincount(first_cell, minlength=nx * ny)
    n_canopy = np.bincount(first_cell[canopy], minlength=nx * ny)
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(n_first > 0, n_canopy / n_first, np.nan)
    cell_labels = {
        "slope": _labels(slope.ravel(), SLOPE_BREAKS),
        "cover": _labels(cover, COVER_BREAKS, scale=100),
    }
    point_cell = cell_of(x, y)

    r = DTM_RESOLUTION
    cx = np.arange(x0 + width + r / 2, x1 - width, r)
    cy = np.arange(y0 + width + r / 2, y1 - width, r)
    mx, my = np.meshgrid(cx, cy)
    cells_xy = np.column_stack((mx.ravel(), my.ravel()))
    cells_z = terrain.evaluate(cells_xy, distances=False)[0]
    dtm_cell = cell_of(cells_xy[:, 0], cells_xy[:, 1])
    return ReferenceContext(
        scored=scored,
        truth=truth,
        interior=interior,
        point_strata={k: v[point_cell] for k, v in cell_labels.items()},
        cells_xy=cells_xy,
        cells_z=cells_z,
        cell_strata={k: v[dtm_cell] for k, v in cell_labels.items()},
    )


@dataclass
class DTMReferences:
    """Full-density DTMs of every classifier run, on a scene's core cells."""

    dtms: dict[str, np.ndarray]  # run id -> DTM (flattened like ReferenceContext.cells_z)
    consensus: np.ndarray  # median of each method's default run
    agreed: np.ndarray  # cells where those runs lie within the tolerance


def run_id(method: str, parameters: str) -> str:
    """Stable short id of a method and its parameters JSON."""
    import hashlib

    return hashlib.sha1(f"{method}|{parameters}".encode()).hexdigest()[:12]


def load_dtm_references(scene) -> DTMReferences | None:
    """Load a scene's ``dtm_references.npz`` file when present."""
    path = Path(scene) / "dtm_references.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return DTMReferences(
        dtms={k[4:]: data[k] for k in data.files if k.startswith("run_")},
        consensus=data["consensus"],
        agreed=data["agreed"].astype(bool),
    )


def _reference_scores(cz, cells, references, rid):
    """Self, consensus and agreed / disputed errors of a candidate DTM on *cells*."""
    if references is None:
        return {}
    agreed = references.agreed & cells
    own = references.dtms.get(rid)
    own_err = terrain_error(own[cells], cz[cells]) if own is not None else {}
    cons = terrain_error(references.consensus[agreed], cz[agreed]) if agreed.any() else {}
    finite = cells & np.isfinite(references.consensus)
    return {
        "self_bias": own_err.get("bias", np.nan),
        "self_mae": own_err.get("mae", np.nan),
        "self_p95": own_err.get("p95", np.nan),
        "consensus_mae": cons.get("mae", np.nan),
        "consensus_bias": cons.get("bias", np.nan),
        "agreed_share": float(agreed.sum() / finite.sum()) if finite.any() else np.nan,
    }


def score_against_reference(
    ref: ReferenceContext,
    points: np.ndarray,
    is_ground: np.ndarray,
    references: DTMReferences | None = None,
    rid: str | None = None,
) -> tuple[dict, list[dict]]:
    """(file-level scores, rows per slope / cover class) for one classifier's ground mask.

    Point scores use the scored points; boundary / interior F1 and the strata
    use interior points only, so tile edges do not bias method comparisons.
    The DTM is a TIN of the predicted ground (noise excluded), compared with
    the reference terrain on interior 1 m cells. With *references* (full-density
    DTMs of every classifier), also against the same run at full density
    (``self_*``: robustness to density) and against the classifiers' median
    where they agree (``consensus_*``); *rid* is the run's :func:`run_id`.
    """
    from alsdb.methods._terrain import TerrainModel

    is_ground = np.asarray(is_ground, bool)
    truth, scored = ref.truth, ref.scored
    overall = score_ground_classification(truth[scored], is_ground[scored])
    inner, edge = scored & ref.interior, scored & ~ref.interior
    keep = is_ground & ~np.isin(points["Classification"], NOISE_CLASSES)
    candidate = TerrainModel(points[keep])
    cz = candidate.evaluate(ref.cells_xy, distances=False)[0]
    dtm = terrain_error(ref.cells_z, cz)
    row = {
        **overall.to_dict(),
        "n_scored": int(scored.sum()),
        "interior_f1": score_ground_classification(truth[inner], is_ground[inner]).f1,
        "boundary_f1": (
            score_ground_classification(truth[edge], is_ground[edge]).f1 if edge.any() else np.nan
        ),
        **{f"dtm_{k}": v for k, v in dtm.items()},
        **_reference_scores(cz, np.ones(len(cz), bool), references, rid),
    }
    strata = []
    groups = [("all", "all", inner, np.ones(len(cz), bool))]
    for kind in ("slope", "cover"):
        labels = [label for label in dict.fromkeys(ref.point_strata[kind][inner]) if label]
        for label in sorted(labels, key=lambda v: float(v.split("-")[0])):
            groups.append(
                (
                    kind,
                    label,
                    inner & (ref.point_strata[kind] == label),
                    ref.cell_strata[kind] == label,
                )
            )
    for kind, label, members, cells in groups:
        s = score_ground_classification(truth[members], is_ground[members])
        d = terrain_error(ref.cells_z[cells], cz[cells])
        strata.append(
            {
                "stratum": kind,
                "label": label,
                "n_points": int(members.sum()),
                "true_ground": s.true_ground,
                "ground_fraction": s.ground_fraction,
                "omission": s.omission,
                "commission": s.commission,
                "total_error": s.total_error,
                "f1": s.f1,
                "n_cells": int(cells.sum()),
                **{f"dtm_{k}": d[k] for k in ("bias", "mae", "rmse", "p95", "coverage")},
                **_reference_scores(cz, cells, references, rid),
            }
        )
    return row, strata


LASGROUND_OPTIONS = ("-ignore_class", "7", "18")


def method_parameters(method: str, ptd: PTDParameters | None = None) -> dict:
    """Return the method settings recorded with benchmark scores."""
    if method in ("csf", "pmf", "smrf"):
        stage = ground_stage(ClassificationConfig(ground_method=method))
        return {k: v for k, v in stage.items() if k not in ("type", "ignore")}
    if method == "ptd":
        return asdict(ptd or PTDParameters())
    if method == "lastools":
        return {"options": list(LASGROUND_OPTIONS)}
    raise ValueError(f"unknown classification method: {method}")


def _file_classifier(
    method: str,
    input_path: Path,
    points: np.ndarray,
    work_dir: Path,
    *,
    lastools_executable: str,
    ptd: PTDParameters | None = None,
) -> ClassificationRun:
    if method in ("csf", "pmf", "smrf"):
        return run_pdal_ground_classifier(points, ClassificationConfig(ground_method=method))
    if method == "ptd":
        # The vendored r-lidar/PTD alsdb itself runs (identical to pyLASR's on
        # identical coordinates, alsdb 9825924).
        from alsdb.methods.ptd import ptd_ground

        start = time.perf_counter()
        ground = ptd_ground(points, **asdict(ptd or PTDParameters()))
        return ClassificationRun("ptd", ground, time.perf_counter() - start)
    neutral_input = work_dir / f"unclassified{input_path.suffix.lower()}"
    neutral_pipeline = [
        {"type": "readers.las", "filename": str(input_path)},
        {
            "type": "filters.assign",
            "value": "Classification = 1 WHERE Classification != 7 && Classification != 18",
        },
        {"type": "writers.las", "filename": str(neutral_input), "forward": "all"},
    ]
    pdal.Pipeline(json.dumps(neutral_pipeline)).execute()
    if method == "lastools":
        return run_lastools_lasground(
            neutral_input,
            work_dir / "lastools.laz",
            executable=lastools_executable,
            options=LASGROUND_OPTIONS,
            reference=points,
        )
    raise ValueError(f"unknown classification method: {method}")


def _runs(methods, ptd):
    """Yield the method, optional PTD settings, and parameter JSON for each run."""
    valid = {"csf", "pmf", "smrf", "ptd", "lastools"}
    unknown = set(methods) - valid
    if unknown:
        raise ValueError(f"unknown classification methods: {sorted(unknown)}")
    return [
        (method, settings, json.dumps(method_parameters(method, settings), sort_keys=True))
        for method in methods
        for settings in (ptd if method == "ptd" else (None,))
    ]


def run_file_benchmark(
    laz_dir: str | Path,
    out_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("csf", "pmf", "smrf"),
    boundary_width: float = 5.0,
    tiles: int | None = None,
    lastools_executable: str = "lasground64",
    ptd: tuple[PTDParameters, ...] = (PTDParameters(),),
    reference_dir: str | Path | None = None,
) -> Path:
    """Compare ground classifiers with the class-2 labels of LAS/LAZ files.

    Writes one row per file, method and parameter set to ``results.csv``
    (type I / II errors, DTM error) and one per slope and cover class to
    ``strata.csv``. Classifiers see every class but noise reset to 1. With
    *reference_dir*, terrain and cover come from the file of the same name
    there (the full-density original). Finished runs are skipped on rerun.
    """
    if boundary_width <= 0:
        raise ValueError("boundary_width must be positive")
    valid = {"csf", "pmf", "smrf", "ptd", "lastools"}
    unknown = set(methods) - valid
    if unknown:
        raise ValueError(f"unknown classification methods: {sorted(unknown)}")
    runs = _runs(methods, ptd)
    laz_dir, out_dir = Path(laz_dir), Path(out_dir)
    files = sorted(path for path in laz_dir.rglob("*") if path.suffix.lower() in (".las", ".laz"))
    if tiles is not None:
        files = files[:tiles]
    if not files:
        raise FileNotFoundError(f"no LAS/LAZ files found in {laz_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.csv"
    strata_csv = out_dir / "strata.csv"
    completed = _completed_file_runs(results)
    for index, input_path in enumerate(files, 1):
        relative = str(input_path.relative_to(laz_dir))
        pending = [r for r in runs if (relative, r[0], r[2]) not in completed]
        if not pending:
            continue
        points = _read_las(input_path)
        bbox = (
            float(points["X"].min()),
            float(points["Y"].min()),
            float(points["X"].max()),
            float(points["Y"].max()),
        )
        print(f"[{index}/{len(files)}] {relative}: {len(points):,} points", flush=True)
        original = _read_las(Path(reference_dir) / input_path.name) if reference_dir else None
        ref = reference_context(points, bbox, boundary_width, original)
        del original
        for method, settings, parameters in pending:
            print(f"  {method} {parameters} ...", end="", flush=True)
            row = {
                "file": relative,
                "method": method,
                "parameters": parameters,
                "n_points": len(points),
            }
            strata = []
            try:
                with tempfile.TemporaryDirectory(prefix=f"alsdb-{method}-", dir=out_dir) as tmp:
                    run = _file_classifier(
                        method,
                        input_path,
                        points,
                        Path(tmp),
                        lastools_executable=lastools_executable,
                        ptd=settings,
                    )
                scores, strata = score_against_reference(ref, points, run.is_ground)
                row.update(scores, status="ok", error="", wall_seconds=run.wall_seconds)
                print(
                    f" F1={scores['f1']:.4f}, type I {scores['omission']:.3f}, "
                    f"type II {scores['commission']:.3f}, DTM MAE {scores['dtm_mae']:.2f} m, "
                    f"{run.wall_seconds:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - one failed tool must not abort a long run
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f" ERROR: {exc}", flush=True)
            for stratum in strata:
                stratum.update(file=relative, method=method, parameters=parameters)
            _append_rows(strata_csv, strata, STRATA_FIELDS)
            _append_rows(results, [row], FILE_RESULT_FIELDS)
    print(f"-> {results}", flush=True)
    return results


def terrain_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Error statistics for matching terrain rasters, ignoring missing cells."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate terrain rasters must have the same shape")
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if not valid.any():
        return {name: np.nan for name in ("bias", "mae", "rmse", "nmad", "p95", "coverage")}
    residual = candidate[valid] - reference[valid]
    median = np.median(residual)
    return {
        "bias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nmad": float(1.4826 * np.median(np.abs(residual - median))),
        "p95": float(np.percentile(np.abs(residual), 95)),
        "coverage": float(valid.sum() / np.isfinite(reference).sum()),
    }


# ---------------------------------------------------------------------------
# Benchmark scenes: synthetic, ULS and decimated ALS against their truth DTM
# ---------------------------------------------------------------------------

REFERENCE_FIELDS = (
    "self_bias",
    "self_mae",
    "self_p95",
    "consensus_mae",
    "consensus_bias",
    "agreed_share",
)
SCENE_RESULT_FIELDS = (*FILE_RESULT_FIELDS, *REFERENCE_FIELDS, "scenario")
SCENE_STRATA_FIELDS = (*STRATA_FIELDS, *REFERENCE_FIELDS, "scenario")


def reference_from_truth(points: np.ndarray, truth, cell: float = STRATUM_CELL) -> ReferenceContext:
    """Build a reference from a scene's 1 m truth DTM and point labels.

    Slope and cover strata come from the truth averaged over *cell* m blocks,
    inside the scored core only.
    """
    dtm = np.asarray(truth["dtm"], dtype=float)
    ny, nx = dtm.shape
    x0, y1 = float(truth["x0"]), float(truth["y1"])
    classes = points["Classification"]
    scored = ~np.isin(classes, UNSCORED_CLASSES)
    if "Withheld" in points.dtype.names:
        scored &= ~points["Withheld"].astype(bool)
    x, y = points["X"], points["Y"]
    interior = (x >= x0) & (x < x0 + nx) & (y > y1 - ny) & (y <= y1)

    b = max(1, round(cell))
    by, bx = max(1, ny // b), max(1, nx // b)

    def blocks(a):
        a = a[: by * b, : bx * b].reshape(by, min(b, ny), bx, min(b, nx))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=(1, 3))

    terrain = blocks(dtm)
    if by > 1 and bx > 1:
        dzdy, dzdx = np.gradient(terrain, float(b))
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    else:
        slope = np.full_like(terrain, np.nan)
    chm = np.asarray(truth["chm_centre"], dtype=float) if "chm_centre" in truth else None
    if chm is not None:
        canopy = np.where(np.isfinite(chm), (chm > COVER_HEIGHT).astype(float), np.nan)
        cover = blocks(canopy)
    else:
        cover = np.full_like(terrain, np.nan)
    labels = {"slope": _labels(slope.ravel(), SLOPE_BREAKS),
              "cover": _labels(cover.ravel(), COVER_BREAKS, scale=100)}  # fmt: skip

    def block_of(px, py):
        row = np.clip(np.floor((y1 - py) / b).astype(np.int64), 0, by - 1)
        col = np.clip(np.floor((px - x0) / b).astype(np.int64), 0, bx - 1)
        return row * bx + col

    point_block = block_of(x, y)
    point_strata = {}
    for kind, values in labels.items():
        point_strata[kind] = values[point_block]
        point_strata[kind][~interior] = ""
    cx, cy = np.meshgrid(x0 + np.arange(nx) + 0.5, y1 - np.arange(ny) - 0.5)
    cells_xy = np.column_stack((cx.ravel(), cy.ravel()))
    cell_block = block_of(cells_xy[:, 0], cells_xy[:, 1])
    return ReferenceContext(
        scored=scored,
        truth=classes == 2,
        interior=interior,
        point_strata=point_strata,
        cells_xy=cells_xy,
        cells_z=dtm.ravel(),
        cell_strata={kind: values[cell_block] for kind, values in labels.items()},
    )


def classify_scene(scene, runs, work_dir, lastools_executable="lasground64"):
    """Run every classifier on one benchmark scene.

    Returns a list of (results row, strata rows); ground masks are kept in
    ``<work_dir>/ground_masks/<scene>.npz``.
    """
    scene = Path(scene)
    laz = sorted(p for p in (scene / "laz").iterdir() if p.suffix.lower() in (".las", ".laz"))
    if len(laz) != 1:
        raise ValueError(f"{scene.name}: expected one LAS/LAZ file in laz/, found {len(laz)}")
    points = _read_las(laz[0])
    ref = reference_from_truth(points, np.load(scene / "truth.npz"))
    references = load_dtm_references(scene)
    meta = scene / "scene.json"
    scenario = json.dumps(json.loads(meta.read_text()).get("scenario", {})) if meta.exists() else ""
    out, masks = [], {}
    for method, settings, parameters in runs:
        row = {"file": scene.name, "method": method, "parameters": parameters,
               "n_points": len(points), "scenario": scenario}  # fmt: skip
        strata = []
        try:
            with tempfile.TemporaryDirectory(prefix=f"alsdb-{method}-", dir=work_dir) as tmp:
                run = _file_classifier(method, laz[0], points, Path(tmp),
                                       lastools_executable=lastools_executable, ptd=settings)  # fmt: skip
            scores, strata = score_against_reference(
                ref, points, run.is_ground, references, run_id(method, parameters)
            )
            row.update(scores, status="ok", error="", wall_seconds=run.wall_seconds)
            masks[run_id(method, parameters)] = (method, parameters, run.is_ground)
        except Exception as exc:  # noqa: BLE001 - one failed tool must not abort a long run
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        for stratum in strata:
            stratum.update(file=scene.name, method=method, parameters=parameters,
                           scenario=scenario)  # fmt: skip
        out.append((row, strata))
    if masks:
        _save_ground_masks(
            Path(work_dir) / "ground_masks" / f"{scene.name}.npz", len(points), masks
        )
    return out


def _save_ground_masks(path, n, masks):
    """Add {run id: (method, parameters, ground mask)} to *path* (bit-packed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(path)) if path.exists() else {}
    index = json.loads(str(data.pop("index"))) if "index" in data else {}
    data.pop("n", None)
    for rid, (method, parameters, mask) in masks.items():
        data[f"run_{rid}"] = np.packbits(np.asarray(mask, bool))
        index[rid] = [method, parameters]
    np.savez_compressed(path, n=n, index=json.dumps(index), **data)


def load_ground_masks(path) -> dict[str, tuple[str, str, np.ndarray]]:
    """{run id: (method, parameters, ground mask)} saved by :func:`classify_scene`."""
    data = np.load(path)
    n, index = int(data["n"]), json.loads(str(data["index"]))
    return {rid: (*index[rid], np.unpackbits(data[f"run_{rid}"])[:n].astype(bool)) for rid in index}


def run_scene_benchmark(
    scenes_dirs,
    out_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("csf", "pmf", "smrf"),
    ptd: tuple[PTDParameters, ...] = (PTDParameters(),),
    workers: int = 1,
    lastools_executable: str = "lasground64",
) -> Path:
    """Classify benchmark scenes and score their DTMs against the scene truth.

    Works on the scene folders of :mod:`~alsdb.benchmark.synthetic`,
    :mod:`~alsdb.benchmark.uls` and :mod:`~alsdb.benchmark.decimated`. Writes
    ``results.csv`` and ``strata.csv`` as :func:`run_file_benchmark`; finished
    runs are skipped on rerun.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    runs = _runs(methods, ptd)
    if isinstance(scenes_dirs, (str, Path)):
        scenes_dirs = [scenes_dirs]
    scenes = sorted(
        (p for d in scenes_dirs for p in Path(d).iterdir()
         if (p / "truth.npz").exists() and (p / "laz").is_dir()),
        key=lambda p: p.name,
    )  # fmt: skip
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, strata_csv = out_dir / "results.csv", out_dir / "strata.csv"
    completed = _completed_file_runs(results)
    todo = [(scene, [r for r in runs if (scene.name, r[0], r[2]) not in completed])
            for scene in scenes]  # fmt: skip
    todo = [(scene, pending) for scene, pending in todo if pending]
    print(f"{len(todo)} of {len(scenes)} scenes to classify", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(classify_scene, scene, pending, out_dir, lastools_executable): scene
                   for scene, pending in todo}  # fmt: skip
        for future in as_completed(futures):
            scene = futures[future]
            try:
                done = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad scene must not stop the run
                print(f"{scene.name}: FAILED {exc}", flush=True)
                continue
            for row, strata in done:
                _append_rows(strata_csv, strata, SCENE_STRATA_FIELDS)
                _append_rows(results, [row], SCENE_RESULT_FIELDS)
            summary = ", ".join(
                f"{row['method']} {row.get('dtm_mae', float('nan')):.2f}"
                if row["status"] == "ok" else f"{row['method']} ERROR"
                for row, _ in done
            )  # fmt: skip
            print(f"{scene.name}: DTM MAE {summary}", flush=True)
    print(f"-> {results}", flush=True)
    return results


def _crop_source(laz_dir, box, max_density=None) -> np.ndarray:
    """Points of the LAS/LAZ files in *laz_dir* inside *box*.

    With *max_density* (points/m2), files denser than that keep every n-th
    point in file order (``filters.decimation``), so ULS surveys of thousands
    of points/m2 fit in memory.
    """
    from alsdb.benchmark.uls import _files_and_header, _header

    files, _, _ = _files_and_header(laz_dir, box)
    bounds = f"([{box[0]},{box[2]}],[{box[1]},{box[3]}])"
    parts = []
    for f in files:
        stages = [str(f)]
        if max_density:
            h = _header(f)
            area = max((h["maxx"] - h["minx"]) * (h["maxy"] - h["miny"]), 1e-6)
            step = int(h["count"] / area // max_density)
            if step > 1:
                stages.append({"type": "filters.decimation", "step": step})
        stages.append({"type": "filters.crop", "bounds": bounds})
        pipeline = pdal.Pipeline(json.dumps(stages))
        pipeline.execute()
        parts.append(pipeline.arrays[0])
    names = set.intersection(*(set(a.dtype.names) for a in parts))
    return np.concatenate([a[[n for n in parts[0].dtype.names if n in names]] for a in parts])


def build_dtm_references(
    scenes_dirs,
    sources: dict[str, str | Path],
    *,
    methods: tuple[str, ...] = ("csf", "pmf", "smrf", "ptd"),
    ptd: tuple[PTDParameters, ...] = (PTDParameters(),),
    tolerance: float = REFERENCE_TOLERANCE,
    out_csv: str | Path | None = None,
    max_density: float | None = None,
) -> list[dict]:
    """Classify the full-density survey of every real scene area with each method.

    Each run's 1 m DTM goes to ``dtm_references.npz`` in the area's scenes. The
    median of each method's default (first) run, where the runs agree within
    *tolerance*, is the consensus reference. *max_density* (points/m2) thins
    denser surveys first. Returns, and writes to *out_csv*, one agreement row
    per area and stratum.
    """
    from alsdb.methods._terrain import TerrainModel

    runs = _runs(methods, ptd)
    defaults = {}
    for method, _, parameters in runs:
        defaults.setdefault(method, run_id(method, parameters))
    areas: dict[tuple, list[Path]] = {}
    for d in [scenes_dirs] if isinstance(scenes_dirs, (str, Path)) else scenes_dirs:
        for scene in sorted(Path(d).iterdir()):
            meta = scene / "scene.json"
            if not (scene / "truth.npz").exists() or not meta.exists():
                continue
            sc = json.loads(meta.read_text()).get("scenario", {})
            if "site" not in sc:
                continue
            key = (sc["site"], sc["x0"], sc["y0"], sc["core"], sc["margin"])
            areas.setdefault(key, []).append(scene)
    rows = []
    for (site, x0, y0, core, margin), members in areas.items():
        if site not in sources:
            print(f"{site} {x0} {y0}: no --source, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        truth = np.load(members[0] / "truth.npz")
        existing = load_dtm_references(members[0])
        if existing is not None and all((m / "dtm_references.npz").exists() for m in members):
            # Already built: report agreement from the stored DTMs.
            empty = np.zeros(0, dtype=[("X", "f8"), ("Y", "f8"), ("Classification", "u1")])
            ref = reference_from_truth(empty, truth)
            dtms, consensus, agreed = existing.dtms, existing.consensus, existing.agreed
            stack = np.stack([dtms[rid] for rid in defaults.values() if rid in dtms])
            spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
            rows += _agreement_rows(site, x0, y0, ref, dtms, defaults, consensus, agreed, spread)
            print(f"{site} {x0:.0f} {y0:.0f}: references exist, "
                  f"{agreed.mean():.0%} of cells agree", flush=True)  # fmt: skip
            continue
        box = (x0 - margin, y0 - margin, x0 + core + margin, y0 + core + margin)
        points = _crop_source(sources[site], box, max_density)
        dtms = {}
        with tempfile.TemporaryDirectory(prefix="alsdb-ref-") as tmp:
            laz = Path(tmp) / "area.laz"
            pdal.Pipeline(
                json.dumps([{"type": "writers.las", "filename": str(laz), "scale_x": 0.001,
                             "scale_y": 0.001, "scale_z": 0.001, "offset_x": "auto",
                             "offset_y": "auto", "offset_z": "auto"}]),
                arrays=[points],
            ).execute()  # fmt: skip
            # Classify the written 1 mm copy, so file-based tools return the same points.
            points = _read_las(laz)
            ref = reference_from_truth(points, truth)
            noise = np.isin(points["Classification"], NOISE_CLASSES)
            for method, settings, parameters in runs:
                run = _file_classifier(method, laz, points, Path(tmp), lastools_executable="lasground64",
                                       ptd=settings)  # fmt: skip
                model = TerrainModel(points[run.is_ground & ~noise])
                dtms[run_id(method, parameters)] = model.evaluate(ref.cells_xy, distances=False)[0]
        stack = np.stack([dtms[rid] for rid in defaults.values()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            consensus = np.nanmedian(stack, axis=0)
            spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        agreed = np.isfinite(stack).all(axis=0) & (spread <= tolerance)
        payload = {f"run_{k}": v.astype(np.float32) for k, v in dtms.items()}
        index = [{"id": run_id(m, prm), "method": m, "parameters": prm,
                  "default": defaults[m] == run_id(m, prm)} for m, _, prm in runs]  # fmt: skip
        for scene in members:
            np.savez_compressed(scene / "dtm_references.npz", consensus=consensus.astype(np.float32),
                                agreed=agreed, **payload)  # fmt: skip
            (scene / "dtm_references.json").write_text(
                json.dumps({"tolerance": tolerance, "runs": index}, indent=2)
            )
        rows += _agreement_rows(site, x0, y0, ref, dtms, defaults, consensus, agreed, spread)
        print(f"{site} {x0:.0f} {y0:.0f}: {len(points):,} points, "
              f"{agreed.mean():.0%} of cells agree within {tolerance} m, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    if out_csv and rows:
        _append_rows(Path(out_csv), rows, tuple(rows[0]))
    return rows


def _agreement_rows(site, x0, y0, ref, dtms, defaults, consensus, agreed, spread):
    """Agreement and distance from the truth DTM per stratum of one area."""
    rows = []
    groups = [("all", "all", np.ones(len(consensus), bool))]
    for kind in ("slope", "cover"):
        for label in sorted({v for v in ref.cell_strata[kind] if v},
                            key=lambda v: float(v.split("-")[0])):  # fmt: skip
            groups.append((kind, label, ref.cell_strata[kind] == label))
    for kind, label, cells in groups:
        ok = cells & np.isfinite(spread)
        row = {"site": site, "x0": x0, "y0": y0, "stratum": kind, "label": label,
               "n_cells": int(ok.sum()),
               "agreed_share": float(agreed[ok].mean()) if ok.any() else np.nan,
               "median_spread": float(np.median(spread[ok])) if ok.any() else np.nan,
               "consensus_vs_truth_mae": terrain_error(ref.cells_z[cells],
                                                       consensus[cells])["mae"]}  # fmt: skip
        for m, rid in defaults.items():
            row[f"{m}_vs_truth_mae"] = terrain_error(ref.cells_z[cells], dtms[rid][cells])["mae"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# End to end: CHMs on each classifier's own ground
# ---------------------------------------------------------------------------

CHM_FIELDS = ("file", "method", "parameters", "chm_method", "chm_mae", "chm_bias",
              "chm_within05", "chm_coverage", "mean_height_bias", "scenario")  # fmt: skip


def chm_on_ground(points, truth, ground, config=None):
    """CHM of the scene's core window with *ground* as the only ground (class 2).

    Every other non-noise point becomes vegetation (5), so the CHM differs from
    the provider's only through the ground. Scored against ``chm_centre``.
    """
    from alsdb.methods._surface import CHMConfig, reconstruct_chm

    config = config or CHMConfig(method="spikefree")
    arr = points.copy()
    noise = np.isin(arr["Classification"], NOISE_CLASSES)
    arr["Classification"][~noise] = 5
    arr["Classification"][np.asarray(ground, bool) & ~noise] = 2
    ref = np.asarray(truth["chm_centre"], dtype=float)
    ny, nx = ref.shape
    x0, y1 = float(truth["x0"]), float(truth["y1"])
    chm = reconstruct_chm(arr, (x0, y1 - ny, x0 + nx, y1), 1.0, config, diagnostics=False)["chm"]
    chm = np.asarray(chm, dtype=float)[:ny, :nx]
    ok = np.isfinite(chm) & np.isfinite(ref)
    if not ok.any():
        return {"chm_mae": np.nan, "chm_bias": np.nan, "chm_within05": np.nan,
                "chm_coverage": 0.0, "mean_height_bias": np.nan}  # fmt: skip
    d = chm[ok] - ref[ok]
    return {
        "chm_mae": float(np.abs(d).mean()),
        "chm_bias": float(d.mean()),
        "chm_within05": float(np.mean(np.abs(d) <= 0.5)),
        "chm_coverage": float(ok.sum() / np.isfinite(ref).sum()),
        "mean_height_bias": float(chm[ok].mean() - ref[ok].mean()),
    }


def _scene_chm_rows(scene, masks_path, done):
    from alsdb.methods._surface import GROUND_CLASSES

    scene = Path(scene)
    laz = sorted(p for p in (scene / "laz").iterdir() if p.suffix.lower() in (".las", ".laz"))
    points = _read_las(laz[0])
    truth = np.load(scene / "truth.npz")
    meta = scene / "scene.json"
    scenario = json.dumps(json.loads(meta.read_text()).get("scenario", {})) if meta.exists() else ""
    runs = {"provider": ("provider", "", np.isin(points["Classification"], GROUND_CLASSES))}
    runs.update(load_ground_masks(masks_path))
    rows = []
    for method, parameters, ground in runs.values():
        if (scene.name, method, parameters) in done:
            continue
        if len(ground) != len(points):
            raise ValueError(f"{scene.name}: ground mask of {method} does not match the points")
        rows.append({"file": scene.name, "method": method, "parameters": parameters,
                     "chm_method": "spikefree", "scenario": scenario,
                     **chm_on_ground(points, truth, ground)})  # fmt: skip
    return rows


def run_scene_chm(scenes_dirs, out_dir, *, workers: int = 1) -> Path:
    """Spike-free CHMs on each classifier's ground, for the scenes in *out_dir*.

    Uses the ground masks of :func:`run_scene_benchmark` and writes
    ``<out_dir>/chm.csv`` (one row per scene and run, plus the provider's
    ground). Needs the native spike-free module.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(out_dir)
    out_csv = out_dir / "chm.csv"
    done = set()
    if out_csv.exists():
        with out_csv.open(newline="") as handle:
            done = {(r["file"], r["method"], r["parameters"]) for r in csv.DictReader(handle)}
    if isinstance(scenes_dirs, (str, Path)):
        scenes_dirs = [scenes_dirs]
    scenes = {
        p.name: p for d in scenes_dirs for p in Path(d).iterdir() if (p / "truth.npz").exists()
    }
    todo = [(scenes[m.stem], m) for m in sorted((out_dir / "ground_masks").glob("*.npz"))
            if m.stem in scenes]  # fmt: skip
    print(f"{len(todo)} classified scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scene_chm_rows, scene, masks, done): scene for scene, masks in todo}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                rows = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad scene must not stop the run
                print(f"{scene.name}: FAILED {exc}", flush=True)
                continue
            if rows:
                _append_rows(out_csv, rows, CHM_FIELDS)
                print(f"{scene.name}: " + ", ".join(f"{r['method']} {r['chm_mae']:.2f}" for r in rows),
                      flush=True)  # fmt: skip
    return out_csv
