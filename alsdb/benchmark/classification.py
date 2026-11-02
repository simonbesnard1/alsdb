"""Ground-classification adapters and accuracy metrics.

Classifiers compared (each cited where it is run):

- CSF, cloth simulation filter (Zhang et al. 2016), PMF, progressive
  morphological filter (Zhang et al. 2003) and SMRF, simple morphological
  filter (Pingel et al. 2013), as implemented in PDAL (Butler et al. 2021);
- PTD, progressive TIN densification (Axelsson 2000), as implemented in lasR
  and its Python binding pyLASR (Roussel et al.);
- LAStools ``lasground`` (rapidlasso GmbH) and FAST-GC (Fareed et al.),
  run as external programs.

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
lasR / pyLASR: Roussel, J.-R., https://github.com/r-lidar/lasR
FAST-GC: Fareed, N. et al., https://github.com/nadeemfareed/FAST-GC
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

from alsdb.core.classification import ClassificationConfig, ground_stage

NOISE_CLASSES = (7, 18)
# Reference points without a ground / non-ground decision are not scored:
# never classified (0), unclassified (1) and noise (7, 18).
UNSCORED_CLASSES = (0, 1, *NOISE_CLASSES)
STRATUM_CELL = 5.0  # m, cells for terrain slope and canopy cover
SLOPE_BREAKS = (0.0, 15.0, 30.0, 45.0, 90.0)  # degrees
COVER_BREAKS = (0.0, 0.5, 0.8, 0.95, 1.0)  # share of first returns above COVER_HEIGHT
COVER_HEIGHT = 2.0  # m above the reference terrain
DTM_RESOLUTION = 1.0  # m


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

    Defaults are lasR's (``classify_with_ptd`` in R; pyLASR's own defaults
    differ: angle 20, distance 3): lowest point per *seed_resolution* cell as
    seeds, iteration *angle* (degrees; lasR suggests ~20 for flat, ~40 for
    mountainous terrain) and *distance* (m), densification stops at
    triangles smaller than *spacing* (m).
    """

    seed_resolution: float = 10.0
    angle: float = 30.0
    distance: float = 2.0
    spacing: float = 0.25


@dataclass(frozen=True)
class SyntheticClassificationScenario:
    """Deterministic synthetic point cloud with exact ground labels."""

    terrain: str = "rolling"
    size: float = 100.0
    density: float = 2.0
    ground_fraction: float = 0.3
    low_vegetation_fraction: float = 0.2
    noise_fraction: float = 0.002
    seed: int = 0

    def __post_init__(self) -> None:
        if self.terrain not in ("flat", "rolling", "steep", "rough"):
            raise ValueError("terrain must be flat, rolling, steep, or rough")
        if self.size <= 0 or self.density <= 0:
            raise ValueError("size and density must be positive")
        fractions = (self.ground_fraction, self.low_vegetation_fraction, self.noise_fraction)
        if any(not 0 <= value <= 1 for value in fractions):
            raise ValueError("scenario fractions must be in [0, 1]")


def _terrain_height(x: np.ndarray, y: np.ndarray, terrain: str) -> np.ndarray:
    if terrain == "flat":
        return np.full_like(x, 100.0)
    if terrain == "rolling":
        return 100 + 2.5 * np.sin(x / 18) + 1.5 * np.cos(y / 24)
    if terrain == "steep":
        return 100 + 0.45 * x + 0.12 * y + 2 * np.sin(y / 12)
    return 100 + 0.18 * x + 3 * np.sin(x / 8) * np.cos(y / 11)


def synthetic_points(
    scenario: SyntheticClassificationScenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points and an exact Boolean ground mask for *scenario*."""
    rng = np.random.default_rng(scenario.seed)
    count = max(4, round(scenario.size**2 * scenario.density))
    dtype = [
        ("X", "f8"),
        ("Y", "f8"),
        ("Z", "f8"),
        ("Classification", "u1"),
        ("ReturnNumber", "u1"),
        ("NumberOfReturns", "u1"),
    ]
    points = np.zeros(count, dtype=dtype)
    points["X"], points["Y"] = rng.uniform(0, scenario.size, (2, count))
    terrain = _terrain_height(points["X"], points["Y"], scenario.terrain)
    truth = rng.random(count) < scenario.ground_fraction
    nonground = ~truth
    low = nonground & (rng.random(count) < scenario.low_vegetation_fraction)
    height = np.zeros(count)
    height[truth] = rng.normal(0, 0.03, truth.sum())
    height[low] = rng.uniform(0.15, 2.0, low.sum())
    height[nonground & ~low] = rng.uniform(2.0, 30.0, (nonground & ~low).sum())
    noise = rng.random(count) < scenario.noise_fraction
    height[noise] = rng.choice((-1, 1), noise.sum()) * rng.uniform(10, 40, noise.sum())
    truth[noise] = False
    points["Z"] = terrain + height
    points["Classification"] = 1
    points["ReturnNumber"] = points["NumberOfReturns"] = 1
    return points, truth


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


def score_boundary_strips(
    points: np.ndarray,
    truth,
    prediction,
    bbox: tuple[float, float, float, float],
    width: float,
) -> dict[str, ClassificationScore]:
    """Score points inside and outside a strip along *bbox* edges."""
    if not np.isfinite(width) or width <= 0:
        raise ValueError("boundary width must be finite and positive")
    x0, y0, x1, y1 = bbox
    if 2 * width >= min(x1 - x0, y1 - y0):
        raise ValueError("boundary width leaves no interior region")
    x, y = points["X"], points["Y"]
    edge = (x < x0 + width) | (x >= x1 - width) | (y < y0 + width) | (y >= y1 - width)
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    if truth.shape != edge.shape or prediction.shape != edge.shape:
        raise ValueError("point and classification arrays must have matching lengths")
    return {
        "boundary": score_ground_classification(truth[edge], prediction[edge]),
        "interior": score_ground_classification(truth[~edge], prediction[~edge]),
    }


def classifier_input(points: np.ndarray) -> np.ndarray:
    """Copy of *points* with every class but noise (7, 18) reset to 1.

    Classifiers then start from scratch but still skip the reference's noise,
    as they would on a delivered survey.
    """
    prepared = points.copy()
    prepared["Classification"][~np.isin(prepared["Classification"], NOISE_CLASSES)] = 1
    # PDAL's ground filters refuse files where only some returns are numbered 0
    # (seen in Robson Creek); count those as single returns.
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


def run_synthetic_grid(
    scenarios: list[SyntheticClassificationScenario],
    methods: tuple[str, ...] = ("csf", "pmf", "smrf"),
    *,
    boundary_width: float = 5.0,
) -> list[dict]:
    """Run PDAL classifiers over a synthetic scenario grid."""
    rows = []
    for scenario in scenarios:
        points, truth = synthetic_points(scenario)
        bbox = (0.0, 0.0, scenario.size, scenario.size)
        for method in methods:
            run = run_pdal_ground_classifier(
                points,
                ClassificationConfig(ground_method=method),
            )
            overall = score_ground_classification(truth, run.is_ground)
            strips = score_boundary_strips(
                points,
                truth,
                run.is_ground,
                bbox,
                min(boundary_width, scenario.size / 4),
            )
            rows.append(
                {
                    **asdict(scenario),
                    "method": method,
                    "wall_seconds": run.wall_seconds,
                    **overall.to_dict(),
                    "boundary_f1": strips["boundary"].f1,
                    "interior_f1": strips["interior"].f1,
                }
            )
    return rows


def run_pylasr_ptd(
    input_path: str | Path,
    output_path: str | Path,
    parameters: PTDParameters | None = None,
) -> ClassificationRun:
    """Run PTD (Axelsson 2000) through pyLASR built from lasR source; noise is skipped."""
    try:
        import pylasr
    except ImportError as exc:
        raise ImportError(
            "PTD benchmarking requires pyLASR built from current lasR source; "
            "the PyPI 0.17.4 wheel does not expose classify_with_ptd"
        ) from exc

    if not hasattr(pylasr, "classify_with_ptd"):
        raise ImportError("Installed pyLASR does not expose classify_with_ptd")
    parameters = parameters or PTDParameters()
    input_path, output_path = Path(input_path), Path(output_path)
    pipeline = pylasr.classify_with_ptd(
        res=parameters.seed_resolution,
        angle=parameters.angle,
        distance=parameters.distance,
        spacing=parameters.spacing,
        classification=2,
        filter=["Classification %out% 7 18"],  # lasR drop_noise()
    ) + pylasr.write_las(str(output_path))
    start = time.perf_counter()
    result = pipeline.execute(str(input_path))
    elapsed = time.perf_counter() - start
    if not result.get("success") or not output_path.is_file():
        raise RuntimeError(f"pyLASR PTD failed: {result}")
    reader = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(output_path)}]))
    reader.execute()
    classified = reader.arrays[0]
    return ClassificationRun(
        method="ptd-pylasr",
        is_ground=np.asarray(classified["Classification"] == 2),
        wall_seconds=elapsed,
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


def run_fastgc(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    executable: str = "fastgc",
    sensor_mode: str = "ALS",
    timeout: float = 3600,
    reference: np.ndarray | None = None,
) -> ClassificationRun:
    """Run FAST-GC as an isolated external classifier."""
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"FAST-GC executable not found: {executable}")
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        resolved,
        "--in_path",
        str(input_path),
        "--out_dir",
        str(output_dir),
        "--sensor_mode",
        sensor_mode,
        "--products",
        "FAST_GC",
    ]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(
            f"FAST-GC failed ({result.returncode}): {(result.stdout + result.stderr)[-4000:]}"
        )
    outputs = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".las", ".laz")
    ]
    preferred = [
        path for path in outputs if "class" in path.name.lower() or "fast_gc" in path.name.lower()
    ]
    candidates = preferred or outputs
    if len(candidates) != 1:
        raise RuntimeError(f"Could not identify one FAST-GC classified output: {candidates}")
    return ClassificationRun("fastgc", _ground_mask_from_las(candidates[0], reference), elapsed)


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
    """Reference labels, terrain and slope / cover strata of one classified file.

    Slope comes from the reference terrain TIN sampled every
    :data:`STRATUM_CELL` m; cover is the share of first returns more than
    :data:`COVER_HEIGHT` m above it in the same cells. Terrain and cover come
    from *reference* when given (e.g. the full-density original of a
    decimated copy), otherwise from *points*; point labels always come from
    *points*.
    """
    from alsdb.processing._terrain import TerrainModel

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


def score_against_reference(
    ref: ReferenceContext, points: np.ndarray, is_ground: np.ndarray
) -> tuple[dict, list[dict]]:
    """(file-level scores, rows per slope / cover class) for one classifier's ground mask.

    Point scores use the scored points; boundary / interior F1 and the strata
    use interior points only, so tile edges do not bias method comparisons.
    The DTM is a TIN of the predicted ground (noise excluded), compared with
    the reference terrain on interior 1 m cells.
    """
    from alsdb.processing._terrain import TerrainModel

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
            }
        )
    return row, strata


LASGROUND_OPTIONS = ("-ignore_class", "7", "18")


def method_parameters(method: str, ptd: PTDParameters | None = None) -> dict:
    """Settings a method runs with, recorded next to its scores."""
    if method in ("csf", "pmf", "smrf"):
        stage = ground_stage(ClassificationConfig(ground_method=method))
        return {k: v for k, v in stage.items() if k not in ("type", "ignore")}
    if method == "ptd":
        return asdict(ptd or PTDParameters())
    if method == "lastools":
        return {"options": list(LASGROUND_OPTIONS)}
    if method == "fastgc":
        return {"sensor_mode": "ALS"}
    raise ValueError(f"unknown classification method: {method}")


def _file_classifier(
    method: str,
    input_path: Path,
    points: np.ndarray,
    work_dir: Path,
    *,
    lastools_executable: str,
    fastgc_executable: str,
    ptd: PTDParameters | None = None,
) -> ClassificationRun:
    if method in ("csf", "pmf", "smrf"):
        return run_pdal_ground_classifier(points, ClassificationConfig(ground_method=method))
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
    if method == "ptd":
        output = work_dir / "ptd.laz"
        run = run_pylasr_ptd(neutral_input, output, ptd)
        return ClassificationRun(
            run.method, _ground_mask_from_las(output, points), run.wall_seconds
        )
    if method == "lastools":
        return run_lastools_lasground(
            neutral_input,
            work_dir / "lastools.laz",
            executable=lastools_executable,
            options=LASGROUND_OPTIONS,
            reference=points,
        )
    if method == "fastgc":
        return run_fastgc(
            neutral_input,
            work_dir / "fastgc",
            executable=fastgc_executable,
            reference=points,
        )
    raise ValueError(f"unknown classification method: {method}")


def _runs(methods, ptd):
    """(method, PTD settings or None, parameters JSON) for every run requested."""
    valid = {"csf", "pmf", "smrf", "ptd", "fastgc", "lastools"}
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
    fastgc_executable: str = "fastgc",
    ptd: tuple[PTDParameters, ...] = (PTDParameters(),),
    reference_dir: str | Path | None = None,
) -> Path:
    """Compare classifiers with existing class 2 labels in LAS/LAZ files.

    One row per file, method and parameter set goes to ``results.csv``
    (point scores, type I / II errors, DTM error), and one row per slope and
    cover class to ``strata.csv``. Each of the *ptd* parameter sets is a
    separate run. Classifiers see every class reset to 1 except noise; the
    reference's unclassified and noise points are not scored. With
    *reference_dir*, the reference terrain and cover of each file come from
    the file of the same name there (the full-density original of decimated
    copies), so DTM errors at every density are against the same terrain.

    Successful runs are skipped on reruns. Failures are recorded and retried
    the next time the command is run.
    """
    if boundary_width <= 0:
        raise ValueError("boundary_width must be positive")
    valid = {"csf", "pmf", "smrf", "ptd", "fastgc", "lastools"}
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
                        fastgc_executable=fastgc_executable,
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

SCENE_RESULT_FIELDS = (*FILE_RESULT_FIELDS, "scenario")
SCENE_STRATA_FIELDS = (*STRATA_FIELDS, "scenario")


def reference_from_truth(points: np.ndarray, truth, cell: float = STRATUM_CELL) -> ReferenceContext:
    """Reference of a benchmark scene: the scene's 1 m truth DTM, labels from its points.

    *truth* is a scene's ``truth.npz`` (``dtm`` and ``chm_centre`` north-up from
    ``x0``, ``y1``): the exact terrain for synthetic scenes, the full-density
    terrain for ULS and decimated ALS. Slope and cover strata come from the
    truth averaged over *cell* m blocks; only points and cells of the scored
    core window are stratified.
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


def classify_scene(scene, runs, work_dir, lastools_executable="lasground64",
                   fastgc_executable="fastgc"):  # fmt: skip
    """[(results row, strata rows)] for each run on one benchmark scene."""
    scene = Path(scene)
    laz = sorted(p for p in (scene / "laz").iterdir() if p.suffix.lower() in (".las", ".laz"))
    if len(laz) != 1:
        raise ValueError(f"{scene.name}: expected one LAS/LAZ file in laz/, found {len(laz)}")
    points = _read_las(laz[0])
    ref = reference_from_truth(points, np.load(scene / "truth.npz"))
    meta = scene / "scene.json"
    scenario = json.dumps(json.loads(meta.read_text()).get("scenario", {})) if meta.exists() else ""
    out = []
    for method, settings, parameters in runs:
        row = {"file": scene.name, "method": method, "parameters": parameters,
               "n_points": len(points), "scenario": scenario}  # fmt: skip
        strata = []
        try:
            with tempfile.TemporaryDirectory(prefix=f"alsdb-{method}-", dir=work_dir) as tmp:
                run = _file_classifier(method, laz[0], points, Path(tmp),
                                       lastools_executable=lastools_executable,
                                       fastgc_executable=fastgc_executable, ptd=settings)  # fmt: skip
            scores, strata = score_against_reference(ref, points, run.is_ground)
            row.update(scores, status="ok", error="", wall_seconds=run.wall_seconds)
        except Exception as exc:  # noqa: BLE001 - one failed tool must not abort a long run
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        for stratum in strata:
            stratum.update(file=scene.name, method=method, parameters=parameters,
                           scenario=scenario)  # fmt: skip
        out.append((row, strata))
    return out


def run_scene_benchmark(
    scenes_dirs,
    out_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("csf", "pmf", "smrf"),
    ptd: tuple[PTDParameters, ...] = (PTDParameters(),),
    workers: int = 1,
    lastools_executable: str = "lasground64",
    fastgc_executable: str = "fastgc",
) -> Path:
    """Classify the points of benchmark scenes and score the DTMs against their truth.

    Works on the scene folders of :mod:`alsdb.benchmark.synthetic`,
    :mod:`~alsdb.benchmark.uls` and :mod:`~alsdb.benchmark.decimated`
    (``laz/`` with one file, ``truth.npz``, ``scene.json``). The reference DTM
    is the scene's truth: exact for synthetic scenes, the full-density survey
    for ULS and decimated ALS. Point labels are scored against the scene's own
    classes (HELIOS++ labels, or the survey provider's). Rows go to
    ``results.csv`` and ``strata.csv`` as in :func:`run_file_benchmark`, with
    the scene's scenario; finished runs are skipped on rerun.
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
        futures = {pool.submit(classify_scene, scene, pending, out_dir, lastools_executable,
                               fastgc_executable): scene for scene, pending in todo}  # fmt: skip
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
