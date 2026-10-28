"""Ground-classification adapters and accuracy metrics."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pdal

from alsdb.core.classification import ClassificationConfig, ground_stage


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
    """pyLASR progressive TIN densification parameters."""

    seed_resolution: float = 10.0
    angle: float = 20.0
    distance: float = 3.0
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


def run_pdal_ground_classifier(
    points: np.ndarray,
    config: ClassificationConfig,
    *,
    reset_classes: bool = True,
) -> ClassificationRun:
    """Run CSF, PMF, or SMRF on an in-memory point array."""
    if config.ground_method not in ("csf", "pmf", "smrf"):
        raise ValueError(f"{config.ground_method!r} is not a PDAL ground method")
    prepared = points.copy()
    if reset_classes:
        prepared["Classification"] = 1
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
    """Run PTD through the optional current-source pyLASR package."""
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
        filter=[""],
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
    "boundary_f1",
    "interior_f1",
)


def _append_file_result(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILE_RESULT_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _completed_file_runs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["file"], row["method"]) for row in csv.DictReader(handle) if row["status"] == "ok"
        }


def _file_classifier(
    method: str,
    input_path: Path,
    points: np.ndarray,
    work_dir: Path,
    *,
    lastools_executable: str,
    fastgc_executable: str,
) -> ClassificationRun:
    if method in ("csf", "pmf", "smrf"):
        return run_pdal_ground_classifier(points, ClassificationConfig(ground_method=method))
    neutral_input = work_dir / f"unclassified{input_path.suffix.lower()}"
    neutral_pipeline = [
        {"type": "readers.las", "filename": str(input_path)},
        {"type": "filters.assign", "value": "Classification = 1"},
        {"type": "writers.las", "filename": str(neutral_input), "forward": "all"},
    ]
    pdal.Pipeline(json.dumps(neutral_pipeline)).execute()
    if method == "ptd":
        output = work_dir / "ptd.laz"
        run = run_pylasr_ptd(neutral_input, output)
        return ClassificationRun(
            run.method, _ground_mask_from_las(output, points), run.wall_seconds
        )
    if method == "lastools":
        return run_lastools_lasground(
            neutral_input,
            work_dir / "lastools.laz",
            executable=lastools_executable,
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


def run_file_benchmark(
    laz_dir: str | Path,
    out_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("csf", "pmf", "smrf"),
    boundary_width: float = 5.0,
    tiles: int | None = None,
    lastools_executable: str = "lasground64",
    fastgc_executable: str = "fastgc",
) -> Path:
    """Compare classifiers with existing class 2 labels in LAS/LAZ files.

    Successful file/method pairs in ``results.csv`` are skipped on reruns.
    Failures are recorded and retried the next time the command is run.
    """
    if boundary_width <= 0:
        raise ValueError("boundary_width must be positive")
    valid = {"csf", "pmf", "smrf", "ptd", "fastgc", "lastools"}
    unknown = set(methods) - valid
    if unknown:
        raise ValueError(f"unknown classification methods: {sorted(unknown)}")
    laz_dir, out_dir = Path(laz_dir), Path(out_dir)
    files = sorted(path for path in laz_dir.rglob("*") if path.suffix.lower() in (".las", ".laz"))
    if tiles is not None:
        files = files[:tiles]
    if not files:
        raise FileNotFoundError(f"no LAS/LAZ files found in {laz_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.csv"
    completed = _completed_file_runs(results)
    for index, input_path in enumerate(files, 1):
        relative = str(input_path.relative_to(laz_dir))
        pending = [method for method in methods if (relative, method) not in completed]
        if not pending:
            continue
        points = _read_las(input_path)
        truth = np.asarray(points["Classification"] == 2)
        x0, x1 = float(points["X"].min()), float(points["X"].max())
        y0, y1 = float(points["Y"].min()), float(points["Y"].max())
        print(f"[{index}/{len(files)}] {relative}: {len(points):,} points", flush=True)
        for method in pending:
            print(f"  {method} ...", end="", flush=True)
            row = {"file": relative, "method": method, "n_points": len(points)}
            try:
                with tempfile.TemporaryDirectory(prefix=f"alsdb-{method}-", dir=out_dir) as tmp:
                    run = _file_classifier(
                        method,
                        input_path,
                        points,
                        Path(tmp),
                        lastools_executable=lastools_executable,
                        fastgc_executable=fastgc_executable,
                    )
                score = score_ground_classification(truth, run.is_ground)
                row.update(score.to_dict())
                row.update(status="ok", error="", wall_seconds=run.wall_seconds)
                if 2 * boundary_width < min(x1 - x0, y1 - y0):
                    strips = score_boundary_strips(
                        points, truth, run.is_ground, (x0, y0, x1, y1), boundary_width
                    )
                    row.update(
                        boundary_f1=strips["boundary"].f1,
                        interior_f1=strips["interior"].f1,
                    )
                print(f" F1={score.f1:.4f}, {run.wall_seconds:.2f}s", flush=True)
            except Exception as exc:  # noqa: BLE001 - one failed tool must not abort a long run
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f" ERROR: {exc}", flush=True)
            _append_file_result(results, row)
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
