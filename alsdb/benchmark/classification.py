"""Ground-classification adapters and accuracy metrics."""

from __future__ import annotations

import json
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
