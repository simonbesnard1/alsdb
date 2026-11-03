"""Validated configuration and PDAL stages for point classification.

Ground filters are PDAL's (Butler et al. 2021) implementations of CSF (Zhang
et al. 2016), PMF (Zhang et al. 2003) and SMRF (Pingel et al. 2013), and
progressive TIN densification (PTD; Axelsson 2000) through the optional
pyLASR package (:mod:`alsdb.core.ptd`). CSF is the default: on the steep and
tropical benchmark sites it held up across pulse densities, where SMRF (better
only at full density) failed below ~2 pulses/m2; PTD gave the lowest errors
from ~1 pulse/m2 up and is recommended where pyLASR is installed
(doc/benchmarking.md). The
optional noise stage is PDAL's extended local minimum filter (Chen et al.
2012) followed by a radius outlier filter.

References
----------
Axelsson, P. (2000). DEM generation from laser scanner data using adaptive TIN
    models. International Archives of Photogrammetry and Remote Sensing,
    33(B4), 110-117.
Butler, H., Chambers, B., Hartzell, P., & Glennie, C. (2021). PDAL: An open
    source library for the processing and analysis of point clouds. Computers
    & Geosciences, 148, 104680. https://doi.org/10.1016/j.cageo.2020.104680
Chen, Z., Devereux, B., Gao, B., & Amable, G. (2012). Upward-fusion urban DTM
    generating method using airborne Lidar data. ISPRS Journal of
    Photogrammetry and Remote Sensing, 72, 121-130.
    https://doi.org/10.1016/j.isprsjprs.2012.07.001
Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). An improved simple
    morphological filter for the terrain classification of airborne LIDAR
    data. ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
    https://doi.org/10.1016/j.isprsjprs.2012.12.002
Zhang, K., Chen, S.-C., Whitman, D., Shyu, M.-L., Yan, J., & Zhang, C. (2003).
    A progressive morphological filter for removing nonground measurements
    from airborne LIDAR data. IEEE Transactions on Geoscience and Remote
    Sensing, 41(4), 872-882. https://doi.org/10.1109/TGRS.2003.810682
Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016). An
    easy-to-use airborne LiDAR data filtering method based on cloth
    simulation. Remote Sensing, 8(6), 501. https://doi.org/10.3390/rs8060501
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ClassificationConfig:
    """Configuration for optional ingestion-time ground and vegetation classification."""

    enabled: bool = True
    ground_method: Literal["csf", "pmf", "smrf", "ptd"] = "csf"
    denoise: bool = False
    outlier_radius: float = 1.0
    outlier_min_neighbors: int = 2
    csf_resolution: float = 0.5
    csf_threshold: float = 0.5
    csf_rigidness: int = 1
    pmf_max_window_size: float = 33.0
    pmf_slope: float = 1.0
    pmf_initial_distance: float = 0.15
    pmf_max_distance: float = 2.5
    smrf_cell: float = 1.0
    smrf_scalar: float = 1.25
    smrf_slope: float = 0.15
    smrf_threshold: float = 0.5
    smrf_window: float = 18.0
    smrf_cut: float = 0.0
    smrf_returns: tuple[str, ...] = ("last", "only")
    # PTD (lasR classify_with_ptd defaults): seed cell, iteration angle and distance,
    # minimum triangle size.
    ptd_seed_resolution: float = 10.0
    ptd_angle: float = 30.0
    ptd_distance: float = 2.0
    ptd_spacing: float = 0.25
    hag_neighbors: int = 10
    allow_hag_extrapolation: bool = True
    vegetation_low: float = 0.5
    vegetation_medium: float = 2.0
    vegetation_high: float = 5.0
    source_classes: tuple[int, ...] = (1,)
    preserve_existing: bool = True
    buffer: float = 0.0

    def __post_init__(self) -> None:
        if self.ground_method not in ("csf", "pmf", "smrf", "ptd"):
            raise ValueError("ground_method must be 'csf', 'pmf', 'smrf', or 'ptd'")
        positive = {
            "outlier_radius": self.outlier_radius,
            "csf_resolution": self.csf_resolution,
            "csf_threshold": self.csf_threshold,
            "pmf_max_window_size": self.pmf_max_window_size,
            "pmf_slope": self.pmf_slope,
            "pmf_initial_distance": self.pmf_initial_distance,
            "pmf_max_distance": self.pmf_max_distance,
            "smrf_cell": self.smrf_cell,
            "smrf_scalar": self.smrf_scalar,
            "smrf_slope": self.smrf_slope,
            "smrf_threshold": self.smrf_threshold,
            "smrf_window": self.smrf_window,
            "ptd_seed_resolution": self.ptd_seed_resolution,
            "ptd_angle": self.ptd_angle,
            "ptd_distance": self.ptd_distance,
            "ptd_spacing": self.ptd_spacing,
        }
        invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"Classification parameters must be finite and positive: {invalid}")
        if self.outlier_min_neighbors < 1 or self.hag_neighbors < 1 or self.csf_rigidness < 1:
            raise ValueError("Neighbor counts and CSF rigidness must be positive integers")
        thresholds = (self.vegetation_low, self.vegetation_medium, self.vegetation_high)
        if not np.isfinite(thresholds).all() or not thresholds[0] < thresholds[1] < thresholds[2]:
            raise ValueError("Vegetation thresholds must be finite and strictly increasing")
        if not self.source_classes or any(not 0 <= value <= 255 for value in self.source_classes):
            raise ValueError("source_classes must contain LAS class codes in the range 0..255")
        valid_returns = {"first", "last", "intermediate", "only"}
        if not self.smrf_returns or not set(self.smrf_returns) <= valid_returns:
            raise ValueError(f"smrf_returns must contain values from {sorted(valid_returns)}")
        if not np.isfinite(self.smrf_cut) or self.smrf_cut < 0:
            raise ValueError("smrf_cut must be finite and nonnegative")
        if not np.isfinite(self.buffer) or self.buffer < 0:
            raise ValueError("buffer must be finite and nonnegative")

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation for provenance metadata."""
        values = asdict(self)
        values["source_classes"] = list(self.source_classes)
        values["smrf_returns"] = list(self.smrf_returns)
        if self.ground_method != "ptd":
            # Added after classified arrays were written: keep other methods'
            # fingerprints (and so their manifests) unchanged.
            for name in ("ptd_seed_resolution", "ptd_angle", "ptd_distance", "ptd_spacing"):
                values.pop(name)
        return values

    @property
    def ptd_parameters(self) -> dict:
        """Keyword arguments for :func:`alsdb.core.ptd.ptd_ground`."""
        return {
            "seed_resolution": self.ptd_seed_resolution,
            "angle": self.ptd_angle,
            "distance": self.ptd_distance,
            "spacing": self.ptd_spacing,
        }

    @property
    def active(self) -> bool:
        """Whether ingestion changes classification labels."""
        return self.enabled or self.denoise

    @property
    def fingerprint(self) -> str:
        """Stable identifier for this classification configuration."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def resolve_classification_config(
    config: ClassificationConfig | None,
    *,
    reclassify: bool = False,
    ground_method: str = "csf",
    denoise: bool = False,
    vegetation_low: float = 0.5,
    vegetation_medium: float = 2.0,
    vegetation_high: float = 5.0,
) -> ClassificationConfig:
    """Resolve the new configuration API or deprecated scalar arguments."""
    if config is not None:
        if reclassify or denoise or ground_method != "csf":
            raise ValueError("classification_config cannot be combined with legacy arguments")
        return config
    return ClassificationConfig(
        enabled=reclassify,
        ground_method=ground_method,
        denoise=denoise,
        vegetation_low=vegetation_low,
        vegetation_medium=vegetation_medium,
        vegetation_high=vegetation_high,
    )


def ground_stage(config: ClassificationConfig) -> dict:
    """Build the configured PDAL ground-classification stage (not for PTD)."""
    if config.ground_method == "ptd":
        raise ValueError("PTD is not a PDAL stage; use alsdb.core.ptd.ptd_ground")
    ignore = "Classification[7:7],Classification[18:18]"
    if config.ground_method == "pmf":
        return {
            "type": "filters.pmf",
            "ignore": ignore,
            "max_window_size": config.pmf_max_window_size,
            "slope": config.pmf_slope,
            "initial_distance": config.pmf_initial_distance,
            "max_distance": config.pmf_max_distance,
        }
    if config.ground_method == "smrf":
        return {
            "type": "filters.smrf",
            "ignore": ignore,
            "cell": config.smrf_cell,
            "scalar": config.smrf_scalar,
            "slope": config.smrf_slope,
            "threshold": config.smrf_threshold,
            "window": config.smrf_window,
            "cut": config.smrf_cut,
            "returns": ", ".join(config.smrf_returns),
        }
    return {
        "type": "filters.csf",
        "ignore": ignore,
        "resolution": config.csf_resolution,
        "threshold": config.csf_threshold,
        "rigidness": config.csf_rigidness,
    }


def denoise_stages(config: ClassificationConfig) -> list[dict]:
    """Build noise-classification stages when denoising is enabled."""
    if not config.denoise:
        return []
    return [
        {"type": "filters.elm"},
        {
            "type": "filters.outlier",
            "method": "radius",
            "radius": config.outlier_radius,
            "min_k": config.outlier_min_neighbors,
        },
    ]


def classification_stages(config: ClassificationConfig) -> list[dict]:
    """Build ground, HAG, and vegetation stages for an enabled configuration.

    With PTD the ground stage is left out: ground is assigned beforehand by
    :func:`alsdb.core.ptd.ptd_ground`.
    """
    if not config.enabled:
        return []
    source = " || ".join(f"Classification == {value}" for value in config.source_classes)
    low, medium, high = (
        config.vegetation_low,
        config.vegetation_medium,
        config.vegetation_high,
    )
    ground = [] if config.ground_method == "ptd" else [ground_stage(config)]
    return [
        *ground,
        {
            "type": "filters.hag_nn",
            "count": config.hag_neighbors,
            "allow_extrapolation": config.allow_hag_extrapolation,
        },
        {"type": "filters.assign", "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0"},
        {
            "type": "filters.assign",
            "value": f"Classification = 3 WHERE ({source}) && HeightAboveGround >= {low} && HeightAboveGround < {medium}",
        },
        {
            "type": "filters.assign",
            "value": f"Classification = 4 WHERE ({source}) && HeightAboveGround >= {medium} && HeightAboveGround < {high}",
        },
        {
            "type": "filters.assign",
            "value": f"Classification = 5 WHERE ({source}) && HeightAboveGround >= {high}",
        },
    ]
