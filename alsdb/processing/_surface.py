"""Reusable, storage-independent CHM reconstruction and quality layers."""

import warnings
from dataclasses import dataclass

import numpy as np

from alsdb.processing._grid import GridSpec
from alsdb.processing._terrain import GROUND_CLASSES, normalize_points

# Bit flags stored as exactly representable float32 integers in ALSZarrStore.
OBSERVED = 1
INTERPOLATED = 2
FILLED = 4
UNSUPPORTED = 8
NO_OBSERVATIONS = 16
EXTRAPOLATED = 32
NEGATIVE_HAG = 64
REJECTED = 128
QUALITY_FLAGS = {
    "observed": OBSERVED,
    "interpolated": INTERPOLATED,
    "filled": FILLED,
    "unsupported_ground": UNSUPPORTED,
    "no_canopy_observations": NO_OBSERVATIONS,
    "terrain_extrapolated": EXTRAPOLATED,
    "negative_hag": NEGATIVE_HAG,
    "rejected_height_or_outlier": REJECTED,
}


@dataclass(frozen=True)
class CHMConfig:
    method: str = "max"
    first_returns_only: bool = True
    height_statistic: str = "max"
    pit_fill: bool = False
    fill_max_cells: int = 4
    max_ground_distance: float | None = None
    ground_extrapolation: bool = False
    veg_classes: tuple = (3, 4, 5)
    ground_classes: tuple = GROUND_CLASSES
    pitfree_thresholds: tuple = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)
    pitfree_max_edge: float | str = "auto"
    max_height: float | None = None
    remove_outliers: bool = False
    outlier_mean_k: int = 8
    # 6, as lasR's classify_with_sor: at 2 the filter removed the apex of an
    # emergent crown (isolated by nature); 6 still drops returns far above canopy.
    outlier_multiplier: float = 6.0
    spikefree_subcell_resolution: float | None = None
    freeze_distance: float | str = "auto"
    height_buffer: float = 0.5
    max_triangle_edge: float | None = None
    lastools_executable: str = "las2dem64"
    lastools_demo: bool = False
    freeze_interval: float = 0.25

    def __post_init__(self):
        if self.method not in ("max", "pitfree", "highest_subcell_tin", "spikefree", "lastools"):
            raise ValueError("Unknown CHM method: " + self.method)
        for name in (
            "max_ground_distance",
            "max_height",
            "spikefree_subcell_resolution",
            "max_triangle_edge",
            "outlier_multiplier",
            "freeze_interval",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(name + " must be finite and positive")
        for name in ("pitfree_max_edge", "freeze_distance"):
            value = getattr(self, name)
            if value != "auto" and (
                not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(name + " must be positive or 'auto'")
        if not np.isfinite(self.height_buffer) or self.height_buffer < 0:
            raise ValueError("height_buffer must be finite and nonnegative")
        for name in ("fill_max_cells", "outlier_mean_k"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(name + " must be a positive integer")
        thresholds = np.asarray(self.pitfree_thresholds)
        if (
            not len(thresholds)
            or not np.isfinite(thresholds).all()
            or np.any(np.diff(thresholds) <= 0)
        ):
            raise ValueError("pitfree_thresholds must be finite and strictly increasing")
        for name in ("veg_classes", "ground_classes"):
            classes = getattr(self, name)
            if not classes or any(not 0 <= c <= 255 for c in classes):
                raise ValueError(name + " must contain LAS classification codes")

    @classmethod
    def from_options(cls, options):
        options = dict(options)
        pitfree, legacy = options.get("pitfree", False), options.get("spikefree", False)
        if (pitfree and legacy) or (options.get("method") is not None and (pitfree or legacy)):
            raise ValueError("method, pitfree and spikefree are mutually exclusive")
        if legacy:
            warnings.warn(
                "spikefree=True is the legacy approximation; use "
                "method='spikefree' for constrained triangulation or "
                "method='highest_subcell_tin' for the old method.",
                DeprecationWarning,
                stacklevel=3,
            )
        options["method"] = options.get("method") or (
            "pitfree" if pitfree else "highest_subcell_tin" if legacy else "max"
        )
        return cls(**{k: v for k, v in options.items() if k in cls.__dataclass_fields__})


def clean_points(arr, *, drop_noise=True):
    """Remove nonfinite/withheld/noise returns before terrain or canopy fitting."""
    keep = np.isfinite(arr["X"]) & np.isfinite(arr["Y"]) & np.isfinite(arr["Z"])
    if drop_noise:
        keep &= ~np.isin(arr["Classification"], (7, 18))
    if "Withheld" in arr.dtype.names:
        keep &= arr["Withheld"] == 0
    return arr[keep]


def surface_mask(arr, config):
    """Returns that build the CHM surface under *config*: vegetation and ground.

    Ground returns (HAG ~0) belong in the surface, as in lidR and LAStools:
    they pin the CHM to zero in crown gaps. With vegetation only, a TIN
    bridges every gap from crown to crown; on Barcelona NFI plots that
    inflated mean pitfree height by ~2 m. Spike-free variants use all
    returns by definition. Callers that prepare normalized points for
    :func:`reconstruct_chm` only need to normalize this subset.
    """
    mask = np.isin(arr["Classification"], (*config.veg_classes, *config.ground_classes))
    if config.first_returns_only and config.method not in (
        "spikefree",
        "highest_subcell_tin",
        "lastools",
    ):
        mask &= arr["ReturnNumber"] == 1
    return mask


def auto_freeze_distance(points):
    """Spike-free freeze distance: three pulse spacings, as lidR recommends.

    Pulses are first returns. The old fixed 1.5 m default suits ~4+ pulses/m2;
    on ~1 pulse/m2 PNOA data it froze too little and left pits (mean CHM
    1.8 m below LAStools on Barcelona NFI plots, vs 0.5 m at 3 m).
    """
    from alsdb.processing.chm import _pulse_spacing

    if "ReturnNumber" in points.dtype.names:
        first = points[points["ReturnNumber"] == 1]
        points = first if len(first) else points
    return 3.0 * _pulse_spacing(points)


DEFAULT_CHM_CONFIG = CHMConfig()


def reconstruct_chm(
    arr, bbox, resolution, config=DEFAULT_CHM_CONFIG, *, prepared=None, diagnostics=True
):
    """Build a CHM and diagnostics from buffered, already ground-cleaned points.

    The caller should rasterize a halo before cropping when hole filling is on.
    This pure function is also used for every uncertainty realization.
    ``diagnostics=False`` returns only ``{"chm": ...}`` and skips quality
    flags, counts, ground distance, and terrain slope.
    """
    from alsdb.processing.chm import (
        _fill_pits,
        _outlier_removal_stages,
        _pitfree_rasterise,
        _rasterise,
        _run,
        _spikefree_rasterise,
    )

    grid = GridSpec.from_bbox(bbox, resolution)
    bbox = grid.bbox
    arr = clean_points(arr)
    normalized, terrain = (
        prepared
        if prepared is not None
        else normalize_points(
            arr,
            max_distance=config.max_ground_distance,
            extrapolate=config.ground_extrapolation,
            ground_classes=config.ground_classes,
            distances=diagnostics,
        )
    )
    gx, gy = grid.centers()
    ground_z, ground_distance, extrapolated = terrain.evaluate(
        np.column_stack((gx.ravel(), gy.ravel())),
        max_distance=config.max_ground_distance,
        extrapolate=config.ground_extrapolation,
        distances=diagnostics,
    )
    supported = np.isfinite(ground_z).reshape(grid.shape)
    candidates = normalized[surface_mask(normalized, config)]
    points = candidates[np.isfinite(candidates["HeightAboveGround"])]
    if config.remove_outliers and len(points) > 2:
        points = _run(
            _outlier_removal_stages(
                min(config.outlier_mean_k, len(points) - 1), config.outlier_multiplier
            ),
            points,
        )

    def bin_counts(source, selected=None):
        inside, cells = grid.point_bins(source)
        if selected is not None:
            cells = cells[np.asarray(selected)[inside]]
        return (
            np.bincount(cells, minlength=grid.nx * grid.ny).reshape(grid.shape).astype(np.float32)
        )

    def raster(points, values, statistic="max"):
        if not len(points):
            return (
                np.zeros(grid.shape, np.float32)
                if statistic == "count"
                else np.full(grid.shape, np.nan, np.float32)
            )
        # Half-open ownership keeps boundary points out of neighbouring tiles.
        mask = (
            (points["X"] >= bbox[0])
            & (points["X"] < bbox[2])
            & (points["Y"] > bbox[1])
            & (points["Y"] <= bbox[3])
        )
        if not mask.any():
            return (
                np.zeros(grid.shape, np.float32)
                if statistic == "count"
                else np.full(grid.shape, np.nan, np.float32)
            )
        # scipy owns Y edges on the opposite side; move exact grid-line Y down.
        y = np.nextafter(points["Y"][mask], -np.inf)
        y = np.minimum(y, bbox[3])
        return _rasterise(
            points["X"][mask], y, np.asarray(values)[mask], bbox, resolution, statistic
        )

    candidate_is_valid = np.isfinite(candidates["HeightAboveGround"])
    candidate_is_canopy = np.isin(candidates["Classification"], config.veg_classes)
    raw_count = bin_counts(candidates)
    invalid = bin_counts(candidates, ~candidate_is_valid) > 0
    raw_canopy_count = bin_counts(candidates, candidate_is_canopy)

    point_is_canopy = np.isin(points["Classification"], config.veg_classes)
    count = bin_counts(points)
    canopy_count = bin_counts(points, point_is_canopy) if diagnostics else None
    out = np.full(grid.shape, np.nan, np.float32)
    freeze_distance = config.freeze_distance
    if freeze_distance == "auto" and len(points):
        freeze_distance = auto_freeze_distance(points)
    if len(points):
        if config.method == "max":
            out = raster(points, points["HeightAboveGround"], config.height_statistic)
        elif config.method == "pitfree":
            out = _pitfree_rasterise(
                points,
                bbox,
                resolution,
                thresholds=config.pitfree_thresholds,
                max_edge=config.pitfree_max_edge,
                base_max_edge=config.max_triangle_edge,
            )
        elif config.method == "highest_subcell_tin":
            out = _spikefree_rasterise(
                points,
                bbox,
                resolution,
                subcell_resolution=config.spikefree_subcell_resolution or resolution / 3,
                max_edge=config.max_triangle_edge,
            )
        elif config.method == "lastools":
            from alsdb.benchmark.lastools import rasterize_lastools

            out = rasterize_lastools(
                points,
                bbox,
                resolution,
                executable=config.lastools_executable,
                freeze_distance=freeze_distance,
                height_buffer=config.height_buffer,
                freeze_interval=config.freeze_interval,
                demo=config.lastools_demo,
                max_triangle_edge=config.max_triangle_edge or 100.0,
            )
        else:
            from alsdb.processing.spikefree import rasterize_spikefree

            out = rasterize_spikefree(
                points,
                bbox,
                resolution,
                freeze_distance=freeze_distance,
                height_buffer=config.height_buffer,
                max_triangle_edge=config.max_triangle_edge,
            )

    rejected = (raw_count > 0) & (count == 0) & ~invalid
    if config.max_height is not None:
        rejected |= np.isfinite(out) & (out > config.max_height)
    out[rejected] = np.nan
    out[~supported] = np.nan
    before = np.isfinite(out)
    if config.pit_fill:
        # Only binned holes are eligible. TIN gaps were deliberately excluded
        # by hull/distance/edge rules and must not be reintroduced by filling.
        eligible = supported & ~rejected & ~invalid & (config.method == "max")
        out = _fill_pits(out, eligible=eligible, max_cells=config.fill_max_cells)
    if not diagnostics:
        return {"chm": out}
    ground = arr[np.isin(arr["Classification"], config.ground_classes)]
    ground_count = bin_counts(ground)
    filled = ~before & np.isfinite(out)
    flags = np.zeros(grid.shape, np.uint16)
    flags[before & (count > 0)] |= OBSERVED
    flags[before & ((count == 0) | (config.method != "max"))] |= INTERPOLATED
    flags[filled] |= FILLED
    flags[~supported | invalid] |= UNSUPPORTED
    flags[raw_canopy_count == 0] |= NO_OBSERVATIONS
    flags[extrapolated.reshape(grid.shape)] |= EXTRAPOLATED
    negative = bin_counts(candidates, candidates["NegativeHAG"] > 0) > 0
    flags[negative] |= NEGATIVE_HAG
    flags[rejected] |= REJECTED
    return {
        "chm": out,
        "chm_quality": flags.astype(np.float32),
        "chm_canopy_count": canopy_count,
        "chm_ground_count": ground_count,
        "chm_ground_distance": np.where(np.isfinite(ground_distance), ground_distance, np.nan)
        .reshape(grid.shape)
        .astype(np.float32),
        "chm_terrain_slope": terrain.slope(np.column_stack((gx.ravel(), gy.ravel())))
        .reshape(grid.shape)
        .astype(np.float32),
    }
