"""Reusable, storage-independent CHM reconstruction and quality layers."""

from dataclasses import dataclass

import numpy as np

from alsdb.methods._grid import GridSpec
from alsdb.methods._terrain import GROUND_CLASSES, normalize_points

# Bit flags stored as exactly representable float32 integers in ALSZarrStore.
OBSERVED = 1
INTERPOLATED = 2
FILLED = 4
UNSUPPORTED = 8
NO_OBSERVATIONS = 16
EXTRAPOLATED = 32
NEGATIVE_HAG = 64
REJECTED = 128
LOW_PULSE_DENSITY = 256
HIGH_SCAN_ANGLE = 512
QUALITY_FLAGS = {
    "observed": OBSERVED,
    "interpolated": INTERPOLATED,
    "filled": FILLED,
    "unsupported_ground": UNSUPPORTED,
    "no_canopy_observations": NO_OBSERVATIONS,
    "terrain_extrapolated": EXTRAPOLATED,
    "negative_hag": NEGATIVE_HAG,
    "rejected_height_or_outlier": REJECTED,
    "low_pulse_density": LOW_PULSE_DENSITY,
    "high_scan_angle": HIGH_SCAN_ANGLE,
}
PULSE_DENSITY_CELL = 5.0  # m, cells for the pulse-density layer and local freeze


def provenance_settings(config):
    """CHM settings recorded in a run's provenance fingerprint."""
    from dataclasses import asdict

    return asdict(config)


@dataclass(frozen=True)
class CHMConfig:
    method: str = "max"
    first_returns_only: bool = True
    height_statistic: str = "max"
    pit_fill: bool = False
    fill_max_cells: int = 4
    max_ground_distance: float | None = None
    # Nearest ground outside the TIN hull, as the area metrics and profiles (and
    # lidR's default); the cells are flagged EXTRAPOLATED.
    ground_extrapolation: bool = True
    veg_classes: tuple = (3, 4, 5)
    ground_classes: tuple = GROUND_CLASSES
    pitfree_thresholds: tuple = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)
    pitfree_max_edge: float | str = "auto"
    max_height: float | None = None
    remove_outliers: bool = False
    outlier_mean_k: int = 8
    # 6, as lasR's classify_with_sor: lower values remove emergent crown apices.
    outlier_multiplier: float = 6.0
    freeze_distance: float | str = "auto"
    spikefree_first_returns: bool = False
    height_buffer: float = 0.5
    max_triangle_edge: float | None = None
    lastools_executable: str = "las2dem64"
    lastools_demo: bool = False
    freeze_interval: float = 0.25
    # Quality flags only (nothing is removed): Fischer et al. (2024) recommend
    # masking below 2 pulses/m2 (4 in rugged terrain with dense vegetation)
    # and scan angles above 20 degrees.
    min_pulse_density: float | None = 2.0
    max_scan_angle: float | None = 20.0

    def __post_init__(self):
        if self.method not in (
            "max",
            "pitfree",
            "tin",
            "spikefree",
            "lastools",
        ):
            raise ValueError("Unknown CHM method: " + self.method)
        for name in (
            "max_ground_distance",
            "max_height",
            "max_triangle_edge",
            "outlier_multiplier",
            "freeze_interval",
            "min_pulse_density",
            "max_scan_angle",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(name + " must be finite and positive")
        for name in ("pitfree_max_edge", "freeze_distance"):
            value = getattr(self, name)
            if name == "freeze_distance" and value == "local":
                if self.method == "lastools":
                    raise ValueError("freeze_distance='local' needs method='spikefree'")
                continue
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
        return cls(**{k: v for k, v in options.items() if k in cls.__dataclass_fields__})


def clean_points(arr, *, drop_noise=True):
    """Remove nonfinite/withheld/noise returns before terrain or canopy fitting.

    Returns *arr* itself (no copy) when there is nothing to remove.
    """
    keep = np.isfinite(arr["X"]) & np.isfinite(arr["Y"]) & np.isfinite(arr["Z"])
    if drop_noise:
        keep &= ~np.isin(arr["Classification"], (7, 18))
    if "Withheld" in arr.dtype.names:
        keep &= arr["Withheld"] == 0
    return arr if keep.all() else arr[keep]


def surface_mask(arr, config):
    """Select the returns that build the CHM surface: vegetation and ground.

    Ground returns pin the CHM to zero in crown gaps, as in lidR and LAStools.
    ``spikefree_first_returns`` restricts spike-free to first returns, as
    Fischer et al. (2024) recommend.
    """
    mask = np.isin(arr["Classification"], (*config.veg_classes, *config.ground_classes))
    if config.first_returns_only and config.method not in (
        "spikefree",
        "lastools",
    ):
        mask &= arr["ReturnNumber"] == 1
    if config.method == "spikefree" and config.spikefree_first_returns:
        mask &= arr["ReturnNumber"] == 1
    return mask


def pulse_density_grid(pulses, bbox, cell=PULSE_DENSITY_CELL):
    """Pulses per m2 in *cell* squares on multiples of *cell* covering *bbox*.

    *pulses* holds one return per pulse (first returns). Returns
    ``(density[row, col], x0, y1)`` north-up; aligned cells give neighbouring
    tiles the same values.
    """
    x0, y0 = np.floor(bbox[0] / cell) * cell, np.floor(bbox[1] / cell) * cell
    x1, y1 = np.ceil(bbox[2] / cell) * cell, np.ceil(bbox[3] / cell) * cell
    nx, ny = max(1, round((x1 - x0) / cell)), max(1, round((y1 - y0) / cell))
    col = np.floor((pulses["X"] - x0) / cell).astype(np.int64)
    row = np.floor((y1 - pulses["Y"]) / cell).astype(np.int64)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    counts = np.bincount(row[inside] * nx + col[inside], minlength=nx * ny)
    return counts.reshape(ny, nx) / cell**2, float(x0), float(y1)


def auto_freeze_distance(points):
    """Spike-free freeze distance: three first-return pulse spacings.

    As lidR's ``spikefree()`` recommends (Roussel et al. 2020). See
    :func:`alsdb.methods.spikefree.local_freeze_distance` for the per-cell
    alternative of Fischer et al. (2024).
    """
    from alsdb.methods.rasterise import _pulse_spacing

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
    ``diagnostics=False`` returns only ``{"chm": ...}`` and skips quality
    flags, counts, ground distance, and terrain slope.
    """
    from alsdb.methods.rasterise import (
        _fill_pits,
        _outlier_removal_stages,
        _pitfree_rasterise,
        _rasterise,
        _run,
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
            distances=False,  # per-point ground distances are not used here
        )
    )
    gx, gy = grid.centers()
    centres = np.column_stack((gx.ravel(), gy.ravel()))
    ground_z, ground_distance, extrapolated = terrain.evaluate(
        centres,
        max_distance=config.max_ground_distance,
        extrapolate=config.ground_extrapolation,
        distances=diagnostics,
    )
    supported = np.isfinite(ground_z).reshape(grid.shape)
    # Callers may pass points already restricted to the surface: no copy then.
    is_surface = surface_mask(normalized, config)
    candidates = normalized if is_surface.all() else normalized[is_surface]
    candidate_is_valid = np.isfinite(candidates["HeightAboveGround"])
    points = candidates if candidate_is_valid.all() else candidates[candidate_is_valid]
    if config.remove_outliers and len(points) > 2:
        points = _run(
            _outlier_removal_stages(
                min(config.outlier_mean_k, len(points) - 1), config.outlier_multiplier
            ),
            points,
        )

    def bin_counts(source, selected=None, bins=None):
        # *bins*: the (inside, cells) of *source*, computed once for several counts
        inside, cells = grid.point_bins(source) if bins is None else bins
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

    candidate_bins = grid.point_bins(candidates)
    raw_count = bin_counts(candidates, bins=candidate_bins)
    invalid = bin_counts(candidates, ~candidate_is_valid, bins=candidate_bins) > 0

    point_bins = candidate_bins if points is candidates else grid.point_bins(points)
    count = raw_count if points is candidates else bin_counts(points, bins=point_bins)
    canopy_count = None
    if diagnostics:
        candidate_is_canopy = np.isin(candidates["Classification"], config.veg_classes)
        raw_canopy_count = bin_counts(candidates, candidate_is_canopy, bins=candidate_bins)
        point_is_canopy = np.isin(points["Classification"], config.veg_classes)
        canopy_count = bin_counts(points, point_is_canopy, bins=point_bins)
    out = np.full(grid.shape, np.nan, np.float32)
    pulses = arr[arr["ReturnNumber"] == 1] if "ReturnNumber" in arr.dtype.names else arr
    freeze_distance = config.freeze_distance
    if freeze_distance == "auto" and len(points):
        freeze_distance = auto_freeze_distance(points)
    elif freeze_distance == "local" and len(points):
        from alsdb.methods.spikefree import local_freeze_grid

        # Pulses: first returns of every non-noise class in the buffered tile.
        extent = (
            min(points["X"].min(), bbox[0]),
            min(points["Y"].min(), bbox[1]),
            max(points["X"].max(), bbox[2]),
            max(points["Y"].max(), bbox[3]),
        )
        freeze_distance = local_freeze_grid(pulses, extent)
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
        elif config.method == "tin":
            from alsdb.methods.rasterise import _delaunay_raster

            out = _delaunay_raster(
                points,
                bbox,
                resolution,
                value_field="HeightAboveGround",
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
            from alsdb.methods.spikefree import rasterize_spikefree

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
    negative = bin_counts(candidates, candidates["NegativeHAG"] > 0, bins=candidate_bins) > 0
    flags[negative] |= NEGATIVE_HAG
    flags[rejected] |= REJECTED
    # Pulse density per 5 m cell of the buffered tile, read at cell centres.
    if len(pulses):
        extent = (pulses["X"].min(), pulses["Y"].min(), pulses["X"].max(), pulses["Y"].max())
        density, dx0, dy1 = pulse_density_grid(pulses, extent)
    else:
        density, dx0, dy1 = np.zeros((1, 1)), bbox[0], bbox[3]
    dcol = np.clip(((gx - dx0) // PULSE_DENSITY_CELL).astype(np.int64), 0, density.shape[1] - 1)
    drow = np.clip(((dy1 - gy) // PULSE_DENSITY_CELL).astype(np.int64), 0, density.shape[0] - 1)
    pulse_density = density[drow, dcol].reshape(grid.shape).astype(np.float32)
    scan_angle = np.full(grid.shape, np.nan, np.float32)
    if "ScanAngleRank" in pulses.dtype.names and len(pulses):
        inside, cells = grid.point_bins(pulses)
        n = np.bincount(cells, minlength=grid.nx * grid.ny)
        total = np.bincount(
            cells, weights=np.abs(pulses["ScanAngleRank"][inside].astype(float)), minlength=n.size
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scan_angle = np.where(n > 0, total / n, np.nan).reshape(grid.shape).astype(np.float32)
    if config.min_pulse_density is not None:
        flags[pulse_density < config.min_pulse_density] |= LOW_PULSE_DENSITY
    if config.max_scan_angle is not None:
        flags[scan_angle > config.max_scan_angle] |= HIGH_SCAN_ANGLE
    return {
        "chm": out,
        "chm_quality": flags.astype(np.float32),
        "chm_canopy_count": canopy_count,
        "chm_ground_count": ground_count,
        "chm_ground_distance": np.where(np.isfinite(ground_distance), ground_distance, np.nan)
        .reshape(grid.shape)
        .astype(np.float32),
        "chm_terrain_slope": terrain.slope(centres).reshape(grid.shape).astype(np.float32),
        "chm_pulse_density": pulse_density,
        "chm_scan_angle": scan_angle,
    }
