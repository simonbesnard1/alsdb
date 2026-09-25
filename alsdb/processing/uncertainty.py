"""Conditional CHM uncertainty from explicit, spatially coherent error models.

Intervals describe the supplied error model. They are not automatically calibrated
confidence intervals and do not recover canopy tops absent from the point cloud.
"""

from dataclasses import asdict, dataclass

import numpy as np

from alsdb.processing._grid import GridSpec
from alsdb.processing._products import (
    backend_versions,
    input_revision,
    software_versions,
    surface_tile,
)
from alsdb.processing._surface import DEFAULT_CHM_CONFIG, clean_points
from alsdb.processing._tiling import (
    _filter_ground_outliers,
    _require_year,
    array_crs,
    array_data_bbox,
    check_bbox_overlap,
    check_year_exists,
    query_to_array,
    run_tiled,
)


@dataclass(frozen=True)
class CHMErrorModel:
    """Standard deviations in metres; parameters must come from survey validation.

    Strip offsets affect ground and canopy together. pulse_z_sigma applies one
    shared vertical error to all returns of a pulse. Terrain errors are a smooth
    random field applied only to ground elevations (correlation length in metres).
    Pulse thinning is sampling sensitivity, not a correction for missed treetops.
    PointSourceId must identify strips; GpsTime must identify pulses within strips.
    """

    strip_z_sigma: float = 0.0
    strip_xy_sigma: float = 0.0
    pulse_z_sigma: float = 0.0
    terrain_sigma: float = 0.0
    terrain_correlation_length: float = 20.0
    pulse_keep_probability: float = 1.0
    calibration: str = "Uncalibrated user-specified error model"

    def __post_init__(self):
        for field in ("strip_z_sigma", "strip_xy_sigma", "pulse_z_sigma", "terrain_sigma"):
            value = getattr(self, field)
            if not np.isfinite(value) or value < 0:
                raise ValueError(field + " must be finite and nonnegative")
        if not np.isfinite(self.terrain_correlation_length) or self.terrain_correlation_length <= 0:
            raise ValueError("terrain_correlation_length must be positive")
        if not 0 < self.pulse_keep_probability <= 1:
            raise ValueError("pulse_keep_probability must be in (0, 1]")


def _mix(values):
    """Stable SplitMix64 finalizer (modular arithmetic is intentional)."""
    values = np.asarray(values, dtype=np.uint64).copy()
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _uniform(keys, salt):
    bits = _mix(keys ^ np.uint64(salt)) >> np.uint64(11)
    return (bits.astype(np.float64) + 0.5) / 2**53


def _normal(keys, salt):
    return np.sqrt(-2 * np.log(_uniform(keys, salt))) * np.cos(
        2 * np.pi * _uniform(keys, salt ^ 0x9E3779B97F4A7C15)
    )


def perturb_points(arr, model, *, seed, realization):
    """Coordinate/ID-keyed random errors agree in overlapping processing tiles."""
    out = arr.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, realization]))
    salts = rng.integers(0, np.iinfo(np.uint64).max, 5, dtype=np.uint64)
    uses_strips = model.strip_z_sigma > 0 or model.strip_xy_sigma > 0
    uses_pulses = model.pulse_z_sigma > 0 or model.pulse_keep_probability < 1
    if (uses_strips or uses_pulses) and "PointSourceId" not in arr.dtype.names:
        raise ValueError("Strip/pulse error models require valid PointSourceId")
    strips = (
        arr["PointSourceId"].astype(np.uint64)
        if "PointSourceId" in arr.dtype.names
        else np.zeros(len(arr), np.uint64)
    )
    if uses_pulses:
        if "GpsTime" not in arr.dtype.names or not np.isfinite(arr["GpsTime"]).all():
            raise ValueError("Pulse error models require finite GpsTime")
        if len(arr) > 1 and np.all(arr["GpsTime"] == 0):
            raise ValueError("GpsTime is all zero; pulse identity is unavailable")
        time_bits = np.asarray(arr["GpsTime"], dtype=np.float64).view(np.uint64)
        pulses = _mix(time_bits) ^ _mix(strips)
        out["Z"] += model.pulse_z_sigma * _normal(pulses, int(salts[3]))
    if uses_strips:
        out["Z"] += model.strip_z_sigma * _normal(strips, int(salts[0]))
        out["X"] += model.strip_xy_sigma * _normal(strips, int(salts[1]))
        out["Y"] += model.strip_xy_sigma * _normal(strips, int(salts[2]))
    if model.terrain_sigma:
        ground = arr["Classification"] == 2
        xy = np.column_stack((arr["X"][ground], arr["Y"][ground]))
        # Random Fourier field approximating a squared-exponential covariance.
        frequencies = rng.normal(size=(64, 2)) / model.terrain_correlation_length
        phases = rng.uniform(0, 2 * np.pi, 64)
        field = np.zeros(len(xy))
        for frequency, phase in zip(frequencies, phases):
            field += np.cos(xy @ frequency + phase)
        out["Z"][ground] += model.terrain_sigma * np.sqrt(2 / 64) * field
    if uses_pulses and model.pulse_keep_probability < 1:
        out = out[_uniform(pulses, int(salts[4])) < model.pulse_keep_probability]
    return out


def ensemble_summary(samples, interval=0.95, min_valid_fraction=0.8):
    """Per-pixel quantiles and sample SD without all-NaN reduction warnings."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 3 or len(samples) < 2:
        raise ValueError("At least two 2-D realizations are required")
    if not 0 < interval < 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Invalid interval or min_valid_fraction")
    n = np.isfinite(samples).sum(axis=0)
    valid = n >= max(2, int(np.ceil(len(samples) * min_valid_fraction)))
    sorted_values = np.sort(np.where(np.isfinite(samples), samples, np.inf), axis=0)
    rows, cols = np.indices(n.shape)

    def quantile(q):
        index = np.maximum(n - 1, 0) * q
        lo = np.floor(index).astype(int)
        hi = np.ceil(index).astype(int)
        result = np.full(n.shape, np.nan, np.float32)
        r, c = rows[valid], cols[valid]
        result[valid] = sorted_values[lo[valid], r, c] * (
            1 - index[valid] + lo[valid]
        ) + sorted_values[hi[valid], r, c] * (index[valid] - lo[valid])
        return result

    mean = np.nansum(samples, axis=0, dtype=np.float64) / np.maximum(n, 1)
    variance = np.nansum((samples - mean) ** 2, axis=0) / np.maximum(n - 1, 1)
    return {
        "chm_uncertainty_lower": quantile((1 - interval) / 2),
        "chm_uncertainty_upper": quantile((1 + interval) / 2),
        "chm_uncertainty_std": np.where(valid, np.sqrt(variance), np.nan).astype(np.float32),
        "chm_uncertainty_n_valid": n.astype(np.float32),
    }


def estimate_chm_uncertainty(
    arr,
    bbox,
    resolution,
    *,
    error_model,
    config=DEFAULT_CHM_CONFIG,
    n_realizations=100,
    seed=0,
    interval=0.95,
    min_valid_fraction=0.8,
    ground_outlier_removal=True,
):
    """Reconstruct each realization; memory is n_realizations × tile pixels × 4.

    The input must include the terrain/fill halo and a displacement margin for
    horizontal errors. Use compute_chm_uncertainty for automatic tiled queries.
    """
    if (
        not isinstance(n_realizations, (int, np.integer))
        or n_realizations < 2
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("n_realizations must be an integer >=2 and seed a nonnegative integer")
    if not 0 < interval < 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Invalid interval or min_valid_fraction")
    arr = clean_points(arr)
    samples = np.empty((n_realizations, *GridSpec.from_bbox(bbox, resolution).shape), np.float32)
    for realization in range(n_realizations):
        perturbed = perturb_points(arr, error_model, seed=seed, realization=realization)
        if ground_outlier_removal:
            perturbed = _filter_ground_outliers(perturbed)
        samples[realization] = surface_tile(perturbed, bbox, resolution, config)["chm"]
    return ensemble_summary(samples, interval, min_valid_fraction)


def _uncertainty_worker(
    provider, query_bbox, crop_bbox, store, tile_index, *, year, resolution, parameters
):
    arr = query_to_array(provider, query_bbox, year=year)
    result = estimate_chm_uncertainty(arr, crop_bbox, resolution, **parameters)
    for name, data in result.items():
        store.write_tile(name, resolution, year, data, crop_bbox)


def compute_chm_uncertainty(
    provider,
    store,
    *,
    error_model,
    year,
    resolution=1.0,
    bbox=None,
    config=DEFAULT_CHM_CONFIG,
    n_realizations=100,
    seed=0,
    interval=0.95,
    min_valid_fraction=0.8,
    tile_size=500.0,
    tile_buffer=50.0,
    n_workers=1,
    overwrite=False,
    source_version=None,
    ground_outlier_removal=True,
):
    """Write conditional uncertainty layers with full error-model provenance.

    No CHM is overwritten. Use identical CHMConfig/input revision for the base
    CHM. Source metadata accuracy and independent interval calibration remain
    the caller's responsibility. Gaussian displacement queries use an additional
    six-sigma margin; arbitrarily large displacements are not bounded.
    """
    _require_year(year)
    backend = backend_versions(config)
    if (
        not isinstance(n_realizations, (int, np.integer))
        or n_realizations < 2
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("n_realizations must be an integer >=2 and seed a nonnegative integer")
    if not 0 < interval < 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Invalid interval or min_valid_fraction")
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    grid = GridSpec.from_bbox(bbox if bbox is not None else array_data_bbox(provider), resolution)
    halo = (config.fill_max_cells + 1) * resolution if config.pit_fill else 0.0
    tiles = list(grid.tiles(tile_size, tile_buffer + halo + 6 * error_model.strip_xy_sigma))
    params = {
        "error_model": error_model,
        "config": config,
        "n_realizations": n_realizations,
        "seed": seed,
        "interval": interval,
        "min_valid_fraction": min_valid_fraction,
        "ground_outlier_removal": ground_outlier_removal,
    }
    revision = input_revision(provider)
    provenance = dict(
        params,
        error_model=asdict(error_model),
        config=asdict(config),
        bbox=list(grid.bbox),
        resolution=resolution,
        tile_size=tile_size,
        tile_buffer=tile_buffer,
        source=str(provider.array_uri),
        source_version=source_version,
        source_revision=revision,
        software=software_versions(),
        backend=backend,
        interpretation="conditional error-model interval; requires external calibration",
    )
    names = (
        "chm_uncertainty_lower",
        "chm_uncertainty_upper",
        "chm_uncertainty_std",
        "chm_uncertainty_n_valid",
    )
    if store.check_run(names[0], resolution, year, provenance, overwrite=overwrite):
        return
    for name in names:
        store.ensure_group(name, resolution, grid.bbox, array_crs(provider), tile_size)
    store.begin_run(names[0], resolution, year, provenance, names)
    try:
        run_tiled(
            _uncertainty_worker,
            provider,
            tiles,
            store,
            n_workers,
            resolution=resolution,
            year=year,
            parameters=params,
        )
        if input_revision(provider) != revision:
            raise RuntimeError("Input fragments changed during processing; rerun on stable input")
    except Exception:
        store.finish_run(names[0], resolution, year, failed=True)
        raise
    store.finish_run(names[0], resolution, year)
