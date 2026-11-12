"""Area-based products (metrics, gap/LAI, biomass) from one query and normalisation per tile.

The engine behind :func:`~alsdb.products.biomass.compute_metrics`,
:func:`~alsdb.products.biomass.compute_biomass` and
:func:`~alsdb.products.gap.compute_gap_fraction`.
"""

import uuid

import numpy as np

from alsdb.methods._grid import GridSpec
from alsdb.methods._surface import (
    QUALITY_FLAGS,
    CHMConfig,
    clean_points,
    provenance_settings,
)
from alsdb.methods._terrain import normalize_points
from alsdb.products._surfaces import (
    CHM_VARIABLES,
    backend_versions,
    input_revision,
    software_versions,
    surface_tile,
)
from alsdb.products._tiling import (
    _filter_ground_outliers,
    _require_year,
    array_crs,
    array_data_bbox,
    check_bbox_overlap,
    check_year_exists,
    query_to_array,
    run_tiled,
)


def _area_worker(
    provider,
    query_bbox,
    crop_bbox,
    store,
    tile_index,
    *,
    grid,
    resolution,
    year,
    needed,
    calculate,
    cached,
    time_index,
    chm,
    config,
    model,
    quality,
    gap_names,
    ground_outlier_removal,
    max_ground_distance,
    ground_extrapolation,
    baba_radius,
    cc_threshold,
    min_density,
    veg_classes,
    k,
    clumping_index,
):
    """Compute and write the requested area products of one tile.

    Module-level with explicit settings, so tiles can run in worker processes.
    *cached* names completed metric slices read from the store (at
    *time_index*) instead of being recomputed.
    """
    from alsdb.products.biomass import _METRIC_NAMES, _extract_metrics, _extract_metrics_baba
    from alsdb.products.gap import _gap_to_lai, gap_statistics
    from alsdb.storage.zarr_store import _res_str

    requires_points = bool(calculate or set(needed) & {"gap", "lai", "chm"})
    values = {}
    if cached:
        r = round((grid.y1 - crop_bbox[3]) / resolution)
        c = round((crop_bbox[0] - grid.x0) / resolution)
        ny, nx = GridSpec.from_bbox(crop_bbox, resolution).shape
        group = store._root[_res_str(resolution)]
        values.update(
            {name: np.asarray(group[name][time_index, r : r + ny, c : c + nx]) for name in cached}
        )
    normalized = terrain = None
    if requires_points:
        attrs = ("Z", "Classification", "ReturnNumber", "Withheld")
        if chm and config.method in ("spikefree", "lastools"):
            attrs += ("NumberOfReturns", "GpsTime", "PointSourceId")
        if chm:
            attrs += ("ScanAngleRank",)
        arr = clean_points(query_to_array(provider, query_bbox, year=year, attributes=attrs))
        if ground_outlier_removal:
            arr = clean_points(_filter_ground_outliers(arr))
        if calculate or "chm" in needed:
            normalized, terrain = normalize_points(
                arr,
                max_distance=max_ground_distance,
                extrapolate=ground_extrapolation,
                distances=False,  # per-point ground distances are not used
            )
        if calculate:
            if baba_radius:
                values.update(
                    _extract_metrics_baba(
                        normalized,
                        resolution,
                        crop_bbox,
                        baba_radius,
                        cc_threshold,
                        min_density,
                        veg_classes,
                        metrics=calculate,
                    )
                )
            else:
                values.update(
                    _extract_metrics(
                        normalized,
                        resolution,
                        crop_bbox,
                        cc_threshold,
                        min_density,
                        veg_classes,
                        metrics=calculate,
                    )
                )
        if set(needed) & {"gap", "lai"}:
            stats = gap_statistics(
                arr, resolution, crop_bbox, min_density, veg_classes, baba_radius
            )
            if "gap" in needed:
                for name in gap_names if quality else ("gap",):
                    store.write_tile(name, resolution, year, stats[name], crop_bbox)
            if "lai" in needed:
                store.write_tile(
                    "lai",
                    resolution,
                    year,
                    _gap_to_lai(stats["gap"], k, clumping_index),
                    crop_bbox,
                )
        if "chm" in needed:
            for name, data in surface_tile(
                arr,
                crop_bbox,
                resolution,
                config,
                prepared=(normalized, terrain),
                diagnostics=quality,
            ).items():
                if name == "chm" or quality:
                    store.write_tile(name, resolution, year, data, crop_bbox)
    for name in needed:
        if name in _METRIC_NAMES:
            store.write_tile(name, resolution, year, values[name], crop_bbox)
    if "biomass" in needed:
        data = np.asarray(model(values), dtype=np.float32)
        if data.shape != GridSpec.from_bbox(crop_bbox, resolution).shape:
            raise ValueError("Biomass model returned an incorrect grid shape")
        store.write_tile("biomass", resolution, year, data, crop_bbox)


def compute_area_products(
    provider,
    store,
    resolution=10.0,
    bbox=None,
    year=None,
    *,
    metrics=None,
    gap=False,
    lai=False,
    biomass=False,
    model_fn=None,
    model_features=None,
    model_id=None,
    chm=False,
    chm_config=None,
    cc_threshold=2.0,
    baba_radius=0.0,
    min_density=0.0,
    veg_classes=(3, 4, 5),
    k=0.5,
    clumping_index=1.0,
    ground_outlier_removal=True,
    max_ground_distance=None,
    ground_extrapolation=True,
    quality=True,
    overwrite=False,
    tile_size=500.0,
    tile_buffer=50.0,
    n_workers=1,
    parallel="threads",
):
    """Write selected metrics, gap/LAI, biomass and CHM on one shared grid.

    ``metrics=None`` requests all structural metrics; ``metrics=()`` requests none.
    Biomass functions can declare ``required_metrics`` and ``model_id`` attributes.
    Supply a stable model_id identifying the fitted model to enable safe resume.
    An unidentified callable is deliberately recomputed. Completed metrics with
    identical provenance are reused for biomass without reading the point cloud.
    Ground cleaning is explicit and shared by every requested product; the gap
    wrapper defaults to disabling it. Quality includes gap support and saturation.
    ``parallel="processes"`` runs the tiles in worker processes (see
    :func:`~alsdb.products._tiling.run_tiled`); a biomass model that cannot be
    pickled (a lambda, a wrapped estimator) keeps the run in threads.
    """
    from alsdb.products.allometry import (
        naesset_model,
    )
    from alsdb.products.biomass import (
        _METRIC_NAMES,
    )
    from alsdb.products.gap import _gap_to_lai

    _require_year(year)
    requested = tuple(_METRIC_NAMES if metrics is None else dict.fromkeys(metrics))
    model = model_fn or naesset_model
    features = (
        tuple(
            model_features
            or getattr(
                model,
                "required_metrics",
                ("h95", "cc") if model is naesset_model else _METRIC_NAMES,
            )
        )
        if biomass
        else ()
    )
    if (set(requested) | set(features)) - set(_METRIC_NAMES):
        raise ValueError("Unknown structural metric")
    if (
        not np.isfinite([min_density, baba_radius, tile_buffer, cc_threshold]).all()
        or min(min_density, baba_radius, tile_buffer) < 0
    ):
        raise ValueError("Density and buffers must be finite and nonnegative")
    if lai:
        _gap_to_lai(np.array([0.5]), k, clumping_index)
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    grid = GridSpec.from_bbox(bbox if bbox is not None else array_data_bbox(provider), resolution)
    config = chm_config or CHMConfig(
        max_ground_distance=max_ground_distance,
        ground_extrapolation=ground_extrapolation,
        veg_classes=tuple(veg_classes),
    )
    # A combined run must use one explicit normalization policy.
    if chm:
        max_ground_distance, ground_extrapolation = (
            config.max_ground_distance,
            config.ground_extrapolation,
        )
    halo = (config.fill_max_cells + 1) * resolution if chm and config.pit_fill else 0
    tiles = list(grid.tiles(tile_size, max(tile_buffer, baba_radius) + halo))
    revision = input_revision(provider)
    base = {
        "algorithm": "area-1",
        "bbox": list(grid.bbox),
        "resolution": resolution,
        "tile_size": tile_size,
        "tile_buffer": max(tile_buffer, baba_radius) + halo,
        "source": str(provider.array_uri),
        "source_revision": revision,
        "software": software_versions(),
        "ground_outlier_removal": ground_outlier_removal,
        "veg_classes": list(veg_classes),
    }
    metric_config = dict(
        base,
        # 2: cover from classified first returns only; Næsset AGB 0 without cover
        metrics_version=2,
        cc_threshold=cc_threshold,
        baba_radius=baba_radius,
        min_density=min_density,
        max_ground_distance=max_ground_distance,
        ground_extrapolation=ground_extrapolation,
    )
    configurations = {name: metric_config for name in requested}
    if gap or lai:
        configurations["gap"] = dict(
            base, baba_radius=baba_radius, min_density=min_density, quality=quality
        )
    if lai:
        configurations["lai"] = dict(configurations["gap"], k=k, clumping_index=clumping_index)
    identified = model_id or getattr(model, "model_id", None)
    if biomass:
        identified = identified or ("naesset-placeholder-v2" if model is naesset_model else None)
        configurations["biomass"] = dict(
            metric_config, features=list(features), model_id=identified or str(uuid.uuid4())
        )
    if chm:
        configurations["chm"] = dict(
            base,
            settings=provenance_settings(config),
            quality=quality,
            backend=backend_versions(config),
            # 2: pulse density and scan angle layers and flags (Fischer et al. 2024).
            **({"quality_layers": 2} if quality else {}),
        )
    needed = [
        name
        for name, cfg in configurations.items()
        if not store.check_run(
            name,
            resolution,
            year,
            cfg,
            overwrite=overwrite or (name == "biomass" and identified is None),
        )
    ]
    if not needed:
        return
    # Only reuse compatible complete slices; a mismatch is a cache miss, not permission to overwrite.
    cached = []
    if "biomass" in needed:
        for name in features:
            if name in needed:
                continue
            try:
                if store.check_run(name, resolution, year, metric_config):
                    cached.append(name)
            except ValueError:
                pass
    time_index = None
    if cached:
        from alsdb.storage.zarr_store import _res_str

        group = store._root[_res_str(resolution)]
        time_index = int(np.flatnonzero(np.asarray(group["time"][:]) == year)[0])
    gap_names = ("gap", "gap_n_ground", "gap_n_classified", "gap_n_first", "gap_saturated")
    crs = array_crs(provider)
    written = []
    for name in needed:
        names = (
            CHM_VARIABLES
            if name == "chm" and quality
            else gap_names
            if name == "gap" and quality
            else (name,)
        )
        for variable in names:
            store.ensure_group(variable, resolution, grid.bbox, crs, tile_size)
        if name == "chm" and quality:
            from alsdb.storage.zarr_store import _res_str

            store._root[_res_str(resolution)]["chm_quality"].attrs["flag_masks"] = QUALITY_FLAGS
        store.begin_run(
            name,
            resolution,
            year,
            configurations[name],
            CHM_VARIABLES if name == "chm" else gap_names if name == "gap" else names,
        )
        written.extend(names)
    # Size the arrays for this year now, so tile workers only write chunks.
    store.prepare_tile_writes(written, resolution, year)
    calculate = (
        tuple(
            dict.fromkeys(
                [name for name in needed if name in _METRIC_NAMES]
                + [name for name in features if name not in cached]
            )
        )
        if "biomass" in needed
        else tuple(name for name in needed if name in _METRIC_NAMES)
    )

    try:
        run_tiled(
            _area_worker,
            provider,
            tiles,
            store,
            n_workers,
            parallel=parallel,
            grid=grid,
            resolution=resolution,
            year=year,
            needed=tuple(needed),
            calculate=calculate,
            cached=tuple(cached),
            time_index=time_index,
            chm=chm,
            config=config,
            model=model,
            quality=quality,
            gap_names=gap_names,
            ground_outlier_removal=ground_outlier_removal,
            max_ground_distance=max_ground_distance,
            ground_extrapolation=ground_extrapolation,
            baba_radius=baba_radius,
            cc_threshold=cc_threshold,
            min_density=min_density,
            veg_classes=tuple(veg_classes),
            k=k,
            clumping_index=clumping_index,
        )
        if input_revision(provider) != revision:
            raise RuntimeError("Input fragments changed during processing; rerun on stable input")
    except Exception:
        for name in needed:
            store.finish_run(name, resolution, year, failed=True)
        raise
    for name in needed:
        store.finish_run(name, resolution, year)
