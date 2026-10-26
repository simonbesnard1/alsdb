"""Shared buffered workers, provenance and restart handling for surface products."""

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from alsdb.processing._grid import GridSpec
from alsdb.processing._surface import (
    QUALITY_FLAGS,
    CHMConfig,
    clean_points,
    reconstruct_chm,
    surface_mask,
)
from alsdb.processing._terrain import TerrainModel, normalize_points
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

CHM_VARIABLES = (
    "chm",
    "chm_quality",
    "chm_canopy_count",
    "chm_ground_count",
    "chm_ground_distance",
    "chm_terrain_slope",
)


def input_revision(provider):
    """Fingerprint visible fragments; ingestion/consolidation invalidates reuse."""
    import hashlib
    import json

    import tiledb

    fragments = tiledb.array_fragments(provider.array_uri, ctx=provider.ctx)
    records = sorted(zip(fragments.uri, fragments.timestamp_range, fragments.cell_num))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def software_versions():
    versions = {}
    for name in ("alsdb", "numpy", "scipy", "pdal", "zarr", "tiledb"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    # 3: ground returns in the CHM surface, pitfree edge trimming, class 9 ground.
    versions["surface_algorithm"] = "3"
    return versions


def backend_versions(config):
    if config.method == "spikefree":
        from alsdb.processing.spikefree import require_backend

        return {"native_spikefree": require_backend().algorithm_version}
    if config.method == "lastools":
        from alsdb.benchmark.lastools import executable_version

        return {"lastools": executable_version(config.lastools_executable)}
    return {}


def _crop(data, outer, inner, resolution):
    r0 = round((outer[3] - inner[3]) / resolution)
    c0 = round((inner[0] - outer[0]) / resolution)
    ny, nx = GridSpec.from_bbox(inner, resolution).shape
    return data[r0 : r0 + ny, c0 : c0 + nx]


def surface_tile(arr, bbox, resolution, config, *, prepared=None):
    """Include enough raster halo for every eligible bounded hole and its rim."""
    halo = (config.fill_max_cells + 1) * resolution if config.pit_fill else 0.0
    x0, y0, x1, y1 = bbox
    outer = (x0 - halo, y0 - halo, x1 + halo, y1 + halo)
    result = reconstruct_chm(arr, outer, resolution, config, prepared=prepared)
    return {name: _crop(data, outer, bbox, resolution) for name, data in result.items()}


def _worker(
    provider,
    query_bbox,
    crop_bbox,
    store,
    tile_index,
    *,
    resolution,
    year,
    products,
    config,
    dtm_method,
    exclude_classes,
    quality,
    ground_outlier_removal,
):
    from alsdb.processing.chm import _rasterise

    attributes = ("Z", "Classification", "ReturnNumber", "Withheld")
    if "chm" in products and config.method in ("spikefree", "lastools"):
        attributes += ("NumberOfReturns", "GpsTime", "PointSourceId")
    arr = clean_points(
        query_to_array(provider, query_bbox, year=year, attributes=attributes), drop_noise=False
    )
    if ground_outlier_removal:
        arr = _filter_ground_outliers(arr)
    grid = GridSpec.from_bbox(crop_bbox, resolution)
    x0, y0, x1, y1 = crop_bbox
    own = (arr["X"] >= x0) & (arr["X"] < x1) & (arr["Y"] > y0) & (arr["Y"] <= y1)

    def binned(points, statistic):
        if not len(points):
            return np.full(grid.shape, np.nan, np.float32)
        y = np.minimum(np.nextafter(points["Y"], -np.inf), y1)
        return _rasterise(points["X"], y, points["Z"], crop_bbox, resolution, statistic)

    is_ground = np.isin(arr["Classification"], config.ground_classes)
    prepared = None
    if "chm" in products:
        cleaned = clean_points(arr)
        # Only surface candidates get a height: on a dense DE-Hai tile that's
        # ~23% of the returns, and the TIN lookup dominated tile time.
        prepared = normalize_points(
            cleaned[surface_mask(cleaned, config)],
            max_distance=config.max_ground_distance,
            extrapolate=config.ground_extrapolation,
            model=TerrainModel(cleaned[np.isin(cleaned["Classification"], config.ground_classes)]),
        )
    if "dtm" in products:
        terrain = prepared[1] if prepared is not None else TerrainModel(arr[is_ground])
        if dtm_method == "min":
            data = binned(arr[own & is_ground], "min")
        else:
            gx, gy = grid.centers()
            data, _, _ = terrain.evaluate(
                np.column_stack((gx.ravel(), gy.ravel())),
                method=dtm_method,
                max_distance=config.max_ground_distance,
                extrapolate=config.ground_extrapolation,
            )
            data = data.reshape(grid.shape).astype(np.float32)
        store.write_tile("dtm", resolution, year, data, crop_bbox)
    if "dsm" in products:
        mask = own & ~np.isin(arr["Classification"], exclude_classes)
        if config.first_returns_only:
            mask &= arr["ReturnNumber"] == 1
        store.write_tile("dsm", resolution, year, binned(arr[mask], "max"), crop_bbox)
    if "chm" in products:
        result = surface_tile(arr, crop_bbox, resolution, config, prepared=prepared)
        for name, data in result.items():
            if name == "chm" or quality:
                store.write_tile(name, resolution, year, data, crop_bbox)
    # Only after every write for this tile succeeded, so a crash mid-tile
    # leaves it unmarked and a resumed run redoes it.
    for product in products:
        store.mark_tile_complete(product, resolution, year, tile_index)


def compute_products(provider, store, products, options):
    """Common orchestration used by compute_chm/dtm/dsm/all."""
    from alsdb.processing.chm import _validate_grid_alignment

    resolution, year = options["resolution"], options["year"]
    tile_size = options.get("tile_size", 500.0)
    _require_year(year)
    _validate_grid_alignment(tile_size, resolution)
    config = CHMConfig.from_options(options)
    dtm_method = options.get("dtm_method", "tin")
    if dtm_method not in ("tin", "idw", "min"):
        raise ValueError("dtm_method must be 'tin', 'idw', or 'min'")
    backend = backend_versions(config) if "chm" in products else {}
    bbox = options.get("bbox")
    if bbox is not None and not check_bbox_overlap(bbox, provider):
        return
    if not check_year_exists(year, provider):
        return
    grid = GridSpec.from_bbox(bbox if bbox is not None else array_data_bbox(provider), resolution)
    bbox = grid.bbox
    buffer = options.get("tile_buffer", 50.0) if ("dtm" in products or "chm" in products) else 0.0
    halo = (
        (config.fill_max_cells + 1) * resolution if config.pit_fill and "chm" in products else 0.0
    )
    tiles = list(grid.tiles(tile_size, buffer + halo))
    quality = options.get("quality", True)
    clean_ground = options.get("ground_outlier_removal", True)
    exclude = options.get("dsm_exclude_classes", options.get("exclude_classes", (7, 18)))
    revision = input_revision(provider)
    base = {
        "bbox": list(bbox),
        "resolution": resolution,
        "tile_size": tile_size,
        "tile_buffer": buffer,
        "source": str(provider.array_uri),
        "source_version": options.get("source_version"),
        "source_revision": revision,
        "ground_outlier_removal": clean_ground,
        "software": software_versions(),
    }
    configurations = {}
    for product in products:
        settings = (
            asdict(config)
            if product == "chm"
            else {
                "method": dtm_method,
                "ground_classes": list(config.ground_classes),
                "max_ground_distance": config.max_ground_distance,
                "ground_extrapolation": config.ground_extrapolation,
            }
            if product == "dtm"
            else {"first_returns_only": config.first_returns_only, "exclude_classes": list(exclude)}
        )
        configurations[product] = dict(
            base,
            settings=settings,
            quality=quality if product == "chm" else False,
            backend=backend if product == "chm" else {},
        )
    needed = [
        p
        for p in products
        if not store.check_run(
            p, resolution, year, configurations[p], overwrite=options.get("overwrite", False)
        )
    ]
    if not needed:
        return
    crs = array_crs(provider)
    # check_run already refused a mismatched configuration (without
    # overwrite), so a surviving non-complete record here is an interrupted
    # or failed attempt of this exact run: resume it instead of redoing tiles.
    resume = not options.get("overwrite", False)
    skip = None
    for product in needed:
        names = CHM_VARIABLES if product == "chm" and quality else (product,)
        for name in names:
            store.ensure_group(name, resolution, bbox, crs, tile_size)
        if product == "chm" and quality:
            from alsdb.storage.zarr_store import _res_str

            store._root[_res_str(resolution)]["chm_quality"].attrs["flag_masks"] = QUALITY_FLAGS
        # Clear old diagnostics even when quality has been disabled for this run.
        resumed = store.begin_run(
            product,
            resolution,
            year,
            configurations[product],
            CHM_VARIABLES if product == "chm" else names,
            resume=resume,
        )
        # A tile is skippable only if every product being computed finished it.
        done = store.completed_tiles(product, resolution, year) if resumed else set()
        skip = done if skip is None else skip & done
    try:
        run_tiled(
            _worker,
            provider,
            tiles,
            store,
            options.get("n_workers", 1),
            skip=skip or frozenset(),
            resolution=resolution,
            year=year,
            products=needed,
            config=config,
            dtm_method=dtm_method,
            exclude_classes=exclude,
            quality=quality,
            ground_outlier_removal=clean_ground,
        )
        if input_revision(provider) != revision:
            raise RuntimeError("Input fragments changed during processing; rerun on stable input")
    except Exception:
        for product in needed:
            store.finish_run(product, resolution, year, failed=True)
        raise
    for product in needed:
        store.finish_run(product, resolution, year)
