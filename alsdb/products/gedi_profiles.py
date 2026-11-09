# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Simulated GEDI canopy profiles and their ALS reference, for attenuation studies.

For every footprint, on one height-above-ground grid:

- the canopy and ground parts of the simulated waveform
  (:func:`alsdb.products.waveform.waveform_from_points`, the GEDI simulator's
  ``-ground`` split by ALS class), rebinned so energy is kept;
- the waveform gap probability as in the GEDI L2B algorithm,
  ``Pgap(h) = 1 - Rv(h) / (Rv + rv_rg * Rg)``, with ``Rv(h)`` the canopy
  energy above ``h``, ``Rv`` and ``Rg`` the canopy and ground energy and
  ``rv_rg`` the canopy-to-ground reflectance ratio (GEDI L2B ``rhov / rhog``);
- the ALS gap probability (MacArthur & Horn 1969) from the same points with
  the same footprint and density weights: once from first returns, once from
  all returns weighted by ``1 / NumberOfReturns``.

All three profiles are gap probabilities on the same heights, so one
retrieval turns them into plant area, e.g. gedidb's
``GEDIVerticalProfiler.compute_profiles`` (``PAI = -ln(Pgap) mu / (G Omega)``,
``PAVD = -dPAI/dz``); with ``G = 0.5`` and ``Omega = 1`` at nadir this is the
MacArthur-Horn estimator of :mod:`alsdb.products.pavd`. The ratio of the
simulated to the ALS profile, against the plant area above, gives the
attenuation (:func:`alsdb.products.pavd.fit_transmittance_model`).

The simulated waveforms are built from the ALS returns and inherit their
occlusion: simulated against ALS isolates what the waveform and the L2B
retrieval add; real GEDI against simulated adds what the simulation cannot
represent.

References
----------
Hancock, S., et al. (2019). The GEDI simulator: A large-footprint waveform
    lidar simulator for calibration and validation of spaceborne missions.
    Remote Sensing of Environment, 220, 309-323.
MacArthur, R. H., & Horn, H. S. (1969). Foliage profile by vertical
    measurements. Ecology, 50(5), 802-804.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alsdb.products import waveform as _wf

HEIGHT_RANGE = (-10.0, 80.0)  # m above ground, default grid of the profiles
# Profiles stop at the signal top: the height above which the canopy holds less
# than this share of the waveform energy (a noise-threshold stand-in). GEDI's
# pgap_theta_z spans only the detected signal, and gedidb's profiler takes the
# highest finite bin as the canopy height.
TOP_THRESHOLD = 1e-3
PROFILE_ARRAYS = ("canopy", "ground", "pgap_waveform", "pgap_als_first", "pgap_als_all")


def height_edges(height_range=HEIGHT_RANGE, z_step=0.15):
    """Return bin edges (m above ground) of the profile grid.

    Parameters
    ----------
    height_range : tuple of float
        Lowest and highest height (m above ground).
    z_step : float
        Bin size (m), the waveform's.

    Returns
    -------
    np.ndarray
        Ascending edges, ``z_step`` apart, covering ``height_range``.
    """
    low, high = height_range
    if not np.isfinite([low, high, z_step]).all() or high <= low or z_step <= 0:
        raise ValueError("Invalid height range or bin size")
    n = int(np.ceil((high - low) / z_step))
    return low + np.arange(n + 1) * z_step


def rebin(z_bins, values, z_ref, edges):
    """Rebin a waveform onto height-above-ground edges, keeping its energy.

    Parameters
    ----------
    z_bins : np.ndarray
        Ascending, equally spaced bin centres (m, elevation).
    values : np.ndarray
        Energy per bin.
    z_ref : float
        Ground elevation (m) the heights are measured from.
    edges : np.ndarray
        Target edges (m above ground).

    Returns
    -------
    np.ndarray
        Energy per target bin; energy outside the edges is dropped.
    """
    step = z_bins[1] - z_bins[0] if len(z_bins) > 1 else 1.0
    source = np.r_[z_bins - step / 2, z_bins[-1] + step / 2] - z_ref
    cumulative = np.r_[0.0, np.cumsum(values)]
    return np.diff(np.interp(edges, source, cumulative))


def gap_probability(canopy, ground, rv_rg=1.0, totals=None):
    """Waveform gap probability at the bin centres, as GEDI L2B.

    ``Pgap(h) = 1 - Rv(h) / (Rv + rv_rg * Rg)``: ``Rv(h)`` is the canopy energy
    above the centre of each bin, ``Rv`` and ``Rg`` the canopy and ground
    totals, ``rv_rg`` the canopy-to-ground reflectance ratio.

    Parameters
    ----------
    canopy, ground : np.ndarray
        Canopy and ground energy per bin, ascending heights.
    rv_rg : float
        Canopy-to-ground reflectance ratio.
    totals : tuple of float, optional
        Canopy and ground energy of the whole waveform (``canopy_energy``,
        ``ground_energy`` of footprints.parquet); default the sums of the
        profiles, which miss energy outside the height grid (e.g. a ground
        return spread below it on steep slopes).

    Returns
    -------
    np.ndarray
        Gap probability per bin: 1 above the canopy, ``1 - cover`` at the ground.
    """
    canopy, ground = np.asarray(canopy, float), np.asarray(ground, float)
    if not np.isfinite(rv_rg) or rv_rg <= 0:
        raise ValueError("rv_rg must be finite and positive")
    rv, rg = (canopy.sum(), ground.sum()) if totals is None else totals
    denominator = rv + rv_rg * rg
    if denominator <= 0:
        return np.full(len(canopy), np.nan)
    above = np.cumsum(canopy[::-1])[::-1] - canopy / 2  # energy above each centre
    return 1.0 - above / denominator


GEDI_FILL = -9999.0  # GEDI's fill value; gedidb treats heights below -1000 as fill


def signal_heights(height, signal_top):
    """Per-footprint heights in GEDI's layout: fill above each signal top.

    GEDI's ``pgap_theta_z`` heights span only the detected signal, and
    gedidb's profiler takes the highest valid height as the canopy height.
    Pass the result with a gap profile to gedidb, e.g.::

        from gedidb.utils.profiles import GEDIVerticalProfiler

        heights = signal_heights(root["height"][:], table["signal_top"])
        cp_pavd, cp_height, H = GEDIVerticalProfiler().compute_profiles(
            root["pgap_waveform"][:], heights, np.pi / 2, 0.5, 1.0
        )  # nadir beam, G = 0.5, Omega = 1

    Parameters
    ----------
    height : np.ndarray
        Bin centres (m above ground), ``profiles.zarr/height``.
    signal_top : array-like
        ``signal_top`` of each footprint (footprints.parquet), in row order.

    Returns
    -------
    np.ndarray
        (footprints, heights) with :data:`GEDI_FILL` above each signal top
        and for footprints without a profile.
    """
    height = np.asarray(height, dtype=float)
    top = np.asarray(signal_top, dtype=float)[:, None]
    out = np.broadcast_to(height, (len(top), len(height))).copy()
    out[~(out < top)] = GEDI_FILL  # NaN tops (no profile) are fill too
    return out


def _als_gap(heights, weights, centres):
    """Return ``1 - (weight above each height) / (total weight)``."""
    total = weights.sum()
    if total <= 0:
        return np.full(len(centres), np.nan)
    order = np.argsort(heights)
    cumulative = np.r_[0.0, np.cumsum(weights[order])]
    # below ground the gap is that at the ground (1 - cover), as for the waveform
    at = np.searchsorted(heights[order], np.maximum(centres, 0.0), side="right")
    return cumulative[at] / total


def heights_above_ground(points, terrain):
    """Return heights above a ground TIN; nearest ground outside it, so no point is lost."""
    from alsdb.methods._terrain import normalize_points

    normalized, _ = normalize_points(points, model=terrain, distances=False, extrapolate=True)
    return np.asarray(normalized["HeightAboveGround"], float)


def als_gap_probability(
    points, center_x, center_y, edges, *, beam_sigma, density_cell, terrain=None, hag=None
):
    """ALS gap probability with the footprint and density weights of the waveform.

    Parameters
    ----------
    points : np.ndarray
        Structured point array of the footprint (X, Y, Z, ReturnNumber,
        NumberOfReturns, Classification).
    center_x, center_y : float
        Footprint centre.
    edges : np.ndarray
        Height-above-ground edges of the profile grid.
    beam_sigma : float or None
        Gaussian footprint sigma (m); None weights all points equally.
    density_cell : float or None
        Cell (m) for the inverse pulse-density weight; None turns it off.
    terrain : TerrainModel, optional
        Ground surface for heights above ground, when ``hag`` is not given.
    hag : np.ndarray, optional
        Heights above ground of *points*, computed once for many footprints.

    Returns
    -------
    first, all_returns : np.ndarray
        Gap probability at the bin centres from first returns, and from all
        returns weighted by ``1 / NumberOfReturns``. Only ground (2) and
        vegetation (3-5) returns count, as in :mod:`alsdb.products.pavd`.
    """
    from alsdb.methods._terrain import VEG_CLASSES

    centres = (edges[:-1] + edges[1:]) / 2
    cls = points["Classification"]
    keep = (cls == 2) | np.isin(cls, VEG_CLASSES)
    used = points[keep]
    nan = np.full(len(centres), np.nan)
    if not len(used):
        return nan, nan
    if hag is None:
        hag = heights_above_ground(used, terrain)
    else:
        hag = np.asarray(hag, float)[keep].copy()
    hag[used["Classification"] == 2] = 0.0
    weights = np.ones(len(used))
    if beam_sigma is not None:
        r2 = (used["X"] - center_x) ** 2 + (used["Y"] - center_y) ** 2
        weights = np.exp(-r2 / (2.0 * beam_sigma**2))
    if density_cell is not None:
        ix = np.floor(used["X"] / density_cell).astype(np.int64)
        iy = np.floor(used["Y"] / density_cell).astype(np.int64)
        ix -= ix.min()
        iy -= iy.min()
        index = ix * (int(iy.max()) + 1) + iy
        first = used["ReturnNumber"] == 1
        pulses = np.bincount(index, weights=first)[index]
        weights = np.divide(weights, pulses, out=np.zeros_like(weights), where=pulses > 0)
    ok = np.isfinite(hag)
    first = ok & (used["ReturnNumber"] == 1)
    gap_first = _als_gap(hag[first], weights[first], centres)
    nret = np.maximum(used["NumberOfReturns"].astype(float), 1.0)
    gap_all = _als_gap(hag[ok], (weights / nret)[ok], centres)
    return gap_first, gap_all


def footprint_profiles(
    data, center_x, center_y, edges, rv_rg=1.0, top_threshold=TOP_THRESHOLD, hag=None, **options
):
    """Return the waveform, canopy/ground profiles and ALS gap profiles of one footprint.

    Parameters
    ----------
    data
        Point arrays of the footprint and its surroundings (as for
        :func:`alsdb.products.waveform.waveform_from_points`).
    center_x, center_y : float
        Footprint centre.
    edges : np.ndarray
        Height-above-ground edges (:func:`height_edges`).
    rv_rg : float
        Canopy-to-ground reflectance ratio for the waveform gap probability.
    top_threshold : float
        Share of the waveform energy above the signal top; the profiles are
        NaN above it (see :data:`TOP_THRESHOLD`).
    hag : np.ndarray, optional
        Heights above ground of *data*, already cleaned (as
        :func:`simulate_profiles` computes them once per chunk); default: a TIN
        of the footprint's own ground returns.
    **options
        Passed to :func:`~alsdb.products.waveform.waveform_from_points`.

    Returns
    -------
    tuple of (WaveformResult, dict, float) or None
        The waveform result, the arrays of :data:`PROFILE_ARRAYS` and the
        signal top (m above ground); None without enough points or without
        ground returns.
    """
    from alsdb.methods._surface import clean_points
    from alsdb.methods._terrain import TerrainModel
    from alsdb.products._footprints import point_array

    result = _wf.waveform_from_points(data, center_x, center_y, **options)
    if result is None or not np.isfinite(result.z_ground_true):
        return None
    canopy = rebin(result.z_bins, result.canopy_waveform, result.z_ground_true, edges)
    ground = rebin(result.z_bins, result.ground_waveform, result.z_ground_true, edges)
    points = point_array(data) if hag is not None else clean_points(point_array(data))
    ground_points = points[points["Classification"] == 2]
    if hag is None and len(ground_points) < 3:
        return None
    radius = options.get("footprint_radius", _wf._FOOTPRINT_RADIUS)
    sigma = options.get("sigma_beam") or _wf._SIGMA_BEAM_FRACTION * radius
    weighted = options.get("gaussian_beam_weighting", True)
    # the waveform's points: those within its query radius (ground beyond it
    # still shapes the terrain model)
    reach = _wf._effective_query_radius(
        radius, weighted, options.get("sigma_beam"),
        options.get("beam_extent_sigma", _wf._BEAM_EXTENT_SIGMA),
    )  # fmt: skip
    inside = (points["X"] - center_x) ** 2 + (points["Y"] - center_y) ** 2 <= reach**2
    gap_first, gap_all = als_gap_probability(
        points[inside], center_x, center_y, edges,
        beam_sigma=sigma if weighted else None,
        density_cell=options.get("density_cell", _wf._DENSITY_CELL),
        terrain=TerrainModel(ground_points) if hag is None else None,
        hag=None if hag is None else np.asarray(hag)[inside],
    )  # fmt: skip
    arrays = {
        "canopy": canopy,
        "ground": ground,
        "pgap_waveform": gap_probability(
            canopy,
            ground,
            rv_rg,
            totals=(result.canopy_waveform.sum(), result.ground_waveform.sum()),
        ),
        "pgap_als_first": gap_first,
        "pgap_als_all": gap_all,
    }
    # the signal top: highest bin with more than top_threshold of the energy above it
    above = np.cumsum(canopy[::-1])[::-1]
    reaching = np.flatnonzero(above > top_threshold * (canopy.sum() + ground.sum()))
    top = float(edges[reaching[-1] + 1]) if len(reaching) else float(edges[0])
    beyond = edges[:-1] >= top
    for name in ("pgap_waveform", "pgap_als_first", "pgap_als_all"):
        arrays[name][beyond] = np.nan
    return result, arrays, top


def _profiles_chunk(task):
    """Profiles of one chunk of nearby shots, read with one query."""
    from alsdb.methods._surface import clean_points
    from alsdb.methods._terrain import TerrainModel
    from alsdb.products._footprints import point_array
    from alsdb.products._tiling import query_to_array

    indices, xy, seeds, rv_rg = task
    w = _wf._WORKER
    options = dict(w["options"])
    edges = options.pop("edges")
    radius = w["radius"]
    low, high = xy.min(axis=0) - radius, xy.max(axis=0) + radius
    points = query_to_array(w["provider"], (low[0], low[1], high[0], high[1]), year=w["year"],
                            attributes=_wf._ATTRIBUTES)  # fmt: skip
    # Clean once and normalise once per chunk: footprints share most of their
    # points, and one TIN of all the chunk's ground serves every footprint.
    points = clean_points(point_array(points)) if len(points) else points
    ground = points[points["Classification"] == 2] if len(points) else points
    hag = heights_above_ground(points, TerrainModel(ground)) if len(ground) >= 3 else None
    grid = _wf._PointGrid(points["X"], points["Y"], low, radius) if hag is not None else None
    output = []
    for k in range(len(indices)):
        neighborhood = grid.disc(xy[k, 0], xy[k, 1], radius) if grid is not None else ()
        found = None
        if len(neighborhood):
            found = footprint_profiles(points[neighborhood], *xy[k], edges, rv_rg=float(rv_rg[k]),
                                       hag=hag[neighborhood], year=w["year"],
                                       rng=np.random.default_rng(seeds[k]), **options)  # fmt: skip
        if found is None:
            output.append((int(indices[k]), None, None))
            continue
        result, arrays, top = found
        metrics = result.to_dict()
        metrics["signal_top"] = top
        metrics["z_ground_true"] = result.z_ground_true
        metrics["canopy_energy"] = float(result.canopy_waveform.sum())
        metrics["ground_energy"] = float(result.ground_waveform.sum())
        output.append(
            (int(indices[k]), metrics, {k_: v.astype(np.float32) for k_, v in arrays.items()})
        )
    return output


def simulate_profiles(
    provider,
    shots: pd.DataFrame,
    out_dir,
    *,
    x_col: str = "center_x",
    y_col: str = "center_y",
    rv_rg: float = 1.0,
    rv_rg_col: str | None = None,
    year: int | None = None,
    height_range: tuple[float, float] = HEIGHT_RANGE,
    n_workers: int = 1,
    rng: np.random.Generator | None = None,
    batch_tile_size: float = 100.0,
    max_shots_per_batch: int = 256,
    **kwargs,
) -> Path:
    """Simulate GEDI canopy profiles and their ALS reference for many shots.

    Writes ``<out_dir>/profiles.zarr`` with arrays of shape (shots, heights):
    ``canopy``, ``ground``, ``pgap_waveform``, ``pgap_als_first``,
    ``pgap_als_all`` (float32, NaN for shots without enough points or ground)
    and ``height`` (bin centres, m above ground); and
    ``<out_dir>/footprints.parquet`` with one row per array row: the shot
    columns, ``shot_index`` (its position in *shots*), the waveform metrics
    (:func:`~alsdb.products.waveform.simulate_batch`), ``signal_top`` (m
    above ground; the gap profiles are NaN above it, as GEDI's
    ``pgap_theta_z`` spans only the signal), ``z_ground_true``,
    ``canopy_energy`` and ``ground_energy`` (shares of the whole waveform) and
    ``rv_rg``. Rows follow the processing order of spatial chunks.

    Pgap for other reflectance ratios follows from the stored canopy and
    ground profiles with :func:`gap_probability` (pass ``totals`` from the
    parquet), without simulating again.

    Parameters
    ----------
    provider : TileDBProvider
    shots : pd.DataFrame
        Shot table with ``x_col`` and ``y_col``; other columns are kept.
    out_dir : str or Path
        Output folder (created).
    x_col, y_col : str
        Easting and northing columns.
    rv_rg : float
        Canopy-to-ground reflectance ratio, when ``rv_rg_col`` is None.
    rv_rg_col : str, optional
        Column with each shot's ratio, e.g. ``rhov / rhog`` of matched GEDI L2B shots.
    year : int, optional
        Survey year.
    height_range : tuple of float
        Heights (m above ground) of the profile grid.
    n_workers : int
        Worker processes (see :func:`~alsdb.products.waveform.simulate_batch`;
        scripts need an ``if __name__ == "__main__":`` guard).
    rng : np.random.Generator, optional
        Generator for reproducible noise.
    batch_tile_size, max_shots_per_batch
        Spatial chunking, as :func:`~alsdb.products.waveform.simulate_batch`.
    **kwargs
        Passed to :func:`~alsdb.products.waveform.waveform_from_points`.

    Returns
    -------
    Path
        ``out_dir``.
    """
    import zarr

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers = shots[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(centers).all():
        raise ValueError("Shot coordinates must be finite")
    ratios = (shots[rv_rg_col].to_numpy(dtype=float) if rv_rg_col is not None
              else np.full(len(shots), float(rv_rg)))  # fmt: skip
    if not (np.isfinite(ratios) & (ratios > 0)).all():
        raise ValueError("Reflectance ratios must be finite and positive")
    edges = height_edges(height_range, kwargs.get("z_step", _wf._Z_STEP))
    radius = _wf._effective_query_radius(
        kwargs.get("footprint_radius", _wf._FOOTPRINT_RADIUS),
        kwargs.get("gaussian_beam_weighting", True),
        kwargs.get("sigma_beam"),
        kwargs.get("beam_extent_sigma", _wf._BEAM_EXTENT_SIGMA),
    )
    random = rng if rng is not None else np.random.default_rng()
    seeds = random.integers(0, np.iinfo(np.uint64).max, size=len(shots), dtype=np.uint64)
    chunks = list(_wf._shot_chunks(centers, batch_tile_size, max_shots_per_batch))
    order = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
    row_of = np.empty(len(shots), np.int64)
    row_of[order] = np.arange(len(order))

    n, nz = len(shots), len(edges) - 1
    root = zarr.open_group(str(out / "profiles.zarr"), mode="w")
    root.create_array("height", data=((edges[:-1] + edges[1:]) / 2).astype(np.float32))
    stores = {
        name: root.create_array(name, shape=(n, nz), chunks=(min(max(n, 1), 512), nz),
                                dtype="float32", fill_value=np.nan)
        for name in PROFILE_ARRAYS
    }  # fmt: skip
    root.attrs.update({"rv_rg_note": "pgap_waveform uses footprints.parquet rv_rg per row",
                       "z_step": float(edges[1] - edges[0])})  # fmt: skip
    metrics = [None] * n
    tasks = ((idx, centers[idx], seeds[idx], ratios[idx]) for idx in chunks)
    options = {**kwargs, "edges": edges}
    for output in _wf._run_chunks(
        _profiles_chunk, tasks, n_workers, provider, options, radius, year
    ):
        rows = np.array([row_of[i] for i, _, _ in output])
        for name in PROFILE_ARRAYS:
            block = np.full((len(output), nz), np.nan, np.float32)
            for k, (_, _, arrays) in enumerate(output):
                if arrays is not None:
                    block[k] = arrays[name]
            start, stop = rows.min(), rows.max() + 1
            if stop - start == len(rows):  # chunks are contiguous rows by construction
                stores[name][start:stop] = block[np.argsort(rows)]
            else:
                for k, row in enumerate(rows):
                    stores[name][row] = block[k]
        for index, m, _ in output:
            metrics[index] = m
    records = []
    for row, index in enumerate(order):
        record = shots.iloc[index].to_dict()
        record["shot_index"] = int(index)
        record["rv_rg"] = float(ratios[index])
        record.update(metrics[index] or {"n_points": 0})
        records.append(record)
    pd.DataFrame(records).to_parquet(out / "footprints.parquet", engine="pyarrow", index=False)
    return out
