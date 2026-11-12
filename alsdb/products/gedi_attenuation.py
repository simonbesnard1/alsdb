# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""GEDI attenuation: real GEDI, simulated GEDI and ALS canopy profiles at the same footprints.

Three plant area profiles are brought to the same footprints and the same
retrieval, so their differences can be attributed:

- the **GEDI L2B** profile of the real shot (``cp_pavd`` of gedidb);
- the **simulated** profile: the waveform alsdb simulates from the ALS points
  at the shot's location, split into canopy and ground, turned into a gap
  probability with the shot's own canopy-to-ground reflectance ratio
  (``rhov / rhog``) as GEDI L2B does;
- the **ALS** profile: the MacArthur-Horn gap probability of the same points
  with the same footprint weights (first returns, and all returns).

Simulated against ALS isolates what the waveform and the L2B retrieval add;
real against simulated what the simulation cannot represent (geolocation,
noise, time between surveys, and the occlusion the ALS itself suffers:
simulated waveforms are built from ALS returns and inherit it).

Steps, each cached in the output folder:

1. :func:`query_gedi_shots` reads the GEDI shots over the ALS extent from a
   gedidb database (Besnard et al. 2025), with reflectances, quality flags
   and profiles;
2. :func:`select_shots` applies quality filters, projects the shots to the
   array's CRS and sets each shot's reflectance ratio;
3. :func:`~alsdb.products.profiles.simulate_gedi_profiles` simulates the
   waveform and the gap profiles;
4. :func:`canopy_profiles` runs gedidb's ``GEDIVerticalProfiler`` on the
   simulated and ALS gap profiles (the retrieval behind ``cp_pavd``) with
   each shot's ``rossg`` and ``omega``;
5. :func:`attenuation_summary` compares the three per relative height and
   fits the transmittance model of
   :func:`~alsdb.products.profiles.fit_transmittance_model`.

:func:`run_gedi_attenuation` runs them in order. Steps 1 and 4 need the
``gedidb`` package (imported only there).

The database does not store the beam elevation angle; the retrieval of the
simulated and ALS profiles assumes a nadir beam (GEDI points within 6 degrees
of nadir, which changes the path length by less than 0.6 %).

``cp_pavd`` holds the plant area in each 1 % step of the canopy height (its
101 values sum to the plant area index), in gedidb and here alike.

References
----------
Besnard, S., Dombrowski, F., & Holcomb, A. (2025). gediDB: A toolbox for
    processing and providing Global Ecosystem Dynamics Investigation (GEDI)
    L2A-B and L4A-C data. Journal of Open Source Software, 10(113), 8593.
    https://doi.org/10.21105/joss.08593
Hancock, S., et al. (2019). The GEDI simulator: A large-footprint waveform
    lidar simulator for calibration and validation of spaceborne missions.
    Remote Sensing of Environment, 220, 309-323.
Tang, H., & Armston, J. (2019). Algorithm theoretical basis document (ATBD)
    for GEDI L2B footprint canopy cover and vertical profile metrics.
    Goddard Space Flight Center, Greenbelt, MD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alsdb.products import profiles as _profiles

logger = logging.getLogger(__name__)

GEDIDB_BUCKET = "dog-ext.gedidb.gedi-l2-l4-v002.0"  # public gedidb database at GFZ
GEDIDB_URL = "https://s3.gfz.de"

# Read for every shot: reflectances and retrieval settings, quality, profiles.
SHOT_VARIABLES = (
    "rhov", "rhog", "rossg", "omega", "pai", "cover", "pgap_theta", "rh", "cp_pavd", "cp_height",
    "l2a_quality_flag", "l2b_quality_flag", "degrade_flag", "sensitivity", "leaf_off_flag",
    "elev_lowestmode", "beam_type", "beam_name", "pft_class",
)  # fmt: skip
PROFILE_KINDS = ("waveform", "als_first", "als_all")  # gap profiles of simulate_gedi_profiles


def als_lonlat_bbox(provider, buffer: float = 0.0) -> tuple[float, float, float, float]:
    """Return the longitude/latitude box around the array's data extent.

    Parameters
    ----------
    provider : TileDBProvider
        ALS array.
    buffer : float
        Margin (m, array CRS) added around the data extent.

    Returns
    -------
    tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)``.
    """
    import pyproj

    from alsdb.products._tiling import array_crs, array_data_bbox

    x0, y0, x1, y1 = array_data_bbox(provider)
    to_lonlat = pyproj.Transformer.from_crs(array_crs(provider), "EPSG:4326", always_xy=True)
    xs = np.linspace(x0 - buffer, x1 + buffer, 9)
    ys = np.linspace(y0 - buffer, y1 + buffer, 9)
    gx, gy = np.meshgrid(xs, ys)  # edges are curved in lon/lat: sample them
    lon, lat = to_lonlat.transform(gx.ravel(), gy.ravel())
    return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def query_gedi_shots(
    provider,
    cache: str | Path,
    *,
    gedi_provider=None,
    start_time: str = "2019-04-01",
    end_time: str | None = None,
    variables: tuple[str, ...] = SHOT_VARIABLES,
    buffer: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return the GEDI shots over the ALS extent, from *cache* or a gedidb database.

    Parameters
    ----------
    provider : TileDBProvider
        ALS array; its data extent (plus *buffer* m) is queried.
    cache : str or Path
        ``.npz`` file the shots are stored in; an existing file is read
        instead of querying again.
    gedi_provider : gedidb.GEDIProvider, optional
        Database to query; default the public gedidb database at GFZ
        (:data:`GEDIDB_BUCKET`).
    start_time, end_time : str
        Acquisition dates to keep; default the whole mission.
    variables : tuple of str
        gedidb variables to read.
    buffer : float
        Margin (m) around the ALS extent.

    Returns
    -------
    dict of np.ndarray
        ``shot_number``, ``longitude``, ``latitude``, ``time`` and every
        variable, one row per shot (profile variables have 101 columns).
        Empty arrays when no shot falls in the extent.
    """
    cache = Path(cache)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}
    import geopandas as gpd
    from shapely.geometry import box

    if gedi_provider is None:
        import gedidb

        gedi_provider = gedidb.GEDIProvider(
            storage_type="s3", s3_bucket=GEDIDB_BUCKET, url=GEDIDB_URL
        )
    lonlat = als_lonlat_bbox(provider, buffer)
    logger.info("Querying GEDI shots in %s", lonlat)
    ds = gedi_provider.get_data(
        variables=list(variables),
        geometry=gpd.GeoDataFrame(geometry=[box(*lonlat)], crs="EPSG:4326"),
        start_time=start_time,
        end_time=end_time or pd.Timestamp.now().strftime("%Y-%m-%d"),
        return_type="xarray",
    )
    shots: dict[str, np.ndarray] = {}
    if ds is None or ds.sizes.get("shot_number", 0) == 0:
        shots["shot_number"] = np.empty(0, np.uint64)
    else:
        for name in ("shot_number", "longitude", "latitude", "time", *variables):
            values = np.asarray(ds[name].values)
            shots[name] = values.astype(str) if values.dtype == object else values
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **shots)
    logger.info("%d GEDI shots stored in %s", len(shots["shot_number"]), cache)
    return shots


def select_shots(
    shots: dict[str, np.ndarray],
    provider,
    *,
    min_sensitivity: float = 0.95,
    leaf_on: bool = True,
    quality: bool = True,
    margin: float = 25.0,
    years: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Return the usable shots inside the ALS extent, in the array's CRS.

    Parameters
    ----------
    shots : dict
        Output of :func:`query_gedi_shots`.
    provider : TileDBProvider
        ALS array (for its CRS and extent).
    min_sensitivity : float
        Lowest beam sensitivity kept (0.95: enough to see the ground under
        95 % cover).
    leaf_on : bool
        Drop shots flagged leaf-off where the plant functional type is
        deciduous (classes 3 and 4 of ``pft_class``), the strata the GEDI
        flag is meant for.
    quality : bool
        Keep only shots with the L2A and L2B quality flags set and no
        degraded geolocation.
    margin : float
        Shots closer than this (m) to the edge of the ALS extent are dropped,
        so the whole footprint has points.
    years : tuple of int, optional
        First and last acquisition year kept (e.g. around the ALS survey).

    Returns
    -------
    pandas.DataFrame
        One row per shot: ``shot_number``, ``center_x``, ``center_y``,
        ``time``, ``rv_rg`` (``rhov / rhog``; the median of the valid shots
        where missing), ``rossg``, ``omega`` and the other scalar variables
        as ``gedi_<name>`` (e.g. ``gedi_pai``, ``gedi_cover``); ``gedi_row``
        is the row in *shots* (for the profile variables).
    """
    import pyproj

    from alsdb.products._tiling import array_crs, array_data_bbox

    n = len(shots["shot_number"])
    if n == 0:
        return pd.DataFrame(columns=["shot_number", "center_x", "center_y", "rv_rg", "gedi_row"])
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", array_crs(provider), always_xy=True)
    x, y = to_xy.transform(shots["longitude"], shots["latitude"])
    x0, y0, x1, y1 = array_data_bbox(provider)
    keep = (x >= x0 + margin) & (x <= x1 - margin) & (y >= y0 + margin) & (y <= y1 - margin)
    counts = {"in the ALS extent": int(keep.sum())}

    def flag(name, wanted):
        return shots[name] == wanted if name in shots else np.ones(n, bool)

    if quality:
        keep &= flag("l2a_quality_flag", 1) & flag("l2b_quality_flag", 1) & flag("degrade_flag", 0)
        counts["good quality"] = int(keep.sum())
    if "sensitivity" in shots:
        keep &= shots["sensitivity"] >= min_sensitivity
        counts[f"sensitivity >= {min_sensitivity}"] = int(keep.sum())
    if leaf_on and "leaf_off_flag" in shots:
        deciduous = np.isin(shots["pft_class"], (3, 4)) if "pft_class" in shots else True
        keep &= ~((shots["leaf_off_flag"] == 1) & deciduous)
        counts["leaf-on"] = int(keep.sum())
    time = pd.to_datetime(shots["time"])
    if years is not None:
        keep &= (time.year >= years[0]) & (time.year <= years[1])
        counts[f"years {years[0]}-{years[1]}"] = int(keep.sum())
    logger.info("GEDI shots: %d queried; %s", n, counts)
    table = pd.DataFrame(
        {
            "shot_number": shots["shot_number"],
            "center_x": x,
            "center_y": y,
            "time": time,
            "gedi_row": np.arange(n),
        }
    )
    for name, values in shots.items():
        if values.ndim != 1 or name in ("shot_number", "time", "longitude", "latitude"):
            continue
        # prefixed: the simulation adds its own cover, rh and ground columns
        table[name if name in ("rossg", "omega") else f"gedi_{name}"] = values
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(shots["rhov"], float) / np.asarray(shots["rhog"], float)
    valid = np.isfinite(ratio) & (ratio > 0)
    fallback = float(np.median(ratio[valid & keep])) if (valid & keep).any() else 1.0
    table["rv_rg"] = np.where(valid, ratio, fallback)
    table.attrs["counts"] = counts
    return table[keep].reset_index(drop=True)


def canopy_profiles(out_dir: str | Path, *, rossg: float = 0.5, omega: float = 1.0) -> Path:
    """Retrieve plant area profiles from the simulated and ALS gap profiles.

    Runs gedidb's ``GEDIVerticalProfiler.compute_profiles`` (the retrieval
    behind its ``cp_pavd``) on ``pgap_waveform``, ``pgap_als_first`` and
    ``pgap_als_all`` of :func:`~alsdb.products.profiles.simulate_gedi_profiles`,
    with each shot's ``rossg`` and ``omega`` where the footprint table has
    them, for a nadir beam.

    Parameters
    ----------
    out_dir : str or Path
        Folder with ``profiles.zarr`` and ``footprints.parquet``.
    rossg, omega : float
        Leaf projection and clumping used where the table has no value.

    Returns
    -------
    Path
        ``<out_dir>/cp_profiles.npz`` with, for every kind in
        :data:`PROFILE_KINDS`, ``cp_pavd_<kind>`` (plant area per 1 % step of
        the canopy height, 101 values from the ground to the top),
        ``cp_height_<kind>`` (m) and ``canopy_height_<kind>``, in the row
        order of ``footprints.parquet``.
    """
    import zarr
    from gedidb.utils.profiles import GEDIVerticalProfiler

    out_dir = Path(out_dir)
    table = pd.read_parquet(out_dir / "footprints.parquet")
    root = zarr.open_group(str(out_dir / "profiles.zarr"), mode="r")
    # no footprint with points: the table has no signal top, and every profile is fill
    tops = table["signal_top"] if "signal_top" in table else np.full(len(table), np.nan)
    heights = _profiles.signal_heights(root["height"][:], tops)

    def per_shot(name, default):
        values = table[name].to_numpy(float) if name in table else np.full(len(table), default)
        return np.where(np.isfinite(values) & (values > 0), values, default)

    g, clumping = per_shot("rossg", rossg), per_shot("omega", omega)
    profiler = GEDIVerticalProfiler()
    result = {}
    for kind in PROFILE_KINDS:
        pavd, height, top = profiler.compute_profiles(
            root[f"pgap_{kind}"][:], heights, np.pi / 2, g, clumping
        )
        result[f"cp_pavd_{kind}"] = pavd
        result[f"cp_height_{kind}"] = height
        result[f"canopy_height_{kind}"] = top
    path = out_dir / "cp_profiles.npz"
    np.savez_compressed(path, **result)
    return path


def _plant_area_above(cp_pavd: np.ndarray) -> np.ndarray:
    """Return the plant area between each level and the canopy top (sum of the steps above)."""
    steps = np.where(np.isfinite(cp_pavd), cp_pavd, 0.0)
    above = np.zeros_like(steps, dtype=float)
    above[:, :-1] = np.cumsum(steps[:, :0:-1], axis=1)[:, ::-1]
    return above


def attenuation_summary(
    out_dir: str | Path,
    shots: dict[str, np.ndarray] | None = None,
    *,
    reference: str = "als_all",
    min_plant_area: float = 5e-3,
    min_cumulative: float = 0.05,
) -> dict:
    """Compare the profiles per relative height and fit the transmittance model.

    Parameters
    ----------
    out_dir : str or Path
        Folder with ``footprints.parquet`` and ``cp_profiles.npz``.
    shots : dict, optional
        Output of :func:`query_gedi_shots`; adds the real GEDI profile
        (``cp_pavd``) of each footprint (matched through ``gedi_row``).
    reference : str
        ALS profile the others are compared with: ``"als_all"`` or
        ``"als_first"``.
    min_plant_area : float
        Reference plant area of a 1 % step below which it is left out of the
        per-level ratios. Near the canopy top most footprints hold almost no
        plant area per step (over 1 km2 of Petit Buëch, 97 % below 0.005 at
        90-100 % of the height), and ratios of such values are noise: the
        median ratio, the ratio of means and the ratio of medians there gave
        0.4, 1.9 and infinity on the same footprints.
    min_cumulative : float
        Reference plant area above a level below which the cumulative ratio
        is left out.

    Returns
    -------
    dict
        ``profiles``: a table per relative height (0-100 %) with the median
        plant area step of every profile over the usable footprints, the
        median ratio to the reference (``ratio_<name>``) and the median ratio
        of the plant area from the canopy top down to that level
        (``cumratio_<name>``, stable where the per-level ratio is not);
        ``fits``: for the simulated waveform (and real
        GEDI), ``beta`` and ``amplitude`` of ``ratio = a * exp(-beta * C)``
        with ``C`` the reference plant area above the level, fitted to the
        median ratio; ``n_footprints``. Also written to
        ``<out_dir>/attenuation_profiles.csv``.
    """
    out_dir = Path(out_dir)
    table = pd.read_parquet(out_dir / "footprints.parquet")
    with np.load(out_dir / "cp_profiles.npz") as stored:
        cp = {name: stored[name] for name in stored.files}
    ref_pavd = cp[f"cp_pavd_{reference}"]
    observed = {"waveform": cp["cp_pavd_waveform"]}
    other = "als_first" if reference == "als_all" else "als_all"
    observed[other] = cp[f"cp_pavd_{other}"]
    if shots is not None and "cp_pavd" in shots and "gedi_row" in table:
        observed["gedi"] = np.asarray(shots["cp_pavd"], float)[table["gedi_row"].to_numpy(int)]
    usable = np.isfinite(ref_pavd).any(axis=1) & np.isfinite(cp["cp_pavd_waveform"]).any(axis=1)
    above = _plant_area_above(ref_pavd)

    def median(values):  # over footprints; levels without any value stay NaN
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(values, axis=0)

    levels = pd.DataFrame({"relative_height": np.arange(101)})
    levels[f"pavd_{reference}"] = median(np.where(usable[:, None], ref_pavd, np.nan))
    levels["plant_area_above"] = median(np.where(usable[:, None], above, np.nan))
    fits = {}
    for name, pavd in observed.items():
        rows = usable & np.isfinite(pavd).any(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(rows[:, None] & (ref_pavd > min_plant_area), pavd / ref_pavd, np.nan)
        levels[f"pavd_{name}"] = median(np.where(rows[:, None], pavd, np.nan))
        levels[f"ratio_{name}"] = median(ratio)
        obs_down = _plant_area_above(pavd) + np.where(np.isfinite(pavd), pavd, 0.0)
        ref_down = above + np.where(np.isfinite(ref_pavd), ref_pavd, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cum = np.where(rows[:, None] & (ref_down > min_cumulative), obs_down / ref_down, np.nan)
        levels[f"cumratio_{name}"] = median(cum)
        if name == other:
            continue
        fit = _profiles.fit_transmittance_model(
            levels["plant_area_above"].to_numpy(),
            np.ones(101),
            levels[f"ratio_{name}"].to_numpy(),
            free_amplitude=True,
        )
        fits[name] = {"beta": fit["beta"], "amplitude": fit["amplitude"], "n": int(rows.sum())}
    levels.to_csv(out_dir / "attenuation_profiles.csv", index=False)
    return {"profiles": levels, "fits": fits, "n_footprints": int(usable.sum())}


def run_gedi_attenuation(
    provider,
    out_dir: str | Path,
    *,
    year: int | None = None,
    gedi_provider=None,
    start_time: str = "2019-04-01",
    end_time: str | None = None,
    min_sensitivity: float = 0.95,
    leaf_on: bool = True,
    years: tuple[int, int] | None = None,
    n_workers: int = 1,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
    **simulation,
) -> dict:
    """Run the GEDI attenuation comparison for one ALS array.

    Queries the GEDI shots over the array (cached in
    ``<out_dir>/gedi_shots.npz``), simulates their waveforms and gap profiles
    from the ALS points with each shot's reflectance ratio, retrieves the
    plant area profiles and compares them with the real GEDI profiles.
    Finished steps are reused unless *overwrite*.

    Parameters
    ----------
    provider : TileDBProvider
        ALS array.
    out_dir : str or Path
        Output folder.
    year : int, optional
        ALS survey year to simulate from.
    gedi_provider, start_time, end_time
        As :func:`query_gedi_shots`.
    min_sensitivity, leaf_on, years
        As :func:`select_shots`.
    n_workers : int
        Worker processes of the simulation (scripts need an
        ``if __name__ == "__main__":`` guard).
    rng : np.random.Generator, optional
        Generator for reproducible waveform noise.
    overwrite : bool
        Redo the simulation and the retrieval (the shot cache is kept:
        delete ``gedi_shots.npz`` to query again).
    **simulation
        Passed to :func:`~alsdb.products.profiles.simulate_gedi_profiles`.

    Returns
    -------
    dict
        Output of :func:`attenuation_summary`, plus ``shots`` (the selected
        shot table) and ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = query_gedi_shots(
        provider,
        out_dir / "gedi_shots.npz",
        gedi_provider=gedi_provider,
        start_time=start_time,
        end_time=end_time,
    )
    selected = select_shots(
        shots, provider, min_sensitivity=min_sensitivity, leaf_on=leaf_on, years=years
    )
    if not len(selected):
        raise ValueError("no usable GEDI shot falls inside the ALS extent")
    if overwrite or not (out_dir / "footprints.parquet").exists():
        _profiles.simulate_gedi_profiles(
            provider,
            selected,
            out_dir,
            rv_rg_col="rv_rg",
            year=year,
            n_workers=n_workers,
            rng=rng,
            **simulation,
        )
        (out_dir / "cp_profiles.npz").unlink(missing_ok=True)
    if not (out_dir / "cp_profiles.npz").exists():
        canopy_profiles(out_dir)
    summary = attenuation_summary(out_dir, shots)
    return {**summary, "shots": selected, "out_dir": out_dir}
