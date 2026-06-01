# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Multi-temporal change detection between two ALS survey years.

Reads two time slices of any gridded variable from an
:class:`~alsdb.storage.ALSZarrStore` and writes three derived products:

- ``{variable}_delta``       — absolute change (year_to − year_from), same units.
- ``{variable}_delta_pct``   — relative change in %, NaN where |year_from| < 1e-6.
- ``{variable}_change_flag`` — +1 gain, −1 loss, 0 no significant change.

Usage::

    from alsdb.storage import ALSZarrStore
    from alsdb.processing.change import compute_change

    store = ALSZarrStore("output/spain.zarr")

    # CHM change between two surveys, ignoring sub-0.5 m differences
    compute_change(store, "chm", year_from=2017, year_to=2022,
                   resolution=1.0, min_delta=0.5)

    # LAI change
    compute_change(store, "lai", year_from=2017, year_to=2022, resolution=10.0)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alsdb.storage.zarr_store import ALSZarrStore

logger = logging.getLogger(__name__)


def compute_change(
    store: "ALSZarrStore",
    variable: str,
    year_from: int,
    year_to: int,
    resolution: float,
    *,
    min_delta: float = 0.0,
    overwrite: bool = False,
) -> None:
    """
    Compute per-pixel change between two survey years for *variable*.

    Reads both time slices from *store* and writes three derived variables:

    - ``{variable}_delta``       : absolute change ``year_to − year_from``,
      same units as the source variable.  NaN where either year has no data.
    - ``{variable}_delta_pct``   : relative change in %.  NaN where
      ``|year_from| < 1e-6`` (avoids division by near-zero).
    - ``{variable}_change_flag`` : +1 gain, −1 loss, 0 no significant change.
      Pixels are flagged only when ``|delta| > min_delta``.

    Parameters
    ----------
    store:
        :class:`~alsdb.storage.ALSZarrStore` containing the source variable.
    variable:
        Variable name, e.g. ``"chm"``, ``"lai"``, ``"biomass"``.
    year_from, year_to:
        Survey years to compare.  Both must already be present in *store*.
    resolution:
        Resolution group in metres (e.g. ``1.0`` or ``10.0``).
    min_delta:
        Minimum absolute change flagged as significant (default 0).
        Set to e.g. ``0.5`` m for CHM to suppress sub-pixel noise.
    overwrite:
        Re-compute even if the output variables already exist for *year_to*.
    """
    from alsdb.storage.zarr_store import _res_str

    if year_from == year_to:
        raise ValueError(f"year_from and year_to must differ; got {year_from}.")

    res_key = _res_str(resolution)
    if res_key not in store._root:
        raise KeyError(f"Resolution group '{res_key}' not found in store.")
    grp = store._root[res_key]
    if variable not in grp:
        raise KeyError(f"Variable '{variable}' not found in group '{res_key}'.")

    time_arr = grp["time"][:]
    for yr, label in [(year_from, "year_from"), (year_to, "year_to")]:
        if yr not in time_arr:
            raise ValueError(
                f"{label}={yr} not present in store for '{variable}' @ {res_key}. "
                f"Available years: {sorted(time_arr.tolist())}"
            )

    delta_name = f"{variable}_delta"
    pct_name = f"{variable}_delta_pct"
    flag_name = f"{variable}_change_flag"

    if not overwrite and all(
        store.has_data(n, resolution, year_to) for n in (delta_name, pct_name, flag_name)
    ):
        logger.info(
            "Change products for %s %d→%d already present at %.0f m — skipping "
            "(pass overwrite=True to recompute)",
            variable,
            year_from,
            year_to,
            resolution,
        )
        return

    t_from = int(np.where(time_arr == year_from)[0][0])
    t_to = int(np.where(time_arr == year_to)[0][0])

    data_from = grp[variable][t_from].astype(np.float32)
    data_to = grp[variable][t_to].astype(np.float32)

    delta = data_to - data_from  # NaN propagates from either input

    with np.errstate(invalid="ignore", divide="ignore"):
        delta_pct = np.where(
            np.abs(data_from) > 1e-6,
            100.0 * delta / data_from,
            np.nan,
        ).astype(np.float32)

    flag = np.where(
        np.isnan(delta),
        np.nan,
        np.where(delta > min_delta, 1.0, np.where(delta < -min_delta, -1.0, 0.0)),
    ).astype(np.float32)

    # Full-grid bbox derived from group attributes
    attrs = dict(grp.attrs)
    x0 = float(attrs["x_origin"])
    y1 = float(attrs["y_origin"])  # top-left corner (north-up)
    nx = int(attrs["nx"])
    ny = int(attrs["ny"])
    full_bbox = (x0, y1 - ny * resolution, x0 + nx * resolution, y1)
    crs_wkt = attrs.get("crs_wkt", "")

    for out_var, out_data in [
        (delta_name, delta),
        (pct_name, delta_pct),
        (flag_name, flag),
    ]:
        store.ensure_group(out_var, resolution, full_bbox, crs_wkt)
        store.write_tile(out_var, resolution, year_to, out_data, full_bbox)

    gain = int((flag == 1).sum())
    loss = int((flag == -1).sum())
    logger.info(
        "Change %s %d→%d @ %.0f m — gain: %d px  loss: %d px  "
        "(delta, delta_pct, change_flag written)",
        variable,
        year_from,
        year_to,
        resolution,
        gain,
        loss,
    )
