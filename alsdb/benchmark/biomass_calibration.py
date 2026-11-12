"""Calibrate the Næsset biomass model against Spanish forest inventory (IFN4) plots.

For every inventory plot inside the ALS array, the structural metrics over the
plot's 25 m circle are computed with :func:`alsdb.products.biomass.plot_metrics`
(the code path of ``compute_metrics(..., baba_radius=25)``), joined with the
field AGB density, and ``AGB = a * h95^b * cc^c`` (Næsset 2002) is fitted with
:func:`alsdb.products.allometry.calibrate_naesset`. Skill is reported out of
fold with spatially grouped cross-validation
(:func:`alsdb.products.allometry.validate_naesset`, 10 km blocks), never in
sample.

Plots whose ALS canopy is far below the tallest field tree are flagged
(``mismatch``): over Barcelona these were neither located elsewhere within
20 m nor shifted as a group, so they are disturbance, a wrong plot or wrong
field heights. Fits are reported with and without them; the flag uses field
heights, so the screened fit is optimistic and both are given.

    pixi run python -m alsdb.benchmark calibrate-biomass <tiledb> <nfi_plot_csv> <out_dir>
        --province 8

Outputs: ``plots.csv`` (field and ALS values per plot), ``calibration.json``
(coefficients, cross-validated skill, settings) and ``calibration.png``.

Field data: Spanish National Forest Inventory (IFN4, MITECO), harmonised plot
table. Model: Næsset, E. (2002), Remote Sens. Environ. 80, 88-99,
https://doi.org/10.1016/S0034-4257(01)00290-5.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("h95", "cc", "hmean", "hmax", "h50")


def load_plots(nfi_csv, provinces=None):
    """Inventory plots (one row each) with field AGB density in Mg/ha."""
    cols = ["PlotID", "Province", "Latitude", "Longitude", "AGBD", "Height_max", "Year_max",
            "Num_trees", "Species_dom"]  # fmt: skip
    plots = pd.read_csv(nfi_csv, usecols=cols, dtype={"Province": str, "PlotID": str},
                        low_memory=False)  # fmt: skip
    if provinces:
        plots = plots[plots.Province.isin([str(p) for p in provinces])]
    return plots.reset_index(drop=True)


def plot_table(provider, plots, *, radius=25.0, n_workers=8, crs=None):
    """Join the plots with their ALS metrics over the plot circle."""
    from pyproj import Transformer

    from alsdb.products.biomass import plot_metrics

    crs = crs or provider.stored_crs()
    if not crs:
        raise ValueError("the array has no CRS: pass crs")
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        plots.Longitude.to_numpy(), plots.Latitude.to_numpy()
    )
    metrics = plot_metrics(provider, x, y, radius, None, metrics=METRICS, n_workers=n_workers)
    table = pd.concat([plots.assign(x=x, y=y), metrics], axis=1)
    return table


def calibrate(table, *, min_points=200, mismatch_m=10.0, block_m=10_000.0, n_splits=5):
    """Fit and cross-validate the Næsset model on all usable plots and on the unflagged ones."""
    from alsdb.products.allometry import calibrate_naesset, validate_naesset

    t = table.copy()
    t["usable"] = (t.n_points >= min_points) & np.isfinite(t.cc) & np.isfinite(t.AGBD)
    t["mismatch"] = (t.Height_max - t.hmax) > mismatch_m
    t["block"] = ((t.x // block_m).astype(int).astype(str) + "_"
                  + (t.y // block_m).astype(int).astype(str))  # fmt: skip
    results = {}
    for name, keep in (("all", t.usable), ("screened", t.usable & ~t.mismatch)):
        d = t[keep]
        h95 = np.nan_to_num(d.h95.to_numpy(), nan=0.0)  # no vegetation return: cc = 0, AGB 0
        cc, agb = d.cc.to_numpy(), d.AGBD.to_numpy()
        (a, b, c), cov = calibrate_naesset(h95, cc, agb, return_cov=True)
        cv = validate_naesset(h95, cc, agb, d.block.to_numpy(), n_splits=n_splits)
        pred = cv["prediction"]
        ok = np.isfinite(pred)
        r2 = 1 - np.sum((agb[ok] - pred[ok]) ** 2) / np.sum((agb[ok] - agb[ok].mean()) ** 2)
        t.loc[d.index, f"cv_pred_{name}"] = pred
        results[name] = {
            "n": len(d), "a": a, "b": b, "c": c,
            "se": [float(v) for v in np.sqrt(np.diag(cov))],
            "cv_rmse": cv["rmse"], "cv_bias": cv["bias"], "cv_r2": float(r2),
            "cv_rmse_relative": cv["rmse"] / float(agb.mean()),
            "field_mean": float(agb.mean()),
        }  # fmt: skip
    settings = {"min_points": min_points, "mismatch_m": mismatch_m, "block_m": block_m,
                "n_splits": n_splits, "n_plots": len(t), "n_usable": int(t.usable.sum()),
                "n_mismatch": int((t.usable & t.mismatch).sum())}  # fmt: skip
    return t, results, settings


def figure(t, results, out):
    """Field against out-of-fold predicted AGBD, for all usable and for unflagged plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, ("all", "screened")):
        r = results[name]
        d = t[t[f"cv_pred_{name}"].notna()]
        ax.scatter(d.AGBD, d[f"cv_pred_{name}"], s=6, alpha=0.4, color="#2d6a4f", lw=0)
        lim = [0, np.nanpercentile(t.AGBD, 99.5)]
        ax.plot(lim, lim, color="#4a5568", lw=0.8)
        ax.set(xlim=lim, ylim=lim, xlabel="field AGBD (Mg/ha)", aspect="equal")
        ax.set_title(f"{name}: n = {r['n']}, RMSE {r['cv_rmse']:.0f} Mg/ha "
                     f"({100 * r['cv_rmse_relative']:.0f} %), R² {r['cv_r2']:.2f}",
                     fontsize=11, loc="left")  # fmt: skip
    axes[0].set_ylabel("predicted AGBD, out of fold (Mg/ha)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_calibration(tiledb_uri, nfi_csv, out_dir, *, provinces=None, radius=25.0, n_workers=8,
                    crs=None, **options):  # fmt: skip
    """Plot metrics, fit, cross-validation and outputs; see the module docstring."""
    from alsdb import ALSDatabase

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "plots.csv"
    if cache.exists():
        table = pd.read_csv(cache, dtype={"PlotID": str, "Province": str})
    else:
        provider = ALSDatabase(storage_type="local", uri=str(tiledb_uri))
        table = plot_table(provider, load_plots(nfi_csv, provinces), radius=radius,
                           n_workers=n_workers, crs=crs)  # fmt: skip
        table = table[table.n_points > 0].reset_index(drop=True)  # plots outside the ALS
        table.to_csv(cache, index=False)
    t, results, settings = calibrate(table, **options)
    t.to_csv(cache, index=False)
    record = {"results": results, "settings": {**settings, "radius": radius,
              "provinces": provinces, "tiledb": str(tiledb_uri), "nfi_csv": str(nfi_csv)}}  # fmt: skip
    (out_dir / "calibration.json").write_text(json.dumps(record, indent=2))
    figure(t, results, out_dir / "calibration.png")
    return record
