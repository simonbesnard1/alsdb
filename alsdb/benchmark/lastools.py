"""LAStools references: in-memory adapter and file-based reference products.

:func:`rasterize_lastools` runs ``las2dem -spike_free`` on points alsdb has
already selected and normalized (used by ``method="lastools"`` and the point
benchmark). :func:`run` builds reference CHM tiles from classified LAS/LAZ
files with LAStools' own normalization, independent of alsdb.

LAStools (rapidlasso GmbH, https://rapidlasso.de; ``las2dem -spike_free``
implements Khosravipour et al. 2016, https://doi.org/10.1016/j.jag.2016.06.005)
is proprietary and not installed by alsdb; executables are called as
separate processes. Without a licence they degrade output on large files, so
every command passes ``-fail`` to error instead.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from alsdb.benchmark._common import (
    auto_parameters,
    list_tiles,
    run_parallel,
    surface_classes,
    write_params,
    write_provenance,
)
from alsdb.methods._grid import GridSpec
from alsdb.methods._terrain import GROUND_CLASSES

logger = logging.getLogger(__name__)


def executable_version(executable="las2dem64", env=None):
    """Return the version reported by a LAStools executable."""
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"LAStools executable not found: {executable}")
    result = subprocess.run(
        [resolved, "-version"], capture_output=True, text=True, timeout=30, check=False, env=env
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or "LAStools version query failed")
    return (result.stdout + result.stderr).strip()


def rasterize_lastools(
    points,
    bbox,
    resolution,
    *,
    executable="las2dem64",
    freeze_distance=1.5,
    height_buffer=0.5,
    freeze_interval=0.25,
    max_triangle_edge=100.0,
    value_field="HeightAboveGround",
    demo=False,
    timeout=600,
):
    """Run las2dem on exactly the provided eligible points and height field.

    Uses a temporary ASCII point cloud to preserve input precision without LAS
    quantization. Demo mode is explicit and limited to fewer than 1.5M points,
    as documented by rapidlasso. No executable is downloaded automatically.
    """
    import rasterio
    from rasterio.transform import from_origin

    grid = GridSpec.from_bbox(bbox, resolution)
    if not np.isfinite([freeze_distance, height_buffer, freeze_interval, max_triangle_edge]).all():
        raise ValueError("LAStools parameters must be finite")
    if min(freeze_distance, freeze_interval, max_triangle_edge) <= 0 or height_buffer < 0:
        raise ValueError("Invalid spike-free parameters")
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"LAStools executable not found: {executable}")
    xyz = np.column_stack((points["X"], points["Y"], points[value_field]))
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if demo and len(xyz) >= 1_500_000:
        raise ValueError("Demo comparison requires fewer than 1.5 million points")
    if len(xyz) < 3:
        return np.full(grid.shape, np.nan, np.float32)
    with tempfile.TemporaryDirectory(prefix="alsdb-lastools-") as directory:
        source, target = Path(directory) / "points.txt", Path(directory) / "surface.tif"
        np.savetxt(source, xyz, fmt="%.17g")
        command = [
            resolved,
            "-i",
            str(source),
            "-iparse",
            "xyz",
            "-o",
            str(target),
            "-spike_free",
            str(freeze_distance),
            str(freeze_interval),
            str(height_buffer),
            "-step",
            str(resolution),
            "-ll",
            str(grid.bbox[0]),
            str(grid.bbox[1]),
            "-ncols",
            str(grid.nx),
            "-nrows",
            str(grid.ny),
            "-nbits",
            "32",
            "-kill",
            str(max_triangle_edge),
            "-no_kml",
            "-demo" if demo else "-fail",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        if not _succeeded(result, target):
            raise RuntimeError("LAStools failed: " + (result.stderr + result.stdout)[-4000:])
        with rasterio.open(target) as dataset:
            expected = from_origin(grid.x0, grid.y1, resolution, resolution)
            if dataset.shape != grid.shape or not dataset.transform.almost_equals(expected):
                raise ValueError("LAStools output grid differs from the requested pixel lattice")
            return dataset.read(1, masked=True).astype(np.float32).filled(np.nan)


# ---------------------------------------------------------------------------
# File-based reference products
# ---------------------------------------------------------------------------

PRODUCTS = ("spikefree", "max", "dtm")


def _tool(name, lastools_dir):
    """Path of LAStools tool *name* (e.g. "las2dem64") and its environment.

    LAStools needs its own ``lib`` on LD_LIBRARY_PATH; setting that only for
    the subprocess keeps it from shadowing the GDAL/SQLite this Python loaded.
    """
    env = os.environ.copy()
    if lastools_dir is None:
        resolved = shutil.which(name)
    else:
        resolved = str(Path(lastools_dir) / name)
        lib = Path(lastools_dir) / "lib"
        if lib.is_dir():
            env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if resolved is None or not Path(resolved).is_file():
        raise FileNotFoundError(
            f"LAStools executable not found: {name} (lastools_dir={lastools_dir})"
        )
    return resolved, env


def _succeeded(result, target):
    """LAStools exits with 1 after warnings; count that as success when the output exists and no ERROR was printed."""
    text = result.stderr + result.stdout
    ok = result.returncode == 0 or (result.returncode == 1 and "ERROR" not in text)
    if ok and result.returncode and Path(target).is_file():
        logger.debug("LAStools warnings for %s: %s", target, text.strip()[-500:])
    return ok and Path(target).is_file()


# A licensed LAStools occasionally reports itself "unlicensed" for one call
# (licence file briefly unreadable): retry after a growing pause.
LICENCE_RETRIES = 5
LICENCE_WAIT = 30.0  # s, times the attempt number


def _licence_failure(result):
    text = result.stderr + result.stdout
    return "license failure" in text or "(unlicensed)" in text


def _call(command, env, target, timeout=None):
    target = Path(target)
    for attempt in range(1, LICENCE_RETRIES + 2):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, env=env, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:  # the child is killed by subprocess.run
            target.unlink(missing_ok=True)  # a hung las2dem leaves an empty output behind
            raise RuntimeError(
                f"LAStools timed out after {timeout} s for {target}\n"
                f"command: {' '.join(map(str, command))}"
            ) from exc
        if not _licence_failure(result) or attempt > LICENCE_RETRIES:
            break
        target.unlink(missing_ok=True)
        logger.warning("LAStools licence failure for %s (attempt %d), retrying in %.0f s",
                       target.name, attempt, LICENCE_WAIT * attempt)  # fmt: skip
        time.sleep(LICENCE_WAIT * attempt)
    if not _succeeded(result, target):
        # LAStools often prints only warnings; say which check failed and what
        # did appear, so a renamed or missing output is distinguishable from a crash.
        siblings = sorted(p.name for p in target.parent.glob(target.stem + "*"))
        raise RuntimeError(
            f"LAStools failed for {target}: exit code {result.returncode}, "
            f"output {'exists' if target.is_file() else 'missing'}, "
            f"files named {target.stem}* in {target.parent}: {siblings}\n"
            f"command: {' '.join(map(str, command))}\n"
            f"output:\n" + (result.stderr + result.stdout)[-3000:]
        )


def run(
    laz_dir,
    out_dir,
    *,
    products=("spikefree", "max"),
    veg_classes=(3, 4, 5),
    ground_classes=GROUND_CLASSES,
    resolution=1.0,
    freeze_distance="auto",
    freeze_interval=0.25,
    height_buffer=0.5,
    max_triangle_edge=100.0,
    lastools_dir=None,
    hag_dir=None,
    n_workers=1,
    tiles=None,
    timeout=3600,
):
    """Generate reference CHM tiles from classified LAS/LAZ with LAStools.

    ``lasheight -replace_z`` normalises points on LAStools' own ground TIN;
    ``spikefree`` runs ``las2dem -spike_free`` on the surface classes and
    ``max`` runs ``lascanopy -max`` on their first returns. ``dtm`` runs
    ``las2dem`` on the ground classes of the raw tile (a TIN of ground and
    water, as alsdb's DTM); it needs no normalisation.
    ``freeze_distance="auto"`` applies alsdb's rule per tile. Tiles have no
    neighbour buffer. Finished outputs are skipped; *hag_dir* reuses existing
    heights; each call is killed after *timeout* seconds.
    """
    products = tuple(products)
    if not set(products) <= set(PRODUCTS):
        raise ValueError(f"products must be among {PRODUCTS}")
    out_dir = Path(out_dir)
    classes = " ".join(map(str, surface_classes(veg_classes, ground_classes)))
    lasheight, env = _tool("lasheight64", lastools_dir)  # products=() runs only lasheight
    needs_dem = {"spikefree", "dtm"} & set(products)
    las2dem, _ = _tool("las2dem64", lastools_dir) if needs_dem else (None, None)
    ground = " ".join(map(str, ground_classes))
    needs_hag = not products or {"spikefree", "max"} & set(products)
    lascanopy, _ = _tool("lascanopy64", lastools_dir) if "max" in products else (None, None)
    subs = ("hag", *products) if needs_hag else products
    for sub in subs:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    files = list_tiles(laz_dir, tiles)
    record = {
        "version": executable_version(las2dem or lascanopy or lasheight, env=env),
        "params": [],
    }

    # Outputs go under .partial/ and are moved into place only on success, so
    # an interrupted run never leaves a truncated file that looks finished.
    for sub in subs:
        shutil.rmtree(out_dir / sub / ".partial", ignore_errors=True)
        (out_dir / sub / ".partial").mkdir()

    def partial(target):
        return target.parent / ".partial" / target.name

    def tile(path):
        stem = path.stem
        targets = {p: out_dir / p / f"{stem}.tif" for p in products}
        if targets and all(t.is_file() for t in targets.values()):
            return None
        hag = Path(hag_dir) / f"{stem}.laz" if hag_dir is not None else None
        if hag is None or not hag.is_file():
            hag = out_dir / "hag" / f"{stem}.laz"
        if needs_hag and not hag.is_file():
            _call(
                [
                    lasheight,
                    "-i",
                    str(path),
                    "-replace_z",
                    "-olaz",
                    "-o",
                    str(partial(hag)),
                    "-fail",
                ],
                env,
                partial(hag),
                timeout,
            )
            os.replace(partial(hag), hag)
        row = {"tile": path.name, "freeze_distance": freeze_distance}
        if "spikefree" in products and not targets["spikefree"].is_file():
            freeze = freeze_distance
            if freeze == "auto":
                freeze, _ = auto_parameters(path, veg_classes, ground_classes, resolution)
                if freeze is None:
                    logger.warning("%s: too few surface first returns, skipped", path.name)
                    return None
                row["freeze_distance"] = round(freeze, 2)
            command = [
                las2dem, "-i", str(hag), "-keep_class", *classes.split(),
                "-spike_free", f"{freeze:.2f}", str(freeze_interval), str(height_buffer),
                "-kill", str(max_triangle_edge), "-step", str(resolution),
                "-nbits", "32", "-no_kml", "-fail", "-o", str(partial(targets["spikefree"])),
            ]  # fmt: skip
            _call(command, env, partial(targets["spikefree"]), timeout)
            os.replace(partial(targets["spikefree"]), targets["spikefree"])
        if "dtm" in products and not targets["dtm"].is_file():
            command = [
                las2dem, "-i", str(path), "-keep_class", *ground.split(),
                "-kill", str(max_triangle_edge), "-step", str(resolution),
                "-nbits", "32", "-no_kml", "-fail", "-o", str(partial(targets["dtm"])),
            ]  # fmt: skip
            _call(command, env, partial(targets["dtm"]), timeout)
            os.replace(partial(targets["dtm"]), targets["dtm"])
        if "max" in products and not targets["max"].is_file():
            # lascanopy appends "_max" to the output name it is given.
            stub = partial(targets["max"]).with_name(f"{stem}.tif")
            written = partial(targets["max"]).with_name(f"{stem}_max.tif")
            command = [
                lascanopy, "-i", str(hag), "-keep_first", "-keep_class", *classes.split(),
                "-max", "-height_cutoff", "0", "-step", str(resolution), "-fail",
                "-o", str(stub),
            ]  # fmt: skip
            _call(command, env, written, timeout)
            os.replace(written, targets["max"])
        return row

    rows = [r for r in run_parallel(tile, files, n_workers) if r is not None]
    if rows:
        write_params(out_dir, rows, ["tile", "freeze_distance"])
    record["params"] = {
        "laz_dir": str(laz_dir),
        "hag_dir": None if hag_dir is None else str(hag_dir),
        "products": products,
        "surface_classes": classes,
        "ground_classes": ground,
        "resolution": resolution,
        "freeze_distance": freeze_distance,
        "freeze_interval": freeze_interval,
        "height_buffer": height_buffer,
        "max_triangle_edge": max_triangle_edge,
        "tiles_processed": len(rows),
    }
    write_provenance(out_dir, {"tool": "lastools", **record})
    return {p: out_dir / p for p in products}
