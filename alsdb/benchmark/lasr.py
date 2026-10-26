"""lasR reference: pitfree and spike-free CHM tiles from classified LAS/LAZ.

lasR (r-lidar, GPL-3) is an independent C++ implementation of the same
algorithms with its own normalization. It runs as a separate ``Rscript``
process on the bundled ``lasr_chm.R``; nothing of lasR is linked into alsdb.
Install it with the optional pixi environment::

    pixi install -e lasr && pixi run -e lasr install-lasr

The script mirrors alsdb surface algorithm 3: noise (7, 18) dropped, ground +
water TIN normalization, surface classes = vegetation + ground + water,
pitfree with highest return per cell and edge-trimmed upper layers, spike-free
on all returns. ``"auto"`` freeze/edge apply alsdb's rule per tile. On a
Barcelona PNOA tile, pitfree matched alsdb on 99.3% of cells (< 1 cm) and
spike-free differed by +0.18 m mean (alsdb higher; seeding differs).
"""

import shutil
import subprocess
from pathlib import Path

from alsdb.benchmark._common import write_provenance

SCRIPT = Path(__file__).with_name("lasr_chm.R")


def lasr_version(rscript="Rscript"):
    """Installed lasR version, or a clear error if R or lasR is missing."""
    resolved = shutil.which(rscript)
    if resolved is None:
        raise FileNotFoundError(
            f"{rscript} not found; use the optional environment: pixi install -e lasr"
        )
    result = subprocess.run(
        [resolved, "-e", "cat(as.character(packageVersion('lasR')))"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "lasR is not installed in this R (pixi run -e lasr install-lasr): "
            + result.stderr[-500:]
        )
    return result.stdout.strip().splitlines()[-1]


def run(
    laz_dir,
    out_dir,
    *,
    veg_classes=(3, 4, 5),
    freeze_distance="auto",
    max_edge="auto",
    n_workers=8,
    tiles=None,
    normalized=False,
    rscript="Rscript",
):
    """Write ``<out_dir>/pitfree/<tile>.tif`` and ``<out_dir>/spikefree/<tile>.tif``.

    Each tile is read with a 50 m buffer from its neighbours. Finished tiles
    are skipped, so rerunning resumes; ``tiles`` limits a run to the first N
    tiles (pilot), as in :func:`alsdb.benchmark.lastools.run`. ``normalized=True`` treats Z as height above
    ground already (lasR's normalization is skipped). Per-tile parameters go
    to ``params.csv``, the run record (lasR version, settings) to
    ``runs.json``.
    """
    version = lasr_version(rscript)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which(rscript),
        str(SCRIPT),
        str(laz_dir),
        str(out_dir),
        f"--ncores={int(n_workers)}",
        f"--veg={','.join(str(int(c)) for c in veg_classes)}",
        f"--freeze={freeze_distance}",
        f"--edge={max_edge}",
    ]
    if tiles is not None:
        command.append(f"--tiles={int(tiles)}")
    if normalized:
        command.append("--normalized")
    subprocess.run(command, check=True)
    write_provenance(
        out_dir,
        {
            "tool": "lasR",
            "version": version,
            "params": {
                "laz_dir": str(laz_dir),
                "veg_classes": list(veg_classes),
                "freeze_distance": freeze_distance,
                "max_edge": max_edge,
                "tiles": tiles,
                "normalized": normalized,
            },
        },
    )
    return {"pitfree": out_dir / "pitfree", "spikefree": out_dir / "spikefree"}
