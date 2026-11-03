"""Build the optional native backends: pixi run -e spikefree build-spikefree.

- ``_spikefree_native``: constrained-Delaunay spike-free (CGAL);
- ``_ptd_native``: progressive TIN densification, the vendored r-lidar/PTD
  (MIT; see ``native/ptd/README.md``).
"""

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def _build(name, sources, extra):
    import pybind11

    root = Path(__file__).parent
    prefix = Path(sys.prefix)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    target = root / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    command = compiler + [
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *(str(root / "native" / s) for s in sources),
        "-o",
        str(target),
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_path("include"),
        "-I" + str(prefix / "include"),
        "-L" + str(prefix / "lib"),
        "-Wl,-rpath," + str(prefix / "lib"),
        *extra,
    ]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    subprocess.run(command, check=True)
    print(target)


def main():
    _build("_spikefree_native", ["spikefree.cpp"], ["-lgmp", "-lmpfr"])
    native = Path(__file__).parent / "native"
    _build(
        "_ptd_native",
        [
            "ptd_binding.cpp",
            "ptd/ptd/PTD.cpp",
            "ptd/hporro/delaunay.cpp",
            "ptd/hporro/index.cpp",
            "ptd/hporro/pred3d.cpp",
        ],
        ["-I" + str(native / "ptd"), "-fopenmp", "-w"],  # -w: upstream code, unchanged
    )


if __name__ == "__main__":
    main()
