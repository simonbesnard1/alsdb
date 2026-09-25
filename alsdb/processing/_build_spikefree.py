"""Build the optional CGAL backend: pixi run -e spikefree build-spikefree."""

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    import pybind11

    root = Path(__file__).parent
    prefix = Path(sys.prefix)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    target = root / ("_spikefree_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = compiler + [
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(root / "native" / "spikefree.cpp"),
        "-o",
        str(target),
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_path("include"),
        "-I" + str(prefix / "include"),
        "-L" + str(prefix / "lib"),
        "-Wl,-rpath," + str(prefix / "lib"),
        "-lgmp",
        "-lmpfr",
    ]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    subprocess.run(command, check=True)
    print(target)


if __name__ == "__main__":
    main()
