"""The example notebooks run as written on the bundled sample tile.

Every code cell of examples/*.ipynb is executed in order, one namespace per
notebook, all in one temporary folder (the notebooks share example_array and
example.zarr, as when a user runs them one after the other).
"""

import json
import os
from pathlib import Path

import matplotlib
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_notebooks_run_on_the_sample(tmp_path):
    pytest.importorskip("sklearn")  # 02_processing trains a random forest
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebooks = sorted(EXAMPLES.glob("*.ipynb"))
    assert len(notebooks) == 4
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for path in notebooks:
            namespace = {"__name__": "__main__"}
            cells = [c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
            for i, cell in enumerate(cells):
                code = "".join(cell["source"])
                try:
                    # the point of the test: run the published code
                    exec(compile(code, f"{path.name}[{i}]", "exec"), namespace)  # noqa: S102
                except Exception as exc:  # noqa: BLE001 - any failure, reported with its cell
                    pytest.fail(f"{path.name} cell {i} failed: {exc!r}\n{code}")
                plt.close("all")
    finally:
        os.chdir(cwd)
    assert (tmp_path / "example.zarr").exists() and (tmp_path / "shots_2014.parquet").exists()
