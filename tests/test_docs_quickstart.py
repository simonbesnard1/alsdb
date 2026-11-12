"""The Quick Overview page runs as written on the bundled sample tile.

Every ``.. code-block:: python`` of doc/user/quick-overview.rst is executed in
order in one namespace, in a temporary folder, except blocks marked
``:class: norun`` (those need the reader's own data).
"""

import os
import re
from pathlib import Path

import matplotlib
import pytest

PAGE = Path(__file__).resolve().parents[1] / "doc" / "user" / "quick-overview.rst"


def code_blocks(text):
    """(options, code) of every python code block, in order."""
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == ".. code-block:: python":
            i += 1
            options = []
            while i < len(lines) and lines[i].strip().startswith(":"):
                options.append(lines[i].strip())
                i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                body.append(lines[i][4:])
                i += 1
            blocks.append((options, "\n".join(body).strip("\n")))
        else:
            i += 1
    return blocks


def test_quick_overview_runs_on_the_sample(tmp_path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = code_blocks(PAGE.read_text())
    runnable = [code for options, code in blocks if ":class: norun" not in options]
    assert len(runnable) >= 6 and len(blocks) > len(runnable)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    try:
        for i, code in enumerate(runnable):
            try:
                # the point of the test: run the documented code
                exec(compile(code, f"quick-overview block {i}", "exec"), namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001 - any failure, reported with its block
                pytest.fail(f"block {i} failed: {exc!r}\n{code}")
    finally:
        os.chdir(cwd)
        plt.close("all")
    assert float(namespace["chm"].max()) > 30  # tall tropical forest
    assert 25 < namespace["result"].rh[98] < 50


def test_code_block_parser_reads_options_and_bodies():
    text = ".. code-block:: python\n   :class: norun\n\n    a = 1\n    b = 2\n\nText\n"
    assert code_blocks(text) == [([":class: norun"], "a = 1\nb = 2")]
    assert not re.search(r"\t", PAGE.read_text())
