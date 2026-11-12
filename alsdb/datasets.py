# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Sample data shipped with alsdb, so the examples run as written.

``sample_laz()`` is 100 m x 100 m of tropical forest (canopy up to ~45 m)
from the 2014 survey of research site RIB_A01 in Acre, Brazil (Sustainable
Landscapes Brazil project), cropped from tile ``RIB_A01_2014_laz_0`` with its
original points (about 66 per m², classes 2-5, up to four returns per pulse).
The CRS (WGS 84 / UTM zone 19S, EPSG:32719, as given in the data guide) and the
survey year are written into the header.

Data: dos-Santos, M.N., M.M. Keller, and D.C. Morton. 2019. LiDAR Surveys over
Selected Forest Research Sites, Brazilian Amazon, 2008-2018. ORNL DAAC, Oak
Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1644 (NASA data, open
for reuse; please cite the dataset).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

SAMPLE = {
    "file": "RIB_A01_2014_sample.laz",
    "crs": "EPSG:32719",
    "year": 2014,
    "bbox": (658_180.0, 8_901_860.0, 658_280.0, 8_901_960.0),
    "citation": (
        "dos-Santos, M.N., M.M. Keller, and D.C. Morton. 2019. LiDAR Surveys over Selected "
        "Forest Research Sites, Brazilian Amazon, 2008-2018. ORNL DAAC, Oak Ridge, Tennessee, "
        "USA. https://doi.org/10.3334/ORNLDAAC/1644"
    ),
}


def sample_laz() -> Path:
    """Return the path of the bundled sample LAZ tile (see the module docstring).

    Returns
    -------
    pathlib.Path
        ``RIB_A01_2014_sample.laz``: 100 m x 100 m, 2014, EPSG:32719, bounds
        ``sample_info()["bbox"]``.
    """
    return Path(str(files("alsdb") / "data" / "sample" / SAMPLE["file"]))


def sample_info() -> dict:
    """Return the sample's CRS, survey year, bounding box and data citation.

    Returns
    -------
    dict
        Keys ``file``, ``crs``, ``year``, ``bbox`` (min_x, min_y, max_x, max_y)
        and ``citation``.
    """
    return dict(SAMPLE)
