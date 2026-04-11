# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""
Shared fixtures for all test modules.

The ``processed_provider`` fixture creates a TileDB array once per session
with a realistic mix of ground (class 2) and vegetation (class 3) points at
year 2021.  The ground points form a regular 10 × 10 grid so that
``filters.hag_delaunay`` can build a valid TIN without edge artefacts.
"""

import numpy as np
import pytest

from alsdb.core.alsdatabase import ALSDatabase
from alsdb.core.alsprovider import ALSProvider
from alsdb.storage.zarr_store import ALSZarrStore
from alsdb.utils.schema import LAS_ATTRIBUTES, TileDBSchemaConfig

# ---------------------------------------------------------------------------
# Constants shared across test modules
# ---------------------------------------------------------------------------

SCHEMA_CFG = TileDBSchemaConfig(
    domain_min_x=100_000.0,
    domain_max_x=900_000.0,
    domain_min_y=3_000_000.0,
    domain_max_y=9_999_900.0,
)

YEAR = 2021
# 1 km × 1 km test tile in ETRS89 / UTM Zone 30N
BBOX = (308_000.0, 4_688_000.0, 309_000.0, 4_689_000.0)
CRS = "EPSG:25830"


def _make_synthetic_points(rng: np.random.Generator):
    """
    Build a structured attribute dict for a synthetic ALS scene.

    Layout
    ------
    * 100 ground points (class 2) on a 10 × 10 regular grid at Z = 800 m.
      A regular grid guarantees a non-degenerate Delaunay TIN for hag_delaunay.
    * 150 vegetation points (class 3) randomly scattered at Z = 810–820 m
      with ReturnNumber = 1.
    """
    # --- ground ---
    gx, gy = np.meshgrid(
        np.linspace(308_050.0, 308_950.0, 10),
        np.linspace(4_688_050.0, 4_688_950.0, 10),
    )
    n_gnd = gx.size
    x_gnd = gx.ravel().astype(np.float64)
    y_gnd = gy.ravel().astype(np.float64)
    a_gnd = {k: np.zeros(n_gnd, dtype=dt) for k, dt in LAS_ATTRIBUTES.items()}
    a_gnd["Z"] = np.full(n_gnd, 800.0)
    a_gnd["Classification"] = np.full(n_gnd, 2, dtype=np.uint8)
    a_gnd["ReturnNumber"] = np.ones(n_gnd, dtype=np.uint8)
    a_gnd["NumberOfReturns"] = np.ones(n_gnd, dtype=np.uint8)

    # --- vegetation ---
    n_veg = 150
    x_veg = rng.uniform(308_100.0, 308_900.0, n_veg).astype(np.float64)
    y_veg = rng.uniform(4_688_100.0, 4_688_900.0, n_veg).astype(np.float64)
    a_veg = {k: np.zeros(n_veg, dtype=dt) for k, dt in LAS_ATTRIBUTES.items()}
    a_veg["Z"] = rng.uniform(815.0, 825.0, n_veg)
    a_veg["Classification"] = np.full(n_veg, 3, dtype=np.uint8)
    a_veg["ReturnNumber"] = np.ones(n_veg, dtype=np.uint8)
    a_veg["NumberOfReturns"] = np.ones(n_veg, dtype=np.uint8)

    x = np.concatenate([x_gnd, x_veg])
    y = np.concatenate([y_gnd, y_veg])
    attrs = {k: np.concatenate([a_gnd[k], a_veg[k]]) for k in LAS_ATTRIBUTES}
    return x, y, attrs


@pytest.fixture(scope="session")
def processed_array_uri(tmp_path_factory):
    """
    Session-scoped TileDB array with ground + vegetation points for year 2021.
    Created once and reused for all integration tests.
    """
    uri = str(tmp_path_factory.mktemp("tiledb") / "als_array")
    db = ALSDatabase(storage_type="local", uri=uri, schema_cfg=SCHEMA_CFG)
    rng = np.random.default_rng(0)
    x, y, attrs = _make_synthetic_points(rng)
    db.write(x, y, YEAR, attrs, crs=CRS)
    return uri


@pytest.fixture(scope="session")
def provider(processed_array_uri):
    """Session-scoped read-only provider over the processed array."""
    return ALSProvider(storage_type="local", uri=processed_array_uri)


@pytest.fixture()
def store(tmp_path):
    """Fresh Zarr store for each test (function scope)."""
    return ALSZarrStore(str(tmp_path / "store.zarr"))
