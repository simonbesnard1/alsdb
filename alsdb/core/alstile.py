# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pdal

from alsdb.tile.Tile import Tile
from alsdb.tile.tile_name import TileNameBase
from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)

# Height-above-ground thresholds for LAS vegetation classes 3 / 4 / 5.
# Values are in the native Z units of the LAZ file (metres or feet depending
# on the CRS).  The subdivision into 3/4/5 is approximate; what matters for
# CHM/gap/biomass is that all vegetation ends up in class 3–5, not the exact
# boundary between them.
_HAG_LOW: float = 0.5  # below this → leave as class 1 (noise / bare ground fringe)
_HAG_MED: float = 2.0  # 0.5–2.0    → class 3 (low vegetation)
_HAG_HIGH: float = 5.0  # 2.0–5.0    → class 4 (medium vegetation)
# ≥ 5.0      → class 5 (high vegetation)


class ALSTile:
    """
    Processes a single LAZ tile into arrays ready for TileDB ingestion.

    Wraps :class:`~alsdb.tile.Tile.Tile` and applies an optional point filter
    before handing data to :class:`~alsdb.core.alsdatabase.ALSDatabase`.

    Parameters
    ----------
    path:
        Path to the ``.laz`` file.
    classification_filter:
        If provided, only points whose ``Classification`` code is in this list
        are passed through.  E.g. ``[2]`` for ground only,
        ``[1, 2, 5]`` for unclassified + ground + high vegetation.
    reclassify:
        If ``True``, run ``filters.smrf`` to identify ground points (class 2),
        then assign LAS vegetation classes (3/4/5) to unclassified (class 1)
        points based on height above ground.  Use this when ingesting surveys
        that were delivered with only minimal classification (e.g. USGS LPC
        files where all non-ground returns are class 1).  Points already
        carrying a non-unclassified label are left unchanged.
    """

    def __init__(
        self,
        path: str | Path,
        classification_filter: Optional[list[int]] = None,
        reclassify: bool = False,
    ) -> None:
        self._tile = Tile(path)
        self._classification_filter = classification_filter
        self._reclassify = reclassify

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tile(self) -> Tile:
        return self._tile

    @property
    def name(self) -> TileNameBase:
        return self._tile.name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        if self._classification_filter is None:
            return data
        mask = np.isin(data["Classification"], self._classification_filter)
        logger.debug(
            "Classification filter %s: kept %d / %d points",
            self._classification_filter,
            int(mask.sum()),
            len(data),
        )
        return data[mask]

    def _z_unit_scale(self) -> float:
        """Return metres-to-native-Z scale factor (1.0 for metric CRS, ~3.281 for feet)."""
        try:
            from pyproj import CRS

            crs = CRS.from_user_input(self.name.crs)
            unit = (
                crs.axis_info[2].unit_name if len(crs.axis_info) > 2 else crs.axis_info[0].unit_name
            )
            if "foot" in unit.lower() or "feet" in unit.lower():
                return 3.280839895
        except Exception:
            pass
        return 1.0

    def _apply_reclassification(self, data: np.ndarray) -> np.ndarray:
        """
        Classify ground and vegetation points using SMRF + HAG.

        Only unclassified (class 1) points are relabelled — existing
        classifications (ground, noise, overlap, etc.) are preserved.
        Thresholds are defined in metres and scaled to the file's native Z units.
        """
        scale = self._z_unit_scale()
        low = _HAG_LOW * scale
        med = _HAG_MED * scale
        high = _HAG_HIGH * scale

        stages = [
            # SMRF: promotes class-1 ground candidates to class 2.
            # ignore high-noise (18) so they don't confuse the surface model.
            {"type": "filters.smrf", "ignore": "Classification[7:7],Classification[18:18]"},
            {"type": "filters.hag_delaunay"},
            {
                "type": "filters.assign",
                "value": "HeightAboveGround = 0 WHERE HeightAboveGround < 0",
            },
            # Assign vegetation classes only to still-unclassified points.
            {
                "type": "filters.assign",
                "value": (
                    f"Classification = 3 WHERE Classification == 1"
                    f" && HeightAboveGround >= {low}"
                    f" && HeightAboveGround < {med}"
                ),
            },
            {
                "type": "filters.assign",
                "value": (
                    f"Classification = 4 WHERE Classification == 1"
                    f" && HeightAboveGround >= {med}"
                    f" && HeightAboveGround < {high}"
                ),
            },
            {
                "type": "filters.assign",
                "value": (
                    f"Classification = 5 WHERE Classification == 1 && HeightAboveGround >= {high}"
                ),
            },
        ]
        p = pdal.Pipeline(json.dumps(stages), arrays=[data])
        p.execute()
        result = p.arrays[0]
        n_veg = int(np.isin(result["Classification"], [3, 4, 5]).sum())
        logger.debug("Reclassification complete: %d vegetation points assigned", n_veg)
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iter_chunks(
        self, chunk_size: Optional[int] = None
    ) -> Generator[tuple[np.ndarray, np.ndarray, dict], None, None]:
        """
        Yield ``(x, y, attrs)`` tuples ready for writing to TileDB.

        Each tuple contains:

        - ``x`` – 1-D ``float64`` array of UTM easting coordinates.
        - ``y`` – 1-D ``float64`` array of UTM northing coordinates.
        - ``attrs`` – dict mapping LAS attribute names to typed numpy arrays.

        Parameters
        ----------
        chunk_size:
            Points per chunk.  Pass ``None`` to read the full tile at once.
        """
        if self._reclassify:
            # SMRF needs to see the full tile to build a reliable ground model,
            # so we read everything at once, reclassify, then re-chunk.
            chunks = list(self._tile.read(chunk_size=None))
            if not chunks:
                return
            data = self._apply_filter(self._apply_reclassification(chunks[0]))
            if len(data) == 0:
                return
            step = chunk_size or len(data)
            for start in range(0, len(data), step):
                chunk = data[start : start + step]
                attrs = {
                    name: chunk[name].astype(dtype)
                    for name, dtype in LAS_ATTRIBUTES.items()
                    if name in chunk.dtype.names
                }
                yield chunk["X"], chunk["Y"], attrs
        else:
            for raw in self._tile.read(chunk_size=chunk_size):
                data = self._apply_filter(raw)
                if len(data) == 0:
                    continue
                attrs = {
                    name: data[name].astype(dtype)
                    for name, dtype in LAS_ATTRIBUTES.items()
                    if name in data.dtype.names
                }
                yield data["X"], data["Y"], attrs
