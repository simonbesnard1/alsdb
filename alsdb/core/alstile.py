# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from alsdb.tile.Tile import Tile
from alsdb.tile.tile_name import PNOATileName
from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)


class ALSTile:
    """
    Processes a single PNOA LAZ tile into arrays ready for TileDB ingestion.

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
    """

    def __init__(
        self,
        path: str | Path,
        classification_filter: Optional[list[int]] = None,
    ) -> None:
        self._tile = Tile(path)
        self._classification_filter = classification_filter

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tile(self) -> Tile:
        return self._tile

    @property
    def name(self) -> PNOATileName:
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
