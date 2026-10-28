# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

import json
import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pdal

from alsdb.core.classification import (
    ClassificationConfig,
    classification_stages,
    denoise_stages,
    resolve_classification_config,
)
from alsdb.tile.tile import Tile
from alsdb.tile.tile_name import TileNameBase
from alsdb.utils.schema import LAS_ATTRIBUTES

logger = logging.getLogger(__name__)

# Height-above-ground thresholds in metres for LAS vegetation classes 3 / 4 / 5.
# Only applied to unclassified (class 1) points when reclassify=True.
# Data is always in metres at the point of reclassification (reprojected if needed).
_HAG_LOW: float = 0.5  # below this → leave as class 1 (noise / bare ground fringe)
_HAG_MED: float = 2.0  # 0.5–2.0 m  → class 3 (low vegetation)
_HAG_HIGH: float = 5.0  # 2.0–5.0 m  → class 4 (medium vegetation)
# ≥ 5.0 m    → class 5 (high vegetation)

_UNRESOLVED = object()  # sentinel for lazy CRS resolution


def _crs_is_feet(crs_str: str) -> bool:
    """Return True if any axis of *crs_str* uses feet as its linear unit."""
    lower = crs_str.lower()
    # US survey foot (EPSG:9003) and international foot (EPSG:9002)
    return bool(
        any(
            kw in lower
            for kw in (
                "ftus",
                "survey foot",
                "survey feet",
                "us foot",
                "international foot",
                '"foot"',
                '"feet"',
            )
        )
    )


def _find_utm_crs(native_crs_str: str, bbox: tuple) -> str:
    """
    Return the WGS 84 UTM CRS that best covers the centre of *bbox*
    (expressed in *native_crs_str* coordinates).

    Uses a one-point PDAL reprojection to WGS84, then derives the UTM zone
    mathematically — no pyproj dependency required.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    dt = np.dtype([("X", "f8"), ("Y", "f8"), ("Z", "f8")])
    arr = np.array([(cx, cy, 0.0)], dtype=dt)
    stages = [{"type": "filters.reprojection", "in_srs": native_crs_str, "out_srs": "EPSG:4326"}]
    p = pdal.Pipeline(json.dumps(stages), arrays=[arr])
    p.execute()
    lon = float(p.arrays[0]["X"][0])
    lat = float(p.arrays[0]["Y"][0])

    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


class ALSTile:
    """
    Processes a single LAZ tile into arrays ready for TileDB ingestion.

    Wraps :class:`~alsdb.tile.tile.Tile` and applies optional reprojection,
    reclassification, and point filtering before handing data to
    :class:`~alsdb.core.alsdatabase.ALSDatabase`.

    Parameters
    ----------
    path:
        Path to the ``.laz`` file.
    classification_filter:
        If provided, only points whose ``Classification`` code is in this list
        are passed through.
    reclassify:
        If ``True``, run ground classification + HAG to assign ground (class 2)
        and vegetation (classes 3/4/5) labels to unclassified (class 1) points.
        Thresholds are always in metres; use together with ``reproject_to`` for
        non-metric source files.
    ground_classifier:
        Algorithm used for ground classification when ``reclassify=True``.

        - ``"csf"`` (default) — Cloth Simulation Filter (Zhang et al. 2016).
          Good general-purpose choice; handles gentle-to-moderate terrain well.
        - ``"pmf"`` — Progressive Morphological Filter (Zhang et al. 2003),
          PDAL's ``filters.pmf``.  More robust on steep or complex terrain.
          Morphological opening, not TIN densification: neither option is
          Axelsson's (2000) PTD used by TerraScan/lasground.
    denoise:
        If ``True``, run noise detection before reclassification:

        - ``filters.elm`` flags below-ground outliers as class 7.
        - ``filters.outlier`` (radius method) flags isolated points as class 7.

        These class-7 points are then excluded from the ground classifier and
        HAG computation.  Useful when the source LAZ has no pre-existing noise
        classification (classes 7/18 not set).
    reproject_to:
        Target CRS for the output points.

        - ``None`` (default) — keep native CRS, no reprojection.
        - ``"auto"`` — detect feet-based CRS and reproject to the appropriate
          WGS 84 UTM zone; metric CRS files are left unchanged.
        - ``"EPSG:XXXX"`` — reproject to an explicit CRS.
    """

    def __init__(
        self,
        path: str | Path,
        classification_filter: list[int] | None = None,
        reclassify: bool = False,
        ground_classifier: str = "csf",
        denoise: bool = False,
        reproject_to: str | None = None,
        hag_low: float = _HAG_LOW,
        hag_med: float = _HAG_MED,
        hag_high: float = _HAG_HIGH,
        classification_config: ClassificationConfig | None = None,
        classification_context: tuple[str | Path, ...] = (),
    ) -> None:
        self._tile = Tile(path)
        self._classification_filter = classification_filter
        self._classification = resolve_classification_config(
            classification_config,
            reclassify=reclassify,
            ground_method=ground_classifier,
            denoise=denoise,
            vegetation_low=hag_low,
            vegetation_medium=hag_med,
            vegetation_high=hag_high,
        )
        self._classification_context = tuple(Path(path) for path in classification_context)
        self._reproject_to = reproject_to
        self._resolved_crs = _UNRESOLVED  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tile(self) -> Tile:
        return self._tile

    @property
    def name(self) -> TileNameBase:
        return self._tile.name

    @property
    def points_bbox(self) -> tuple[float, float, float, float]:
        """Native-CRS extent of the points (exact after reading; header before)."""
        return self._tile.native_bbox

    @property
    def target_crs(self) -> str:
        """CRS of the data after ingestion (native or reprojected)."""
        out = self._get_out_crs()
        return out if out is not None else self.name.crs

    @property
    def classification_config(self) -> ClassificationConfig:
        """Resolved ingestion-time classification configuration."""
        return self._classification

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_z_scale(self) -> float | None:
        """Return foot→metre scale factor when reprojecting a feet-based CRS, else None."""
        if self._get_out_crs() is None:
            return None
        lower = self.name.crs.lower()
        # US survey foot (EPSG:9003): 0.30480060960121924 m/ft
        if any(
            kw in lower
            for kw in ("ftus", "us_survey_foot", "survey foot", "survey feet", "us foot")
        ):
            return 0.3048006096
        # International foot (EPSG:9002): exactly 0.3048 m/ft
        if any(kw in lower for kw in ("international foot", '"foot"', '"feet"')):
            return 0.3048
        return None

    def _get_out_crs(self) -> str | None:
        """Resolve the reprojection target CRS (cached). Returns None if no reprojection."""
        if self._resolved_crs is not _UNRESOLVED:
            return self._resolved_crs  # type: ignore[return-value]

        if self._reproject_to is None:
            self._resolved_crs = None
        elif self._reproject_to != "auto":
            self._resolved_crs = self._reproject_to
        else:
            native = self.name.crs
            if _crs_is_feet(native):
                target = _find_utm_crs(native, self.name.bbox_native)
                import re as _re

                m = _re.search(r'AUTHORITY\["EPSG","(\d+)"\]\s*\]?\s*$', native)
                native_label = f"EPSG:{m.group(1)}" if m else native.split('"')[1]
                logger.info("Auto-reprojection: %s → %s", native_label, target)
                self._resolved_crs = target
            else:
                self._resolved_crs = None  # already metric

        return self._resolved_crs  # type: ignore[return-value]

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

    def _apply_denoise(self, data: np.ndarray) -> np.ndarray:
        """
        Flag noise points using ELM (below-ground outliers) and radius-based
        outlier detection (isolated points).  Flagged points receive class 7
        and are excluded from subsequent ground classification and HAG.
        """
        stages = denoise_stages(self._classification)
        if not stages:
            return data
        p = pdal.Pipeline(json.dumps(stages), arrays=[data])
        p.execute()
        result = p.arrays[0] if p.arrays else data
        n_noise = int((result["Classification"] == 7).sum())
        logger.debug("Denoise: %d noise points flagged (class 7)", n_noise)
        return result

    def _apply_reclassification(self, data: np.ndarray) -> np.ndarray:
        """
        Classify ground and vegetation points using CSF or PMF + HAG.

        Thresholds are in metres.  Data must already be in a metric CRS
        (use reproject_to to ensure this for feet-based source files).
        Only unclassified (class 1) points are relabelled.
        """
        original = data["Classification"].copy()
        stages = classification_stages(self._classification)
        p = pdal.Pipeline(json.dumps(stages), arrays=[data])
        p.execute()
        result = p.arrays[0]
        if self._classification.preserve_existing:
            restore = ~np.isin(original, self._classification.source_classes)
            result["Classification"][restore] = original[restore]
        n_veg = int(np.isin(result["Classification"], [3, 4, 5]).sum())
        logger.debug("Reclassification complete: %d vegetation points assigned", n_veg)
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def iter_chunks(
        self, chunk_size: int | None = None
    ) -> Generator[tuple[np.ndarray, np.ndarray, dict], None, None]:
        """
        Yield ``(x, y, attrs)`` tuples ready for writing to TileDB.

        Each tuple contains:

        - ``x`` – 1-D ``float64`` array of easting coordinates.
        - ``y`` – 1-D ``float64`` array of northing coordinates.
        - ``attrs`` – dict mapping LAS attribute names to typed numpy arrays.

        Parameters
        ----------
        chunk_size:
            Points per chunk.  Pass ``None`` to read the full tile at once.
        """
        out_crs = self._get_out_crs()
        z_scale = self._get_z_scale()

        if self._classification.enabled:
            # Ground classifier needs to see the full tile for a reliable ground model.
            chunks = list(self._tile.read(chunk_size=None, out_crs=out_crs))
            if not chunks:
                return
            raw = chunks[0]
            core_size = len(raw)
            if self._classification.buffer > 0 and self._classification_context:
                x0, x1 = float(raw["X"].min()), float(raw["X"].max())
                y0, y1 = float(raw["Y"].min()), float(raw["Y"].max())
                buffer = self._classification.buffer
                context = []
                for path in self._classification_context:
                    arrays = list(Tile(path).read(chunk_size=None, out_crs=out_crs))
                    if not arrays:
                        continue
                    neighbor = arrays[0]
                    inside = (
                        (neighbor["X"] >= x0 - buffer)
                        & (neighbor["X"] <= x1 + buffer)
                        & (neighbor["Y"] >= y0 - buffer)
                        & (neighbor["Y"] <= y1 + buffer)
                    )
                    if inside.any():
                        aligned = np.zeros(inside.sum(), dtype=raw.dtype)
                        for name in raw.dtype.names:
                            if name in neighbor.dtype.names:
                                aligned[name] = neighbor[name][inside]
                        context.append(aligned)
                if context:
                    raw = np.concatenate((raw, *context))
            if z_scale is not None:
                raw = raw.copy()
                raw["Z"] = raw["Z"] * z_scale
            if self._classification.denoise:
                raw = self._apply_denoise(raw)
            data = self._apply_reclassification(raw)[:core_size]
            data = self._apply_filter(data)
            if len(data) == 0:
                return
            step = chunk_size or len(data)
            for start in range(0, len(data), step):
                chunk = data[start : start + step]
                attrs = {
                    name: chunk[name].astype(dtype, copy=False)
                    for name, dtype in LAS_ATTRIBUTES.items()
                    if name in chunk.dtype.names
                }
                yield chunk["X"], chunk["Y"], attrs
        else:
            for raw in self._tile.read(chunk_size=chunk_size, out_crs=out_crs):
                if z_scale is not None:
                    raw = raw.copy()
                    raw["Z"] = raw["Z"] * z_scale
                if self._classification.denoise:
                    raw = self._apply_denoise(raw)
                data = self._apply_filter(raw)
                if len(data) == 0:
                    continue
                attrs = {
                    name: data[name].astype(dtype, copy=False)
                    for name, dtype in LAS_ATTRIBUTES.items()
                    if name in data.dtype.names
                }
                yield data["X"], data["Y"], attrs
