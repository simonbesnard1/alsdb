"""One north-up pixel lattice shared by processing and storage."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    x0: float
    y1: float
    resolution: float
    nx: int
    ny: int

    @classmethod
    def from_bbox(cls, bbox, resolution):
        x0, y0, x1, y1 = map(float, bbox)
        if not np.isfinite([x0, y0, x1, y1, resolution]).all():
            raise ValueError("Grid coordinates and resolution must be finite")
        if resolution <= 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("Grid requires positive resolution and nonempty bbox")
        # Preserve the requested upper-left origin; extend east and south.
        nx = math.ceil(round((x1 - x0) / resolution, 9))
        ny = math.ceil(round((y1 - y0) / resolution, 9))
        return cls(x0, y1, float(resolution), max(1, nx), max(1, ny))

    @property
    def bbox(self):
        return (
            self.x0,
            self.y1 - self.ny * self.resolution,
            self.x0 + self.nx * self.resolution,
            self.y1,
        )

    @property
    def shape(self):
        return self.ny, self.nx

    def centers(self):
        return np.meshgrid(
            self.x0 + (np.arange(self.nx) + 0.5) * self.resolution,
            self.y1 - (np.arange(self.ny) + 0.5) * self.resolution,
        )

    def point_bins(self, points):
        """Return inside mask and flat north-up cell IDs, with half-open ownership."""
        x, y = points["X"], points["Y"]
        x0, y0, x1, y1 = self.bbox
        inside = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x < x1) & (y > y0) & (y <= y1)
        col = np.floor((x[inside] - x0) / self.resolution).astype(np.int64)
        row = np.floor((y1 - y[inside]) / self.resolution).astype(np.int64)
        # Protect against roundoff at the far outside edge, after ownership.
        return inside, np.minimum(row, self.ny - 1) * self.nx + np.minimum(col, self.nx - 1)

    def tiles(self, tile_size, buffer=0.0):
        ratio = tile_size / self.resolution
        if tile_size <= 0 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError("tile_size must be a positive whole multiple of resolution")
        if not np.isfinite(buffer) or buffer < 0:
            raise ValueError("tile_buffer must be finite and nonnegative")
        n = round(ratio)
        for row in range(0, self.ny, n):
            for col in range(0, self.nx, n):
                crop = (
                    self.x0 + col * self.resolution,
                    self.y1 - min(row + n, self.ny) * self.resolution,
                    self.x0 + min(col + n, self.nx) * self.resolution,
                    self.y1 - row * self.resolution,
                )
                x0, y0, x1, y1 = crop
                yield ((x0 - buffer, y0 - buffer, x1 + buffer, y1 + buffer), crop)
