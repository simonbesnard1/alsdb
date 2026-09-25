"""Terrain interpolation with explicit support and extrapolation diagnostics."""

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree


class TerrainModel:
    def __init__(self, ground):
        xy = np.column_stack((ground["X"], ground["Y"]))
        z = np.asarray(ground["Z"], dtype=float)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(z)
        xy, z = xy[valid], z[valid]
        # Duplicate XY ground returns have no unique TIN elevation. Use mean Z.
        self.xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        self.z = np.bincount(inverse, weights=z) / np.bincount(inverse) if len(z) else np.empty(0)
        self.tree = cKDTree(self.xy) if len(self.xy) else None
        self.tin = None
        if len(self.xy) >= 3:
            try:
                self.tin = Delaunay(self.xy)
            except QhullError:
                pass  # Collinear/sparse support is explicitly marked extrapolated.

    def slope(self, xy):
        """Local TIN slope in degrees; undefined outside nondegenerate support."""
        result = np.full(len(xy), np.nan)
        if self.tin is None:
            return result
        simplex = self.tin.find_simplex(xy)
        inside = simplex >= 0
        z = self.z[self.tin.simplices[simplex[inside]]]
        gradient = np.einsum(
            "ij,ijk->ik", z[:, :2] - z[:, 2, None], self.tin.transform[simplex[inside], :2]
        )
        result[inside] = np.degrees(np.arctan(np.linalg.norm(gradient, axis=1)))
        return result

    def evaluate(self, xy, *, max_distance=None, extrapolate=False, method="tin"):
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
            raise ValueError("max_distance must be finite and positive")
        if method not in ("tin", "idw"):
            raise ValueError("Terrain interpolation must be tin or idw")
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        z = np.full(len(xy), np.nan)
        distance = np.full(len(xy), np.inf)
        extrapolated = np.ones(len(xy), dtype=bool)
        if self.tree is None or not len(xy):
            return z, distance, extrapolated
        distance, nearest = self.tree.query(xy)
        simplex = self.tin.find_simplex(xy) if self.tin is not None else np.full(len(xy), -1)
        inside = simplex >= 0
        extrapolated = ~inside
        if method == "tin":
            if inside.any():
                transform = self.tin.transform[simplex[inside]]
                bary = np.einsum("ijk,ik->ij", transform[:, :2], xy[inside] - transform[:, 2])
                weights = np.column_stack((bary, 1 - bary.sum(axis=1)))
                z[inside] = (weights * self.z[self.tin.simplices[simplex[inside]]]).sum(axis=1)
            if extrapolate:
                z[~inside] = self.z[nearest[~inside]]
        elif method == "idw":
            d, idx = self.tree.query(xy, k=min(8, len(self.xy)))
            if d.ndim == 1:
                z = self.z[idx].copy()
            else:
                w = 1 / np.maximum(d, 1e-10) ** 2
                z = (w * self.z[idx]).sum(axis=1) / w.sum(axis=1)
            if not extrapolate:
                z[~inside] = np.nan
        else:
            raise ValueError("Terrain interpolation must be tin or idw")
        if max_distance is not None:
            z[distance > max_distance] = np.nan
        return z, distance, extrapolated


def normalize_points(
    arr, *, max_distance=None, extrapolate=False, terrain_method="tin", model=None
):
    """Return a copy with HAG plus support fields; invalid HAG remains NaN."""
    model = model if model is not None else TerrainModel(arr[arr["Classification"] == 2])
    xy = np.column_stack((arr["X"], arr["Y"]))
    ground_z, distance, extrapolated = model.evaluate(
        xy, max_distance=max_distance, extrapolate=extrapolate, method=terrain_method
    )
    hag = arr["Z"] - ground_z
    negative = np.isfinite(hag) & (hag < 0)
    hag = np.maximum(hag, 0)
    fields = {
        "HeightAboveGround": hag,
        "GroundDistance": distance,
        "GroundExtrapolated": extrapolated.astype(np.uint8),
        "NegativeHAG": negative.astype(np.uint8),
    }
    # Allocate once rather than copying the entire record array for each field.
    dtype = arr.dtype.descr + [
        (name, values.dtype) for name, values in fields.items() if name not in arr.dtype.names
    ]
    out = np.empty(len(arr), dtype=dtype)
    for name in arr.dtype.names:
        out[name] = arr[name]
    for name, values in fields.items():
        out[name] = values
    return out, model
