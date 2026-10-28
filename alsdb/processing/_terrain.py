"""Terrain interpolation with explicit support and extrapolation diagnostics."""

import json

import numpy as np
from pdal import Pipeline as _PdalPipeline  # bound here: tests patch pdal.Pipeline
from scipy.spatial import cKDTree

# ASPRS ground (2) and water (9): water returns are the terrain surface over
# water bodies, as in lidR's normalize_height/rasterize_terrain defaults.
GROUND_CLASSES = (2, 9)


class _Tin:
    """Planar Delaunay TIN with a grid-bucket point locator.

    ``simplices`` and ``transform`` follow SciPy's ``Delaunay`` layout so
    interpolation and slope calculations can share the same representation.

    :meth:`find_simplex` registers every triangle in the grid cells its
    bounding box covers and tests a query only against triangles in its cell.
    """

    _CHUNK = 262_144  # queries per vectorised batch, bounds memory

    def __init__(self, local):
        pts = np.zeros(len(local), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        pts["X"], pts["Y"] = local[:, 0], local[:, 1]
        pipeline = _PdalPipeline(json.dumps([{"type": "filters.delaunay"}]), arrays=[pts])
        pipeline.execute()  # RuntimeError for collinear input
        mesh = pipeline.meshes[0]
        if not len(mesh):
            raise ValueError("no triangles")
        self.simplices = np.column_stack((mesh["A"], mesh["B"], mesh["C"])).astype(np.intp)
        v = local[self.simplices]  # (n, 3, 2)
        a, b = v[:, 0] - v[:, 2], v[:, 1] - v[:, 2]  # columns of the barycentric matrix
        det = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
        self.transform = np.empty((len(v), 3, 2))
        with np.errstate(divide="ignore", invalid="ignore"):  # degenerate: never "inside"
            self.transform[:, 0, 0] = b[:, 1] / det
            self.transform[:, 0, 1] = -b[:, 0] / det
            self.transform[:, 1, 0] = -a[:, 1] / det
            self.transform[:, 1, 1] = a[:, 0] / det
        self.transform[:, 2] = v[:, 2]
        # Bucket grid: cells about one point spacing wide.
        self._g0 = local.min(axis=0)
        extent = np.maximum(local.max(axis=0) - self._g0, 1e-9)
        self._cell = max(float(np.sqrt(extent[0] * extent[1] / len(local))), 1e-6)
        lo = np.floor((v.min(axis=1) - self._g0) / self._cell).astype(np.int64)
        hi = np.floor((v.max(axis=1) - self._g0) / self._cell).astype(np.int64)
        self._shape = (int(hi[:, 1].max()) + 1, int(hi[:, 0].max()) + 1)  # rows (y), cols (x)
        width = hi[:, 0] - lo[:, 0] + 1
        counts = width * (hi[:, 1] - lo[:, 1] + 1)
        tri = np.repeat(np.arange(len(v)), counts)
        k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        cols = np.repeat(lo[:, 0], counts) + k % np.repeat(width, counts)
        rows = np.repeat(lo[:, 1], counts) + k // np.repeat(width, counts)
        cell = rows * self._shape[1] + cols
        order = np.argsort(cell, kind="stable")
        self._cell_tris = tri[order]
        self._starts = np.concatenate(
            ([0], np.cumsum(np.bincount(cell, minlength=self._shape[0] * self._shape[1])))
        )

    def find_simplex(self, xy, eps=1e-10):
        """Index of the triangle containing each point, -1 outside the TIN."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(xy), -1, dtype=np.intp)
        col = np.floor((xy[:, 0] - self._g0[0]) / self._cell)
        row = np.floor((xy[:, 1] - self._g0[1]) / self._cell)
        inside_grid = (col >= 0) & (row >= 0) & (col < self._shape[1]) & (row < self._shape[0])
        query = np.flatnonzero(inside_grid)
        cell = (row[query] * self._shape[1] + col[query]).astype(np.int64)
        for s in range(0, len(query), self._CHUNK):
            q, c = query[s : s + self._CHUNK], cell[s : s + self._CHUNK]
            n = self._starts[c + 1] - self._starts[c]
            pid = np.repeat(np.arange(len(q)), n)
            k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            tri = self._cell_tris[np.repeat(self._starts[c], n) + k]
            t = self.transform[tri]
            d = xy[q[pid]] - t[:, 2]
            b0 = t[:, 0, 0] * d[:, 0] + t[:, 0, 1] * d[:, 1]
            b1 = t[:, 1, 0] * d[:, 0] + t[:, 1, 1] * d[:, 1]
            hit = np.flatnonzero((b0 >= -eps) & (b1 >= -eps) & (1 - b0 - b1 >= -eps))
            first = hit[np.unique(pid[hit], return_index=True)[1]]
            out[q[pid[first]]] = tri[first]
        return out


class TerrainModel:
    def __init__(self, ground):
        xy = np.column_stack((ground["X"], ground["Y"]))
        z = np.asarray(ground["Z"], dtype=float)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(z)
        xy, z = xy[valid], z[valid]
        # Duplicate XY ground returns have no unique TIN elevation. Use mean Z.
        self.xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        self.z = np.bincount(inverse, weights=z) / np.bincount(inverse) if len(z) else np.empty(0)
        # Centering avoids precision loss when triangulating large map coordinates.
        self._origin = self.xy.mean(axis=0) if len(self.xy) else np.zeros(2)
        local = self.xy - self._origin
        self.tree = cKDTree(local) if len(local) else None
        self.tin = None
        if len(local) >= 3:
            try:
                self.tin = _Tin(local)
            except (RuntimeError, ValueError):
                pass  # Collinear/sparse support is explicitly marked extrapolated.

    def _find_simplex(self, xy):
        return self.tin.find_simplex(xy)

    def slope(self, xy):
        """Local TIN slope in degrees; undefined outside nondegenerate support."""
        result = np.full(len(xy), np.nan)
        if self.tin is None:
            return result
        simplex = self._find_simplex(np.asarray(xy, dtype=float) - self._origin)
        inside = simplex >= 0
        z = self.z[self.tin.simplices[simplex[inside]]]
        gradient = np.einsum(
            "ij,ijk->ik", z[:, :2] - z[:, 2, None], self.tin.transform[simplex[inside], :2]
        )
        result[inside] = np.degrees(np.arctan(np.linalg.norm(gradient, axis=1)))
        return result

    def evaluate(self, xy, *, max_distance=None, extrapolate=False, method="tin", distances=True):
        """Ground elevation at *xy*, nearest-ground distance and outside-TIN flags.

        ``distances=False`` skips the nearest-ground query when nothing needs it
        (no *max_distance*, no extrapolation, TIN method); distances are NaN.
        """
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
            raise ValueError("max_distance must be finite and positive")
        if method not in ("tin", "idw"):
            raise ValueError("Terrain interpolation must be tin or idw")
        xy = np.asarray(xy, dtype=float).reshape(-1, 2) - self._origin  # TIN/tree frame
        z = np.full(len(xy), np.nan)
        distance = np.full(len(xy), np.inf)
        extrapolated = np.ones(len(xy), dtype=bool)
        if self.tree is None or not len(xy):
            return z, distance, extrapolated
        if distances or max_distance is not None or extrapolate or method != "tin":
            distance, nearest = self.tree.query(xy)
        else:
            distance = np.full(len(xy), np.nan)
        simplex = self._find_simplex(xy) if self.tin is not None else np.full(len(xy), -1)
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
    arr,
    *,
    max_distance=None,
    extrapolate=False,
    terrain_method="tin",
    model=None,
    ground_classes=GROUND_CLASSES,
    distances=True,
):
    """Return a copy with HAG plus support fields; invalid HAG remains NaN.

    ``distances=False`` leaves GroundDistance NaN unless *max_distance* or
    *extrapolate* needs it (see :meth:`TerrainModel.evaluate`).
    """
    if model is None:
        model = TerrainModel(arr[np.isin(arr["Classification"], ground_classes)])
    xy = np.column_stack((arr["X"], arr["Y"]))
    ground_z, distance, extrapolated = model.evaluate(
        xy,
        max_distance=max_distance,
        extrapolate=extrapolate,
        method=terrain_method,
        distances=distances,
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
