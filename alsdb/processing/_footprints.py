"""Bounded spatial batching for nearby footprint queries."""

import numpy as np
from scipy.spatial import cKDTree

from alsdb.processing._tiling import bounded_map, query_to_array


def footprint_batches(
    provider,
    centers,
    radius,
    *,
    year=None,
    attributes=(),
    batch_tile_size=100.0,
    n_workers=1,
    max_shots_per_batch=256,
    process=None,
):
    """Yield positional (index, points) groups; query each spatial block once.

    The cap on shots and number of pending blocks bounds scheduling overhead.
    Point memory still depends on survey density and the chosen block size.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    if not np.isfinite(centers).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Footprint centers and positive radius must be finite")
    if not np.isfinite(batch_tile_size) or batch_tile_size <= 0 or max_shots_per_batch < 1:
        raise ValueError("Batch size and maximum shot count must be positive")
    blocks = {}
    for index, block in enumerate(np.floor(centers / batch_tile_size).astype(np.int64)):
        blocks.setdefault(tuple(block), []).append(index)

    def chunks():
        for indices in blocks.values():
            for start in range(0, len(indices), max_shots_per_batch):
                yield indices[start : start + max_shots_per_batch]

    def query(indices):
        xy = centers[indices]
        low, high = xy.min(axis=0) - radius, xy.max(axis=0) + radius
        points = query_to_array(
            provider, (low[0], low[1], high[0], high[1]), year=year, attributes=attributes
        )
        tree = cKDTree(np.column_stack((points["X"], points["Y"])))
        # Lists refer into one point block, avoiding a copy for every overlapping shot.
        neighborhoods = tree.query_ball_point(xy, radius)
        return (
            process(indices, points, neighborhoods)
            if process is not None
            else (indices, points, neighborhoods)
        )

    yield from bounded_map(query, chunks(), n_workers)


def point_array(data):
    """Accept query dictionaries and structured point arrays without repeated copies."""
    if isinstance(data, np.ndarray):
        return data
    names = [name for name in data if name != "Year"]
    arr = np.empty(len(data[names[0]]), dtype=[(name, data[name].dtype) for name in names])
    for name in names:
        arr[name] = data[name]
    return arr
