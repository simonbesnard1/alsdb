# SPDX-License-Identifier: EUPL-1.2
# SPDX-FileCopyrightText: 2026 Simon Besnard
# SPDX-FileCopyrightText: 2026 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences

"""Robust-envelope ground classification.

Progressive TIN densification (PTD; Axelsson 2000), run as alsdb's tilt
ensemble, seeds a robust surface fit to all last returns in the spirit of Kraus & Pfeifer's
(1998) robust interpolation: a thin-plate grid is fitted by iteratively
reweighted least squares, where returns above the surface lose weight so the
surface settles on the lower envelope of the data. Two changes make it work
across forest types:

* **Vegetation evidence.** The asymmetry of the weights is scaled by the local
  share of multi-return pulses, lidar's own signature of vegetation. Under
  canopy (share above ``evidence_threshold``) the fit is a lower envelope and
  sheds the low vegetation that PTD accepts as ground; on bare rock (no
  multiple returns) the weights are symmetric, so the surface passes through
  the points and recovers the cliff and crest returns that PTD's angle and
  distance criteria reject at low density.
* **PTD initialisation.** The first fit uses PTD's ground alone. Starting from
  all last returns puts the surface inside a closed canopy, from where the
  reweighting never descends.

Ground is then every seed point within ``keep`` (vegetated) to ``keep_bare``
(bare) metres above the final surface, plus the other candidates within that
tolerance that lie in a gap of the seed, farther than ``gap`` from every kept
seed point. Without the gap rule the envelope accepts grass and low shrubs
within ``keep`` of the surface wherever the cloud is dense: at full density
that doubled PTD's commission; at low density the seed has gaps everywhere
and the rule changes nothing. The surface is solved in overlapping
blocks, each pulled weakly (``anchor``) towards one coarse fit of the same
data: where no return supports the surface (closed canopy without ground
hits) this keeps the thin plate on the terrain trend instead of a free
extrapolation, which is worth a third of the error in dense tropical forest;
a stronger pull trades cliffs for shrub patches (see the benchmark pages). The DTM itself is built from those
points as usual (TIN); the surface only selects them.

Against hand-corrected IGN Lidar HD (Petit Buech, Vizzavona) and Robson Creek
decimated to 0.5-8 pulses/m2 it lowered the DTM error of the PTD tilt ensemble
most in dense tropical forest at low density; it is weaker on bare rock under
sparse trees at full density and on closed shrub patches without ground
returns beneath them (see doc/benchmarking.md).

References
----------
Axelsson, P. (2000). DEM generation from laser scanner data using adaptive TIN
    models. International Archives of Photogrammetry and Remote Sensing,
    33(B4), 110-117.
Kraus, K., & Pfeifer, N. (1998). Determination of terrain models in wooded
    areas with airborne laser scanner data. ISPRS Journal of Photogrammetry
    and Remote Sensing, 53(4), 193-203.
    https://doi.org/10.1016/S0924-2716(98)00009-4
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

try:  # CHOLMOD (scikit-sparse >= 0.5): 3x faster than SuperLU per block, but it holds the
    # GIL, so it is used for single-threaded calls only; SuperLU scales across threads.
    from sksparse.cholmod import cho_solve as _cho_solve
except ImportError:  # pragma: no cover - optional dependency
    _cho_solve = None

DEFAULTS = {
    "cell": 1.0,  # grid spacing of the surface (m)
    "smoothness": 0.1,  # weight of the thin-plate (squared Laplacian) penalty
    "iterations": 5,  # reweighting passes
    "evidence_cell": 5.0,  # cell of the multi-return share (m); averaged over 3 x 3 cells
    "evidence_threshold": 0.2,  # multi-return share at which the fit is a full lower envelope
    "keep": 0.1,  # ground: at most this far above the surface under vegetation (m)
    "keep_bare": 1.0,  # ... and on bare ground (m)
    "weight_scale": 1.0,  # Kraus & Pfeifer weight 1 / (1 + (a r)^b) for returns r m above
    "weight_power": 4.0,
    "symmetric_scale": 3.0,  # Cauchy scale of the symmetric (bare-ground) weights (m)
    "prior": 0.5,  # weight of last returns PTD did not call ground, relative to PTD ground
    "anchor": 1e-3,  # pull of every grid node towards one coarse (4 x cell) fit of the same data
    "block": 128,  # nodes per side of the blocks the grid is solved in, and their overlap
    "overlap": 32,
    "gap": 2.0,  # non-seed candidates are ground only farther than this from every kept seed point (m)
}
# Revision of the algorithm, recorded with benchmark scores. 1: anchor only on grids above
# 4 x block^2 nodes; 2: anchor whenever the grid is split into blocks; 3: seeded by the PTD
# tilt ensemble (was a single run) and non-seed points added only in seed gaps (``gap``).
REVISION = 3
# The PTD run that seeds the surface: alsdb's tuned settings and tilt ensemble (ptd.DEFAULTS).
PTD_SEED = {"seed_resolution": 10.0, "angle": 45.0, "distance": 1.0, "spacing": 0.25,
            "tilt_runs": 8, "tilt": 10.0, "votes": 3}  # fmt: skip


def vegetation_evidence(points, cell=5.0):
    """Share of multi-return points around each point, on *cell* m cells averaged 3 x 3.

    Returns ones when the points carry no ``NumberOfReturns``.
    """
    from scipy.ndimage import uniform_filter

    if "NumberOfReturns" not in points.dtype.names or not len(points):
        return np.ones(len(points))
    x, y = points["X"], points["Y"]
    x0, y0 = np.floor(x.min() / cell) * cell, np.floor(y.min() / cell) * cell
    j = ((x - x0) / cell).astype(np.intp)
    i = ((y - y0) / cell).astype(np.intp)
    shape = (int(i.max()) + 1, int(j.max()) + 1)
    n_all = np.zeros(shape)
    n_multi = np.zeros(shape)
    np.add.at(n_all, (i, j), 1.0)
    np.add.at(n_multi, (i, j), (points["NumberOfReturns"] > 1).astype(float))
    n_all = uniform_filter(n_all, 3, mode="nearest")
    n_multi = uniform_filter(n_multi, 3, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(n_all > 0, n_multi / n_all, 0.0)
    return share[i, j]


class _Grid:
    """Regular grid of surface nodes covering a set of points, with bilinear sampling."""

    def __init__(self, x, y, cell):
        self.cell = float(cell)
        self.x0 = np.floor(x.min() / cell) * cell - cell
        self.y0 = np.floor(y.min() / cell) * cell - cell
        self.nx = int(np.ceil((x.max() - self.x0) / cell)) + 2
        self.ny = int(np.ceil((y.max() - self.y0) / cell)) + 2

    @property
    def size(self):
        return self.nx * self.ny

    def _locate(self, x, y):
        u = (np.asarray(x, float) - self.x0) / self.cell
        v = (np.asarray(y, float) - self.y0) / self.cell
        j = np.clip(np.floor(u).astype(np.intp), 0, self.nx - 2)
        i = np.clip(np.floor(v).astype(np.intp), 0, self.ny - 2)
        return i, j, np.clip(u - j, 0.0, 1.0), np.clip(v - i, 0.0, 1.0)

    def sampling_matrix(self, x, y):
        """Sparse (n, size) matrix of bilinear weights of the nodes at each point."""
        i, j, fu, fv = self._locate(x, y)
        base = i * self.nx + j
        n = len(x)
        rows = np.repeat(np.arange(n), 4)
        cols = np.column_stack((base, base + 1, base + self.nx, base + self.nx + 1)).ravel()
        vals = np.column_stack(((1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv)).ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, self.size))

    def sample(self, nodes, x, y):
        """Bilinear value of the node field *nodes* (flat, size) at the points."""
        s = nodes.reshape(self.ny, self.nx)
        i, j, fu, fv = self._locate(x, y)
        return (s[i, j] * (1 - fu) * (1 - fv) + s[i, j + 1] * fu * (1 - fv)
                + s[i + 1, j] * (1 - fu) * fv + s[i + 1, j + 1] * fu * fv)  # fmt: skip

    def laplacian(self):
        """Sparse 5-point Laplacian over the interior nodes (zero rows on the border)."""
        idx = np.arange(self.size).reshape(self.ny, self.nx)
        c = idx[1:-1, 1:-1].ravel()
        rows = np.tile(c, 5)
        cols = np.concatenate([c, c - 1, c + 1, c - self.nx, c + self.nx])
        vals = np.concatenate([np.full(len(c), -4.0)] + [np.ones(len(c))] * 4)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.size, self.size))


def _blocks(n, block, overlap):
    """(start, stop) node ranges of blocks of *block* nodes overlapping by *overlap*."""
    if n <= block:
        return [(0, n)]
    step = block - overlap
    starts = list(range(0, n - block, step)) + [n - block]
    return [(s, s + block) for s in starts]


def _ramp(n, lo_edge, hi_edge, overlap):
    """Blend weights along one axis: 1 inside, 0 on the outer quarter of an inner overlap
    (a free block boundary is unreliable, badly so across a cliff), linear in between."""
    w = np.ones(n)
    margin = max(1, overlap // 4)
    ramp = np.concatenate([np.zeros(margin), np.linspace(0.0, 1.0, overlap - margin + 2)[1:-1]])
    if not lo_edge:
        w[:overlap] = ramp
    if not hi_edge:
        w[n - overlap:] = ramp[::-1]
    return w


def _solve_block(args):
    """Direct solve of one block's weighted thin-plate system; returns its node values."""
    from scipy.sparse.linalg import spsolve

    bny, bnx, rows, cols, vals, z, w, smoothness, anchor, anchor_weight, cholmod = args
    size = bny * bnx
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(z), size))
    idx = np.arange(size).reshape(bny, bnx)
    c = idx[1:-1, 1:-1].ravel()
    L = sp.csr_matrix(
        (np.concatenate([np.full(len(c), -4.0)] + [np.ones(len(c))] * 4),
         (np.tile(c, 5), np.concatenate([c, c - 1, c + 1, c - bnx, c + bnx]))), shape=(size, size))  # fmt: skip
    AtW = A.T @ sp.diags(w)
    K = (AtW @ A) + smoothness * (L.T @ L) + anchor_weight * sp.identity(size)
    b = AtW @ z
    if anchor is not None:  # weak pull towards the coarse global surface
        b = b + anchor_weight * anchor.ravel()
    if cholmod:
        return _cho_solve(K.tocsc(), b)
    return spsolve(K.tocsc(), b)


def fit_surface(grid, located, z, w, smoothness, *, block=128, overlap=32, workers=1, anchor=1e-3, x=None, y=None):
    """Weighted thin-plate grid fit: min sum w (z - S(x, y))^2 + smoothness * sum (lap S)^2.

    The grid is solved in overlapping blocks (direct factorisation each; free
    boundaries hidden by linear blending over the overlap), so the cost grows
    linearly with the tile and no large factorisation is needed. Where no
    point supports the surface the blocks would extrapolate differently, so
    each is pulled with weight *anchor* towards one coarse global fit (4 x the
    cell), which also removes the near-singular directions. *located* is
    ``grid._locate(x, y)`` of the points; *x*, *y* are needed for the anchor.
    """
    from concurrent.futures import ThreadPoolExecutor

    i, j, fu, fv = located
    coarse_nodes = None
    if anchor and x is not None and (grid.nx > block or grid.ny > block):  # split into blocks
        coarse = _Grid(x, y, grid.cell * 4)
        coarse_nodes = fit_surface(coarse, coarse._locate(x, y), z, w, smoothness * 16,
                                   block=10**6, overlap=0, workers=1, anchor=0.0)  # fmt: skip
        gx = grid.x0 + np.arange(grid.nx) * grid.cell
        gy = grid.y0 + np.arange(grid.ny) * grid.cell
        GX, GY = np.meshgrid(gx, gy)
        coarse_nodes = coarse.sample(coarse_nodes, GX.ravel(), GY.ravel()).reshape(grid.ny, grid.nx)
    anchor_weight = anchor if coarse_nodes is not None else 1e-6
    cholmod = _cho_solve is not None and workers <= 1
    corner = np.column_stack(((1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv))
    jobs, places = [], []
    for i0, i1 in _blocks(grid.ny, block, overlap):
        for j0, j1 in _blocks(grid.nx, block, overlap):
            inside = np.flatnonzero((i >= i0) & (i < i1 - 1) & (j >= j0) & (j < j1 - 1))
            bny, bnx = i1 - i0, j1 - j0
            li, lj = i[inside] - i0, j[inside] - j0
            base = li * bnx + lj
            rows = np.repeat(np.arange(len(inside)), 4)
            cols = np.column_stack((base, base + 1, base + bnx, base + bnx + 1)).ravel()
            jobs.append((bny, bnx, rows, cols, corner[inside].ravel(), z[inside], w[inside], smoothness,
                         None if coarse_nodes is None else coarse_nodes[i0:i1, j0:j1], anchor_weight,
                         cholmod))  # fmt: skip
            blend = np.outer(_ramp(bny, i0 == 0, i1 == grid.ny, overlap),
                             _ramp(bnx, j0 == 0, j1 == grid.nx, overlap))  # fmt: skip
            places.append((slice(i0, i1), slice(j0, j1), blend))
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            solutions = list(pool.map(_solve_block, jobs))
    else:
        solutions = [_solve_block(job) for job in jobs]
    acc = np.zeros((grid.ny, grid.nx))
    norm = np.zeros((grid.ny, grid.nx))
    for (si, sj, blend), nodes in zip(places, solutions):
        acc[si, sj] += blend * nodes.reshape(blend.shape)
        norm[si, sj] += blend
    return (acc / norm).ravel()


def envelope_ground(points, *, workers=1, seed=None, ptd_parameters=None, **parameters):
    """Classify ground with the robust envelope.

    Parameters
    ----------
    points : numpy.ndarray
        Structured array with X, Y, Z, Classification and, for the vegetation
        evidence and the last-return candidates, ReturnNumber and
        NumberOfReturns. Noise (classes 7, 18) is never ground.
    workers : int
        Threads of the seeding PTD run and of the block solves.
    seed : numpy.ndarray, optional
        Boolean ground mask to seed the surface with instead of running PTD.
    ptd_parameters : dict, optional
        Overrides of :data:`PTD_SEED` for the seeding run.
    **parameters
        Overrides of :data:`DEFAULTS`.

    Returns
    -------
    numpy.ndarray
        Boolean ground mask in the order of *points*.
    """
    unknown = set(parameters) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown envelope parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **parameters}
    n = len(points)
    if n == 0:
        return np.zeros(0, bool)
    usable = ~np.isin(points["Classification"], (7, 18))
    if seed is None:
        from alsdb.methods.ptd import ptd_ground

        seed = ptd_ground(points, workers=workers, **{**PTD_SEED, **(ptd_parameters or {})})
    seed = np.asarray(seed, bool) & usable
    names = points.dtype.names
    if "ReturnNumber" in names and "NumberOfReturns" in names:
        last = points["ReturnNumber"] == points["NumberOfReturns"]
    else:
        last = np.ones(n, bool)
    cand = np.flatnonzero((last & usable) | seed)
    if len(cand) < 4:
        return seed
    x = np.ascontiguousarray(points["X"][cand], float)
    y = np.ascontiguousarray(points["Y"][cand], float)
    z = np.ascontiguousarray(points["Z"][cand], float)
    is_seed = seed[cand]
    evidence = vegetation_evidence(points, p["evidence_cell"])[cand]
    v = np.clip(evidence / p["evidence_threshold"], 0.0, 1.0)

    grid = _Grid(x, y, p["cell"])
    located = grid._locate(x, y)
    prior = np.where(is_seed, 1.0, p["prior"])
    w = is_seed.astype(float)  # the first surface is PTD's ground alone
    a, b_, s_sym = p["weight_scale"], p["weight_power"], p["symmetric_scale"]
    for _ in range(int(p["iterations"])):
        nodes = fit_surface(grid, located, z, w, p["smoothness"], workers=workers, x=x, y=y,
                            anchor=p["anchor"], block=int(p["block"]), overlap=int(p["overlap"]))  # fmt: skip
        r = z - grid.sample(nodes, x, y)
        w_env = np.where(r <= 0, 1.0, 1.0 / (1.0 + (a * np.maximum(r, 0)) ** b_))
        w_sym = 1.0 / (1.0 + (r / s_sym) ** 2)
        w = prior * ((1 - v) * w_sym + v * w_env)
    tolerance = p["keep"] + (p["keep_bare"] - p["keep"]) * (1 - v)
    within = r <= tolerance
    kept_seed = within & is_seed
    added = np.flatnonzero(within & ~is_seed)
    if len(added) and p["gap"] > 0:
        from scipy.spatial import cKDTree

        if kept_seed.any():
            tree = cKDTree(np.column_stack((x[kept_seed], y[kept_seed])))
            d, _ = tree.query(np.column_stack((x[added], y[added])), k=1,
                              distance_upper_bound=p["gap"], workers=max(1, workers))  # fmt: skip
            added = added[~np.isfinite(d)]
    ground = np.zeros(n, bool)
    ground[cand[kept_seed]] = True
    ground[cand[added]] = True
    logger.debug("Envelope ground: %d seed, %d candidates, %d ground", int(seed.sum()), len(cand),
                 int(ground.sum()))  # fmt: skip
    return ground
