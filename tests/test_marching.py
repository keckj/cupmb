#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
"""
Correctness of marching blocks against analytic scalar fields: zero crossings on grid edges (1D),
marching squares (2D) and marching cubes (3D).
"""

import numpy as np
import pytest

from pmb_testing import (
    FIELDS,
    edges_of,
    ftype_eps,
    grid_coordinates,
    non_degenerate,
    sample_field,
    scalar_source,
    signed_volume,
    sphere,
    weld,
)

from cupmb import AllocationPolicy, VertexProperty

DIMS = (1, 2, 3)
FTYPES = (np.float32, np.float64)
ITYPES = (np.uint32, np.uint64)
POLICIES = (AllocationPolicy.ALLOCATE_EXACT_SIZE, AllocationPolicy.ALLOCATE_UPPER_POW2)

X0, L = -0.5, 1.0
RESOLUTION = {1: 1021, 2: 61, 3: 29}  # not multiples of any block size


def run(pmb, **kwds):
    out = pmb(to_numpy=True, **kwds)
    V, I = out[:2]
    assert V is not None and I is not None, "no output, the analytic field has zeros in the domain"
    return V, I


def check_roots(pmb, field, V, I, ftype, itype):
    """
    1D: vertices are the zero crossings of the scalar field on grid edges and each simplex is a single vertex.
    Every root of the analytic field is found exactly once, within the linear interpolation error bound.
    """
    assert pmb.dim == 1
    dx = float(pmb.dx[0])
    lo = float(pmb.x0[0])
    hi = lo + float(pmb.L[0])

    # Types and shapes: V (n,1) and I (n,1), simplex k references vertex k
    assert isinstance(V, np.ndarray) and isinstance(I, np.ndarray)
    assert V.dtype == ftype, V.dtype
    assert I.dtype == itype, I.dtype
    assert V.ndim == 2 and V.shape[1] == 1, V.shape
    assert I.shape == V.shape, (I.shape, V.shape)
    np.testing.assert_array_equal(I[:, 0], np.arange(V.shape[0]))

    eps = ftype_eps(ftype)
    P = V[:, 0].astype(np.float64)
    assert np.isfinite(P).all()
    assert (P >= lo - 16 * eps).all() and (P <= hi + 16 * eps).all()

    # Test precondition: no root lies on a grid point (such a root would give a vertex on each adjacent edge)
    roots, slopes = field.roots(lo, hi)
    assert roots.size > 0
    x = grid_coordinates(pmb)[0]
    assert np.abs(roots[:, None] - x[None, :]).min() > 1e-6 * dx

    # Every root is found exactly once (vertices are never duplicated on block boundaries in 1D)
    assert np.unique(P).size == P.size, "duplicated vertices"
    assert P.size == roots.size, (P.size, roots.size)
    P = np.sort(P)

    # Residual: linear interpolation error of f along a grid edge is bounded by max|f''| dx^2 / 8
    rounding = 64 * eps * max(1.0, field.gradient_bound)
    residual_tol = field.hessian_bound * dx**2 / 8 * 1.05 + rounding
    residual = np.abs(field.f(P[:, None]))
    assert residual.max() <= residual_tol, (residual.max(), residual_tol)

    # Position error: |f(v)| <= M dx^2/8 and |f'| >= |f'(root)| - M dx on the grid edge containing the root
    min_slope = np.abs(slopes) - field.hessian_bound * dx
    assert (min_slope > 0).all(), "test precondition: grid is too coarse"
    position_tol = residual_tol / min_slope + 16 * eps * max(1.0, abs(lo), abs(hi))
    error = np.abs(P - roots)
    assert (error <= position_tol).all(), (error.max(), position_tol.min())
    assert (error <= dx).all()

    return V, I


def check_mesh(pmb, field, V, I, ftype, itype):
    dim = pmb.dim
    if dim == 1:
        return check_roots(pmb, field, V, I, ftype, itype)
    dx = np.asarray(pmb.dx, dtype=np.float64)

    # Types and shapes
    assert isinstance(V, np.ndarray) and isinstance(I, np.ndarray)
    assert V.dtype == ftype, V.dtype
    assert I.dtype == itype, I.dtype
    assert V.ndim == 2 and V.shape[1] == dim, V.shape
    assert I.ndim == 2 and I.shape[1] == dim, I.shape
    assert V.shape[0] > 0 and I.shape[0] > 0

    # Indices are in range and all vertices are referenced
    assert I.min() >= 0
    assert I.max() < V.shape[0]
    assert np.unique(I).size == V.shape[0], "some vertices are not referenced by any simplex"

    # Vertices are finite and inside the domain
    assert np.isfinite(V).all()
    eps = ftype_eps(ftype)
    lo = np.asarray(pmb.x0, dtype=np.float64)
    hi = lo + np.asarray(pmb.L, dtype=np.float64)
    tol = 16 * eps
    assert (V >= lo - tol).all() and (V <= hi + tol).all()

    P = V.astype(np.float64)
    hmax = dx.max()

    # Residual: linear interpolation error of f along a grid edge is bounded by max|f''| dx^2 / 8
    rounding = 64 * eps * max(1.0, field.gradient_bound)
    residual = np.abs(field.f(P))
    residual_tol = field.hessian_bound * hmax**2 / 8 * 1.05 + rounding
    assert residual.max() <= residual_tol, (residual.max(), residual_tol)

    # Distance to the analytic surface
    if field.distance is not None:
        distance = field.distance(P)
        if hasattr(field, "radius"):
            distance_tol = hmax**2 / (4 * field.radius) * 1.05 + rounding
        else:
            distance_tol = rounding
        assert distance.max() <= distance_tol, (distance.max(), distance_tol)

    # Simplices are small: all vertices of a simplex belong to the same grid cell
    extent = np.ptp(P[np.asarray(I, dtype=np.int64)], axis=1)
    assert (extent <= dx * (1 + 1e-3) + 4 * eps).all(), extent.max(axis=0) / dx

    # Topology after welding duplicated block face vertices
    Vw, Iw = weld(V, I)
    assert Vw.shape[0] <= V.shape[0]
    Iw = non_degenerate(Iw)
    E, counts = np.unique(edges_of(Iw), axis=0, return_counts=True)
    Pw = Vw.astype(np.float64)
    if dim == 3:
        assert counts.max() <= 2, "non manifold edge"
        # Consistent orientation: an interior edge is traversed once in each direction by its two triangles
        directed = np.concatenate([Iw[:, [0, 1]], Iw[:, [1, 2]], Iw[:, [2, 0]]])
        assert np.unique(directed, axis=0).shape[0] == directed.shape[0], "inconsistently oriented triangles"
        # Counterclockwise triangles: the geometric normal (v1-v0)x(v2-v0) points from negative to positive values
        a, b, c = Pw[Iw[:, 0]], Pw[Iw[:, 1]], Pw[Iw[:, 2]]
        normals, centers = np.cross(b - a, c - a), (a + b + c) / 3
    else:
        assert np.bincount(Iw.ravel()).max() <= 2, "non manifold vertex"
        # Consistent orientation: a vertex starts at most one segment and ends at most one segment
        assert np.bincount(Iw[:, 0]).max() <= 1 and np.bincount(Iw[:, 1]).max() <= 1, "inconsistently oriented segments"
        # Counterclockwise segments around negative regions: the normal (dy,-dx) points from negative to positive values
        d = Pw[Iw[:, 1]] - Pw[Iw[:, 0]]
        normals, centers = np.stack([d[:, 1], -d[:, 0]], axis=-1), 0.5 * (Pw[Iw[:, 0]] + Pw[Iw[:, 1]])
    h = 1e-6
    gradients = np.stack([field.f(centers + h * e) - field.f(centers - h * e) for e in np.eye(dim)], axis=-1) / (2 * h)
    assert (np.einsum("ij,ij->i", normals, gradients) > 0).mean() > 0.99, "simplex normals do not follow the gradient"

    if field.closed:
        if dim == 3:
            assert (counts == 2).all(), f"mesh is not watertight ({(counts == 1).sum()} boundary edges)"
            assert Vw.shape[0] - E.shape[0] + Iw.shape[0] == 2, "Euler characteristic of a sphere is 2"
            # Counterclockwise triangles seen from outside: positive signed volume
            volume = signed_volume(Vw, Iw)
            exact = 4.0 / 3.0 * np.pi * field.radius**3
        else:
            assert (np.bincount(Iw.ravel(), minlength=Vw.shape[0]) == 2).all(), "curve is not closed"
            assert Vw.shape[0] == E.shape[0], "Euler characteristic of a closed curve is 0"
            # Counterclockwise segments around the negative region: positive signed area (shoelace formula)
            a, b = Pw[Iw[:, 0]], Pw[Iw[:, 1]]
            volume = 0.5 * np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
            exact = np.pi * field.radius**2
        assert abs(volume - exact) <= 0.02 * exact, (volume, exact)
    elif dim == 3:
        # Boundary edges can only lie on the domain boundary
        boundary = E[counts == 1]
        Pb = Vw[boundary.ravel()].astype(np.float64)
        on_boundary = (np.abs(Pb - lo) <= 1e-3 * dx) | (np.abs(Pb - hi) <= 1e-3 * dx)
        assert on_boundary.any(axis=1).all(), "mesh has holes inside the domain"

    return Vw, Iw


@pytest.mark.parametrize("policy", POLICIES, ids=str)
@pytest.mark.parametrize("itype", ITYPES, ids=lambda t: np.dtype(t).name)
@pytest.mark.parametrize("ftype", FTYPES, ids=lambda t: np.dtype(t).name)
@pytest.mark.parametrize("dim", DIMS, ids=lambda d: f"{d}D")
def test_sphere(make_pmb, dim, ftype, itype, policy):
    field = sphere(dim)
    pmb = make_pmb(
        dim, field.fdef, N=RESOLUTION[dim], x0=X0, L=L, ftype=ftype, itype=itype, default_allocation_policy=policy
    )
    V, I = run(pmb)
    check_mesh(pmb, field, V, I, ftype, itype)

    # Second call reuses the allocated buffers and gives the same mesh size
    V2, I2 = run(pmb)
    assert V2.shape == V.shape and I2.shape == I.shape


FIELD_CASES = [(dim, name) for dim in DIMS for name in ("plane", "sines")] + [(1, "polynomial")]


@pytest.mark.parametrize("ftype", FTYPES, ids=lambda t: np.dtype(t).name)
@pytest.mark.parametrize("dim, field_name", FIELD_CASES, ids=lambda v: f"{v}D" if isinstance(v, int) else v)
def test_fields(make_pmb, dim, ftype, field_name):
    field = FIELDS[field_name](dim)
    pmb = make_pmb(dim, field.fdef, N=RESOLUTION[dim], x0=X0, L=L, ftype=ftype)
    V, I = run(pmb)
    check_mesh(pmb, field, V, I, ftype, np.uint32)


@pytest.mark.parametrize("dim", DIMS, ids=lambda d: f"{d}D")
def test_anisotropic_grid(make_pmb, dim):
    """Different resolutions, lengths and origins per axis."""
    field = sphere(dim)
    N = (37, 23, 31)[:dim]  # in 1D: a single block, the roots are found anyway
    x0 = (-0.45, -0.52, -0.38)[:dim]
    Ls = (0.93, 1.07, 0.81)[:dim]
    pmb = make_pmb(dim, field.fdef, N=N, x0=x0, L=Ls, ftype=np.float64)
    assert tuple(pmb.N) == N
    np.testing.assert_allclose(pmb.dx, np.asarray(Ls) / np.asarray(N))
    V, I = run(pmb)
    check_mesh(pmb, field, V, I, np.float64, np.uint32)


@pytest.mark.parametrize("itype", (np.int32, np.int64, np.int16, np.uint16), ids=lambda t: np.dtype(t).name)
def test_unsupported_index_types_are_rejected(make_pmb, itype):
    """Only uint32 and uint64 simplex index types are supported."""
    with pytest.raises((TypeError, ValueError, AssertionError)):
        make_pmb(3, sphere(3).fdef, N=12, x0=X0, L=L, itype=itype)


@pytest.mark.parametrize("dim", DIMS, ids=lambda d: f"{d}D")
def test_device_output(make_pmb, dim):
    """Without to_numpy the output buffers stay on the device (cupy arrays)."""
    import cupy as cp

    field = sphere(dim)
    pmb = make_pmb(dim, field.fdef, N=16, x0=X0, L=L)
    V, I = pmb()
    assert isinstance(V, cp.ndarray) and isinstance(I, cp.ndarray)
    Vh, Ih = pmb(to_numpy=True)
    assert isinstance(Vh, np.ndarray) and isinstance(Ih, np.ndarray)
    assert V.shape == Vh.shape and I.shape == Ih.shape


@pytest.mark.parametrize("dim", DIMS, ids=lambda d: f"{d}D")
def test_vertex_count_matches_grid_edge_crossings(make_pmb, dim):
    """One welded vertex per grid edge with a sign change of the sampled scalar field."""
    field = FIELDS["sines"](dim)
    pmb = make_pmb(dim, field.fdef, N=RESOLUTION[dim], x0=X0, L=L, ftype=np.float64)
    V, I = run(pmb)
    Vw, _ = weld(V, I)
    S = sample_field(pmb, field.f) < 0
    crossings = sum(int((np.diff(S, axis=k) != 0).sum()) for k in range(dim))
    assert Vw.shape[0] == crossings


def test_vertex_count_vs_skimage(make_pmb):
    skimage_measure = pytest.importorskip("skimage.measure")
    field = FIELDS["sines"](3)
    pmb = make_pmb(3, field.fdef, N=RESOLUTION[3], x0=X0, L=L, ftype=np.float64)
    V, I = run(pmb)
    Vw, Iw = weld(V, I)
    Iw = non_degenerate(Iw)

    S = sample_field(pmb, field.f)
    verts, faces, _, _ = skimage_measure.marching_cubes(S, level=0.0, spacing=tuple(pmb.dx), method="lorensen")
    assert Vw.shape[0] == verts.shape[0]
    assert abs(Iw.shape[0] - faces.shape[0]) <= 0.02 * faces.shape[0]

    # Same vertex positions (skimage vertices are relative to the domain origin)
    cKDTree = pytest.importorskip("scipy.spatial").cKDTree
    distance, _ = cKDTree(Vw).query(verts + np.asarray(pmb.x0))
    assert distance.max() <= 1e-6


def test_output_is_deterministic_in_size(make_pmb):
    field = sphere(3)
    pmb = make_pmb(3, field.fdef, N=40, x0=X0, L=L)
    shapes = {tuple(a.shape for a in pmb(to_numpy=True)[:2]) for _ in range(5)}
    assert len(shapes) == 1, shapes


@pytest.mark.parametrize("N", (32, 62, 63, 64, 93, 94, 2047, 100003))
def test_roots_1D_block_boundaries(make_pmb, force_block_size, N):
    """
    1D: roots in cells adjacent to logical block boundaries (B=32, 31 cells per block) are found exactly once,
    for grids that exactly tile the blocks or end with a sliver of cells.
    """
    force_block_size((32,))
    field = FIELDS["sines"](1)
    pmb = make_pmb(1, field.fdef, N=N, x0=X0, L=L, ftype=np.float64)
    V, I = run(pmb)
    check_roots(pmb, field, V, I, np.float64, np.uint32)


def test_root_on_grid_point_1D(make_pmb):
    """
    1D: zero values are considered negative. A sign changing root lying exactly on a grid point gives a single
    vertex, a zero touching root (both neighbors positive) gives a vertex on both adjacent grid edges (same position).
    """
    pmb = make_pmb(1, scalar_source(1, "x - T(0.25)"), N=8, x0=0.0, L=1.0, ftype=np.float64)
    V, I = run(pmb)
    np.testing.assert_array_equal(V[:, 0], [0.25])
    np.testing.assert_array_equal(I[:, 0], [0])

    pmb = make_pmb(1, scalar_source(1, "(x - T(0.25))*(x - T(0.25))"), N=8, x0=0.0, L=1.0, ftype=np.float64)
    V, I = run(pmb)
    np.testing.assert_array_equal(V[:, 0], [0.25, 0.25])
    Vw, Iw = weld(V, I)
    assert Vw.shape == (1, 1) and (Iw == 0).all()


ORIENTATION_PROPERTIES = r"""
namespace cupmb {
    /* Normal of a 0-simplex: sign of the derivative of the scalar field (from negative to positive values) */
    template <typename T, typename I, I dim>
    __device__ __forceinline__ trove::array<T,dim> normal(const trove::array<T,dim>& p) {
        const T h = T(1e-4);
        trove::array<T,dim> pp(p), pm(p);
        trove::get<0>(pp) += h;
        trove::get<0>(pm) -= h;
        const trove::array<I,dim> i;
        trove::array<T,dim> n;
        trove::get<0>(n) = (scalar<T,I,dim>(pp, i) - scalar<T,I,dim>(pm, i) > T(0) ? T(+1) : T(-1));
        return n;
    }

    template <typename T, typename I, I dim>
    __device__ __forceinline__ trove::array<T,1> value(const trove::array<T,dim>& p) {
        return trove::make_array<T>(scalar<T,I,dim>(p, trove::array<I,dim>()));
    }
}
"""


@pytest.mark.parametrize("ftype", FTYPES, ids=lambda t: np.dtype(t).name)
def test_vertex_properties_1D(make_pmb, ftype):
    """
    1D vertex properties: a 0-simplex carries no orientation, its normal is the sign of the derivative of the
    scalar field (here computed as a vertex property of shape (n, 1)), scalar property has shape (n,).
    """
    field = FIELDS["sines"](1)
    properties = (
        VertexProperty("normal", fname="normal", dtype="T", shape=("dim",)),
        VertexProperty("value", fname="value", dtype="T", shape=None),
    )
    pmb = make_pmb(
        1, field.fdef + ORIENTATION_PROPERTIES, N=RESOLUTION[1], x0=X0, L=L, ftype=ftype, vertex_properties=properties
    )
    V, I, normal, value = pmb(to_numpy=True)
    check_roots(pmb, field, V, I, ftype, np.uint32)
    assert normal.shape == V.shape and normal.dtype == ftype
    assert value.shape == V.shape[:1] and value.dtype == ftype

    _, slopes = field.roots(X0, X0 + L)
    np.testing.assert_array_equal(normal[np.argsort(V[:, 0]), 0], np.sign(slopes))
    assert np.abs(value).max() <= field.hessian_bound * (L / RESOLUTION[1]) ** 2 / 8 * 1.05 + 1e-5


@pytest.mark.parametrize("with_predicate", (False, True), ids=("no_predicate", "predicate"))
def test_autotuning_measure_1D(make_pmb, with_predicate):
    """1D kernels are benchmarked (block size and thread size) and give the same roots as the estimated sizes."""
    from pmb_testing import ALWAYS_TRUE_PREDICATE, predicate_source

    from cupmb import AutotuningRigor

    field = FIELDS["sines"](1)
    pdef = predicate_source(1, ALWAYS_TRUE_PREDICATE) if with_predicate else ""
    kwds = dict(pdef=pdef, N=100003, x0=X0, L=L, ftype=np.float64)
    ref = run(make_pmb(1, field.fdef, autotuning_rigor=AutotuningRigor.ESTIMATE, **kwds))
    pmb = make_pmb(1, field.fdef, autotuning_rigor=AutotuningRigor.MEASURE, **kwds)
    V, I = run(pmb)
    check_roots(pmb, field, V, I, np.float64, np.uint32)
    np.testing.assert_array_equal(np.sort(V, axis=0), np.sort(ref[0], axis=0))
