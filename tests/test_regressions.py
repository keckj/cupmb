#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
"""
Regression tests for previously reported bugs.
"""

import collections

import numpy as np
import pytest

from pmb_testing import (
    edges_of,
    non_degenerate,
    ftype_eps,
    grid_coordinates,
    predicate_source,
    run_python,
    scalar_source,
    sphere,
    sphere_predicate_body,
    weld,
)

from cupmb import ExtraParameter

X0, L = -0.5, 1.0


def counts(out):
    V, I = out[:2]
    return (0 if V is None else V.shape[0], 0 if I is None else I.shape[0])


@pytest.mark.parametrize("axis", (0, 1, 2), ids=lambda k: "xyz"[k])
def test_exported_scalar_field_layout_3D(make_pmb, axis):
    """The exported scalar field is indexed S[iz, iy, ix] (x fastest) and has shape N[::-1]+1."""
    N = (4, 5, 6)
    pmb = make_pmb(3, scalar_source(3, "xyz"[axis]), N=N, x0=(-0.3, -0.35, -0.45), export_scalar_field=True)
    out = pmb(to_numpy=True)
    assert len(out) == 3
    S = out[2]
    assert S.shape == (7, 6, 5) and S.flags.c_contiguous
    coords = np.meshgrid(*grid_coordinates(pmb), indexing="ij")
    np.testing.assert_allclose(S, coords[axis].T, rtol=0, atol=1e-6)


@pytest.mark.parametrize("axis", (0, 1), ids=lambda k: "xy"[k])
def test_exported_scalar_field_layout_2D(make_pmb, axis):
    """The exported scalar field is indexed S[iy, ix] (x fastest, as matplotlib) and has shape N[::-1]+1."""
    N = (4, 5)
    pmb = make_pmb(2, scalar_source(2, "xy"[axis]), N=N, x0=(-0.3, -0.35), export_scalar_field=True)
    S = pmb(to_numpy=True)[2]
    assert S.shape == (6, 5) and S.flags.c_contiguous
    coords = np.meshgrid(*grid_coordinates(pmb), indexing="ij")
    np.testing.assert_allclose(S, coords[axis].T, rtol=0, atol=1e-6)


def test_exported_scalar_field_layout_1D(make_pmb):
    """The exported scalar field is indexed S[ix] and has shape N+1."""
    pmb = make_pmb(1, scalar_source(1, "x"), N=7, x0=-0.3, export_scalar_field=True)
    V, I, S = pmb(to_numpy=True)
    assert S.shape == (8,)
    np.testing.assert_allclose(S, grid_coordinates(pmb)[0], rtol=0, atol=1e-6)
    # Single zero crossing at x=0 (vertex k is simplex k)
    assert V.shape == I.shape == (1, 1)
    assert abs(float(V[0, 0])) <= 1e-6 and int(I[0, 0]) == 0


@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_exported_scalar_field_matches_vertices(make_pmb, dim):
    """Exported values are the field values at the grid points."""
    field = sphere(dim)
    pmb = make_pmb(dim, field.fdef, N=(17, 19, 23)[:dim], x0=X0, L=L, ftype=np.float64, export_scalar_field=True)
    V, I, S = pmb(to_numpy=True)
    coords = np.stack(np.meshgrid(*grid_coordinates(pmb), indexing="ij"), axis=-1)
    np.testing.assert_allclose(S, field.f(coords).T, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_nan_field_gives_no_nan_vertices(make_pmb, dim):
    """NaN values (here undefined for x > 0.2) must never produce NaN vertices."""
    field = sphere(dim)
    body = f"(x > T(0.2)) ? T(nan(\"\")) : ({field.body})"
    pmb = make_pmb(dim, scalar_source(dim, body), N=40, x0=X0, L=L)
    V, I = pmb(to_numpy=True)
    assert V is not None and I is not None
    assert np.isfinite(V).all(), "NaN vertices"
    assert I.min() >= 0 and I.max() < V.shape[0]
    assert (V[:, 0] <= 0.2 + 1.0 / 40).all()


@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_full_nan_field_is_empty(make_pmb, dim):
    pmb = make_pmb(dim, scalar_source(dim, 'T(nan(""))'), N=20)
    V, I = pmb(to_numpy=True)
    assert V is None and I is None


@pytest.mark.parametrize("ftype", (np.float32, np.float64), ids=lambda t: np.dtype(t).name)
@pytest.mark.parametrize("ctype", ("float", "double", "T"))
def test_extra_parameter_from_python_float(make_pmb, ftype, ctype):
    """Extra parameters with an explicit C type accept Python floats (converted to the right type)."""
    field = sphere(3)
    c = field.center
    body = " + ".join(f"({a} - T({float(ck)!r}))*({a} - T({float(ck)!r}))" for (a, ck) in zip("xyz", c))
    body += " - T(r)*T(r)"
    extra_params = collections.OrderedDict(r=ExtraParameter(ctype))
    fdef = scalar_source(3, body, extra_args=f", const {ctype} r")
    pmb = make_pmb(3, fdef, N=32, x0=X0, L=L, ftype=ftype, extra_params=extra_params)

    for r in (0.31, 0.2):
        V, I = pmb(r=r, to_numpy=True)
        assert V is not None
        d = np.abs(np.linalg.norm(V.astype(np.float64) - c, axis=1) - r)
        assert d.max() <= (1.0 / 32) ** 2 / (4 * r) * 1.05 + 1e-5, d.max()

    # Positional extra parameters
    V, I = pmb(0.25, to_numpy=True)
    d = np.abs(np.linalg.norm(V.astype(np.float64) - c, axis=1) - 0.25)
    assert d.max() <= (1.0 / 32) ** 2 + 1e-5


def test_extra_parameter_callable(make_pmb):
    """Callable extra parameters are evaluated when the parameter is not given."""
    field = sphere(3)
    radii = iter((0.3, 0.2))
    extra_params = collections.OrderedDict(r=ExtraParameter("T", lambda: next(radii)))
    body = " + ".join(f"({a} - T({float(ck)!r}))*({a} - T({float(ck)!r}))" for (a, ck) in zip("xyz", field.center))
    fdef = scalar_source(3, body + " - r*r", extra_args=", const T& r")
    pmb = make_pmb(3, fdef, N=32, x0=X0, L=L, extra_params=extra_params)
    n1 = counts(pmb(to_numpy=True))[1]
    n2 = counts(pmb(to_numpy=True))[1]
    assert n1 > n2 > 0
    assert counts(pmb(r=0.3, to_numpy=True))[1] == n1


@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_change_types_after_construction(make_pmb, dim):
    """Changing pmb.ftype / pmb.itype after a first call rebuilds the kernels with the new types."""
    field = sphere(dim)
    pmb = make_pmb(dim, field.fdef, N=33, x0=X0, L=L, ftype=np.float32, itype=np.uint32)
    V32, I32 = pmb(to_numpy=True)
    assert V32.dtype == np.float32 and I32.dtype == np.uint32

    pmb.ftype = np.float64
    V64, I64 = pmb(to_numpy=True)
    assert V64.dtype == np.float64
    assert I64.shape == I32.shape
    d = field.distance(V64)
    assert d.max() <= (1.0 / 33) ** 2 / (4 * field.radius) * 1.05 + 1e-12, d.max()
    np.testing.assert_allclose(np.unique(V64, axis=0), np.unique(V32, axis=0), atol=1e-5)

    pmb.itype = np.uint64
    V, I = pmb(to_numpy=True)
    assert V.dtype == np.float64 and I.dtype == np.uint64
    assert I.shape == I32.shape

    pmb.ftype = np.float32
    V, I = pmb(to_numpy=True)
    assert V.dtype == np.float32 and I.dtype == np.uint64
    assert I.shape == I32.shape


def test_change_resolution_after_construction(make_pmb):
    field = sphere(3)
    pmb = make_pmb(3, field.fdef, N=24, x0=X0, L=L)
    n24 = counts(pmb(to_numpy=True))
    pmb.N = 48
    n48 = counts(pmb(to_numpy=True))
    assert n48[1] > 3 * n24[1]
    ref = make_pmb(3, field.fdef, N=48, x0=X0, L=L)
    assert counts(ref(to_numpy=True))[1] == n48[1]


@pytest.mark.parametrize("with_predicate", (False, True), ids=("mask", "predicate"))
def test_grid_size_is_not_a_template_parameter(make_pmb, force_block_size, monkeypatch, with_predicate):
    """The grid size is a kernel argument: changing it builds the same kernels (no recompilation)."""
    from cupmb import ParallelMarchingBlocks

    build = ParallelMarchingBlocks.build_module_and_extract_kernels_from_source
    names = []

    def recording_build(self, kernel_source_code, kernel_names, device=None):
        names.append(tuple(kernel_names))
        return build(self, kernel_source_code, kernel_names, device)

    monkeypatch.setattr(ParallelMarchingBlocks, "build_module_and_extract_kernels_from_source", recording_build)
    force_block_size((4, 4, 4))

    field = sphere(3)
    pdef = predicate_source(3, sphere_predicate_body(field.center, field.radius)) if with_predicate else ""
    built = {}
    for N in (24, 48, (40, 17, 33)):
        names.clear()
        pmb = make_pmb(3, field.fdef, pdef=pdef, N=N, x0=X0, L=L)
        assert counts(pmb(to_numpy=True))[1] > 0
        built[str(N)] = set(names)
    assert len(built["24"]) > 0
    assert built["24"] == built["48"] == built["(40, 17, 33)"], built


def test_legacy_scalar_signature(make_pmb):
    """User scalar functions declared with the former grid size template pack (I... N) are still accepted."""
    field = sphere(3)
    fdef = field.fdef.replace(
        "template <typename T, typename I, I dim>", "template <typename T, typename I, I dim, I... N>"
    )
    assert fdef != field.fdef
    ref = counts(make_pmb(3, field.fdef, N=24, x0=X0, L=L)(to_numpy=True))
    assert counts(make_pmb(3, fdef, N=24, x0=X0, L=L)(to_numpy=True)) == ref


@pytest.mark.parametrize("to_numpy", (False, True))
def test_non_blocking_stream(make_pmb, to_numpy):
    """Results computed on a non-blocking stream are synchronized properly (stable counts and values)."""
    import cupy as cp

    field = sphere(3)
    ref = make_pmb(3, field.fdef, N=96, x0=X0, L=L)(to_numpy=True)
    pmb = make_pmb(3, field.fdef, N=96, x0=X0, L=L)
    stream = cp.cuda.Stream(non_blocking=True)
    for _ in range(20):
        V, I = pmb(stream=stream, to_numpy=to_numpy)
        assert counts((V, I)) == counts(ref)
        if not to_numpy:
            with stream:
                V, I = cp.asnumpy(V, stream=stream), cp.asnumpy(I, stream=stream)
            stream.synchronize()
        assert np.isfinite(V).all()
        assert I.max() < V.shape[0]
        np.testing.assert_allclose(np.sort(V, axis=0), np.sort(ref[0], axis=0), atol=1e-6)


@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_empty_result(make_pmb, dim):
    """A field without zeros gives None buffers, return_elapsed_time is honored."""
    pmb = make_pmb(dim, scalar_source(dim, "T(1)"), N=32)
    V, I = pmb(to_numpy=True)
    assert V is None and I is None

    res = pmb(return_elapsed_time=True)
    assert isinstance(res, tuple) and len(res) == 2, res
    (V, I), elapsed = res
    assert V is None and I is None
    assert isinstance(elapsed, float) and elapsed >= 0.0


@pytest.mark.parametrize("dim", (1, 3), ids=lambda d: f"{d}D")
def test_empty_result_with_exported_scalar_field(make_pmb, dim):
    pmb = make_pmb(dim, scalar_source(dim, "x + T(1)"), N=(4, 5, 6)[:dim], export_scalar_field=True)
    V, I, S = pmb(to_numpy=True)
    assert V is None and I is None
    assert S is not None and S.shape == (5, 6, 7)[:dim][::-1]


@pytest.mark.parametrize("dim", (0, 4, -1))
def test_invalid_dimension_is_rejected(make_pmb, dim):
    with pytest.raises(ValueError):
        make_pmb(dim, sphere(3).fdef)
    with pytest.raises(TypeError):
        make_pmb(True, sphere(3).fdef)


def test_change_dimension_after_construction(make_pmb):
    """Changing pmb.dim resets the bounding box to the unit box and keeps an isotropic grid size."""
    pmb = make_pmb(2, scalar_source(2, "x - T(0.3)"), N=40)
    V, I = pmb(to_numpy=True)
    assert V.shape[1] == I.shape[1] == 2
    pmb.fdef = scalar_source(1, "x - T(0.3)")
    pmb.dim = 1
    assert tuple(pmb.N) == (40,)
    V, I = pmb(to_numpy=True)
    assert V.shape == I.shape == (1, 1)
    np.testing.assert_allclose(V[0, 0], 0.3, atol=1e-6)


def test_return_elapsed_time(make_pmb):
    field = sphere(3)
    pmb = make_pmb(3, field.fdef, N=32, x0=X0, L=L)
    (V, I), elapsed = pmb(to_numpy=True, return_elapsed_time=True)
    assert V is not None and I is not None
    assert isinstance(elapsed, float) and elapsed > 0.0


@pytest.mark.slow
def test_float32_large_indices_are_exact(make_pmb):
    """
    More than 2^24 vertices with ftype=float32: indices must not go through float32 and stay exact,
    each simplex must fit in a single grid cell.
    """
    import cupy as cp

    N = 256
    # Integer hash of the grid point index: random signs, about half of the grid edges are crossed.
    body = (
        "[&]() -> T { unsigned int h = (unsigned int)(ix)*73856093u ^ (unsigned int)(iy)*19349663u"
        " ^ (unsigned int)(iz)*83492791u; h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;"
        " return T(h & 0xffffu) / T(65536) - T(0.5) + T(1) / T(131072); }()"
    )
    pmb = make_pmb(3, scalar_source(3, body), N=N, x0=X0, L=L, ftype=np.float32, itype=np.uint32)
    V, I = pmb()
    assert V is not None and I is not None
    assert int(I.max()) >= 2**24, "test is not meaningful, increase N"
    assert int(I.max()) < V.shape[0]

    dx = L / N
    P = V[I.astype(cp.int64)]  # (ntris, 3, 3)
    extent = (P.max(axis=1) - P.min(axis=1)).max(axis=0)
    assert (extent <= dx * 1.01).all(), extent / dx
    edges = cp.linalg.norm(P - cp.roll(P, 1, axis=1), axis=-1)
    assert float(edges.max()) <= np.sqrt(3) * dx * 1.01


@pytest.mark.slow
@pytest.mark.parametrize("N", (255, 256, 257))
def test_large_grids(make_pmb, N):
    field = sphere(3)
    pmb = make_pmb(3, field.fdef, N=N, x0=X0, L=L, ftype=np.float32)
    V, I = pmb(to_numpy=True)
    d = field.distance(V.astype(np.float64))
    assert d.max() <= (L / N) ** 2 / (4 * field.radius) * 1.05 + 64 * ftype_eps(np.float32)
    Vw, Iw = weld(V, I)
    Iw = non_degenerate(Iw)
    E, edge_counts = np.unique(edges_of(Iw), axis=0, return_counts=True)
    assert (edge_counts == 2).all()
    assert Vw.shape[0] - E.shape[0] + Iw.shape[0] == 2


PYCUDA_SCRIPT = """
import numpy as np
from cupmb import ParallelMarchingBlocks, Backend, AutotuningRigor
from pmb_testing import sphere, predicate_source, sphere_predicate_body

dim = {dim}
field = sphere(dim)
for pdef in ("", predicate_source(dim, sphere_predicate_body(field.center, field.radius))):
    pmb = ParallelMarchingBlocks(
        dim, field.fdef, pdef=pdef, N=32, x0=-0.5, backend=Backend.PYCUDA, autotuning_rigor=AutotuningRigor.MEASURE
    )
    V, I = pmb(to_numpy=True)
    assert V is not None and I is not None
    assert V.shape[1] == I.shape[1] == dim
    assert I.max() < V.shape[0]
    d = field.distance(V.astype(np.float64)).max()
    assert d <= (1.0 / 32) ** 2 / (4 * field.radius) * 1.05 + 1e-5, d
    if dim == 1:
        assert V.shape == I.shape == (2, 1) and (I[:, 0] == np.arange(2)).all()
print("OK", V.shape, I.shape)
"""


@pytest.mark.parametrize("dim", (1, 3), ids=lambda d: f"{d}D")
def test_pycuda_backend(backend, dim):
    """The pycuda backend imports without OpenGL support and produces a sphere mesh (own process/context)."""
    pytest.importorskip("pycuda")
    proc = run_python(PYCUDA_SCRIPT.format(dim=dim))
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-4000:]
