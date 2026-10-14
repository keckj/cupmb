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
Runtime KD-tree domain decomposition from block predicates (cupmb::may_contain_zeros), 2D and 3D.
"""

import numpy as np
import pytest

from pmb_testing import (
    ALWAYS_TRUE_PREDICATE,
    assert_same_points,
    predicate_source,
    run_python,
    sphere,
    sphere_predicate_body,
)

X0, L = -0.5, 1.0

# Grid sizes chosen so that the block grid G = ceil(N / (B-1)) is either a power of two, exactly tiles
# the domain or ends with a sliver of blocks, for the block sizes below.
SIZES = (49, 50, 63, 95, 96)
BLOCK_SIZES = {
    3: ((4, 4, 4), (8, 8, 8), (8, 4, 4)),
    2: ((8, 8), (16, 16), (16, 8)),
}


def counts(out):
    V, I = out[:2]
    return (0 if V is None else V.shape[0], 0 if I is None else I.shape[0])


def mesh(make_pmb, force_block_size, dim, field, N, B, pdef="", **kwds):
    if B is not None:
        force_block_size(B)
    pmb = make_pmb(dim, field.fdef, pdef=pdef, N=N, x0=X0, L=L, **kwds)
    return pmb, pmb(to_numpy=True)


@pytest.mark.parametrize("B", BLOCK_SIZES[3], ids=lambda B: "x".join(map(str, B)))
@pytest.mark.parametrize("N", SIZES)
def test_always_true_predicate_3D(make_pmb, force_block_size, N, B):
    """A predicate that never culls a block must give exactly the same mesh size as no predicate."""
    field = sphere(3, radius=0.37)
    _, ref = mesh(make_pmb, force_block_size, 3, field, N, B)
    _, out = mesh(make_pmb, force_block_size, 3, field, N, B, pdef=predicate_source(3, ALWAYS_TRUE_PREDICATE))
    assert counts(ref)[1] > 0
    assert counts(out) == counts(ref)


@pytest.mark.parametrize("B", BLOCK_SIZES[2], ids=lambda B: "x".join(map(str, B)))
@pytest.mark.parametrize("N", SIZES)
def test_always_true_predicate_2D(make_pmb, force_block_size, N, B):
    field = sphere(2, radius=0.37)
    _, ref = mesh(make_pmb, force_block_size, 2, field, N, B)
    _, out = mesh(make_pmb, force_block_size, 2, field, N, B, pdef=predicate_source(2, ALWAYS_TRUE_PREDICATE))
    assert counts(ref)[1] > 0
    assert counts(out) == counts(ref)


@pytest.mark.parametrize(
    "dim, axis", [(dim, axis) for dim in (2, 3) for axis in range(dim)], ids=lambda v: f"{v}D" if v > 1 else "xyz"[v]
)
def test_sphere_predicate_keeps_all_simplices(make_pmb, force_block_size, dim, axis):
    """
    Conservative interval predicate for a sphere offset along one axis (non symmetric problem):
    the predicate domain axes follow the scalar field axes (domain.a[0] is x), no simplex can be lost.
    """
    center = np.zeros(dim)
    center[axis] = 0.23
    field = sphere(dim, center=tuple(center), radius=0.19)
    pdef = predicate_source(dim, sphere_predicate_body(field.center, field.radius))

    B = BLOCK_SIZES[dim][1]
    _, ref = mesh(make_pmb, force_block_size, dim, field, 95, B)
    _, out = mesh(make_pmb, force_block_size, dim, field, 95, B, pdef=pdef)
    assert counts(ref)[1] > 0
    assert counts(out) == counts(ref)

    # Same welded vertex sets (output order may differ)
    assert_same_points(ref[0], out[0], atol=1e-5)


def test_predicate_with_autotuned_block_size(make_pmb):
    """Without forcing the block size (blocks may differ): same simplices and same welded vertices."""
    field = sphere(3, center=(0.0, 0.0, 0.23), radius=0.19)
    pdef = predicate_source(3, sphere_predicate_body(field.center, field.radius))
    ref = make_pmb(3, field.fdef, N=95, x0=X0, L=L)(to_numpy=True)
    out = make_pmb(3, field.fdef, pdef=pdef, N=95, x0=X0, L=L)(to_numpy=True)
    assert counts(out)[1] == counts(ref)[1] > 0
    assert_same_points(ref[0], out[0], atol=1e-5)


def test_cusvna_affine_arithmetic_predicate(make_pmb):
    """Predicate written with cusvna affine arithmetic (requires the cusvna submodule)."""
    field = sphere(3, center=(0.23, 0.0, 0.0), radius=0.19)
    pdef = predicate_source(
        3,
        includes=("cusvna/af1.cuh",),
        body="""
        using Interval = cusvna::Interval<T>;
        using AF = cusvna::AF1<T,3>;
        const AF x = AF::template from_interval<0>(Interval(domain.a[0], domain.b[0])) - T(0.23);
        const AF y = AF::template from_interval<1>(Interval(domain.a[1], domain.b[1]));
        const AF z = AF::template from_interval<2>(Interval(domain.a[2], domain.b[2]));
        return possibly_zero(x*x + y*y + z*z - T(0.19)*T(0.19));
        """,
    )
    ref = make_pmb(3, field.fdef, N=95, x0=X0, L=L)(to_numpy=True)
    out = make_pmb(3, field.fdef, pdef=pdef, N=95, x0=X0, L=L, enable_cuda_std_lib=True)(to_numpy=True)
    assert counts(out)[1] == counts(ref)[1] > 0
    assert_same_points(ref[0], out[0], atol=1e-5)


ALWAYS_FALSE_SCRIPT = """
import numpy as np, cupy as cp
from cupmb import ParallelMarchingBlocks
from pmb_testing import sphere, predicate_source, ALWAYS_FALSE_PREDICATE

dim = {dim}
field = sphere(dim)
pmb = ParallelMarchingBlocks(dim, field.fdef, pdef=predicate_source(dim, ALWAYS_FALSE_PREDICATE), N=64, x0=-0.5)
V, I = pmb(to_numpy=True)
assert V is None and I is None, (V, I)

# Same empty result on subsequent calls, elapsed time is still returned
res = pmb(to_numpy=True, return_elapsed_time=True)
assert isinstance(res, tuple) and len(res) == 2, res
(V, I), elapsed = res
assert V is None and I is None, (V, I)
assert isinstance(elapsed, float) and elapsed >= 0.0, elapsed

# The CUDA context is not corrupted
cp.cuda.runtime.deviceSynchronize()
V, I = ParallelMarchingBlocks(dim, field.fdef, N=32, x0=-0.5)(to_numpy=True)
assert I is not None and I.shape[0] > 0
print("OK")
"""


@pytest.mark.parametrize("dim", (2, 3), ids=lambda d: f"{d}D")
def test_always_false_predicate(backend, dim):
    """
    All blocks culled: empty result (None buffers), no CUDA error and the CUDA context is still usable afterwards.
    Runs in a subprocess because a CUDA error would poison the context of the whole test session.
    """
    proc = run_python(ALWAYS_FALSE_SCRIPT.format(dim=dim))
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-4000:]


@pytest.mark.parametrize("dim", (2, 3), ids=lambda d: f"{d}D")
def test_predicate_after_construction(make_pmb, dim):
    """Setting pmb.pdef after a first call rebuilds the kernels and keeps the same output size."""
    field = sphere(dim)
    pmb = make_pmb(dim, field.fdef, N=50, x0=X0, L=L)
    ref = pmb(to_numpy=True)
    pmb.pdef = predicate_source(dim, sphere_predicate_body(field.center, field.radius))
    out = pmb(to_numpy=True)
    assert counts(out)[1] == counts(ref)[1] > 0
    assert_same_points(ref[0], out[0], atol=1e-5)
    pmb.pdef = None
    assert counts(pmb(to_numpy=True))[1] == counts(ref)[1]


def test_predicate_is_incompatible_with_scalar_field_export(make_pmb):
    field = sphere(3)
    with pytest.raises(RuntimeError):
        make_pmb(
            3, field.fdef, pdef=predicate_source(3, ALWAYS_TRUE_PREDICATE), N=16, x0=X0, L=L, export_scalar_field=True
        )
