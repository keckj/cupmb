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
Repeated calls without intermediate device to host copies: once the buffers are allocated, kernels read the
active block, vertex and simplex counts from device counters, grids are sized from the buffer capacities and
the counts are read back once. When a buffer is too small, kernels are launched again step by step.
"""

import collections

import numpy as np
import pytest

from pmb_testing import AXES, as_numpy, run_python, scalar_source

from cupmb import ExtraParameter, VertexProperty

X0, L = -0.5, 1.0
CENTER = (0.041, -0.027, 0.013)
RESOLUTION = {1: 4099, 2: 67, 3: 33}
BLOCK_SIZE = {1: (32,), 2: (8, 8), 3: (4, 4, 4)}

# Concentric level sets cos(k*|p - c|) = 0, their number grows with k (no zero for k=1): buffers are
# reallocated when k grows and reused when k shrinks, the empty level set is also tested.
FREQUENCIES = (6.0, 12.0, 1.0, 9.0, 20.0, 4.0, 20.0, 14.0, 1.0, 25.0)


def field_source(dim):
    r2 = " + ".join(f"({a} - T({c!r}))*({a} - T({c!r}))" for (a, c) in zip(AXES, CENTER[:dim]))
    return scalar_source(dim, f"cos(k*sqrt({r2}))", extra_args=", const T& k")


def predicate_source(dim):
    """Conservative interval predicate: the block may contain a zero if k*[rmin, rmax] contains (n+1/2)*pi."""
    lines = []
    for i, c in enumerate(CENTER[:dim]):
        lines.append(
            f"{{ const T a = domain.a[{i}] - T({c!r}), b = domain.b[{i}] - T({c!r}); const T aa = a*a, bb = b*b; "
            "lo += ((a <= T(0)) && (b >= T(0))) ? T(0) : (aa < bb ? aa : bb); hi += (aa > bb ? aa : bb); }"
        )
    body = "\n        ".join(lines)
    return rf"""
namespace cupmb {{
    template <typename T, typename I, I dim, typename... F>
    __device__ __forceinline__ bool may_contain_zeros(const BoundingBox<T,dim>& domain, const T& k) {{
        T lo = T(0), hi = T(0);
        {body}
        const T inv_pi = T(0.3183098861837907), eps = T(1e-3);
        const T a = k*sqrt(lo)*inv_pi - T(0.5) - eps;
        const T b = k*sqrt(hi)*inv_pi - T(0.5) + eps;
        return floor(b) >= ceil(a);
    }}
}}
"""


def phase_source(dim):
    """Vertex property k*|p - c| (uses the extra parameter)."""
    r2 = " + ".join(
        f"(trove::get<{i}>(p) - T({c!r}))*(trove::get<{i}>(p) - T({c!r}))" for (i, c) in enumerate(CENTER[:dim])
    )
    return rf"""
namespace cupmb {{
    template <typename T, typename I, I dim>
    __device__ __forceinline__ trove::array<T,1> phase(const trove::array<T,dim>& p, const T& k) {{
        return trove::make_array<T>(k*sqrt({r2}));
    }}
}}
"""


def make(make_pmb, dim, with_predicate, with_properties, **kwds):
    fdef = field_source(dim)
    if with_properties:
        fdef += phase_source(dim)
        kwds["vertex_properties"] = (VertexProperty("phase", fname="phase", dtype="T", shape=None),)
    return make_pmb(
        dim,
        fdef,
        pdef=(predicate_source(dim) if with_predicate else ""),
        N=RESOLUTION[dim],
        x0=X0,
        L=L,
        extra_params=collections.OrderedDict(k=ExtraParameter("T")),
        **kwds,
    )


def canonical(out):
    """Simplices as rows of vertex coordinates (and vertex properties), sorted (simplex order is not deterministic)."""
    V, I, *P = (as_numpy(x) for x in out)
    if I is None:
        assert V is None and all(p is None for p in P)
        return None
    VP = np.hstack([V] + [p.reshape(V.shape[0], -1) for p in P])
    S = VP[I].reshape(I.shape[0], -1)
    return S[np.lexsort(S.T[::-1])]


def assert_same_output(out, ref):
    S, Sref = canonical(out), canonical(ref)
    if Sref is None:
        assert S is None
    else:
        assert S is not None and S.shape == Sref.shape
        np.testing.assert_array_equal(S, Sref)


@pytest.fixture
def count_readbacks(monkeypatch, backend):
    """Count the device to host copies of the backend."""
    from cupmb.backend import CudaBackend

    backend_cls = CudaBackend(backend)
    asnumpy = backend_cls.asnumpy
    calls = []

    def counting_asnumpy(cls, a, stream=None):
        calls.append(a.size)
        return asnumpy(a, stream)

    monkeypatch.setattr(backend_cls, "asnumpy", classmethod(counting_asnumpy))
    return calls


@pytest.mark.parametrize("with_properties", (False, True), ids=("", "properties"))
@pytest.mark.parametrize("with_predicate", (False, True), ids=("mask", "predicate"))
@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_repeated_calls_match_fresh_instances(make_pmb, force_block_size, dim, with_predicate, with_properties):
    """Each call of a reused instance gives the same output as a new instance (buffers grow, shrink, empty)."""
    force_block_size(BLOCK_SIZE[dim])
    pmb = make(make_pmb, dim, with_predicate, with_properties)
    nonempty = 0
    for k in FREQUENCIES:
        out = pmb(k=k)
        ref = make(make_pmb, dim, with_predicate, with_properties)(k=k)
        assert_same_output(out, ref)
        nonempty += ref[1] is not None
    assert 0 < nonempty < len(FREQUENCIES)


@pytest.mark.parametrize("with_predicate", (False, True), ids=("mask", "predicate"))
@pytest.mark.parametrize("dim", (1, 2, 3), ids=lambda d: f"{d}D")
def test_single_readback_in_steady_state(make_pmb, force_block_size, count_readbacks, dim, with_predicate):
    """Once the buffers are large enough, counts are copied back to the host once per call."""
    force_block_size(BLOCK_SIZE[dim])
    pmb = make(make_pmb, dim, with_predicate, with_properties=True)

    pmb(k=20.0)
    assert len(count_readbacks) > 1  # first call: buffers are allocated step by step

    for k in (20.0, 6.0, 1.0, 14.0):
        count_readbacks.clear()
        pmb(k=k)
        assert len(count_readbacks) == 1, (k, count_readbacks)

    # buffers are too small: fall back to step by step launches
    count_readbacks.clear()
    pmb(k=60.0)
    assert len(count_readbacks) > 1

    count_readbacks.clear()
    V, I, _ = pmb(k=60.0, to_numpy=True)
    assert len(count_readbacks) == 4  # counts, vertices, simplices and vertex property


PYCUDA_SCRIPT = """
import collections, sys
import numpy as np
from cupmb import ParallelMarchingBlocks, Backend, ExtraParameter
from test_device_driven import FREQUENCIES, field_source, predicate_source, canonical

def make(dim, with_predicate):
    return ParallelMarchingBlocks(dim, field_source(dim), pdef=(predicate_source(dim) if with_predicate else ""),
        N=33, x0=-0.5, extra_params=collections.OrderedDict(k=ExtraParameter("T")), backend=Backend.PYCUDA)

for dim in (2, 3):
    for with_predicate in (False, True):
        pmb = make(dim, with_predicate)
        for k in FREQUENCIES:
            S, Sref = canonical(pmb(k=k, to_numpy=True)), canonical(make(dim, with_predicate)(k=k, to_numpy=True))
            assert (S is None) == (Sref is None), (dim, with_predicate, k)
            if Sref is not None:
                np.testing.assert_array_equal(S, Sref)
print("OK")
"""


def test_repeated_calls_pycuda(backend):
    pytest.importorskip("pycuda")
    proc = run_python(PYCUDA_SCRIPT, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("OK")
