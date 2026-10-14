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
On disk cache of the compiled kernels (CUPMB_CACHE_DIR, CUPMB_DISABLE_KERNEL_CACHE).
"""

import os, uuid

import numpy as np
import pytest

from pmb_testing import as_numpy, run_python, sphere


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CUPMB_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUPMB_DISABLE_KERNEL_CACHE", raising=False)
    from cupmb import kernel_cache_dir

    assert kernel_cache_dir() == os.path.join(str(tmp_path), "kernels")
    return kernel_cache_dir()


def cache_entries(cache_dir):
    if not os.path.isdir(cache_dir):
        return set()
    return {f for f in os.listdir(cache_dir) if f.endswith(".bin")}


@pytest.fixture
def tag():
    # CuPy keeps the modules it already built in memory (the compiled module is then not available
    # for caching), each test uses its own kernel sources to start from a cold compilation
    return uuid.uuid4().hex


def mesh(make_pmb, field, tag, **kwds):
    fdef = f"{field.fdef}\n// {tag}\n"
    pmb = make_pmb(field.dim, fdef, N=32, x0=-0.5, L=1.0, **kwds)
    V, I = pmb()
    return as_numpy(V), as_numpy(I)


def assert_same_mesh(mesh0, mesh1):
    # simplices are generated in a non deterministic order (atomics)
    (V0, I0), (V1, I1) = mesh0, mesh1
    assert I0.shape == I1.shape
    S0, S1 = (V[I].reshape(I.shape[0], -1) for (V, I) in ((V0, I0), (V1, I1)))
    S0, S1 = (S[np.lexsort(S.T[::-1])] for S in (S0, S1))
    np.testing.assert_array_equal(S0, S1)


@pytest.fixture
def forbid_build(monkeypatch, backend):
    """Make any kernel compilation fail (cache hits only)."""
    from cupmb.backend import CudaBackend

    backend_cls = CudaBackend(backend)

    def build_kernels(*args, **kwds):
        raise AssertionError("kernels should have been loaded from the cache")

    def forbid():
        monkeypatch.setattr(backend_cls, "build_kernels", build_kernels)
        monkeypatch.setattr(backend_cls, "build_module_and_extract_kernels_from_source", build_kernels)

    return forbid


@pytest.mark.parametrize("dim", (1, 2, 3))
def test_cache_hit(make_pmb, cache_dir, forbid_build, tag, dim):
    field = sphere(dim)
    mesh0 = mesh(make_pmb, field, tag)
    entries = cache_entries(cache_dir)
    assert entries
    assert all(os.path.isfile(os.path.join(cache_dir, e[:-4] + ".json")) for e in entries)

    forbid_build()
    mesh1 = mesh(make_pmb, field, tag)
    assert cache_entries(cache_dir) == entries
    assert_same_mesh(mesh0, mesh1)


def test_cache_invalidation(make_pmb, cache_dir, tag):
    mesh(make_pmb, sphere(3, radius=0.31), tag)
    entries = cache_entries(cache_dir)
    # different source code, build options and floating point type give new entries
    mesh(make_pmb, sphere(3, radius=0.29), tag)
    entries_source = cache_entries(cache_dir)
    assert entries < entries_source
    mesh(make_pmb, sphere(3, radius=0.29), tag, extra_build_options=("-DCUPMB_TEST_OPTION=1",))
    entries_options = cache_entries(cache_dir)
    assert entries_source < entries_options
    mesh(make_pmb, sphere(3, radius=0.29), tag, ftype=np.float64)
    assert entries_options < cache_entries(cache_dir)


def test_cache_disabled(make_pmb, cache_dir, tag, monkeypatch):
    monkeypatch.setenv("CUPMB_DISABLE_KERNEL_CACHE", "1")
    mesh(make_pmb, sphere(2), tag)
    assert not cache_entries(cache_dir)
    monkeypatch.setenv("CUPMB_DISABLE_KERNEL_CACHE", "0")
    mesh(make_pmb, sphere(2), tag + "_enabled")
    assert cache_entries(cache_dir)


def test_corrupted_entries_are_ignored(make_pmb, cache_dir, tag):
    field = sphere(3)
    mesh0 = mesh(make_pmb, field, tag)
    entries = cache_entries(cache_dir)
    for e in entries:
        with open(os.path.join(cache_dir, e), "wb") as f:
            f.write(b"garbage")
    mesh1 = mesh(make_pmb, field, tag)
    assert_same_mesh(mesh0, mesh1)


def test_clear_kernel_cache(make_pmb, cache_dir, tag):
    from cupmb import clear_kernel_cache

    mesh(make_pmb, sphere(2), tag)
    assert cache_entries(cache_dir)
    clear_kernel_cache()
    assert not os.path.exists(cache_dir)
    clear_kernel_cache()


PYCUDA_SCRIPT = """
from cupmb import ParallelMarchingBlocks, Backend
from pmb_testing import sphere

field = sphere(3)
pmb = ParallelMarchingBlocks(3, field.fdef, N=32, x0=-0.5, backend=Backend.PYCUDA)
V, I = pmb(to_numpy=True)
print("SIMPLICES", I.shape[0])
"""


def test_pycuda_cache(backend, tmp_path):
    """Same checks with the pycuda backend (own process/context)."""
    pytest.importorskip("pycuda")
    env = {"CUPMB_CACHE_DIR": str(tmp_path), "CUPMB_LOGLEVEL": "DEBUG", "CUPMB_DISABLE_KERNEL_CACHE": ""}
    outputs = []
    for _ in range(2):
        proc = run_python(PYCUDA_SCRIPT, env=env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outputs.append(proc.stdout + proc.stderr)
    cache_dir = os.path.join(str(tmp_path), "kernels")
    assert cache_entries(cache_dir)
    assert "Stored kernels in cache" in outputs[0]
    assert "Loaded kernels from cache" not in outputs[0]
    assert "Loaded kernels from cache" in outputs[1]
    assert "Stored kernels in cache" not in outputs[1]
    simplices = [line for out in outputs for line in out.splitlines() if line.startswith("SIMPLICES")]
    assert len(simplices) == 2 and simplices[0] == simplices[1] and simplices[0] != "SIMPLICES 0"
