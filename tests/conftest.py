#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import functools

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="also run tests marked as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@functools.lru_cache(None)
def _gpu_status():
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover - depends on the environment
        return f"cupy is not available ({e})"
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as e:  # pragma: no cover - depends on the environment
        return f"no usable CUDA driver ({e})"
    if count == 0:  # pragma: no cover - depends on the environment
        return "no CUDA device found"
    return None


@pytest.fixture(scope="session")
def backend():
    """Cupy backend, skips the test when cupy or a CUDA device is not available."""
    status = _gpu_status()
    if status is not None:
        pytest.skip(status)
    from cupmb import Backend

    return Backend.CUPY


@pytest.fixture
def make_pmb(backend):
    """Factory for ParallelMarchingBlocks instances using the session backend."""
    from cupmb import ParallelMarchingBlocks

    def factory(dim, fdef, **kwds):
        kwds.setdefault("backend", backend)
        return ParallelMarchingBlocks(dim, fdef, **kwds)

    return factory


@pytest.fixture
def force_block_size(monkeypatch):
    """
    Force the cuda block size used by the marching blocks kernels (a single autotuning candidate),
    usage: force_block_size((8, 8, 8)) before the first call to the ParallelMarchingBlocks instance.
    """
    from cupmb import ParallelMarchingBlocks

    def force(B):
        B = tuple(int(b) for b in B)

        def generate_candidate_block_size(self, device):
            assert len(B) == self.dim, (B, self.dim)
            return (np.asarray(B, dtype=self.itype),)

        monkeypatch.setattr(ParallelMarchingBlocks, "_generate_candidate_block_size", generate_candidate_block_size)

    return force
