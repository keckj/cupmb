#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
"""
Public API, packaging and import time behaviour (no GPU required).
"""

import os, re

import pytest

from pmb_testing import run_python

import cupmb


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", cupmb.__version__), cupmb.__version__
    assert isinstance(cupmb.__version_info__, tuple)
    assert ".".join(cupmb.__version_info__) == cupmb.__version__


def test_all_contains_names():
    assert all(isinstance(name, str) for name in cupmb.__all__)
    for name in cupmb.__all__:
        assert hasattr(cupmb, name), name
    for name in ("ParallelMarchingBlocks", "Backend", "AllocationPolicy", "AutotuningRigor", "ExtraParameter"):
        assert name in cupmb.__all__


def test_star_import():
    namespace = {}
    exec("from cupmb import *", namespace)
    assert "ParallelMarchingBlocks" in namespace
    assert "Backend" in namespace


def test_kernel_headers_are_available():
    """Device sources and the trove/cusvna submodule headers are shipped with the package."""
    kernels = os.path.join(os.path.dirname(cupmb.__file__), "kernels")
    for name in ("pmb.cuh", "pmb1D.cuh", "pmb2D.cuh", "pmb3D.cuh", "trove/array.h", "cusvna/af1.cuh"):
        assert os.path.isfile(os.path.join(kernels, name)), name


@pytest.mark.parametrize("level", ("debug", "DEBUG", "info", "Warning", "10"))
def test_loglevel_environment_variable(level):
    proc = run_python("import cupmb; print(cupmb.logger.getEffectiveLevel())", env={"CUPMB_LOGLEVEL": level})
    assert proc.returncode == 0, proc.stderr[-4000:]
    expected = {"debug": 10, "info": 20, "warning": 30, "10": 10}[level.lower()]
    assert int(proc.stdout.strip().splitlines()[-1]) == expected


def test_import_does_not_initialize_cuda():
    """Importing cupmb must not require a GPU, cupy or pycuda."""
    proc = run_python("import sys, cupmb; print('cupy' in sys.modules, 'pycuda' in sys.modules)")
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.split()[-2:] == ["False", "False"]
