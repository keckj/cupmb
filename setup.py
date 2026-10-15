#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#

# All metadata lives in pyproject.toml. This shim only fails early with a helpful message when the
# git submodule providing the trove device headers has not been initialized: in a git checkout
# cupmb/kernels/trove is a symlink to ext/trove, in an sdist it is a regular directory.
import os, glob

from setuptools import setup

kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cupmb", "kernels")

if not glob.glob(os.path.join(kernel_dir, "trove", "*.h")):
    raise RuntimeError(
        "Trove dependency is missing (cupmb/kernels/trove is empty), please initialize git submodules:"
        f"{os.linesep}  git submodule update --init --recursive"
    )

setup()
