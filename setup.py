#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
# All metadata lives in pyproject.toml. This shim only fails early with a helpful message when the
# git submodules providing the device headers (trove, cusvna) have not been initialized: in a git
# checkout cupmb/kernels/{trove,cusvna} are symlinks to ext/*, in an sdist they are regular directories.
import os, glob, warnings

from setuptools import setup

kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cupmb", "kernels")

if not glob.glob(os.path.join(kernel_dir, "trove", "*.h")):
    raise RuntimeError(
        "Trove dependency is missing (cupmb/kernels/trove is empty), please initialize git submodules:"
        f"{os.linesep}  git submodule update --init --recursive"
    )

if not glob.glob(os.path.join(kernel_dir, "cusvna", "*.cuh")) and not glob.glob(
    os.path.join(kernel_dir, "cusvna", "*.h")
):
    warnings.warn(
        "Optional cusvna dependency is missing (cupmb/kernels/cusvna is empty), predicates based on "
        "interval/affine arithmetic will not be available. To fix this, initialize git submodules:"
        f"{os.linesep}  git submodule update --init --recursive"
    )

setup()
