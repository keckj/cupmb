#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
import os

import cusvna

module_dir = os.path.dirname(os.path.realpath(os.path.join(__file__, "..")))
cuda_kernels_dir = os.path.join(module_dir, "kernels")
utils_dir = os.path.join(module_dir, "utils")
# cusvna headers (interval and affine arithmetic for block predicates) come from the cusvna package
cusvna_include_dir = cusvna.get_include()

if not os.path.exists(module_dir):
    msg = f"Cannot find module directory '{module_dir}'."
    raise RuntimeError(msg)

if not os.path.exists(utils_dir):
    msg = f"Cannot find utils directory '{utils_dir}'."
    raise RuntimeError(msg)

if not os.path.exists(cuda_kernels_dir):
    msg = f"Cannot find cuda directory '{cuda_kernels_dir}'."
    raise RuntimeError(msg)
