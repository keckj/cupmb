#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os

module_dir = os.path.dirname(os.path.realpath(os.path.join(__file__, "..")))
cuda_dir = os.path.join(module_dir, "cuda")
cuda_kernels_dir = os.path.join(cuda_dir, "kernels")

if not os.path.exists(module_dir):
    msg = f"Cannot find module directory '{module_dir}'."
    raise RuntimeError(msg)

if not os.path.exists(cuda_dir):
    msg = f"Cannot find cuda directory '{cuda_dir}'."
    raise RuntimeError(msg)
