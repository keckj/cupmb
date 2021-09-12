#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#

__version__ = "2021.09.dev0"
""":noindex: The version of this package as a string."""

__version_info__ = ("2021", "09", "dev0")
""":noindex: The version of this package as a tuple of strings."""

__author__ = "Jean-Baptiste Keck"
""":noindex: The author of this package."""

from cupmb.utils.logger import logger

logger.info(f"Initializing {__name__} version {__version__}.")


def _extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}"


import cupy as cp

int_driver_version = cp.cuda.runtime.driverGetVersion()
int_cuda_version = cp.cuda.runtime.runtimeGetVersion()
driver_version = _extract_cuda_version(int_driver_version)
cuda_version = _extract_cuda_version(int_cuda_version)

if int_driver_version < 11000:
    msg = (
        "You are using an outdated NVIDIA driver, expected a CUDA >= 11.0 compatible driver but found version "
        f"{driver_version} ({int_driver_version}). Please upgrade your NVidia driver to the latest stable "
        "driver supported by your device at https://www.nvidia.fr/Download/index.aspx."
    )
    logger.warning(msg)

if int_cuda_version < 11000:
    msg = (
        "You are using an outdated CUDA version, expected at least CUDA 11.0 but found version "
        f"{cuda_version} ({int_cuda_version}). Please download and install lastest CUDA release from "
        "https://developer.nvidia.com/cuda-downloads and upgrade cupy."
    )
    logger.warning(msg)

msg = (
    f"Loaded cuda runtime {cuda_version} ({int_cuda_version}), "
    f"driver version is {driver_version} ({int_driver_version})."
)
logger.info(msg)

from cupmb.mc import (
    BufferAllocationPolicy,
    KernelAutotuningRigor,
    ParallelMarchingBlocks,
)

__all__ = [
    BufferAllocationPolicy,
    KernelAutotuningRigor,
    ParallelMarchingBlocks,
    logger,
]
