#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#

__version__ = "2026.10.dev0"
""":noindex: The version of this package as a string (single source of truth for packaging)."""

__version_info__ = tuple(__version__.split("."))
""":noindex: The version of this package as a tuple of strings."""

__author__ = "Jean-Baptiste Keck"
""":noindex: The author of this package."""

from cupmb.utils.logger import logger, log_once

log_once("info", f"Initializing {__name__} version {__version__}.", logger)

from cupmb.enums import (
    Backend,
    BufferType,
    AllocationPolicy,
    AutotuningRigor,
)

from cupmb.bbox import BoundingBox
from cupmb.parameters import ExtraParameter
from cupmb.properties import VertexProperty

from cupmb.core import (
    ParallelMarchingBlocks,
)
from cupmb.utils.kernel_cache import kernel_cache_dir, clear_kernel_cache

__all__ = [
    "Backend",
    "BufferType",
    "AllocationPolicy",
    "AutotuningRigor",
    "BoundingBox",
    "ExtraParameter",
    "VertexProperty",
    "ParallelMarchingBlocks",
    "kernel_cache_dir",
    "clear_kernel_cache",
]
