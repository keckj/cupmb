#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, logging

from colorama import init as init_colorama
from termcolor import colored

init_colorama(autoreset=True)

__version__ = "2021.09.dev0"
""":noindex: The version of this package as a string."""

__version_info__ = ("2021", "09", "dev0")
""":noindex: The version of this package as a tuple of strings."""

__author__ = "Jean-Baptiste Keck"
""":noindex: The author of this package."""


class ColoredFormatter(logging.Formatter):
    colored_levels = {
        "DEBUG": dict(color="white", attrs=["bold"]),
        "VERBOSE": dict(color="blue", attrs=["bold"]),
        "INFO": dict(color="green", attrs=["bold"]),
        "WARNING": dict(color="yellow", attrs=["bold"]),
        "ERROR": dict(color="red", attrs=["bold"]),
        "CRITICAL": dict(color="red", attrs=["bold", "underline"]),
    }

    def __init__(self, fmt, use_color=True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in self.colored_levels:
            suffix = colored(record.getMessage(), **self.colored_levels[levelname])
            record.msg = " "
        else:
            suffix = ""

        return super().format(record) + suffix


class StreamHandler(logging.StreamHandler):
    def emit(self, record):
        print(self.format(record))


FORMAT = " ".join(
    [
        colored("%(asctime)s", "green", attrs=["bold"]),
        colored("%(name)s", "blue", attrs=["bold"]),
        colored("%(levelname)5s", "white", attrs=["bold", "dark"]),
        "%(message)s",
    ]
)

logger = logging.getLogger(__name__)

stream = StreamHandler()
stream.setFormatter(ColoredFormatter(FORMAT))

logger.addHandler(stream)
logger.setLevel(os.environ.get("CUPMB_LOGLEVEL", "DEBUG"))
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
