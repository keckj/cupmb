#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, logging, functools

from colorama import init as init_colorama
from termcolor import colored

init_colorama()


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


_FORMAT = " ".join(
    [
        colored("%(asctime)s", "green", attrs=["bold"]),
        colored("%(name)s", "blue", attrs=["bold"]),
        colored("%(levelname)5s", "white", attrs=["bold", "dark"]),
        "%(message)s",
    ]
)

logger = logging.getLogger("cupmb")
if not logger.handlers:
    level = os.environ.get("CUPMB_LOGLEVEL", "WARNING")

    stream = logging.StreamHandler()
    stream.setFormatter(ColoredFormatter(_FORMAT))
    stream.setLevel(level)

    logger.propagate = False
    logger.addHandler(stream)
    logger.setLevel(level)


@functools.lru_cache(None)
def log_once(category, msg, logger=logger):
    getattr(logger, category)(msg)
