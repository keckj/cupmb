#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, tempfile

from pmb import __name__, __author__, __version__
from appdirs import AppDirs

project_dir: str = os.path.dirname(os.path.realpath(os.path.join(__file__, "..", "..")))

module_dir: str = os.path.join(project_dir, "pmb")

cuda_dir: str = os.path.join(module_dir, "cuda")
cuda_utils_dir: str = os.path.join(cuda_dir, "utils")
cuda_kernels_dir: str = os.path.join(cuda_dir, "kernels")

appdirs: AppDirs = AppDirs(appname=__name__, appauthor=__author__, version=__version__)
app_log_dir: str = appdirs.user_log_dir
app_data_dir: str = appdirs.user_data_dir
app_config_dir: str = appdirs.user_config_dir
app_cache_dir: str = appdirs.user_cache_dir

persistent_tmp_dir: str = tempfile.gettempdir()

if not os.path.exists(project_dir):
    msg = f"Cannot find project directory '{project_dir}'."
    raise RuntimeError(msg)

if not os.path.exists(cuda_dir):
    msg = f"Cannot find cuda directory '{cuda_dir}'."
    raise RuntimeError(msg)


# runtime_tmp_dir: str = tempfile.mkdtemp(prefix=__name__)
# logger.debug(f"Creating runtime temporary directory '{runtime_tmp_dir}'.")
# def _remove_runtime_tmp_dir():
# logger.debug(f"Removing runtime temporary directory '{runtime_tmp_dir}'.")
# if os.path.isdir(runtime_tmp_dir):
# shutil.rmtree(runtime_tmp_dir)
# atexit.register(_remove_runtime_tmp_dir)
