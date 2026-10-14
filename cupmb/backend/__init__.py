#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
import os

from cupmb.enums import Backend


def find_cuda_include_dirs(logger, extra_candidates=()):
    """
    Determine the CUDA toolkit include directories (used to build kernels against the CUDA standard library).

    The CUDA root is determined from the CUPMB_CUDA_ROOT environment variable if it is set. Otherwise, the
    first existing directory among extra_candidates (backend specific), CUDA_PATH, CUDA_HOME and /usr/local/cuda
    is used. For CUDA >= 13 toolkits, the include/cccl directory is returned as well when it exists.
    """
    if "CUPMB_CUDA_ROOT" in os.environ:
        cuda_root = os.environ["CUPMB_CUDA_ROOT"].strip()
        if not os.path.isdir(cuda_root):
            msg = f"CUPMB_CUDA_ROOT has been set to '{cuda_root}' but this directory does not exist."
            logger.error(msg)
            raise RuntimeError(msg)
    else:
        candidates = tuple(extra_candidates) + (
            ("CUDA_PATH", os.environ.get("CUDA_PATH")),
            ("CUDA_HOME", os.environ.get("CUDA_HOME")),
            ("default CUDA root", "/usr/local/cuda"),
        )
        cuda_root = None
        for source, candidate in candidates:
            if not candidate:
                continue
            candidate = candidate.strip()
            if os.path.isdir(os.path.join(candidate, "include")):
                logger.debug(f"Using CUDA root '{candidate}' from {source}.")
                cuda_root = candidate
                break
            logger.debug(f"Ignoring CUDA root '{candidate}' from {source} (no include directory).")
        if cuda_root is None:
            msg = (
                "Could not determine the CUDA toolkit root directory, please set the CUPMB_CUDA_ROOT "
                "(or CUDA_PATH) environment variable to your CUDA toolkit root directory."
            )
            logger.error(msg)
            raise RuntimeError(msg)

    cuda_include_dir = os.path.join(cuda_root, "include")
    if not os.path.isdir(cuda_include_dir):
        msg = (
            f"Could not find CUDA include directory '{cuda_include_dir}', please set the CUPMB_CUDA_ROOT "
            "environment variable to your CUDA toolkit root directory."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    include_dirs = [cuda_include_dir]
    cccl_include_dir = os.path.join(cuda_include_dir, "cccl")
    if os.path.isdir(cccl_include_dir):
        # CUDA >= 13 toolkits ship libcu++/CUB/Thrust headers in include/cccl
        include_dirs.append(cccl_include_dir)

    return tuple(map(os.path.realpath, include_dirs))


class CudaBackend:
    def __new__(cls, backend, initialize_context=True):
        if backend is Backend.CUPY:
            from cupmb.backend.cupy_backend import CupyBackend

            return CupyBackend
        elif backend is Backend.PYCUDA:
            from cupmb.backend.pycuda_backend import PyCudaBackend

            if initialize_context:
                import pycuda.autoinit  # noqa: F401
            return PyCudaBackend
        else:
            raise NotImplementedError(backend)

    @classmethod
    def determine_cuda_include_dirs(cls, logger):
        return find_cuda_include_dirs(logger)

    @classmethod
    def determine_cuda_include_dir(cls, logger):
        return cls.determine_cuda_include_dirs(logger)[0]
