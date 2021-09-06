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
import cupy as cp
import ctypes as ctypes_module

from pmb import logger
from pmb.utils.types import to_tuple
from pmb.utils.IO import cuda_dir, persistent_tmp_dir


class CudaBackend:
    """
    Access to the Cuda compute backend, build kernels, modules and provide an easy
    access to all accessible devices.
    """

    def __new__(cls):
        msg = "CudaBackend is not instantiable." ""
        raise NotImplementedError(msg)

    @staticmethod
    def is_32bit():
        """Flag identifying 32 bit platforms."""
        return ctypes_module.sizeof(ctypes_module.c_void_p) == 4

    @staticmethod
    def is_64bit():
        """Flag identifying 32 bit platforms."""
        return ctypes_module.sizeof(ctypes_module.c_void_p) == 8

    @staticmethod
    def available_device_count():
        """Returns the number of available Cuda devices."""
        cupy_dc = cp.cuda.runtime.getDeviceCount()
        return int(cupy_dc)

    @staticmethod
    def cuda_runtime_version():
        """Returns the Cuda runtime version as an integer."""
        return int(cp.cuda.runtime.runtimeGetVersion())

    @staticmethod
    def cuda_driver_version():
        """Returns the Cuda driver version as an integer."""
        return int(cp.cuda.runtime.driverGetVersion())

    @staticmethod
    def cuda_runtime_versions():
        """Returns the Cuda runtime version as a tuple of integers (minor, major)."""
        v = int(cp.cuda.runtime.runtimeGetVersion())
        return (v // 1000, (v % 1000) // 10)

    @staticmethod
    def cuda_driver_versions():
        """Returns the Cuda driver version as a tuple of integers (minor, major)."""
        v = int(cp.cuda.runtime.driverGetVersion())
        return (v // 1000, (v % 1000) // 10)

    @classmethod
    def build_kernel(
        cls,
        kernel_name,
        code=None,
        filepath=None,
        options=None,
        backend="nvrtc",
        **extra_raw_kernel_kwds,
    ):
        """
        Build a custom Cuda kernel from source.

        This class can be used to define a custom kernel using raw Cuda source of
        a raw Cuda file.

        The kernel is compiled at an invocation of the __call__() method,
        which is cached for each device. The compiled binary is also cached into
        a file under the $HOME/.cupy/kernel_cache/ directory with a hashed file name.

        The cached binary is reused by other processes.

        Parameters
        ----------
        kernel_name
            Name of the kernel function to build, cannot be a template instantiation.
        code
            CUDA source code as a string.
        filepath
            Path to a file containing CUDA source, mutually exclusive with `code`.
        options
            Compiler options passed to the backend (NVRTC or NVCC).
            If None, defaults to device default build options.
        backend:
            Either nvrtc or nvcc. Defaults to nvrtc.
        extra_raw_kernel_kwds:
            Extra keyword parameters passed to cupy.RawKernel.
        """
        if code is None:
            if not os.path.isfile(filepath):
                msg = "Cuda source file '{}' does not exist."
                raise RuntimeError(msg.format(filepath))
            with open(filepath) as f:
                code = f.read()
        else:
            assert filepath is None, filepath

        if options is None:
            options = cls.default_build_options()

        try:
            kernel = cp.RawKernel(
                code=code,
                name=kernel_name,
                options=options,
                backend=backend,
                **extra_raw_kernel_kwds,
            )
        except:
            fname = os.path.join(persistent_tmp_dir, "failed.cu")
            msg = f"Failed to build kernel {kernel_name}; source has been dumped to file://{fname}."
            logger.error(msg)
            with open(fname, "w") as f:
                f.write(code)
            raise

        return kernel

    @classmethod
    def build_module(
        cls,
        kernel_names,
        code=None,
        filepath=None,
        options=None,
        backend="nvrtc",
        extract_kernels=False,
        **extra_raw_module_kwds,
    ):
        r"""
        Build a custom Cuda module from source.

        This class can be used to define a custom module using raw Cuda source of
        a raw Cuda file (\*.cu, \*.cuh) or cuda modules (\*.cubin, \*.ptx).

        For the former case, the Cuda source code is compiled when any method is called.
        For the latter case, an existing CUDA binary (\*.cubin) or a PTX file can be loaded by providing its path.

        Cuda kernels in a cupy.RawModule can be retrieved by calling get_function(),
        which will return an instance of cupy.RawKernel.

        Parameters
        ----------
        kernel_names
            Name of the kernels function to build, can contain explicit template instantiations.
        code
            CUDA source code as a string, mutually exclusive with filepath.
        filepath
            Path to cuda source or cuda module.
            Filenames ending in \*.cubin and \*.ptx are treated as modules,
            else the file is considered to contain raw cuda source code.
        options
            Compiler options passed to the backend (NVRTC or NVCC).
            If None, defaults to device default build options.
        backend:
            Either nvrtc or nvcc. Defaults to nvrtc.
        extract_kernels:
            Return directly a tuple of cupy.RawKernels instead of a cupy.RawModule.
        extra_raw_module_kwds:
            Extra keyword parameters passed to cupy.RawModule.
        """
        kernel_names = to_tuple(kernel_names)

        if filepath is None:
            assert code is not None, code
        elif filepath.endswith(".ptx") or filepath.endswith(".cubin"):
            pass
        else:
            if not os.path.isfile(filepath):
                msg = "Cuda source file '{}' does not exist."
                raise RuntimeError(msg.format(filepath))
            with open(filepath) as f:
                code = f.read()
            filepath = None
        assert (filepath is None) ^ (code is None), (filepath, len(code))

        if options is None:
            options = cls.default_build_options()

        try:
            module = cp.RawModule(
                code=code,
                path=filepath,
                name_expressions=kernel_names,
                options=options,
                backend=backend,
                **extra_raw_module_kwds,
            )
            kernels = tuple(map(module.get_function, kernel_names))
        except:
            fname = os.path.join(persistent_tmp_dir, "failed.cu")
            knames = ", ".join(kernel_names)
            msg = f"Failed to build module kernels {knames}; source has been dumped to file://{fname}."
            logger.error(msg)
            with open(fname, "w") as f:
                f.write(code)
            raise

        if extract_kernels:
            return kernels
        else:
            return module

    @classmethod
    def defines_as_options(cls, defines):
        """
        Convert a dictionnary of compiler defines -D{key}={value} to a tuple of option strings
        that can be passed to build_kernel or build_module. Values can also be empty or None.
        """
        options = tuple()
        if defines is None:
            return options
        for k in sorted(defines):
            v = defines[k]
            init = f"={v}" if v else ""
            options += (f"-D{k}{init}",)
        return options

    @classmethod
    def default_build_options(cls):
        """Return platform default build options."""
        return (
            "-std=c++17",
            f"-I{cuda_dir}",
            "--builtin-move-forward=true",
            "--builtin-initializer-list=true",
        )
