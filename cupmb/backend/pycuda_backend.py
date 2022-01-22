#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, sys, tempfile, subprocess

from cupmb.backend import CudaBackend
from cupmb.utils.logger import logger
from cupmb.utils.IO import utils_dir

logger.info("Importing PyCuda backend.")

import pycuda
import pycuda.driver
import pycuda.gpuarray
import pycuda.compiler

import numpy as np


def _extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}.{v%10}"


def _extract_int_version(v):
    return 1000 * v[0] + 10 * v[1] + v[2]


# Check PyCuda driver and cuda version
int_driver_version = pycuda.driver.get_driver_version()
driver_version = _extract_cuda_version(int_driver_version)

cuda_version = pycuda.driver.get_version()
int_cuda_version = _extract_int_version(cuda_version)
cuda_version = ".".join(map(str, cuda_version))

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
        "https://developer.nvidia.com/cuda-downloads and rebuild PyCuda."
    )
    logger.warning(msg)

msg = (
    f"PyCuda loaded cuda runtime {cuda_version} ({int_cuda_version}), "
    f"driver version is {driver_version} ({int_driver_version})."
)
logger.info(msg)


class PyCudaDevice:
    da = pycuda.driver.device_attribute

    def __init__(self, device_id):
        self._pycuda_device = pycuda.driver.Device(device_id)
        self._device_attrs = self._pycuda_device.get_attributes()
        self._device_id = device_id

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def handle(self):
        return self._pycuda_device

    @property
    def id(self):
        return self._device_id

    @property
    def max_grid_dim(self):
        return (self.max_grid_dim_x, self.max_grid_dim_y, self.max_grid_dim_z)

    @property
    def max_block_dim(self):
        return (self.max_block_dim_x, self.max_block_dim_y, self.max_block_dim_z)

    @property
    def major(self):
        return self._pycuda_device.compute_capability()[0]

    @property
    def minor(self):
        return self._pycuda_device.compute_capability()[1]

    @property
    def name(self):
        return self._pycuda_device.name()

    @property
    def pci_bus_id(self):
        return self._pycuda_device.pci_bus_id()

    @property
    def warp_size(self):
        return self._device_attrs[self.da.WARP_SIZE]

    @property
    def max_threads_per_block(self):
        return self._device_attrs[self.da.MAX_THREADS_PER_BLOCK]

    @property
    def max_block_dim_x(self):
        return self._device_attrs[self.da.MAX_BLOCK_DIM_X]

    @property
    def max_block_dim_y(self):
        return self._device_attrs[self.da.MAX_BLOCK_DIM_Y]

    @property
    def max_block_dim_z(self):
        return self._device_attrs[self.da.MAX_BLOCK_DIM_Z]

    @property
    def max_grid_dim_x(self):
        return self._device_attrs[self.da.MAX_GRID_DIM_X]

    @property
    def max_grid_dim_y(self):
        return self._device_attrs[self.da.MAX_GRID_DIM_Y]

    @property
    def max_grid_dim_z(self):
        return self._device_attrs[self.da.MAX_GRID_DIM_Z]

    @property
    def max_shared_memory_per_block(self):
        return self._device_attrs[self.da.MAX_SHARED_MEMORY_PER_BLOCK]

    @property
    def max_shared_memory_per_block_optin(self):
        try:
            return self._device_attrs[self.da.MAX_SHARED_MEMORY_PER_BLOCK]
        except AttributeError:
            return self.max_shared_memory_per_block

    @property
    def free_memory_bytes(self):
        return pycuda.driver.mem_get_info()[0]

    @property
    def global_memory_bytes(self):
        return pycuda.driver.mem_get_info()[1]


class PyCudaKernel:
    def __init__(self, handle):
        assert isinstance(handle, pycuda.driver.Function), type(handle)
        self._pycuda_kernel = handle

    @property
    def handle(self):
        return self._pycuda_kernel

    @property
    def max_threads_per_block(self):
        return self._pycuda_kernel.max_threads_per_block

    def __call__(
        self, grid, block, args, stream=None, shared_mem=None, time_kernel=False
    ):
        assert isinstance(grid, tuple)
        assert isinstance(block, tuple)
        assert len(grid) <= 3, len(grid)
        assert len(block) <= 3, len(block)
        grid = tuple(map(int, grid)) + (1,) * (3 - len(grid))
        block = tuple(map(int, block)) + (1,) * (3 - len(block))
        shared_mem = int(shared_mem) if (shared_mem is not None) else 0
        if time_kernel:
            return self._pycuda_kernel(
                *args,
                block=block,
                grid=grid,
                shared=shared_mem,
                stream=None,
                time_kernel=True,
            )
        else:
            self._pycuda_kernel(
                *args,
                grid=grid,
                block=block,
                shared=shared_mem,
                stream=stream,
                time_kernel=False,
            )


class PointerHolderBase(pycuda.driver.PointerHolderBase):
    def __init__(self, int_ptr, size):
        super().__init__()
        self._int_ptr = int_ptr
        self._size = size

    @property
    def int_ptr(self):
        return self._int_ptr

    @property
    def size(self):
        return self._size

    def get_pointer(self):
        return self._int_ptr


class CudaOpenGLMappedBuffer(pycuda.gl.RegisteredBuffer):
    def __init__(self, gl_buffer, flags=0):
        super().__init__(int(gl_buffer), flags)
        self._registered_mapping = None
        self._registered = True

    @property
    def gl_buffer(self):
        return self.gl_handle()

    @property
    def cuda_buffer(self):
        assert self.mapped
        ptr, size = self._registered_mapping.device_ptr_and_size()
        return PointerHolderBase(ptr, size)

    @property
    def registered(self):
        return self._registered

    @property
    def mapped(self):
        return self._registered_mapping is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        assert self._registered
        return self

    def unregister(self):
        if self._registered:
            super().unregister()
            self._registered = False
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")

        if not self.mapped:
            self._registered_mapping = super().map(stream)

        return self.cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")

        if self.mapped:
            self._registered_mapping.unmap(stream)
            self._registered_mapping = None

        return self


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order="C"):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped

        def allocator(nbytes, buf=self._cuda_buffer):
            assert nbytes == buf.size
            return buf.get_pointer()

        return pycuda.gpuarray.GPUArray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            allocator=allocator,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array


class PyCudaBackend(CudaBackend):
    ArrayBackend = pycuda.gpuarray
    Array = pycuda.gpuarray.GPUArray
    Buffer = PointerHolderBase
    Stream = pycuda.driver.Stream
    Device = PyCudaDevice
    Kernel = PyCudaKernel
    AllocationError = pycuda.driver.MemoryError

    CudaOpenGLMappedArray = CudaOpenGLMappedArray
    CudaOpenGLMappedBuffer = CudaOpenGLMappedBuffer

    class NullPtr:
        gpudata = 0x0

    @classmethod
    def default_stream(cls):
        return pycuda.driver.Stream()

    @classmethod
    def asnumpy(cls, a):
        return a.get()

    @classmethod
    def asarray(cls, a):
        return pycuda.gpuarray.to_gpu(a)

    @classmethod
    def get_nbytes(cls, a):
        if a is None:
            return 0
        elif isinstance(a, cls.Array):
            return a.nbytes
        elif isinstance(a, cls.Buffer):
            return a.size
        else:
            raise NotImplementedError(type(a))

    @classmethod
    def synchronize_stream(cls, stream):
        stream.synchronize()

    @classmethod
    def get_data_pointers(cls, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return tuple(cls.NullPtr if (a is None) else a for a in args)

    @classmethod
    def array_from_pointer(cls, ptr, shape, dtype):
        assert isinstance(ptr, cls.Buffer)
        req_bytes = np.prod(shape, dtype=np.float64) * dtype(0).itemsize
        ptr_bytes = cls.get_nbytes(ptr)
        assert req_bytes == ptr_bytes, (req_bytes, ptr_bytes)
        return cls.Array(shape=shape, dtype=dtype, gpudata=ptr)

    @classmethod
    def fill_zeros(cls, a, stream):
        assert isinstance(a, cls.Array), type(a)
        a.fill(0, stream=stream)

    @classmethod
    def determine_cuda_include_dir(cls, logger):
        if "CUPMB_CUDA_ROOT" in os.environ:
            cuda_root = os.environ["CUPMB_CUDA_ROOT"].strip()
            if not os.path.isdir(cuda_root):
                msg = f"CUPMB_CUDA_ROOT has been set to '{cuda_root}' but this directory does not exist."
                logger.error(msg)
                raise RuntimeError(msg)
            cuda_include_dir = os.path.join(cuda_root, "include")
        else:
            (cuda_libdir, _) = pycuda.compiler.DynamicModule()._locate_cuda_libdir()
            if not os.path.isdir(cuda_libdir):
                msg = f"Cupy reports cuda libdir '{cuda_libdir}' but this directory does not exist."
                logger.error(msg)
                raise RuntimeError(msg)
            cuda_include_dir = os.path.join(cuda_libdir, "..", "include")

        if not os.path.isdir(cuda_include_dir):
            msg = (
                f"Could not find cuda include directory '{cuda_include_dir}', please "
                "set the CUDA_ROOT environment variable to your cuda root directory "
                "or rebuild pycuda by providing a correct CUDA_ROOT during configuration."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return os.path.realpath(cuda_include_dir)

    @classmethod
    def build_module_and_extract_kernels_from_source(
        cls,
        logger,
        device_major,
        device_minor,
        source_code,
        kernel_names,
        build_options,
    ):
        # Here we use the nvrtc module because pycuda cannot handle templated __global__ functions
        # due to name mangling (see https://wiki.tiker.net/PyCuda/Examples/C++FunctionTemplates).
        # We use a subprocess to generate a ptx or cubin file that we then load back with pycuda
        # to avoid any eventual cuda context clash due to the use of official nvidia cuda python
        # modules. Please note that the subprocess also generates the list of mangled kernel names.
        # We use temporary files to communicate build options and kernel names to the subprocess,
        # because it is easier than via command line arguments. The same technique is used to get
        # back kernel ptx/cubin binary data and mangled kernel names back from the subprocess.

        nvrtc_build_program = os.path.join(utils_dir, "nvrtc_build.py")
        if not os.path.isfile(nvrtc_build_program):
            msg = "nvrtc build program file:///{nvrtc_build_program} is missing."
            logger.error(msg)
            raise RuntimeError(msg)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_knames = os.path.join(tmpdir, "input_knames")
            input_options = os.path.join(tmpdir, "input_options")
            output_knames = os.path.join(tmpdir, "output_knames")
            output_binary = os.path.join(tmpdir, "output_binary")
            source_file = os.path.join(tmpdir, "source.cu")

            with open(input_options, "w") as f:
                for opt in build_options:
                    f.write(opt)
                    f.write(os.linesep)

            with open(input_knames, "w") as f:
                for kname in kernel_names:
                    f.write(kname)
                    f.write(os.linesep)

            with open(source_file, "w") as f:
                f.write(source_code)

            args = [
                sys.executable,
                nvrtc_build_program,
                "--major",
                str(device_major),
                "--minor",
                str(device_minor),
                "--source-file",
                source_file,
                "--input-knames",
                input_knames,
                "--input-options",
                input_options,
                "--output-knames",
                output_knames,
                "--output-binary",
                output_binary,
            ]

            try:
                subprocess.run(args, check=True)
            except Exception:
                cmd = " ".join(args)
                logger.error(cmd)
                logger.error("Failed to build kernels by using nvrtc_build.py")
                raise

            if not os.path.isfile(output_knames):
                msg = f"Source file:///{output_knames} has not been generated."
                logger.error(msg)
                raise RuntimeError(msg)

            mangled_kernel_names = tuple(
                filter(len, open(output_knames).read().split(os.linesep))
            )

            if not os.path.isfile(output_binary):
                msg = f"Source file:///{output_binary} has not been generated."
                raise RuntimeError(msg)

            with open(output_binary, "rb") as f:
                module = pycuda.driver.module_from_buffer(f.read())

            kernels = tuple(map(module.get_function, mangled_kernel_names))

            return tuple(map(cls.Kernel, kernels))

        msg = "Failed to build kernels by using nvrtc_build.py"
        logger.error(msg)
        raise RuntimeError(msg)
