#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os

from cupmb.backend import CudaBackend
from cupmb.utils.logger import logger

logger.info("Importing CuPy backend.")

import cupy as cp
import numpy as np

from cuda import cudart


def _extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}"


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


def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CupyDevice:
    def __init__(self, device_id):
        self._cupy_device = cp.cuda.Device(device_id)
        self._device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        self._device_attrs = self._cupy_device.attributes

    def __enter__(self):
        self._cupy_device.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cupy_device.__exit__()

    @property
    def handle(self):
        return self._cupy_device

    @property
    def id(self):
        return self._cupy_device.id

    @property
    def max_grid_dim(self):
        return (self.max_grid_dim_x, self.max_grid_dim_y, self.max_grid_dim_z)

    @property
    def max_block_dim(self):
        return (self.max_block_dim_x, self.max_block_dim_y, self.max_block_dim_z)

    @property
    def major(self):
        return self._device_props["major"]

    @property
    def minor(self):
        return self._device_props["minor"]

    @property
    def name(self):
        return self._device_props["name"].decode("utf-8")

    @property
    def pci_bus_id(self):
        return self._cupy_device.pci_bus_id

    @property
    def warp_size(self):
        return self._device_attrs["WarpSize"]

    @property
    def max_threads_per_block(self):
        return self._device_attrs["MaxThreadsPerBlock"]

    @property
    def max_block_dim_x(self):
        return self._device_attrs["MaxBlockDimX"]

    @property
    def max_block_dim_y(self):
        return self._device_attrs["MaxBlockDimY"]

    @property
    def max_block_dim_z(self):
        return self._device_attrs["MaxBlockDimZ"]

    @property
    def max_grid_dim_x(self):
        return self._device_attrs["MaxGridDimX"]

    @property
    def max_grid_dim_y(self):
        return self._device_attrs["MaxGridDimY"]

    @property
    def max_grid_dim_z(self):
        return self._device_attrs["MaxGridDimZ"]

    @property
    def max_shared_memory_per_block(self):
        return self._device_attrs["MaxSharedMemoryPerBlock"]

    @property
    def max_shared_memory_per_block_optin(self):
        return self._device_attrs["MaxSharedMemoryPerBlockOptin"]

    @property
    def free_memory_bytes(self):
        return self._cupy_device.mem_info[0]

    @property
    def global_memory_bytes(self):
        return self._cupy_device.mem_info[1]


class CupyKernel:
    def __init__(self, handle):
        assert isinstance(handle, cp.RawKernel), type(handle)
        self._cupy_kernel = handle

    @property
    def handle(self):
        return self._cupy_kernel

    @property
    def max_threads_per_block(self):
        return self._cupy_kernel.max_threads_per_block

    def __call__(self, grid, block, args, stream, shared_mem=None, time_kernel=False):
        shared_mem = shared_mem if (shared_mem is not None) else 0
        if time_kernel:
            start = stream.record()
            with stream:
                self._cupy_kernel(grid, block, args, shared_mem=shared_mem)
            end = stream.record()
            stream.synchronize()
            return 1e-3 * cp.cuda.get_elapsed_time(start, end)
        else:
            with stream:
                self._cupy_kernel(grid, block, args, shared_mem=shared_mem)


class CudaOpenGLMappedBuffer:
    def __init__(self, gl_buffer, flags=0):
        self._gl_buffer = int(gl_buffer)
        self._flags = int(flags)

        self._graphics_ressource = None
        self._cuda_buffer = None

        self.register()

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_ressource(self):
        assert self.registered
        return self._graphics_ressource

    @property
    def registered(self):
        return self._graphics_ressource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        if self.registered:
            return self._graphics_ressource
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
        )
        return self._graphics_ressource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
        )
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(
            cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
        )

        ptr, size = check_cudart_err(
            cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
        )

        self._cuda_buffer = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(ptr, size, self), 0
        )

        return self._cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
        )

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
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array


class CupyBackend(CudaBackend):
    ArrayBackend = cp
    Array = cp.ndarray
    Buffer = cp.cuda.memory.MemoryPointer
    Stream = cp.cuda.Stream
    Device = CupyDevice
    Kernel = CupyKernel
    AllocationError = cp.cuda.memory.OutOfMemoryError

    CudaOpenGLMappedArray = CudaOpenGLMappedArray
    CudaOpenGLMappedBuffer = CudaOpenGLMappedBuffer

    NullMem = cp.cuda.UnownedMemory(ptr=0x0, size=0, owner=None, device_id=0)
    NullPtr = cp.cuda.memory.MemoryPointer(NullMem, 0)

    @classmethod
    def default_stream(cls):
        return cp.cuda.Stream.null

    @classmethod
    def asnumpy(cls, a):
        return cp.asnumpy(a)

    @classmethod
    def asarray(cls, a):
        return cp.asarray(a)

    @classmethod
    def get_nbytes(cls, a):
        if a is None:
            return 0
        elif isinstance(a, cls.Array):
            return a.nbytes
        elif isinstance(a, cls.Buffer):
            return a.mem.size
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
        return cls.Array(shape=shape, dtype=dtype, memptr=ptr)

    @classmethod
    def fill_zeros(cls, a, stream):
        assert isinstance(a, cls.Array), type(a)
        cp.cuda.runtime.memsetAsync(
            a.data.ptr,
            0,
            a.nbytes,
            stream.ptr,
        )

    @classmethod
    def determine_cuda_include_dir(cls, logger):
        if "CUPMB_CUDA_ROOT" in os.environ:
            cuda_root = os.environ["CUPMB_CUDA_ROOT"].strip()
            if not os.path.isdir(cuda_root):
                msg = f"CUPMB_CUDA_ROOT has been set to '{cuda_root}' but this directory does not exist."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            cuda_root = cp.cuda.get_cuda_path()
            if not os.path.isdir(cuda_root):
                msg = f"Cupy reports cuda path '{cuda_root}' but this directory does not exist."
                logger.error(msg)
                raise RuntimeError(msg)

        cuda_include_dir = os.path.join(cuda_root, "include")

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
        try:
            module = cp.RawModule(
                code=source_code,
                name_expressions=kernel_names,
                backend="nvrtc",
                options=build_options,
            )
        except Exception as e:
            logger.error(f"Failed to build module, build options were {build_options}.")
            logger.error(e)
            raise e

        kernels = tuple(map(cls.Kernel, map(module.get_function, kernel_names)))
        return kernels
