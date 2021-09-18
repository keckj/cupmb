#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
from cupmb.backend import CudaBackend
from cupmb.utils.logger import logger

logger.info("Importing CuPy backend.")

import cupy as cp
import numpy as np


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


class CupyBackend(CudaBackend):
    ArrayBackend = cp
    Array = cp.ndarray
    Buffer = cp.cuda.memory.MemoryPointer
    Stream = cp.cuda.Stream
    Device = CupyDevice
    Kernel = CupyKernel

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
        if isinstance(a, None):
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
        return args

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
    def build_module_and_extract_kernels(
        cls,
        logger,
        device_major,
        device_minor,
        kernel_source_file,
        kernel_names,
        build_options,
    ):
        with open(kernel_source_file) as f:
            code = f.read()

        try:
            module = cp.RawModule(
                code=code,
                name_expressions=kernel_names,
                backend="nvrtc",
                options=build_options,
            )
        except Exception as e:
            logger.error(
                f"Failed to compile {kernel_source_file}, build options were {build_options}."
            )
            logger.error(e)
            raise e

        kernels = tuple(map(cls.Kernel, map(module.get_function, kernel_names)))
        return kernels
