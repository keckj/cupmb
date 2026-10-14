#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, sys, tempfile, subprocess, threading, functools

from cupmb.backend import CudaBackend
from cupmb.utils.logger import logger
from cupmb.utils.IO import utils_dir

logger.info("Importing PyCuda backend.")

import pycuda
import pycuda.driver
import pycuda.gpuarray

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
    """
    PyCuda device wrapper. Instances are cached per device id because querying device
    attributes is expensive. Entering the device makes sure that the current CUDA context
    is bound to this device (the device primary context is pushed if required).
    """

    da = pycuda.driver.device_attribute

    _instances = {}
    _instances_lock = threading.Lock()

    def __new__(cls, device_id):
        device_id = int(device_id)
        with cls._instances_lock:
            instance = cls._instances.get(device_id)
            if instance is None:
                instance = super().__new__(cls)
                instance._initialize(device_id)
                cls._instances[device_id] = instance
        return instance

    def _initialize(self, device_id):
        self._pycuda_device = pycuda.driver.Device(device_id)
        self._device_attrs = self._pycuda_device.get_attributes()
        self._device_id = device_id
        self._pci_bus_id = self._pycuda_device.pci_bus_id()
        self._compute_capability = self._pycuda_device.compute_capability()
        self._primary_context = None
        self._tls = threading.local()

    def _current_context_is_on_device(self):
        try:
            current_device = pycuda.driver.Context.get_device()
        except (pycuda.driver.LogicError, pycuda.driver.Error):
            return False
        return current_device.pci_bus_id() == self._pci_bus_id

    def __enter__(self):
        stack = self._tls.__dict__.setdefault("pushed", [])
        if self._current_context_is_on_device():
            stack.append(False)
        else:
            if self._primary_context is None:
                self._primary_context = self._pycuda_device.retain_primary_context()
            self._primary_context.push()
            stack.append(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tls.pushed.pop():
            pycuda.driver.Context.pop()

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
        return self._compute_capability[0]

    @property
    def minor(self):
        return self._compute_capability[1]

    @property
    def name(self):
        return self._pycuda_device.name()

    @property
    def pci_bus_id(self):
        return self._pci_bus_id

    @property
    def warp_size(self):
        return self._device_attrs[self.da.WARP_SIZE]

    @property
    def max_threads_per_multiprocessor(self):
        return self._device_attrs[self.da.MAX_THREADS_PER_MULTIPROCESSOR]

    @property
    def max_blocks_per_multiprocessor(self):
        return self._device_attrs[self.da.MAX_BLOCKS_PER_MULTIPROCESSOR]

    @property
    def multiprocessor_count(self):
        return self._device_attrs[self.da.MULTIPROCESSOR_COUNT]

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
        with self:
            return pycuda.driver.mem_get_info()[0]

    @property
    def global_memory_bytes(self):
        return self._pycuda_device.total_memory()


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

    def __call__(self, grid, block, args, stream=None, shared_mem=None, time_kernel=False):
        assert isinstance(grid, tuple)
        assert isinstance(block, tuple)
        assert len(grid) <= 3, len(grid)
        assert len(block) <= 3, len(block)
        grid = tuple(map(int, grid)) + (1,) * (3 - len(grid))
        block = tuple(map(int, block)) + (1,) * (3 - len(block))
        shared_mem = int(shared_mem) if (shared_mem is not None) else 0
        if time_kernel:
            # pycuda's own time_kernel records events on the null stream, record them on our stream instead
            start, end = pycuda.driver.Event(), pycuda.driver.Event()
            start.record(stream)
        self._pycuda_kernel(
            *args,
            grid=grid,
            block=block,
            shared=shared_mem,
            stream=stream,
        )
        if time_kernel:
            end.record(stream)
            end.synchronize()
            return 1e-3 * start.time_till(end)


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


@functools.cache
def _opengl_interop_classes():
    # pycuda.gl is only available when PyCuda has been built with OpenGL support (CUDA_ENABLE_GL),
    # it is thus imported lazily, only when OpenGL interoperability is actually requested.
    try:
        import pycuda.gl
    except ImportError as e:
        msg = (
            "PyCuda OpenGL interoperability is not available, please rebuild PyCuda with "
            f"OpenGL support (CUDA_ENABLE_GL=True) to use this feature ({e})."
        )
        logger.error(msg)
        raise ImportError(msg) from e

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

            def allocator(nbytes, buf=self.cuda_buffer):
                assert nbytes <= buf.size, (nbytes, buf.size)
                return buf

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

    return CudaOpenGLMappedBuffer, CudaOpenGLMappedArray


class _LazyOpenGLInteropClass:
    def __init__(self, index):
        self._index = index

    def __get__(self, instance, owner):
        return _opengl_interop_classes()[self._index]


class PyCudaBackend(CudaBackend):
    ArrayBackend = pycuda.gpuarray
    Array = pycuda.gpuarray.GPUArray
    Buffer = PointerHolderBase
    Stream = pycuda.driver.Stream
    Device = PyCudaDevice
    Kernel = PyCudaKernel
    AllocationError = pycuda.driver.MemoryError

    CudaOpenGLMappedBuffer = _LazyOpenGLInteropClass(0)
    CudaOpenGLMappedArray = _LazyOpenGLInteropClass(1)

    class NullPtr:
        gpudata = 0x0

    @classmethod
    def default_stream(cls):
        # None means the (legacy) default stream for all pycuda launches and copies
        return None

    @classmethod
    def asnumpy(cls, a, stream=None):
        if stream is None:
            return a.get()
        h = a.get_async(stream=stream)
        stream.synchronize()
        return h

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
        if stream is None:
            pycuda.driver.Context.synchronize()
        else:
            stream.synchronize()

    @classmethod
    def get_data_pointers(cls, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return tuple(cls.NullPtr if (a is None) else a for a in args)

    @classmethod
    def array_from_pointer(cls, ptr, shape, dtype):
        assert isinstance(ptr, cls.Buffer)
        req_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        ptr_bytes = cls.get_nbytes(ptr)
        if req_bytes > ptr_bytes:
            msg = (
                f"Buffer of {ptr_bytes} bytes is too small to hold an array of shape {shape} "
                f"and dtype {np.dtype(dtype)}."
            )
            raise ValueError(msg)
        return cls.Array(shape=shape, dtype=dtype, gpudata=ptr)

    @classmethod
    def fill_zeros(cls, a, stream):
        assert isinstance(a, cls.Array), type(a)
        if a.nbytes > 0:
            pycuda.driver.memset_d8_async(int(a.gpudata), 0, a.nbytes, stream)

    @classmethod
    @functools.cache
    def compiler_version(cls):
        nvrtc_build_program = os.path.join(utils_dir, "nvrtc_build.py")
        result = subprocess.run(
            [sys.executable, nvrtc_build_program, "--nvrtc-version"], check=True, capture_output=True, text=True
        )
        return f"pycuda-{pycuda.VERSION_TEXT}-nvrtc-{result.stdout.strip()}"

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
        kernels, _, _ = cls.build_kernels(logger, device_major, device_minor, source_code, kernel_names, build_options)
        return kernels

    @classmethod
    def build_kernels(
        cls,
        logger,
        device_major,
        device_minor,
        source_code,
        kernel_names,
        build_options,
    ):
        """
        Build the kernels, return the kernels, the compiled module (cubin or ptx) and the mangled kernel names.
        """
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
            msg = f"nvrtc build program file:///{nvrtc_build_program} is missing."
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

            with open(output_knames) as f:
                mangled_kernel_names = tuple(filter(len, f.read().split(os.linesep)))

            if not os.path.isfile(output_binary):
                msg = f"Source file:///{output_binary} has not been generated."
                logger.error(msg)
                raise RuntimeError(msg)

            with open(output_binary, "rb") as f:
                binary = f.read()

        module = pycuda.driver.module_from_buffer(binary)
        kernels = tuple(map(cls.Kernel, map(module.get_function, mangled_kernel_names)))
        return kernels, binary, mangled_kernel_names

    @classmethod
    def load_kernels(cls, logger, binary_path, mangled_kernel_names):
        """Load kernels from a compiled module file (cubin or ptx) and the mangled kernel names."""
        with open(binary_path, "rb") as f:
            module = pycuda.driver.module_from_buffer(f.read())
        return tuple(map(cls.Kernel, map(module.get_function, mangled_kernel_names)))
