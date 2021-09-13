from cupmb.backend import CudaBackend
from cupmb.utils.logger import logger

logger.info("Importing PyCuda backend and Cuda-Python.")

import pycuda
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit

from cuda import cuda, cudart, nvrtc

import numpy as np


def cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def check_cuda_error(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={} ({})".format(
                result[0].value, cuda_get_error_enum(result[0]).decode("utf-8")
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


CCE = check_cuda_error


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


# Check Python-Cuda driver and cuda version
int_driver_version = CCE(cudart.cudaDriverGetVersion())
int_cuda_version = CCE(cudart.cudaRuntimeGetVersion())
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
        "https://developer.nvidia.com/cuda-downloads and rebuild Python-Cuda."
    )
    logger.warning(msg)

msg = (
    f"Python-Cuda loaded cuda runtime {cuda_version} ({int_cuda_version}), "
    f"driver version is {driver_version} ({int_driver_version})."
)
logger.info(msg)


class PyCudaDevice:
    da = pycuda.driver.device_attribute

    def __init__(self, device_id):
        self._pycuda_device = pycuda.driver.Device(device_id)
        self._device_attrs = self._pycuda_device.get_attributes()
        self._device_id = device_id
        # self._context = self._pycuda_device.retain_primary_context()

    def __enter__(self):
        # self._context.push()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self._context.pop()
        pass

    @property
    def handle(self):
        return self._pycuda_device

    @property
    def id(self):
        return self._device_id

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
        return self._device_attrs[self.da.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN]


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
        shared_mem = shared_mem if (shared_mem is not None) else 0
        if time_kernel:
            return self._pycuda_kernel(
                *args,
                grid=grid,
                block=block,
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


class PyCudaBackend(CudaBackend):
    ArrayBackend = pycuda.gpuarray
    Array = pycuda.gpuarray.GPUArray
    Buffer = pycuda.driver.DeviceAllocation
    Stream = pycuda.driver.Stream
    Device = PyCudaDevice
    Kernel = PyCudaKernel

    NullPtr = Array(shape=(), dtype=np.uint8)

    @classmethod
    def default_stream(cls):
        return None

    @classmethod
    def asnumpy(cls, a):
        return a.get()

    @classmethod
    def asarray(cls, a):
        return pycuda.gpuarray.to_gpu(a)

    @classmethod
    def get_nbytes(cls, a):
        if isinstance(a, None):
            return 0
        elif isinstance(a, cls.Array):
            return a.nbytes
        elif isinstance(a, cls.Buffer):
            return a.size
        else:
            raise NotImplementedError(type(a))

    @classmethod
    def get_data_pointers(cls, *args):
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
    def build_module_and_extract_kernels(
        cls, logger, program_name, device_id, code, kernel_names, build_options
    ):
        # Here we use nvrtc because pycuda cannot handle templates __global__ functions
        # due to name mangling (see https://wiki.tiker.net/PyCuda/Examples/C++FunctionTemplates).

        def encode_arg(a):
            return a.encode("ascii")

        def decode_arg(a):
            return a.decode("ascii")

        major = CCE(
            cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device_id
            )
        )
        minor = CCE(
            cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device_id
            )
        )
        _, nvrtc_minor = CCE(nvrtc.nvrtcVersion())
        use_cubin = nvrtc_minor >= 1

        prefix = "sm" if use_cubin else "compute"
        arch_arg = f"--gpu-architecture={prefix}_{major}{minor}"

        build_options = tuple(build_options)
        build_options += (arch_arg,)

        prog = CCE(
            nvrtc.nvrtcCreateProgram(
                encode_arg(code), encode_arg(program_name), 0, [], []
            )
        )
        for kname in map(encode_arg, kernel_names):
            CCE(nvrtc.nvrtcAddNameExpression(prog, kname))

        try:
            CCE(
                nvrtc.nvrtcCompileProgram(
                    prog, len(build_options), list(map(encode_arg, build_options))
                )
            )
        except RuntimeError as e:
            log_size = CCE(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * log_size
            CCE(nvrtc.nvrtcGetProgramLog(prog, log))
            logger.error(
                f"Failed to compile program {program_name}, build options were {build_options}."
            )
            logger.error(decode_arg(log))
            raise e

        mangled_kernel_names = []
        for kname in map(encode_arg, kernel_names):
            mangled_name = CCE(nvrtc.nvrtcGetLoweredName(prog, kname))
            mangled_kernel_names.append(decode_arg(mangled_name))

        if use_cubin:
            dataSize = CCE(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * dataSize
            CCE(nvrtc.nvrtcGetCUBIN(prog, data))
            binary_format = "CUBIN"
        else:
            dataSize = CCE(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * dataSize
            CCE(nvrtc.nvrtcGetPTX(prog, data))
            binary_format = "PTX"

        def message_handler(compile_success, info_str, error_str):
            if compile_success:
                return
            logger.error(
                f"PyCuda failed to build program {program_name} from nvrtc {binary_format} buffer."
            )
            if info_str:
                logger.error(info_str)
            if error_str:
                logger.error(info_str)
            raise RuntimeError

        module = pycuda.driver.module_from_buffer(
            np.char.array(data), message_handler=message_handler
        )

        kernels = []
        for mangled_kname in mangled_kernel_names:
            kernels.append(cls.Kernel(module.get_function(mangled_kname)))

        return tuple(kernels)
