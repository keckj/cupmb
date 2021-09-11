import os, sys, re, tempfile
import sympy as sm
import numpy as np
import cupy as cp

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pmb import logger
from pmb.cuda.backend import CudaBackend
from pmb.cuda.device import CudaDevice, CudaDeviceGroup

from pmb.utils.IO import cuda_kernels_dir
from pmb.utils.units import bytes2str


def extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}"


logger.info("Loading CUDA backend.")

int_driver_version = cp.cuda.runtime.driverGetVersion()
int_cuda_version = cp.cuda.runtime.runtimeGetVersion()
driver_version = extract_cuda_version(int_driver_version)
cuda_version = extract_cuda_version(int_cuda_version)

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

logger.info("Creating default CUDA device group containing all visible devices.")
device_count = CudaBackend.available_device_count()
if device_count <= 0:
    msg = "Could not find any CUDA device."
    logger.fatal(msg)
    raise RuntimeError(msg)
logger.info(f"Found {device_count} visible CUDA device(s).")

devices = CudaDeviceGroup()
for i in range(device_count):
    logger.debug(f"Initializing cuda device {i}...")
    devices.insert_device(i, CudaDevice(i))
    logger.info(f"Found cuda device {devices[i]}")

    attrs = devices[i].attrs
    logger.debug(f"Device UUID:                       {attrs.uuid.hex()}")
    logger.debug(
        "Compute capability:                "
        f"{'.'.join(map(str, devices[i].compute_capability))} (sm_{devices[i].architecture})"
    )
    logger.debug(f"Concurrent kernels:                {attrs.concurrent_kernels}")
    logger.debug(f"Async engine count:                {attrs.async_engine_count}")
    logger.debug(f"Multiprocessor count:              {attrs.multi_processor_count}")
    logger.debug(
        f"Global memory size:                {bytes2str(attrs.total_global_memory)}"
    )
    logger.debug(
        f"Constant memory size:              {bytes2str(attrs.total_constant_memory)}"
    )
    logger.debug(f"L2 cache size:                     {bytes2str(attrs.l2_cache_size)}")
    logger.debug(f"Warp size:                         {attrs.warp_size}")
    logger.debug(
        f"Max blocks per SM:                 {attrs.max_blocks_per_multi_processor}"
    )
    logger.debug(
        f"Max threads per SM:                {attrs.max_threads_per_multi_processor}"
    )
    logger.debug(f"Max threads per block:             {attrs.max_threads_per_block}")
    logger.debug(f"Max blockdim.x:                    {attrs.max_block_dim_x}")
    logger.debug(f"Max blockdim.y:                    {attrs.max_block_dim_y}")
    logger.debug(f"Max griddim.x:                     {attrs.max_grid_dim_x}")
    logger.debug(f"Max griddim.y:                     {attrs.max_grid_dim_y}")
    logger.debug(
        f"Max 32B registers per SM:          {attrs.max_registers_per_multiprocessor} "
        f"({bytes2str(32/8*attrs.max_registers_per_multiprocessor)})"
    )
    logger.debug(
        f"Max 32B registers per block:       {attrs.max_registers_per_block} "
        f"({bytes2str(32/8*attrs.max_registers_per_block)})"
    )
    logger.debug(
        f"Max shared memory per SM:          {bytes2str(attrs.max_shared_memory_per_multiprocessor)}"
    )
    logger.debug(
        f"Max shared memory per block:       {bytes2str(attrs.max_shared_memory_per_block)}"
    )
    logger.debug(
        f"Max shared memory per block optin: {bytes2str(attrs.max_shared_memory_per_block_optin)}"
    )

device_id = 0
logger.info(f"Using device {device_id}.")
device = devices[device_id]

dtype_to_ctype = {
    np.float32: "float",
    np.float64: "double",
}
itype_to_ctype = {
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long int",
}

sdef = """
template <typename T>
constexpr T pi = T(3.14159265358979323846);

template <typename T, typename I, I Nx, I Ny, I Nz>
__device__ __forceinline__ T scalar(
        const T& x,  const T& y,  const T& z,
        const I& ix, const I& iy, const I& iz) {
    return cos(2*pi<T>*x)*sin(2*pi<T>*y)*cos(2*pi<T>*z);
}
"""
tmpdir = tempfile.TemporaryDirectory()
tmpfile = os.path.join(tmpdir.name, "cupmb_func_definition.cuh")
with open(tmpfile, "w") as f:
    f.write(sdef)

source = os.path.join(cuda_kernels_dir, "pmb.cuh")
options = CudaBackend.default_build_options()
options += ("--restrict", f"-I{cuda_kernels_dir}", f"-I{tmpdir.name}")

triangle_table = np.asarray(
    [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1],
        [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1],
        [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1],
        [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
        [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1],
        [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1],
        [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1],
        [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1],
        [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1],
        [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
        [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1],
        [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1],
        [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1],
        [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1],
        [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
        [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
        [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1],
        [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
        [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1],
        [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
        [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1],
        [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1],
        [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
        [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1],
        [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1],
        [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1],
        [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1],
        [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1],
        [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int8,
)

genx = np.asarray([int(0 in xi) for xi in triangle_table], dtype=np.uint64)
geny = np.asarray([int(3 in xi) for xi in triangle_table], dtype=np.uint64)
genz = np.asarray([int(8 in xi) for xi in triangle_table], dtype=np.uint64)
local_vertex_count = genx + geny + genz
print("Vertex count: ", local_vertex_count)

triangles_count = np.asarray(
    [(xi != -1).sum() // 3 for xi in triangle_table], dtype=np.uint8
)
print("Triangle count:   ", triangles_count)

tt = triangle_table.copy()
tt[tt == -1] = 15
tt = tt.astype(np.uint64)
unused_bit = 0
ttf = np.zeros(shape=256, dtype=np.uint64)
ttf += genx << 0
ttf += geny << 1
ttf += genz << 2
ttf += unused_bit << 3
for i in range(15):
    ttf += tt[:, i] << (4 * (i + 1))


def hexify(x):
    return "    " + ", ".join(f"0x{xi:016x}ull" for xi in x)


print(
    "unsigned long long int cases[256] = {\n"
    + "\n".join(map(hexify, ttf.reshape(-1, 8)))
    + "\n}"
)


def boolean_expression(
    values,
    nbits=None,
    ignore=None,
    prefix=None,
    postfix=None,
    subs=None,
    ctype=None,
):
    nbits = np.iinfo(values.dtype).bits if (nbits is None) else nbits
    assert nbits > 0, nbits

    ignore = set() if (ignore is None) else set(ignore)
    prefix = "" if (prefix is None) else prefix
    postfix = "" if (postfix is None) else postfix
    ctype = "" if (ctype is None) else ctype
    subs = (lambda x: x) if (subs is None) else subs

    expr = ""
    written = False
    for i in range(nbits):
        M = np.full_like(values, dtype=bool, fill_value=False)
        for ignored_value in ignore:
            M |= values == ignored_value
        (U,) = np.where(M)
        (I,) = np.where(((values >> (nbits - 1 - i)) & 1) & (~M))
        sop = sm.SOPform(ss, I.tolist(), U.tolist())
        if sop == False:  # noqa: E712
            continue
        sops = str(sop).replace("~", "!").replace("&", "&&").replace("|", "||")
        if written:
            expr += " + "
        written = True
        if nbits - 1 - i == 0:
            if sop == True:  # noqa: E712
                expr += "{ctype}(1)"
            else:
                expr += f"{ctype}({sops})"
        else:
            if sop == True:  # noqa: E712
                expr += "{ctype}({1 << (nbits - 1 - i)})"
            else:
                expr += f"({ctype}({sops}) << {nbits-1-i})"
    if not written:
        expr += "0"
    return subs(prefix + expr + postfix)


ss = sm.symbols("e3 e2 e1 e0")


def subs(x):
    return re.sub(r"e(\d)", r"e[\1]", x)


def required_shared_memory_bytes(device, B, dtype, itype):
    warpsize = device.attrs.warp_size
    assert B % warpsize == 0, B
    assert B // warpsize <= warpsize, (B, warpsize)
    warps_per_block = B // warpsize
    m0 = B * dtype(0).itemsize
    m1 = B * (itype(0).itemsize + np.uint8(0).itemsize)
    m2 = 2 * (warps_per_block + 1) * itype(0).itemsize
    return max(m0, m1, m2)


dtype = np.float32
itype = np.uint32
ndim = 3
n = 21
Nx = itype(n)
Ny = itype(n)
Nz = itype(n)
N = Nx * Ny * Nz

logger.info(f"Discretization is Nx={Nx}, Ny={Ny} and Nz={Nz}")
Bx = itype(8)
By = itype(8)
Bz = itype(8)
B = Bx * By * Bz
shared_memory_bytes = required_shared_memory_bytes(device, B, dtype, itype)
logger.info(
    f"Block size is Bx={Bx} and By={By}, requires {shared_memory_bytes//1024}kiB shared memory"
)

Tx = np.uint32(128)
warpsize = device.attrs.warp_size

Gx = (Nx + Bx - 2) // (Bx - 1)
Gy = (Ny + By - 2) // (By - 1)
Gz = (Nz + Bz - 2) // (Bz - 1)
Nblocks = Gx * Gy * Gz
Px = (Nblocks + Tx - 1) // Tx

isolevel = dtype(0.5 ** 2)
x0 = dtype(0)
y0 = dtype(0)
z0 = dtype(0)
Lx = dtype(1)
Ly = dtype(1)
Lz = dtype(1)
dx = dtype(Lx / Nx)
dy = dtype(Ly / Ny)
dz = dtype(Lz / Nz)


def scalar(x, y, z):
    if np.isscalar(x):
        # return (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
        return (
            np.cos(2 * np.pi * x, dtype=x.dtype)
            * np.sin(2 * np.pi * y, dtype=y.dtype)
            * np.cos(2 * np.pi * z, dtype=z.dtype)
        )
    else:
        assert x.shape == y.shape, (x.shape, y.shape)
        inside_domain = np.logical_and(
            np.logical_and(
                np.logical_and(x >= x0, x <= x0 + Lx),
                np.logical_and(y >= y0, y <= y0 + Ly),
            ),
            np.logical_and(z >= z0, z <= z0 + Lz),
        )
        output = np.empty_like(x * y * z)
        output[...] = np.where(
            inside_domain,
            # (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2,
            np.cos(2 * np.pi * x, dtype=x.dtype)
            * np.sin(2 * np.pi * y, dtype=y.dtype)
            * np.cos(2 * np.pi * z, dtype=z.dtype),
            float("nan"),
        )
        return output


fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    sharex=True,
    sharey=True,
    figsize=(12, 6),
    subplot_kw=dict(projection="3d"),
)
fig.suptitle(
    f"Marching squares: Nx={Nx}, Ny={Ny}, Nz={Nz}, Bx={Bx}, By={By}, Bz={Bz}, Gx={Gx}, Gy={Gy}, Gz={Gz}"
)
axes[0].set_title("CPU")
axes[1].set_title("GPU")
x = x0 + np.arange(Nx + 1) * dx
y = y0 + np.arange(Ny + 1) * dy
z = z0 + np.arange(Nz + 1) * dz
x, y, z = np.meshgrid(x, y, z)
I = scalar(x, y, z)
E = (I - isolevel) <= 0
C = np.where(np.isnan(I), "g", np.where(E, "m", "b"))
V = np.where(abs(I - isolevel) < 0.1)
for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    if False:
        ax.scatter(
            x[V].ravel(),
            y[V].ravel(),
            z[V].ravel(),
            c=C[V].ravel(),
            s=10,
            zorder=100000,
            alpha=1.0,
        )

h_minvalues = np.empty((Gz, Gy, Gx), dtype=dtype)
h_maxvalues = np.empty((Gz, Gy, Gx), dtype=dtype)

Ix = np.arange(Bx)
Iy = np.arange(By)
Iz = np.arange(Bz)
for i in range(Gz):
    for j in range(Gy):
        for k in range(Gx):
            zidx = i * (Bz - 1) + Iz
            yidx = j * (By - 1) + Iy
            xidx = k * (Bx - 1) + Ix
            z = (z0 + zidx * dz).astype(dtype)
            y = (y0 + yidx * dy).astype(dtype)
            x = (x0 + xidx * dx).astype(dtype)
            x, y, z = np.meshgrid(x, y, z)
            s = scalar(x, y, z) - isolevel
            h_minvalues[i, j, k] = np.nanmin(s)
            h_maxvalues[i, j, k] = np.nanmax(s)

h_block_mask = (
    (h_minvalues <= 0.0) & (h_maxvalues >= 0.0) & (h_minvalues < h_maxvalues)
).astype(np.uint8)
h_block_count = h_block_mask.sum().reshape((1,)).astype(itype)
h_compacted_block_indices = (
    np.arange(Nblocks, dtype=itype)[h_block_mask.astype(np.bool_).ravel()]
).copy()

logger.info(f"Found {h_block_count.item()} active blocks containing solutions.")

vertices = []
triangles = []
for i in range(h_block_count.item()):
    bid = h_compacted_block_indices[i]
    bidx = bid
    bidz = bidx // (Gx * Gy)
    bidx -= bidz * Gx * Gy
    bidy = bidx // Gx
    bidx -= bidy * Gx

    xidx = bidx * (Bx - 1) + Ix
    yidx = bidy * (By - 1) + Iy
    zidx = bidz * (Bz - 1) + Iz

    x = (x0 + xidx * dx).astype(dtype)
    y = (y0 + yidx * dy).astype(dtype)
    z = (z0 + zidx * dz).astype(dtype)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    s = scalar(X, Y, Z) - isolevel
    xx = X.ravel()
    yy = Y.ravel()
    zz = Z.ravel()

    S = tuple(np.full_like(s, fill_value=np.nan) for i in range(8))
    S[0][+0:, +0:, +0:] = s[+0:, +0:, +0:]
    S[1][+0:, +0:, :-1] = s[+0:, +0:, +1:]
    S[2][+0:, :-1, :-1] = s[+0:, +1:, +1:]
    S[3][+0:, :-1, +0:] = s[+0:, +1:, +0:]
    S[4][:-1, +0:, +0:] = s[+1:, +0:, +0:]
    S[5][:-1, +0:, :-1] = s[+1:, +0:, +1:]
    S[6][:-1, :-1, :-1] = s[+1:, +1:, +1:]
    S[7][:-1, :-1, +0:] = s[+1:, +1:, +0:]

    S = tuple(map(np.ravel, S))
    caseid = np.zeros_like(s, dtype=np.uint8).ravel()
    for i in range(8):
        caseid += (1 << i) * (S[i] <= 0.0).astype(np.uint8)

    thread_first_vertex_id = []
    thread_xid, thread_yid, thread_zid = [], [], []
    thread_nvertices = []

    # generate vertices
    for j in range(caseid.size):
        voffset = len(vertices)
        thread_first_vertex_id.append(voffset)
        config = triangle_table[caseid[j]]

        local_idx = 0
        xid, yid, zid = np.nan, np.nan, np.nan
        genx, geny, genz = 0, 0, 0
        visited = (set(range(12)) - {0, 3, 8}) | {-1}
        for k, e in enumerate(config):
            if e in visited:
                continue
            oz = j // (Bx * By)
            oy = (j - oz * Bx * By) // Bx
            ox = j % Bx
            xs, ys, zs = xx[j], yy[j], zz[j]
            if e == 0:
                genx = 1
                xid = local_idx
                visited.add(0)
                xs += (S[0][j] / (S[0][j] - S[1][j])) * dx
            elif e == 3:
                geny = 1
                yid = local_idx
                visited.add(3)
                ys += (S[0][j] / (S[0][j] - S[3][j])) * dy
            elif e == 8:
                genz = 1
                zid = local_idx
                visited.add(8)
                zs += (S[0][j] / (S[0][j] - S[4][j])) * dz
            else:
                raise NotImplementedError(e)
            vertices.append((xs, ys, zs))
            local_idx += 1
        assert local_idx <= 3, local_idx
        thread_xid.append(xid)
        thread_yid.append(yid)
        thread_zid.append(zid)
        thread_nvertices.append(local_idx)

    # generate triangles
    assert caseid.size == Bx * By * Bz
    for j in range(caseid.size):
        tx = j
        tz = tx // (Bx * By)
        tx -= tz * Bx * By
        ty = tx // Bx
        tx -= ty * Bx
        if (tx == Bx - 1) or (ty == By - 1) or (tz == Bz - 1):
            continue
        case_id = caseid[j]
        config = triangle_table[case_id]
        local_indices = []
        for k, e in enumerate(config):
            if e == -1:
                break
            ox = int(e in {1, 5, 9, 10})
            oy = int(e in {2, 6, 10, 11})
            oz = int(e in {4, 5, 6, 7})
            offset = j + 1 * ox + Bx * oy + Bx * By * oz

            dirx = int(e in {0, 2, 4, 6})
            diry = int(e in {1, 3, 5, 7})
            dirz = int(e in {8, 9, 10, 11})
            if dirx:
                delta = thread_xid[offset]
            elif diry:
                delta = thread_yid[offset]
            elif dirz:
                delta = thread_zid[offset]
            else:
                raise NotImplementedError((dirx, diry, dirz))
            assert np.isfinite(delta), delta
            assert thread_nvertices[offset] > delta, (
                f"case_id={case_id}, e={e}, ox={ox}, oy={oy}, oz={oz}, "
                f"offset={offset}, nvertices={thread_nvertices[offset]}"
            )

            idx = thread_first_vertex_id[offset] + delta
            local_indices.append(idx)

        assert len(local_indices) % 3 == 0, len(local_indices)
        triangles.extend(local_indices)

h_vertices = np.asarray(vertices, dtype=dtype)
h_triangles = np.asarray(triangles, dtype=itype).reshape(-1, 3)
h_vertex_and_triangle_count = np.asarray(
    [h_vertices.shape[0], h_triangles.shape[0]], dtype=itype
)

X, Y, Z = h_vertices[:, 0], h_vertices[:, 1], h_vertices[:, 2]
axes[0].scatter(X, Y, Z, zorder=10, c="y", marker="x", s=100)

triangles = [[tuple(h_vertices[j]) for j in i] for i in h_triangles]
poly3d = Poly3DCollection(triangles, facecolors="m", alpha=0.5, lw=1)
axes[0].add_collection3d(poly3d)
logger.info(
    f"Found {h_vertex_and_triangle_count[0]} python solutions: max(|f(x)|) = {np.nanmax(abs(scalar(X,Y,Z)))}"
)

assert h_block_mask.dtype == np.uint8
assert h_block_count.dtype == itype
assert h_vertices.dtype == dtype
assert h_triangles.dtype == itype
assert h_vertex_and_triangle_count.dtype == itype

d_block_mask = cp.empty(shape=h_block_mask.shape, dtype=h_block_mask.dtype)
d_block_count = cp.empty(shape=h_block_count.shape, dtype=h_block_count.dtype)
d_vertex_and_triangle_count = cp.empty(
    shape=h_vertex_and_triangle_count.shape, dtype=h_vertex_and_triangle_count.dtype
)
d_compacted_block_indices = cp.empty(
    shape=h_compacted_block_indices.shape,
    dtype=h_compacted_block_indices.dtype,
)
d_vertices = cp.empty(shape=h_vertices.shape, dtype=h_vertices.dtype)
d_triangles = cp.empty(shape=h_triangles.shape, dtype=h_triangles.dtype)

flt_type = dtype_to_ctype[dtype]
int_type = itype_to_ctype[itype]
knames = (
    f"computeBlockMask{ndim}D<{flt_type},{int_type},{Bx}u,{By}u,{Bz}u,{Nx}u,{Ny}u,{Nz}u,{warpsize}u>",
)
knames += (f"countActiveBlocks<{int_type},{Nblocks}u,{Tx}u,{warpsize}u>",)
knames += (f"compactActiveBlocks<{int_type},{Nblocks}u,{Tx}u,{warpsize}u>",)
knames += (
    f"computeSimplices{ndim}D<{flt_type},{int_type},{Bx}u,{By}u,{Bz}u,{Nx}u,{Ny}u,{Nz}u,{warpsize}u>",
)
for k in knames:
    logger.info(k)
(
    compute_block_mask,
    count_active_blocks,
    compact_active_blocks,
    compute_simplices,
) = CudaBackend.build_module(
    knames,
    filepath=source,
    extract_kernels=True,
    options=options,
)
compute_simplices.max_dynamic_shared_size_bytes = shared_memory_bytes

if B > compute_block_mask.max_threads_per_block:
    raise RuntimeError((B, compute_block_mask.max_threads_per_block))
if B > compute_simplices.max_threads_per_block:
    raise RuntimeError((B, compute_simplices.max_threads_per_block))
if Tx > count_active_blocks.max_threads_per_block:
    raise RuntimeError((Tx, count_active_blocks.max_threads_per_block))
if Tx > compact_active_blocks.max_threads_per_block:
    raise RuntimeError((Tx, compact_active_blocks.max_threads_per_block))

compute_block_mask(
    (Gx, Gy, Gz), (Bx, By, Bz), (isolevel, x0, dx, y0, dy, z0, dz, d_block_mask)
)
block_mask = cp.asnumpy(d_block_mask)
logger.info("Block active mask:")
print("CPU block mask")
print(h_block_mask)
print("GPU block mask")
print(block_mask)
np.testing.assert_equal(block_mask, h_block_mask)

cp.cuda.runtime.memsetAsync(d_block_count.data.ptr, 0x0, d_block_count.nbytes, 0)
count_active_blocks((Px,), (Tx,), (d_block_mask, d_block_count))
block_count = cp.asnumpy(d_block_count)
logger.info(
    f"CPU counted {h_block_count.item()} active blocks and GPU {block_count.item()}."
)
np.testing.assert_equal(block_count, h_block_count)

cp.cuda.runtime.memsetAsync(d_block_count.data.ptr, 0x0, d_block_count.nbytes, 0)
compact_active_blocks(
    (Px,),
    (Tx,),
    (
        itype(d_compacted_block_indices.size),
        d_block_mask,
        d_compacted_block_indices,
        d_block_count,
    ),
)
block_count_bis = cp.asnumpy(d_block_count)
compacted_block_indices = cp.asnumpy(d_compacted_block_indices)
logger.info(
    f"GPU successfully compacted {block_count_bis.item()} active block triangles."
)
assert block_count == block_count_bis, (block_count, block_count_bis)
compacted_block_indices.sort()
print("CPU block triangles")
print(h_compacted_block_indices)
print("GPU block triangles")
print(compacted_block_indices)
np.testing.assert_equal(compacted_block_indices, h_compacted_block_indices)

Nactive = block_count.item()
logger.info(f"Generating triangles by using {Nactive} active blocks...")
cp.cuda.runtime.memsetAsync(
    d_vertex_and_triangle_count.data.ptr, 0x0, d_vertex_and_triangle_count.nbytes, 0
)
compute_simplices(
    (Nactive,),
    (Bx, By, Bz),
    (
        itype(d_vertices.shape[0]),
        itype(d_triangles.shape[0]),
        isolevel,
        x0,
        dx,
        y0,
        dy,
        z0,
        dz,
        d_compacted_block_indices,
        d_vertices,
        d_triangles,
        d_vertex_and_triangle_count,
    ),
    shared_mem=shared_memory_bytes,
)
vertex_and_triangle_count = cp.asnumpy(d_vertex_and_triangle_count)
vertices = cp.asnumpy(d_vertices)
triangles = cp.asnumpy(d_triangles)
X, Y, Z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
axes[1].scatter(X, Y, Z, zorder=10, c="y", marker="x", s=100)

triangles = [[tuple(vertices[j]) for j in i] for i in triangles]
poly3d = Poly3DCollection(triangles, facecolors="m", alpha=0.5, lw=1)
axes[1].add_collection3d(poly3d)

logger.info(
    f"GPU found {vertex_and_triangle_count[0]} grid-aligned zeros "
    f"on domain [0,1] while CPU found {h_vertex_and_triangle_count[0]}."
)
logger.info(
    f"GPU found {vertex_and_triangle_count[1]} vertex triangles to represent isolines "
    f"while CPU found {h_vertex_and_triangle_count[1]}."
)

try:
    np.testing.assert_equal(vertex_and_triangle_count, h_vertex_and_triangle_count)
except Exception as e:
    print(e)
plt.show()
