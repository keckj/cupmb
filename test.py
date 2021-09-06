# import numpy as np
import cupy as cp

from pmb import logger
from pmb.cuda.backend import CudaBackend
from pmb.cuda.device import CudaDevice, CudaDeviceGroup

# from pmb.utils.IO import cuda_kernels_dir
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
    logger.debug(f"Max griddim.x:                     {attrs.max_grid_dim_x}")
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
