import os
import numpy as np
import cupy as cp

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

dtype_to_ctype = {
    np.float32: "float",
    np.float64: "double",
}

source = os.path.join(cuda_kernels_dir, "pmb.cuh")
options = CudaBackend.default_build_options() + ("--restrict", f"-I{cuda_kernels_dir}")


def scalar(x):
    inside_domain = np.logical_and(x >= 0.0, x <= 1.0)
    y = np.empty_like(x)
    y[...] = np.where(
        inside_domain, np.sin(153 * np.pi * x, dtype=x.dtype), float("nan")
    )
    return y


Nx = np.int32(1024 * 1024)
for ndim in (1,):
    for dtype in (np.float64,):
        for Bx in (1024,):
            Bx = np.uint32(Bx)
            Tx = np.uint32(32)

            isolevel = dtype(0.0)
            x0 = dtype(0.0)
            Lx = dtype(1.0)
            dx = dtype(Lx / Nx)

            warpsize = device.attrs.warp_size
            assert Bx % warpsize == 0, (Bx, warpsize)

            Gx = (Nx + Bx - 2) // (Bx - 1)
            Nblocks = Gx
            Px = (Nblocks + Tx - 1) // Tx

            h_minvalues = np.empty(Gx, dtype=dtype)
            h_maxvalues = np.empty(Gx, dtype=dtype)

            Ix = np.arange(Bx)
            for i in range(Gx):
                idx = i * (Bx - 1) + Ix
                x = (x0 + idx * dx).astype(dtype)
                s = scalar(x) - isolevel
                h_minvalues[i] = np.nanmin(s)
                h_maxvalues[i] = np.nanmax(s)

            h_block_mask = (
                (h_minvalues <= 0.0)
                & (h_maxvalues >= 0.0)
                & (h_minvalues < h_maxvalues)
            ).astype(np.uint8)
            h_block_count = h_block_mask.sum().reshape((1,)).astype(np.uint32)
            h_compacted_block_indices = (
                np.arange(Nblocks, dtype=np.int32)[h_block_mask.astype(np.bool_)]
            ).copy()

            vertices = []
            for i in range(h_block_count.item()):
                bidx = h_compacted_block_indices[i]
                idx = bidx * (Bx - 1) + Ix
                x = (x0 + idx * dx).astype(dtype)
                s = scalar(x) - isolevel
                caseid = (s[:-1] <= 0.0).astype(np.uint32) + 2 * (s[1:] <= 0.0).astype(
                    np.uint32
                )
                for j in range(caseid.size):
                    if caseid[j] in (0, 3):
                        continue
                    if s[j] == s[j + 1]:
                        continue
                    elif caseid[j] == 1:
                        xs = x[j] + (-s[j] / (-s[j] + s[j + 1])) * dx
                        vertices.append(xs)
                    elif caseid[j] == 2:
                        xs = x[j] + (+s[j] / (+s[j] - s[j + 1])) * dx
                        vertices.append(xs)
                    else:
                        raise NotImplementedError(caseid[j])
            h_vertices = np.asarray(vertices, dtype=dtype)
            h_vertex_count = np.asarray([h_vertices.size], dtype=np.uint32)

            d_block_mask = cp.empty(shape=h_block_mask.shape, dtype=h_block_mask.dtype)
            d_block_count = cp.empty(
                shape=h_block_count.shape, dtype=h_block_count.dtype
            )
            d_vertex_count = cp.empty(
                shape=h_vertex_count.shape, dtype=h_vertex_count.dtype
            )
            d_compacted_block_indices = cp.empty(
                shape=h_compacted_block_indices.shape,
                dtype=h_compacted_block_indices.dtype,
            )
            d_vertices = cp.empty(shape=h_vertices.shape, dtype=h_vertices.dtype)

            ctype = dtype_to_ctype[dtype]
            knames = (f"computeBlockMask{ndim}D<{ctype},{Bx}u,{Nx}u,{warpsize}u>",)
            knames += (f"countActiveBlocks<{Tx}u,{Nblocks}u,{warpsize}u>",)
            knames += (f"compactActiveBlocks<{Tx}u,{Nblocks}u,{warpsize}u>",)
            knames += (f"computeSimplices{ndim}D<{ctype},{Bx}u,{Nx}u,{warpsize}u>",)
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

            if Bx > compute_block_mask.max_threads_per_block:
                raise RuntimeError((Bx, compute_block_mask.max_threads_per_block))
            if Bx > compute_simplices.max_threads_per_block:
                raise RuntimeError((Bx, compute_simplices.max_threads_per_block))
            if Tx > count_active_blocks.max_threads_per_block:
                raise RuntimeError((Tx, count_active_blocks.max_threads_per_block))
            if Tx > compact_active_blocks.max_threads_per_block:
                raise RuntimeError((Tx, compact_active_blocks.max_threads_per_block))

            compute_block_mask((Gx,), (Bx,), (isolevel, x0, dx, d_block_mask))
            block_mask = cp.asnumpy(d_block_mask)

            cp.cuda.runtime.memsetAsync(
                d_block_count.data.ptr, 0x0, d_block_count.nbytes, 0
            )
            count_active_blocks((Px,), (Tx,), (d_block_mask, d_block_count))
            block_count = cp.asnumpy(d_block_count)

            cp.cuda.runtime.memsetAsync(
                d_block_count.data.ptr, 0x0, d_block_count.nbytes, 0
            )
            compact_active_blocks(
                (Px,),
                (Tx,),
                (
                    np.uint32(d_compacted_block_indices.size),
                    d_block_mask,
                    d_compacted_block_indices,
                    d_block_count,
                ),
            )
            block_count_bis = cp.asnumpy(d_block_count)
            compacted_block_indices = cp.asnumpy(d_compacted_block_indices)
            compacted_block_indices.sort()

            cp.cuda.runtime.memsetAsync(
                d_vertex_count.data.ptr, 0x0, d_vertex_count.nbytes, 0
            )
            compute_simplices(
                (block_count.item(),),
                (Bx,),
                (
                    np.uint32(d_vertices.size),
                    isolevel,
                    x0,
                    dx,
                    d_compacted_block_indices,
                    d_vertices,
                    d_vertex_count,
                ),
            )
            vertex_count = cp.asnumpy(d_vertex_count)
            vertices = cp.asnumpy(d_vertices)
            vertices.sort()

            assert block_count == block_count_bis, (block_count, block_count_bis)

            np.testing.assert_equal(block_count, h_block_count)
            np.testing.assert_equal(block_mask, h_block_mask)
            np.testing.assert_equal(compacted_block_indices, h_compacted_block_indices)
            np.testing.assert_equal(vertex_count, h_vertex_count)
            np.testing.assert_allclose(vertices, h_vertices)
            logger.info(
                f"Found {vertex_count.item()} zeros of function sin(153*π*x) on domain [0,1]."
            )
