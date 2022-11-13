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

os.environ["CUPMB_LOGLEVEL"] = "INFO"

import jinja2

import numpy as np
import cupy as cp

from cupmb import logger
from cupmb.backend import CudaBackend

from cupmb.enums import Backend

logger.info("Creating default CUDA device group containing all visible devices.")
device_count = cp.cuda.runtime.getDeviceCount()
if device_count <= 0:
    msg = "Could not find any CUDA device."
    logger.fatal(msg)
    raise RuntimeError(msg)
logger.info(f"Found {device_count} visible CUDA device(s).")

backend = CudaBackend(Backend.CUPY)
device = backend.Device(0)
stream = backend.Stream()

dtype_to_ctype = {
    np.float32: "float",
    np.float64: "double",
}

script_dir = os.path.realpath(os.path.dirname(__file__))
cuda_kernels_dir = os.path.realpath(os.path.join(script_dir, "..", "cupmb", "kernels"))
options = ("-std=c++17", "--restrict", f"-I{cuda_kernels_dir}", "-G")

template_file = os.path.join(cuda_kernels_dir, "pmb1D.cuh")
with open(template_file) as f:
    template = f.read()
template = jinja2.Template(template)

fdef = r"""
namespace cupmb {
    template <typename T>
    __device__ __forceinline__ T scalar(const T& x) {
        return sin(153*x);
    }
}
    """
source = template.render(cupmb_func_definitions=fdef, cupmb_vertex_properties="")


def scalar(x):
    inside_domain = np.logical_and(x >= 0.0, x <= 1.0)
    y = np.empty_like(x)
    y[...] = np.where(inside_domain, np.sin(153 * x, dtype=x.dtype), float("nan"))
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

            warpsize = device.warp_size
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
            itype = "unsigned int"
            knames = (
                f"cupmb::computeBlockMask{ndim}D<{ctype},{Bx}u,{Nx}u,{warpsize}u>",
            )
            knames += (
                f"cupmb::countActiveBlocks<{itype},{Nblocks}u,{Tx}u,{warpsize}u>",
            )
            knames += (
                f"cupmb::compactActiveBlocks<{itype},{Nblocks}u,{Tx}u,{warpsize}u>",
            )
            knames += (
                f"cupmb::computeSimplices{ndim}D<{ctype},{Bx}u,{Nx}u,{warpsize}u>",
            )
            for k in knames:
                logger.info(k)

            (
                compute_block_mask,
                count_active_blocks,
                compact_active_blocks,
                compute_simplices,
            ) = backend.build_module_and_extract_kernels_from_source(
                logger, None, None, source, knames, options
            )

            if Bx > compute_block_mask.max_threads_per_block:
                raise RuntimeError((Bx, compute_block_mask.max_threads_per_block))
            if Bx > compute_simplices.max_threads_per_block:
                raise RuntimeError((Bx, compute_simplices.max_threads_per_block))
            if Tx > count_active_blocks.max_threads_per_block:
                raise RuntimeError((Tx, count_active_blocks.max_threads_per_block))
            if Tx > compact_active_blocks.max_threads_per_block:
                raise RuntimeError((Tx, compact_active_blocks.max_threads_per_block))

            compute_block_mask((Gx,), (Bx,), (isolevel, x0, dx, d_block_mask), stream)
            block_mask = cp.asnumpy(d_block_mask)

            cp.cuda.runtime.memsetAsync(
                d_block_count.data.ptr, 0x0, d_block_count.nbytes, stream.ptr
            )
            count_active_blocks((Px,), (Tx,), (d_block_mask, d_block_count), stream)
            block_count = cp.asnumpy(d_block_count)

            cp.cuda.runtime.memsetAsync(
                d_block_count.data.ptr, 0x0, d_block_count.nbytes, stream.ptr
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
                stream,
            )
            block_count_bis = cp.asnumpy(d_block_count)
            compacted_block_indices = cp.asnumpy(d_compacted_block_indices)
            compacted_block_indices.sort()

            cp.cuda.runtime.memsetAsync(
                d_vertex_count.data.ptr, 0x0, d_vertex_count.nbytes, stream.ptr
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
                stream,
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
                f"Found {vertex_count.item()} zeros of function sin(153*x) on domain [0,1]."
            )
