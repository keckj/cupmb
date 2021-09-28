/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package cupmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

#ifndef PMB1D_CUH_JACGPBQT
#define PMB1D_CUH_JACGPBQT

#include "pmb.cuh"

namespace cupmb {

template <typename T, unsigned int Bx, unsigned int Nx, unsigned int warpsize=32>
__launch_bounds__(Bx)
__global__ void computeBlockMask1D(const T isolevel, const T x0, const T dx,
        unsigned char *const __restrict block_mask) {

    static_assert(Bx%warpsize==0, "Bx%warpsize != 0");
    static_assert(Bx/warpsize<=warpsize, "Bx/warpsize > warpsize");

    constexpr unsigned int warps_per_block = Bx/warpsize;
    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int laneid = threadIdx.x % warpsize;
    const unsigned int warpid = threadIdx.x / warpsize;
    const unsigned int idx = blockIdx.x*(Bx-1) + threadIdx.x;

    const T NaN(nan(""));
    const T x = x0 + idx*dx;
    const T v = (idx > Nx ? NaN : scalar(x) - isolevel);
    T minval(v), maxval(v);

    /* intra warp reduction */
    {
        #pragma unroll
        for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
            minval = min(minval, __shfl_down_sync(0xffffffff, minval, offset));
            maxval = max(maxval, __shfl_down_sync(0xffffffff, maxval, offset));
        }
        if (laneid == 0) {
            minbuf[warpid] = minval;
            maxbuf[warpid] = maxval;
        }
        __syncthreads();
    }

    /* intra block reduction */
    if (warpid == 0) {
        minval = (laneid >= warps_per_block ? NaN : minbuf[laneid]);
        maxval = (laneid >= warps_per_block ? NaN : maxbuf[laneid]);

        #pragma unroll
        for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
            minval = min(minval, __shfl_down_sync(0xffffffff, minval, offset));
            maxval = max(maxval, __shfl_down_sync(0xffffffff, maxval, offset));
        }
        if (laneid == 0) {
            const bool block_is_active = (minval <= T(0)) && (maxval >= T(0)) && (minval < maxval);
            block_mask[blockIdx.x] = (unsigned char)(block_is_active);
        }
    }
}


template <typename T, unsigned int Bx, unsigned int Nx, unsigned int warpsize=32>
__launch_bounds__(Bx)
__global__ void computeSimplices1D(unsigned int allocated_vertex_count,
        const T isolevel, const T x0, const T dx,
        const unsigned int *const __restrict compacted_block_indices,
        T* const __restrict output_vertices,
        unsigned int *const __restrict output_vertex_count) {
    static_assert(Bx%warpsize==0, "Bx%warpsize != 0");
    static_assert(Bx/warpsize<=warpsize, "Bx/warpsize > warpsize");

    constexpr unsigned int warps_per_block = Bx/warpsize;
    constexpr unsigned int shared_mem_bytes = constexpr_max(Bx*sizeof(T), (warps_per_block+1)*sizeof(unsigned int));
    __shared__ unsigned char shared_mem_buffer[shared_mem_bytes];

    T* const __restrict scalars = reinterpret_cast<T*>(shared_mem_buffer);
    unsigned int* const __restrict vertex_offsets = reinterpret_cast<unsigned int*>(shared_mem_buffer);

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = compacted_block_indices[blockIdx.x];
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;
    const unsigned int idx = bid*(Bx-1) + tid;

    const T NaN(nan(""));
    const T x = x0 + idx*dx;

    /* load scalars */
    T s[2];
    {
        s[0] = (idx > Nx ? NaN : scalar(x) - isolevel);
        scalars[tid] = s[0];
        __syncthreads();
        s[1] = (tid == Bx-1 ? s[0] : scalars[tid+1]);
        __syncthreads();
    }

    /* compute case id */
    unsigned int case_id = 0u;
    {
        #pragma unroll
        for (unsigned int k = 0u; k < 2u; ++k) {
            case_id += ((s[k]<=T(0)) << k);
            if (s[k] != s[k]) {
                case_id = 0u;
                break;
            }
        }
    }

    /* compute vertex */
    T local_vertices[1];
    unsigned int nvertices;
    switch(case_id) {
        case 0:
        case 3:
            nvertices=0u;
            break;
        case 1:
        case 2:
            nvertices=1u;
            local_vertices[0] = x + (s[0]/(s[0]-s[1]))*dx;
            break;
        default: __trap();
    }

    /* prefix scan output vertex id */
    unsigned int vertex_id = prefix_sum<unsigned int, warps_per_block, warpsize>(
            nvertices, laneid, warpid, vertex_offsets, output_vertex_count);

    /* write vertices to global memory */
    #pragma unroll 1
    for (unsigned int k = 0u; (k < nvertices) && (vertex_id < allocated_vertex_count); ++k) {
        output_vertices[vertex_id] = local_vertices[k];
        ++vertex_id;
    }
}

} /* end of namespace cupmb */

#endif /* end of include guard: PMB1D_CUH_JACGPBQT */
