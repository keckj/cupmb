/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package pmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

#include "cuda_fp16.h"

template <typename T>
constexpr T pi = T(3.14159265358979323846);

template<typename T> constexpr
__device__ T const& constexpr_max(T const& a, T const& b) {
  return (a > b ? a : b);
}

template <typename T>
__device__ __forceinline__ T scalar(const T x) {
    return sin(153*pi<T>*x);
}

template <typename T, unsigned int warps_per_block, unsigned int warpsize>
__device__ T prefix_sum(T prefix, const unsigned int laneid, const unsigned int warpid,
        T* const __restrict volatile warp_offset, T* const __restrict volatile global_block_offset) {

    /* intra warp prefix scan, all warp shuffle scan their data */
    T local_prefix(prefix);
    {
        #pragma unroll
        for (unsigned int offset=1u; offset<warpsize; offset<<=1u) {
            const T other_prefix = __shfl_up_sync(0xffffffff, local_prefix, offset);
            if (laneid >= offset) {
                local_prefix += other_prefix;
            }
        }
        __syncthreads();
        if ((laneid == warpsize-1u) && (warpid<warps_per_block)) {
            warp_offset[warpid] = local_prefix;
        }
        __syncthreads();
    }

    /* intra block prefix scan into shared memory, only first warp participates */
    {
        if (warpid == 0) {
            T warp_prefix = (laneid >= warps_per_block ? 0u : warp_offset[laneid]);
            #pragma unroll
            for (unsigned int offset=1; offset<warpsize; offset<<=1u) {
                const T other_prefix = __shfl_up_sync(0xffffffff, warp_prefix, offset);
                if (laneid >= offset) {
                    warp_prefix += other_prefix;
                }
            }
            if (laneid < warps_per_block) {
                warp_offset[laneid] = warp_prefix;
            }
        }
        __syncthreads();
        local_prefix += (warpid==0 ? 0u : warp_offset[warpid-1]);
    }

    /* determine an inter block offset by using atomic operation in global memory (order does not matter) */
    {
        if ((warpid == 0) && (laneid == 0)) {
            warp_offset[warps_per_block] = atomicAdd(global_block_offset, warp_offset[warps_per_block-1]);
        }
        __syncthreads();
    }

    /* inter block prefix scan */
    local_prefix += warp_offset[warps_per_block];
    __syncthreads();

    return local_prefix - prefix;
}


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
        case 0: nvertices=0u; break;
        case 1: nvertices=1u; local_vertices[0] = x + (-s[0]/(-s[0]+s[1]))*dx; break;
        case 2: nvertices=1u; local_vertices[0] = x + (+s[0]/(+s[0]-s[1]))*dx; break;
        case 3: nvertices=0u; break;
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


template<unsigned int Tx, unsigned int Nblocks, unsigned int warpsize>
__launch_bounds__(Tx)
__global__ void countActiveBlocks(const unsigned char *const __restrict block_mask,
        unsigned int *const __restrict active_blocks_count) {
    static_assert(Tx%warpsize==0, "Tx%warpsize != 0");
    static_assert(Tx/warpsize<=warpsize, "Tx/warpsize > warpsize");
    constexpr unsigned int warps_per_block = Tx/warpsize;
    __shared__ unsigned int block_count[warps_per_block];

    const unsigned int laneid = threadIdx.x % warpsize;
    const unsigned int warpid = threadIdx.x / warpsize;
    const unsigned int block_index = blockIdx.x*blockDim.x + threadIdx.x;
    const bool block_is_active = (block_index >= Nblocks ? false : block_mask[block_index]);

    unsigned int active_block_count;

    /* intra warp reduction */
    {
        const unsigned int warp_sum = __popc(__ballot_sync(0xffffffff, block_is_active));
        if (laneid == 0) {
            block_count[warpid] = warp_sum;
        }
        __syncthreads();
    }

    /* intra block reduction and final atomic add */
    if (warpid == 0) {
        active_block_count = (laneid >= warps_per_block ? 0u : block_count[laneid]);

        #pragma unroll
        for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
            active_block_count += __shfl_down_sync(0xffffffff, active_block_count, offset);
        }
        if (laneid == 0) {
            atomicAdd(active_blocks_count, active_block_count);
        }
    }
}


template<unsigned int Tx, unsigned int Nblocks, unsigned int warpsize>
__launch_bounds__(Tx)
__global__ void compactActiveBlocks(unsigned int allocated_active_blocks,
        const unsigned char *const __restrict block_mask,
        unsigned int *const __restrict compacted_block_indices,
        unsigned int *const __restrict active_blocks_count) {
    static_assert(Tx%warpsize==0, "Tx%warpsize != 0");
    static_assert(Tx/warpsize<=warpsize, "Tx/warpsize > warpsize");
    constexpr unsigned int warps_per_block = Tx/warpsize;

    __shared__ unsigned int warp_offset[warps_per_block+1];

    const unsigned int laneid = threadIdx.x % warpsize;
    const unsigned int warpid = threadIdx.x / warpsize;
    const unsigned int block_index = blockIdx.x*blockDim.x + threadIdx.x;
    const bool block_is_active = (block_index >= Nblocks ? false : block_mask[block_index]);

    const unsigned int offset = prefix_sum<unsigned int, warps_per_block, warpsize>(
            block_is_active, laneid, warpid, warp_offset, active_blocks_count);
    if (block_is_active && (offset<allocated_active_blocks)) {
        compacted_block_indices[offset] = block_index;
    }
}
