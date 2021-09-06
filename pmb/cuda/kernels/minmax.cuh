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

template <typename T>
__device__ __forceinline__ T scalar(const T x) {
    return sin(11*pi<T>*x);
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
__global__ void computeSimplices1D(const T isolevel, const T x0, const T dx,
        unsigned char *const __restrict compacted_block_indices) {
    static_assert(Bx%warpsize==0, "Bx%warpsize != 0");
    static_assert(Bx/warpsize<=warpsize, "Bx/warpsize > warpsize");

    constexpr unsigned int warps_per_block = Bx/warpsize;
    __shared__ T scalars[Bx];

    const unsigned int tid = threadIdx.x
    const unsigned int bid = compacted_block_indices[blockIdx.x];
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;
    const unsigned int idx = bid*(Bx-1) + tid;

    const T NaN(nan(""));
    const T x = x0 + idx*dx;
    T S0, S1;

    /* load scalars */
    {
        const T s0 = (idx > Nx ? NaN : scalar(x) - isolevel);
        scalars[tid] = s0;
        __syncthreads();

        const T s1 = scalars[tid+1];
        S0 = min(s0, s1);
        S1 = max(s0, s1);
    }
    bool generate_vertex = (S0<=T(0)) && (S1>=T(0)) && (S0 < S1);
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
__global__ void compactActiveBlocks(const unsigned char *const __restrict block_mask,
        unsigned int *const __restrict compacted_block_indices,
        unsigned int *const __restrict active_blocks_count) {
    static_assert(Tx%warpsize==0, "Tx%warpsize != 0");
    static_assert(Tx/warpsize<=warpsize, "Tx/warpsize > warpsize");
    constexpr unsigned int warps_per_block = Tx/warpsize;

    __shared__ unsigned int warp_offset[warps_per_block];
    __shared__ unsigned int block_offset[1];

    const unsigned int laneid = threadIdx.x % warpsize;
    const unsigned int warpid = threadIdx.x / warpsize;
    const unsigned int block_index = blockIdx.x*blockDim.x + threadIdx.x;
    const bool block_is_active = (block_index >= Nblocks ? false : block_mask[block_index]);

    /* intra warp prefix scan */
    unsigned int local_prefix(block_is_active);
    {
        #pragma unroll
        for (unsigned int offset=1u; offset<warpsize; offset<<=1u) {
            const unsigned int other_prefix = __shfl_up_sync(0xffffffff, local_prefix, offset);
            if (laneid >= offset) {
                local_prefix += other_prefix;
            }
        }
        if (laneid == warpsize-1u) {
            warp_offset[warpid] = local_prefix;
        }
        __syncthreads();
    }

    /* intra block prefix scan */
    {
        if (warpid == 0) {
            unsigned int warp_prefix = (laneid >= warps_per_block ? 0u : warp_offset[laneid]);
            #pragma unroll
            for (unsigned int offset=1; offset<warpsize; offset<<=1u) {
                const unsigned int other_prefix = __shfl_up_sync(0xffffffff, warp_prefix, offset);
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

    /* determine an inter block offset (order does not matter) */
    {
        if ((warpid == 0) && (laneid == 0)) {
            block_offset[0] = atomicAdd(active_blocks_count, warp_offset[warps_per_block-1]);
        }
        __syncthreads();
    }

    /* inter block prefix scan */
    if (block_is_active) {
        local_prefix += block_offset[0];
        compacted_block_indices[local_prefix-1] = block_index;
    }
}
