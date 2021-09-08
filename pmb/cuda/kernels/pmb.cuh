/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package pmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

template<typename T> constexpr
__device__ T const& constexpr_max(T const& a, T const& b) {
  return (a > b ? a : b);
}

__forceinline__ __device__ uint2 operator-(const uint2 &a, const uint2 &b) {
    return uint2{a.x-b.x, a.y-b.y};
}

__forceinline__ __device__ uint2& operator+=(uint2 &a, const uint2 &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__forceinline__ __device__ uint2 atomicAdd(uint2* address, uint2 val) {
    uint2 out;
    out.x = atomicAdd(reinterpret_cast<unsigned int*>(address)+0, val.x);
    out.y = atomicAdd(reinterpret_cast<unsigned int*>(address)+1, val.y);
    return out;
}

template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  typedef T value_type;
  typedef integral_constant<T,v> type;
  constexpr __device__ operator T() { return v; }
};

typedef integral_constant<bool,false> false_type;
typedef integral_constant<bool,true> true_type;

template<class T, class U>
struct is_same : false_type {};

template<class T>
struct is_same<T, T> : true_type {};

static_assert(sizeof(uint2)==sizeof(unsigned long long int), "sizeof(uint2) != sizeof(unsigned long long int)");

union Uint2 {
    uint2 u2;
    unsigned long long int ull;
};

template <typename T>
__forceinline__ __device__ T shfl_up_sync(unsigned int mask, T var, unsigned int delta) {
    if constexpr(is_same<T,uint2>::value) {
        Uint2 val;
        val.u2 = var;
        val.ull = __shfl_up_sync(mask, val.ull, delta);
        return val.u2;
    }
    else {
        return __shfl_up_sync(mask, var, delta);
    }
    return T{};
}


template <typename T, unsigned int warps_per_block, unsigned int warpsize>
__device__ T prefix_sum(T prefix, const unsigned int laneid, const unsigned int warpid,
        T* const __restrict volatile warp_offset, T* const __restrict volatile global_block_offset) {
    const T Z = {};

    /* intra warp prefix scan, all warp shuffle scan their data */
    T local_prefix(prefix);
    {
        #pragma unroll
        for (unsigned int offset=1u; offset<warpsize; offset<<=1u) {
            const T other_prefix = shfl_up_sync<T>(0xffffffff, local_prefix, offset);
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
            T warp_prefix = (laneid >= warps_per_block ? Z : warp_offset[laneid]);
            #pragma unroll
            for (unsigned int offset=1; offset<warpsize; offset<<=1u) {
                const T other_prefix = shfl_up_sync<T>(0xffffffff, warp_prefix, offset);
                if (laneid >= offset) {
                    warp_prefix += other_prefix;
                }
            }
            if (laneid < warps_per_block) {
                warp_offset[laneid] = warp_prefix;
            }
        }
        __syncthreads();
        local_prefix += (warpid==0 ? Z : warp_offset[warpid-1]);
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

template <typename T>
constexpr T pi = T(3.14159265358979323846);

#include "pmb1D.cuh"
#include "pmb2D.cuh"
