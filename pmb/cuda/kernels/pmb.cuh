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
__forceinline__ __device__ ulonglong2 operator-(const ulonglong2 &a, const ulonglong2 &b) {
    return ulonglong2{a.x-b.x, a.y-b.y};
}


__forceinline__ __device__ uint2& operator+=(uint2 &a, const uint2 &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}
__forceinline__ __device__ ulonglong2& operator+=(ulonglong2 &a, const ulonglong2 &b) {
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
__forceinline__ __device__ ulonglong2 atomicAdd(ulonglong2* address, ulonglong2 val) {
    ulonglong2 out;
    out.x = atomicAdd(reinterpret_cast<unsigned long long int*>(address)+0, val.x);
    out.y = atomicAdd(reinterpret_cast<unsigned long long int*>(address)+1, val.y);
    return out;
}

template <typename I> struct I2 { using type = void; };
template <> struct I2<unsigned int> { using type = uint2; };
template <> struct I2<unsigned long long int> { using type = ulonglong2; };

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
    if constexpr(is_same<T,ulonglong2>::value) {
        var.x = __shfl_up_sync(mask, var.x, delta);
        var.y = __shfl_up_sync(mask, var.y, delta);
        return var;
    }
    else if constexpr(is_same<T,uint2>::value) {
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


template <typename I, unsigned int warps_per_block, unsigned int warpsize>
__device__ I prefix_sum(I prefix, const unsigned int laneid, const unsigned int warpid,
        I* const __restrict volatile warp_offset, I* const __restrict volatile global_block_offset) {
    const I Z = {};

    /* intra warp prefix scan, all warp shuffle scan their data */
    I local_prefix(prefix);
    {
        #pragma unroll
        for (unsigned int offset=1u; offset<warpsize; offset<<=1u) {
            const I other_prefix = shfl_up_sync<I>(0xffffffff, local_prefix, offset);
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
            I warp_prefix = (laneid >= warps_per_block ? Z : warp_offset[laneid]);
            #pragma unroll
            for (unsigned int offset=1; offset<warpsize; offset<<=1u) {
                const I other_prefix = shfl_up_sync<I>(0xffffffff, warp_prefix, offset);
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

template <typename T, typename I, unsigned int warps_per_block, unsigned int warpsize>
__device__ void compute_block_mask(const T local_value, const I bid,
        const unsigned int laneid, const unsigned int warpid,
        T *const __restrict minbuf, T *const __restrict maxbuf,
        unsigned char *const __restrict block_mask) {

    T minval(local_value);
    T maxval(local_value);

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
        minval = (laneid >= warps_per_block ? nan("") : minbuf[laneid]);
        maxval = (laneid >= warps_per_block ? nan("") : maxbuf[laneid]);

        #pragma unroll
        for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
            minval = min(minval, __shfl_down_sync(0xffffffff, minval, offset));
            maxval = max(maxval, __shfl_down_sync(0xffffffff, maxval, offset));
        }
        if (laneid == 0) {
            const bool block_is_active = (minval <= T(0)) && (maxval >= T(0)) && (minval < maxval);
            block_mask[bid] = (unsigned char)(block_is_active);
        }
    }
}


template<typename I, I Nblocks, I Tx, unsigned int warpsize>
__launch_bounds__(Tx)
__global__ void countActiveBlocks(
        const unsigned char *const __restrict block_mask,
                          I *const __restrict active_blocks_count) {
    static_assert(Tx%warpsize==0, "Tx%warpsize != 0");
    static_assert(Tx/warpsize<=warpsize, "Tx/warpsize > warpsize");
    constexpr unsigned int warps_per_block = Tx/warpsize;
    __shared__ I block_count[warps_per_block];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const I block_index = bid*Tx + tid;
    const bool block_is_active = (block_index >= Nblocks ? false : block_mask[block_index]);

    /* intra warp reduction */
    {
        const unsigned int warp_activity = __ballot_sync(0xffffffff, block_is_active);
        if (laneid == 0) {
            block_count[warpid] = __popc(warp_activity);
        }
        __syncthreads();
    }

    /* intra block reduction and final atomic add */
    if (warpid == 0) {
        I active_block_count = (laneid >= warps_per_block ? I(0) : block_count[laneid]);

        #pragma unroll
        for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
            active_block_count += __shfl_down_sync(0xffffffff, active_block_count, offset);
        }
        if (laneid == 0) {
            atomicAdd(active_blocks_count, active_block_count);
        }
    }
}


template<typename I, I Nblocks, I Tx, unsigned int warpsize>
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
    const I block_index = blockIdx.x*Tx + threadIdx.x;
    const bool block_is_active = (block_index >= Nblocks ? false : block_mask[block_index]);

    const unsigned int offset = prefix_sum<unsigned int, warps_per_block, warpsize>(
            block_is_active, laneid, warpid, warp_offset, active_blocks_count);
    if (block_is_active && (offset<allocated_active_blocks)) {
        compacted_block_indices[offset] = block_index;
    }
}

#include "pmb1D.cuh"
#include "pmb2D.cuh"
#include "pmb3D.cuh"
