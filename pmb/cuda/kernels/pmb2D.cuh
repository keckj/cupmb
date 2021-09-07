/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package pmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

template <typename T>
__device__ __forceinline__ T scalar(const T x, const T y) {
    return sin(3*pi<T>*x)*cos(5*pi<T>*y);
}

template <typename T, unsigned int Bx, unsigned int By, unsigned int Nx, unsigned int Ny, unsigned int warpsize=32>
__launch_bounds__(Bx*By)
__global__ void computeBlockMask2D(const T isolevel,
        const T x0, const T dx,
        const T y0, const T dy,
        unsigned char *const __restrict block_mask) {

    constexpr unsigned int B = Bx*By;
    constexpr unsigned int Mx = (Nx+Bx-2)/(Bx-1);
    constexpr unsigned int warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize");

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = threadIdx.y*Bx + threadIdx.x;
    const unsigned int bid = blockIdx.y*Mx + blockIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;
    const unsigned int xidx = blockIdx.x*(Bx-1) + threadIdx.x;
    const unsigned int yidx = blockIdx.y*(By-1) + threadIdx.y;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;
    const T v = ( ((xidx > Nx) || (yidx > Ny)) ? NaN : scalar(x,y) - isolevel);
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
            block_mask[bid] = (unsigned char)(block_is_active);
        }
    }
}


template <typename T, unsigned int Bx, unsigned int By, unsigned int Nx, unsigned int Ny, unsigned int warpsize=32>
__launch_bounds__(Bx*By)
__global__ void computeSimplices2D(unsigned int allocated_vertex_count, unsigned int allocated_index_count,
        const T isolevel, const T x0, const T dx, const T y0, const T dy,
        const unsigned int *const __restrict compacted_block_indices,
        T* const __restrict output_vertices,
        unsigned int *const __restrict output_indices,
        unsigned int *const __restrict output_vertex_and_index_count) {

    constexpr unsigned int B = Bx*By;
    constexpr unsigned int warps_per_block = B/warpsize;
    constexpr unsigned int Mx = (Nx+Bx-2)/(Bx-1);
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize");

    constexpr unsigned int shared_mem_bytes = constexpr_max(B*sizeof(T), (warps_per_block+1)*sizeof(unsigned int));
    __shared__ unsigned char shared_mem_buffer[shared_mem_bytes];

    T* const __restrict scalars = reinterpret_cast<T*>(shared_mem_buffer);
    unsigned int* const __restrict vertex_offsets = reinterpret_cast<unsigned int*>(shared_mem_buffer);

    const unsigned int tid = threadIdx.y*Bx + threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const unsigned int bid = compacted_block_indices[blockIdx.x];
    const unsigned int xidx = (bid%Mx)*(Bx-1) + threadIdx.x;
    const unsigned int yidx = (bid/Mx)*(By-1) + threadIdx.y;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;
}
