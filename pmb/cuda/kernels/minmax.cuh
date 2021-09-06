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
__device__ __forceinline__ scalar(const T x) {
    return sin(4*x);
}

template <typename T, unsigned int Bx, unsigned int Nx, unsigned int warpsize=32>
__launch_bounds__(Bx)
__global__ void computeMinMax(const T x0, const T dx,
        T *const __restrict block_minval,
        T *const __restrict block_maxval) {

    static_assert(Bx%warpsize==0, "Bx%32 != 0");
    static_assert(Bx/warpsize<=warpsize, "Bx/warpsize > warpsize");

    constexpr unsigned int warps_per_block = Bx/warpsize;
    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int laneid = threadIdx.x % warpsize;
    const unsigned int warpid = threadIdx.x / warpsize;
    const unsigned int idx = blockIdx.x*(Bx-1) + threadIdx.x;
    const T x = x0 + idx*dx;
    const T v = (idx >= Nx ? NAN : scalar(x));
    {
        T minval(v), maxval(v);

        #pragma unroll
        for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
            minval = min(minval, __shfl_down_sync(0xffffffff, minval, offset));
            maxval = max(maxval, __shfl_down_sync(0xffffffff, maxval, offset));
        }

        if (laneId == 0) {
            minbuf[warpid] = minval;
            maxbuf[warpid] = maxval;
        }
        __syncthreads();

        if (warpid == 0) {
            minval = (laneid >= warps_per_block ? T(NAN) : minbuf[laneid]);
            maxval = (laneid >= warps_per_block ? T(NAN) : maxbuf[laneid]);

            #pragma unroll
            for (unsigned int offset=warpsize/2u; offset>0u; offset>>=1u) {
                minval = min(minval, __shfl_down_sync(0xffffffff, minval, offset));
                maxval = max(maxval, __shfl_down_sync(0xffffffff, maxval, offset));
            }

            if (laneid == 0) {
                block_minval[blockIdx.x] = minval;
                block_maxval[blockIdx.x] = maxval;
            }
        }
    }
}
