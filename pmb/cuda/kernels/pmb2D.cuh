/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package pmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

__constant__ char line_table[16][4] = {
    {-1, -1, -1, -1},
    {0, 3, -1, -1},
    {0, 1, -1, -1},
    {3, 1, -1, -1},
    {1, 2, -1, -1},
    {0, 1, 3, 2},
    {0, 2, -1, -1},
    {3, 2, -1, -1},
    {3, 2, -1, -1},
    {0, 2, -1, -1},
    {0, 3, 1, 2},
    {1, 2, -1, -1},
    {3, 1, -1, -1},
    {0, 1, -1, -1},
    {0, 3, -1, -1},
    {-1, -1, -1, -1},
};

/*__constant__ char4 line_table[16] = {*/
    /*char4{-1, -1, -1, -1},  */
    /*char4{0, 3, -1, -1},  */
    /*char4{0, 1, -1, -1},  */
    /*char4{3, 1, -1, -1},  */
    /*char4{1, 2, -1, -1},  */
    /*char4{0, 1, 3, 2},  */
    /*char4{0, 2, -1, -1},  */
    /*char4{3, 2, -1, -1},  */
    /*char4{3, 2, -1, -1},  */
    /*char4{0, 2, -1, -1},  */
    /*char4{0, 3, 1, 2},  */
    /*char4{1, 2, -1, -1},  */
    /*char4{3, 1, -1, -1},  */
    /*char4{0, 1, -1, -1},  */
    /*char4{0, 3, -1, -1},  */
    /*char4{-1, -1, -1, -1},  */
/*};*/

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
    constexpr unsigned int Gx = (Nx+Bx-2)/(Bx-1);
    constexpr unsigned int warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize");

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = threadIdx.y*Bx + threadIdx.x;
    const unsigned int bid = blockIdx.y*Gx + blockIdx.x;
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
        unsigned int *const __restrict output_vertex_and_index_count) {

    constexpr unsigned int B = Bx*By;
    constexpr unsigned int warps_per_block = B/warpsize;
    constexpr unsigned int Gx = (Nx+Bx-2)/(Bx-1);
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize");

    constexpr unsigned int shared_mem_bytes = constexpr_max(B*sizeof(T), (warps_per_block+1)*sizeof(unsigned int));
    __shared__ unsigned char shared_mem_buffer[shared_mem_bytes];

    T* const __restrict scalars = reinterpret_cast<T*>(shared_mem_buffer);
    unsigned int* const __restrict vertex_offsets = reinterpret_cast<unsigned int*>(shared_mem_buffer);

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tid = ty*Bx + tx;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const unsigned int bid = compacted_block_indices[blockIdx.x];
    const unsigned int bidx = bid%Gx;
    const unsigned int bidy = bid/Gx;
    const unsigned int xidx = bidx*(Bx-1) + tx;
    const unsigned int yidx = bidy*(By-1) + ty;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;

    /* load scalars */
    T s[4];
    {
        s[0] = ( ((xidx > Nx) || (yidx > Ny)) ? NaN : scalar(x,y) - isolevel);
        scalars[tid] = s[0];
        __syncthreads();
        s[1] = (tx == Bx-1 ? s[0] : scalars[tid+1]);
        s[2] = (ty == By-1 ? s[1] : scalars[tid+Bx+1]);
        s[3] = (ty == By-1 ? s[0] : scalars[tid+Bx+0]);
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

    /* compute local vertices (at most two out of four) */
    unsigned int nvertices = 0u;
    T vx[2], vy[2];
    {
        const char* edges = line_table[case_id];

        #pragma unroll
        for (unsigned int k = 0u; k<4u; ++k) {
            char i0 = edges[k];
            if (i0==-1) {
                break;
            }
            if ((i0==1)||(i0==2)) {
                continue;
            }
            /*i1 = (i0+1)%4;*/
            /*w = s[i0]/(s[i0]-s[i1]);*/
            /*vx[nvertices] = x + (i0==0)*w*dx;*/
            /*vy[nvertices] = y + (i0==3)*(T(1)-w)*dy;*/
            vx[nvertices] = x;// + (i0==0)*(s[0]/(s[0]-s[1]))*dx;
            vy[nvertices] = y;// + (i0==3)*(s[0]/(s[3]-s[0]))*dy;
            ++nvertices;
        }
    }

    /* prefix scan output vertex id */
    unsigned int vertex_id = prefix_sum<unsigned int, warps_per_block, warpsize>(
            nvertices, laneid, warpid, vertex_offsets, output_vertex_and_index_count);

    if (nvertices) {
        printf("bid=%u, bidx=%u, bidy=%u, tidx=%u, tidy=%u, x=%f, y=%f, case_id=%u, nvertices=%u, vertex_id=%u\n", bid, bidx, bidy, tx, ty, x, y, case_id, nvertices, vertex_id);
    }

    /* write vertices to global memory */
    #pragma unroll 2
    for (unsigned int k = 0u; (k < nvertices) && (vertex_id < allocated_vertex_count); ++k) {
        output_vertices[2*vertex_id+0] = x;
        output_vertices[2*vertex_id+1] = y;
        ++vertex_id;
    }
}
