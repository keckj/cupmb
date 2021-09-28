/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package cupmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

#ifndef PMB2D_CUH_MKPYXQV3
#define PMB2D_CUH_MKPYXQV3

#include "pmb.cuh"

namespace cupmb {

/* table that stores edges ids to be generated (up to nedges_to_generates) */
__constant__ char4 pmb2d_case_table[16] = {
    {-1, -1, -1, -1},  // case 0
    {+0, +3, -1, -1},  // case 1
    {+0, +1, -1, -1},  // case 2
    {+3, +1, -1, -1},  // case 3
    {+1, +2, -1, -1},  // case 4
    {+0, +1, +3, +2},  // case 5
    {+0, +2, -1, -1},  // case 6
    {+3, +2, -1, -1},  // case 7
    {+3, +2, -1, -1},  // case 8
    {+0, +2, -1, -1},  // case 9
    {+0, +3, +1, +2},  // case 10
    {+1, +2, -1, -1},  // case 11
    {+3, +1, -1, -1},  // case 12
    {+0, +1, -1, -1},  // case 13
    {+0, +3, -1, -1},  // case 14
    {-1, -1, -1, -1},  // case 15
};


template <typename T, typename I, I Bx, I By, I Nx, I Ny, unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx*By)
__global__ void computeBlockMask2D(
        const T isolevel,
        const T x0, const T y0,
        const T dx, const T dy,
        unsigned char *const __restrict block_mask,
        T *const __restrict output_scalar_field,
        F... fargs) {

    static_assert(is_same<T, float>::value || is_same<T, double>::value,
            "T has to be float or double.");
    static_assert(is_same<I, unsigned int>::value || is_same<I, unsigned long long>::value,
            "I has to be unsigned int or unsigned long long.");

    constexpr I B = Bx*By;
    constexpr I warps_per_block = B/warpsize;
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = threadIdx.y*Bx + threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const I bid = blockIdx.y*Gx + blockIdx.x;
    const trove::array<I,2> idx{blockIdx.x*(Bx-1) + threadIdx.x,
                                blockIdx.y*(By-1) + threadIdx.y};
    const I& ix = trove::get<0>(idx);
    const I& iy = trove::get<1>(idx);

    const trove::array<T,2> pos{x0 + ix*dx,
                                y0 + iy*dy};

    const bool inside_domain = (ix <= Nx) && (iy <= Ny);
    const T v = (inside_domain ? cupmb::scalar<T,I,2,Nx,Ny>(pos, idx, fargs...) : nan(""));

    if (inside_domain && (output_scalar_field != nullptr)) {
        output_scalar_field[iy*(Nx+1)+ix] = v;
    }

    compute_block_mask<T, I, warps_per_block, warpsize>(v - isolevel, bid, laneid, warpid, minbuf, maxbuf, block_mask);
}


template <typename T, typename I, I Bx, I By, I Nx, I Ny,
    unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx*By)
__global__ void computeSimplices2D(
        const I allocated_vertex_count, const I allocated_edge_count,
        const T isolevel,
        const T x0, const T dx,
        const T y0, const T dy,
        const I *const __restrict compacted_block_indices,
        trove::array<T,2> *const __restrict output_vertices,
        trove::array<I,2> *const __restrict output_edges,
        I *const __restrict output_vertex_and_edge_count,
        const F... fargs) {

    static_assert(is_same<T,float>::value || is_same<T,double>::value,
            "T has to be float or double.");
    static_assert(is_same<I,unsigned int>::value || is_same<I,unsigned long long int>::value,
            "I has to be unsigned int or unsigned long long int.");
    using II = typename I2<I>::type;

    constexpr I B = Bx*By;
    constexpr I warps_per_block = B/warpsize;
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");

    /* shared memory is allocated to share data of various types accross threads
     *  1) B*sizeof(T) shared_scalars
     *  2) B*(sizeof(I)+sizeof(unsigned char)) shared_vertex_offsets and shared_vertex_count
     *  3) (warps_per_block+1)*sizeof(II) = 2*(Bx*By/warpsize)*sizeof(I) shared_offsets
     * User should ensure to allocate at least the max of those three values. */
    extern __shared__ unsigned char shared_mem_buffer[];

    T  *const __restrict shared_scalars = reinterpret_cast<T*>(shared_mem_buffer);
    II *const __restrict shared_offsets = reinterpret_cast<II*>(shared_mem_buffer);
    I  *const __restrict shared_vertex_offset = reinterpret_cast<I*>(shared_mem_buffer);
    unsigned char* const __restrict shared_vertex_count = reinterpret_cast<unsigned char*>(shared_vertex_offset + B);

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tid = ty*Bx + tx;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    /* compute vertex position */
    trove::array<I,2> idx;
    trove::array<T,2> pos;
    I &ix(trove::get<0>(idx)), &iy(trove::get<1>(idx));
    T  &x(trove::get<0>(pos)),  &y(trove::get<1>(pos));
    {
        I bid = compacted_block_indices[blockIdx.x];

        const I bidx = bid%Gx;
        const I bidy = bid/Gx;

        ix = bidx*(Bx-1) + tx;
        iy = bidy*(By-1) + ty;

        x = x0 + ix*dx;
        y = y0 + iy*dy;
    }

    /* domain boundary test */
    const bool nx = (ix == Nx);
    const bool ny = (iy == Ny);
    const bool nn = nx || ny;
    const bool inside_domain = (ix <= Nx) && (iy <= Ny);

    /* thread block boundary test */
    const bool bx = (tx == Bx-1);
    const bool by = (ty == By-1);
    const bool bb = bx || by;

    /* load scalars and compute case id */
    T s[4];
    unsigned char case_id = 0u;
    {
        const trove::array<I,2>& cidx = idx;
        const trove::array<T,2>& cpos = pos;

        s[0] = cupmb::scalar<T,I,2,Nx,Ny>(cpos, cidx, fargs...) - isolevel;

        shared_scalars[tid] = s[0];
        __syncthreads();

        /* Thread divergence at block boundaries here: caseid at block boundary is only required to determine the number
         * of vertices to generate. This information could be deduced directly from lower x/y/z-threads neighbors caseid. */
        x += dx; ix += 1; s[1] = (bx ? cupmb::scalar<T,I,2,Nx,Ny>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+0*Bx+1]);
        y += dy; iy += 1; s[2] = (bb ? cupmb::scalar<T,I,2,Nx,Ny>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+1*Bx+1]);
        x -= dx; ix -= 1; s[3] = (by ? cupmb::scalar<T,I,2,Nx,Ny>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+1*Bx+0]);
        y -= dy; iy -= 1;
        __syncthreads();

        #pragma unroll 4
        for (unsigned int k = 0u; k < 4u; ++k) {
            case_id += (s[k]<T(0)) << k;
        }
    }

    /* load or compute edge table, vertex count, edge count and vertex generation */
    char4 edges = pmb2d_case_table[case_id];
    const char *const __restrict edges_ptr = reinterpret_cast<const char*>(&edges);

    unsigned char nedges(0), nvertices(0);
    bool genvertex[2] = {false, false};

    #pragma unroll 4
    for (unsigned int k=0u; k<4u; k++) {
        if (edges_ptr[k] == -1) { break; }
        genvertex[0] |= (edges_ptr[k]==0);
        genvertex[1] |= (edges_ptr[k]==3);
        nedges += (k==0 || k==2);
    }
    const bool can_generate_edges = inside_domain && !bb && !nn;
    nedges = (can_generate_edges ? nedges : 0);

    genvertex[0] &= inside_domain && !bx && !nx;
    genvertex[1] &= inside_domain && !by && !ny;
    nvertices = genvertex[0] + genvertex[1];

    /* prefix scan output vertex id and output edge id */
    II offsets;
    I& vertex_offset = offsets.x;
    I&   edge_offset = offsets.y;
    {
        II icounts;
        icounts.x = nvertices;
        icounts.y = nedges;
        offsets = prefix_sum<II, warps_per_block, warpsize>(
                icounts, laneid, warpid, shared_offsets,
                reinterpret_cast<II*>(output_vertex_and_edge_count));
        shared_vertex_offset[tid] = vertex_offset;
        shared_vertex_count[tid] = nvertices;
        __syncthreads();
    }

    /* compute thread-local vertices (at most two out of four) and store them in global memory */
    {
        #pragma unroll 2
        for (unsigned int k = 0u; (k<2u) && (vertex_offset < allocated_vertex_count); ++k) {
            T vertex[2];
            if (genvertex[k]) {
                vertex[0] = x + (k==0 ? (s[0]/(s[0]-s[1]))*dx : T(0));
                vertex[1] = y + (k==1 ? (s[0]/(s[0]-s[3]))*dy : T(0));
                output_vertices[vertex_offset] = trove::make_array<T>(vertex[0], vertex[1]);
                ++vertex_offset;
            }
        }
    }

    /* compute edge vertex indices (at most two edges, representing four indices) and store them to global memory */
    I vidx;
    #pragma unroll 2
    for (unsigned int k = 0u; (k<nedges) && (edge_offset < allocated_edge_count); ++k) {
        I simplex[2];
        #pragma unroll 2
        for(unsigned int i = 0u; i<2; ++i) {
            const char e = edges_ptr[2*k+i];
            const int neighbor_tid = tid + (e==2)*Bx + (e==1);
            vidx = shared_vertex_offset[neighbor_tid] + (e==1 || e==3)*(shared_vertex_count[neighbor_tid]==2);
            simplex[i] = min(vidx, allocated_vertex_count-1);
        }
        output_edges[edge_offset] = trove::make_array<I>(simplex[0], simplex[1]);
        ++edge_offset;
    }
}

} /* end of namespace cupmb */

#endif /* end of include guard: PMB2D_CUH_MKPYXQV3 */
