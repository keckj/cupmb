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

/* The following code has been generated at runtime */
{{cupmb_func_definitions}}
{{cupmb_vertex_properties}}
/****************************************************/

namespace cupmb {

/* One-dimensional marching blocks: vertices are the zero crossings of the scalar field on the grid
 * edges [x_i, x_{i+1}] (linear interpolation) and each simplex is a single vertex (0-simplex), so that
 * vertices and simplices have shape (n,1) and simplex k references vertex k.
 * A grid edge belongs to exactly one logical block (the last thread of a block does not generate any
 * vertex), hence vertices are never duplicated on block boundaries.
 * 0-simplices carry no orientation: the normal (negative to positive scalar values) is the sign of
 * f(x_{i+1}) - f(x_i), it can be computed with a vertex property if required. */


template <typename T, typename I, I Bx, I Nx, unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx)
__global__ void computeBlockMask1D(
        const T x0, const T dx,
        unsigned char *const __restrict block_mask,
        T *const __restrict output_scalar_field,
        F... fargs) {

    static_assert(is_same<T, float>::value || is_same<T, double>::value,
            "T has to be float or double.");
    static_assert(is_same<I, unsigned int>::value || is_same<I, unsigned long long>::value,
            "I has to be unsigned int or unsigned long long.");

    constexpr I warps_per_block = Bx/warpsize;
    static_assert(Bx%warpsize==0, "Bx%warpsize != 0");
    static_assert(Bx/warpsize<=warpsize, "Bx/warpsize > warpsize (in practice Bx <= warpsize**2 = 32**2 = 1024)");

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const I bid = blockIdx.x;
    const trove::array<I,1> idx{bid*(Bx-1) + tid};
    const I& ix = trove::get<0>(idx);

    const trove::array<T,1> pos{x0 + ix*dx};

    const bool inside_domain = (ix <= Nx);
    const T v = (inside_domain ? cupmb::scalar<T,I,1,Nx>(pos, idx, fargs...) : quiet_nan<T>());

    if (inside_domain && (output_scalar_field != nullptr)) {
        /* 64 bit indexing for consistency with 2D and 3D */
        using ull = unsigned long long;
        output_scalar_field[ull(ix)] = v;
    }

    /* out of domain values are NaN (ignored), inside domain NaN values are considered positive (see mask_value) */
    compute_block_mask<T, I, warps_per_block, warpsize>((inside_domain ? mask_value(v) : v),
            bid, laneid, warpid, minbuf, maxbuf, block_mask);
}


template <typename T, typename I,
    I Nthreads,
    I Bx, /* threads per logical block for simplex extraction (next step) */
    I Nx, /* cells in the grid (corresponds to number of points - 1) */
    unsigned int warpsize=32, typename... F>
__launch_bounds__(Nthreads)
__global__ void compactActiveBlocksFromPredicate1D(
        const T x0,
        const T dx,
        const I lx, /* current level of splitting */
        const I sx, /* new splits to be generated, this is equal to 0 only at start_level */
        const I current_active_blocks,
        const I allocated_output_index_count,
        const I *const __restrict input_block_indices,
              I *const __restrict output_block_indices,
              I *const __restrict active_blocks_count,
        F... fargs) {

    static_assert(is_same<T, float>::value || is_same<T, double>::value,
            "T has to be float or double.");
    static_assert(is_same<I, unsigned int>::value || is_same<I, unsigned long long>::value,
            "I has to be unsigned int or unsigned long long.");

    const unsigned int laneid = threadIdx.x % warpsize;
    const unsigned int warpid = threadIdx.x / warpsize;
    const I tid = blockIdx.x*Nthreads + threadIdx.x;
    const bool thread_is_idle = (tid >= current_active_blocks);

    constexpr I warps_per_block = Nthreads/warpsize;
    static_assert(Nthreads%warpsize==0, "Nthreads%warpsize != 0");
    static_assert(Nthreads/warpsize<=warpsize, "Nthreads/warpsize > warpsize (in practice Nthreads <= warpsize**2 = 32**2 = 1024)");
    __shared__ I warp_offset[warps_per_block+1];

    /* number of final logical blocks in the grid */
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);

    /* end level E corresponds to log2(H) with H = next_pow2(G) */
    constexpr I Ex = log_of_next_pow2(Gx);

    /* level after this step, note that p = l at start_level because s=0 */
    const I px = lx+sx; assert(px <= Ex);

    const bool is_first_level = (sx == 0);

    /* get current virtual block index (binary tree node at level lx) */
    const I bx = (thread_is_idle ? I(0) : (is_first_level ? tid : input_block_indices[tid]));

    /* subblock size in number of cells (a subblock at level p contains 2**(E-p) final blocks) */
    const I cells_x = (Bx-1) << (Ex-px);

    /* Iterate on subblocks. Emptiness is tested with integers and bounds are computed with the same
     * formula as sample points (x0 + ix*dx) to avoid rounding issues (duplicated or missing blocks).
     * In 1D the subblock index is the same at intermediate and last levels (nbx < Gx at last level).
     * Bounding box axis follows scalar(x) argument: a[0]/b[0] = x. */
    BoundingBox<T,1> domain;
    #pragma unroll 2
    for(I kx=0; kx<(I(1)<<sx); ++kx) {
        const I nbx = (bx << sx) + kx;
        const I iax = nbx*cells_x;
        domain.a[0] = x0 + min(iax,           Nx)*dx;
        domain.b[0] = x0 + min(iax + cells_x, Nx)*dx;
        const bool empty = (iax >= Nx);
        const bool active = (thread_is_idle || empty ? false : may_contain_zeros<T, I, 1, F...>(domain, fargs...));
        const I offset = prefix_sum<I, warps_per_block, warpsize>(
                I(active), laneid, warpid, warp_offset, active_blocks_count);
        if (active && offset < allocated_output_index_count) {
            output_block_indices[offset] = nbx;
        }
    }
}


template <typename T, typename I, I Bx, I Nx,
    unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx)
__global__ void computeSimplices1D(
        const I allocated_vertex_count, const I allocated_simplex_count,
        const T x0,
        const T dx,
        const I *const __restrict compacted_block_indices,
        trove::array<T,1> *const __restrict output_vertices,
        trove::array<I,1> *const __restrict output_simplices,
        I *const __restrict output_vertex_and_simplex_count,
        const F... fargs) {

    static_assert(is_same<T,float>::value || is_same<T,double>::value,
            "T has to be float or double.");
    static_assert(is_same<I,unsigned int>::value || is_same<I,unsigned long long int>::value,
            "I has to be unsigned int or unsigned long long int.");
    using II = typename I2<I>::type;

    constexpr I warps_per_block = Bx/warpsize;
    static_assert(Bx%warpsize==0, "Bx%warpsize != 0");
    static_assert(Bx/warpsize<=warpsize, "Bx/warpsize > warpsize (in practice Bx <= warpsize**2 = 32**2 = 1024)");

    /* shared memory is allocated to share data of various types accross threads
     *  1) Bx*sizeof(T) shared_scalars
     *  2) (warps_per_block+1)*sizeof(II) = 2*(Bx/warpsize+1)*sizeof(I) shared_offsets
     * User should ensure to allocate at least the max of those two values. */
    extern __shared__ __align__(16) unsigned char shared_mem_buffer[];

    T  *const __restrict shared_scalars = reinterpret_cast<T*>(shared_mem_buffer);
    II *const __restrict shared_offsets = reinterpret_cast<II*>(shared_mem_buffer);

    const unsigned int tid = threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    /* compute grid point position */
    trove::array<I,1> idx;
    trove::array<T,1> pos;
    I &ix(trove::get<0>(idx));
    T  &x(trove::get<0>(pos));
    {
        const I bid = compacted_block_indices[blockIdx.x];
        ix = bid*(Bx-1) + tid;
        x = x0 + ix*dx;
    }

    /* domain boundary test */
    const bool nx = (ix == Nx);
    const bool inside_domain = (ix <= Nx);

    /* thread block boundary test */
    const bool bx = (tid == Bx-1);

    /* load scalars */
    T s[2];
    {
        const trove::array<I,1>& cidx = idx;
        const trove::array<T,1>& cpos = pos;

        /* the user function is never evaluated outside of [0,N] */
        s[0] = (inside_domain ? cupmb::scalar<T,I,1,Nx>(cpos, cidx, fargs...) : T(1));

        shared_scalars[tid] = s[0];
        __syncthreads();

        /* Out-of-block neighbors are not evaluated: block boundary threads never generate vertices. */
        s[1] = (bx ? s[0] : shared_scalars[tid+1]);
        __syncthreads();
    }

    /* vertex (and simplex) generation on edge [x_i, x_{i+1}], NaN scalars are considered positive */
    const bool genvertex = inside_domain && !bx && !nx && ((s[0]<=T(0)) != (s[1]<=T(0)));

    /* Prefix scan output vertex id, both vertex and simplex counters are incremented by the same amount.
     * The two global counters are updated by two independent atomic operations, so the simplex offset
     * of a block may differ from its vertex offset: the vertex offset is used for both outputs, simplex k
     * is stored at index k and references vertex k. */
    II offsets;
    const I& vertex_offset = offsets.x;
    {
        II icounts;
        icounts.x = I(genvertex);
        icounts.y = I(genvertex);
        offsets = prefix_sum<II, warps_per_block, warpsize>(
                icounts, laneid, warpid, shared_offsets,
                reinterpret_cast<II*>(output_vertex_and_simplex_count));
    }

    /* store vertex and simplex in global memory */
    if (genvertex) {
        if (vertex_offset < allocated_vertex_count) {
            /* NaN scalars never produce NaN vertices (see edge_interpolation_parameter) */
            const T vertex = x + edge_interpolation_parameter(s[0], s[1])*dx;
            assert(!isnan(vertex));
            output_vertices[vertex_offset] = trove::make_array<T>(vertex);
        }
        if (vertex_offset < allocated_simplex_count) {
            output_simplices[vertex_offset] = trove::make_array<I>(vertex_offset);
        }
    }
}

} /* end of namespace cupmb */

#endif /* end of include guard: PMB1D_CUH_JACGPBQT */
