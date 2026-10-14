/*
 * Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef PMB2D_CUH_MKPYXQV3
#define PMB2D_CUH_MKPYXQV3

#include "pmb.cuh"

/* The following code has been generated at runtime */
{{cupmb_func_definitions}}
{{cupmb_vertex_properties}}
/****************************************************/

namespace cupmb {

/* table that stores edges ids to be generated (up to nedges_to_generates)
 * Segments are oriented such that the negative region (s<=0) lies on their left: isolines are counterclockwise
 * around negative regions, the outward normal (dy,-dx) points towards positive values and the signed area
 * computed with the shoelace formula is the area of the negative region. */
__device__ __constant__ char4 pmb2d_case_table[16] = {
    {-1, -1, -1, -1},  // case 0
    {+0, +3, -1, -1},  // case 1
    {+1, +0, -1, -1},  // case 2
    {+1, +3, -1, -1},  // case 3
    {+2, +1, -1, -1},  // case 4
    {+0, +1, +2, +3},  // case 5
    {+2, +0, -1, -1},  // case 6
    {+2, +3, -1, -1},  // case 7
    {+3, +2, -1, -1},  // case 8
    {+0, +2, -1, -1},  // case 9
    {+3, +0, +1, +2},  // case 10
    {+1, +2, -1, -1},  // case 11
    {+3, +1, -1, -1},  // case 12
    {+0, +1, -1, -1},  // case 13
    {+3, +0, -1, -1},  // case 14
    {-1, -1, -1, -1},  // case 15
};


template <typename T, typename I, I Bx, I By, unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx*By)
__global__ void computeBlockMask2D(
        const T x0, const T y0,
        const T dx, const T dy,
        const I Nx, const I Ny, /* cells in the grid (number of points - 1 in each direction) */
        unsigned char *const __restrict block_mask,
        T *const __restrict output_scalar_field,
        F... fargs) {

    static_assert(is_same<T, float>::value || is_same<T, double>::value,
            "T has to be float or double.");
    static_assert(is_same<I, unsigned int>::value || is_same<I, unsigned long long>::value,
            "I has to be unsigned int or unsigned long long.");

    constexpr I B = Bx*By;
    constexpr I warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    const I Gx = number_of_blocks(Nx, Bx);

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
    const T v = (inside_domain ? cupmb::scalar<T,I,2>(pos, idx, fargs...) : quiet_nan<T>());

    if (inside_domain && (output_scalar_field != nullptr)) {
        /* x-fastest layout, 64 bit indexing as (Nx+1)*(Ny+1) may overflow I */
        using ull = unsigned long long;
        output_scalar_field[ull(iy)*ull(Nx+1) + ull(ix)] = v;
    }

    /* out of domain values are NaN (ignored), inside domain NaN values are considered positive (see mask_value) */
    compute_block_mask<T, I, warps_per_block, warpsize>((inside_domain ? mask_value(v) : v),
            bid, laneid, warpid, minbuf, maxbuf, block_mask);
}


template <typename T, typename I,
    I Nthreads,
    I Bx, I By, /* threads per logical block for simplex extraction (next step) */
    unsigned int warpsize=32, typename... F>
__launch_bounds__(Nthreads)
__global__ void compactActiveBlocksFromPredicate2D(
        const T x0, const T y0,
        const T dx, const T dy,
        const I Nx, const I Ny, /* cells in the grid (number of points - 1 in each direction) */
        const I lx, const I ly, /* current level of splitting */
        const I sx, const I sy, /* new splits to be generated, this is equal to (0,0) only at start_level */
        const I current_active_blocks, /* number of input blocks, or upper bound when input_count is not null */
        const I *const __restrict input_count, /* number of input blocks computed on the device (can be null) */
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
    const I input_blocks = (input_count == nullptr ? current_active_blocks : min(current_active_blocks, *input_count));
    if (I(blockIdx.x)*Nthreads >= input_blocks) {
        /* the whole thread block is idle */
        return;
    }
    const bool thread_is_idle = (tid >= input_blocks);

    constexpr I warps_per_block = Nthreads/warpsize;
    static_assert(Nthreads%warpsize==0, "B%warpsize != 0");
    static_assert(Nthreads/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    __shared__ I warp_offset[warps_per_block+1];

    /* number of final logical blocks in the grid */
    const I Gx = number_of_blocks(Nx, Bx);
    const I Gy = number_of_blocks(Ny, By);

    /* end level E corresponds to log2(H) with H = next_pow2(G) */
    const I Ex = log_of_next_pow2(Gx);
    const I Ey = log_of_next_pow2(Gy);

    /* level after this step, note that p = l at start_level because s=0 */
    const I px = lx+sx; assert(px <= Ex);
    const I py = ly+sy; assert(py <= Ey);

    const bool is_first_level = (sx == 0) && (sy == 0);
    const bool is_last_level  = (px == Ex) && (py == Ey);

    /* get current virtual block index */
    const I bid = (thread_is_idle ? I(0) : (is_first_level ? tid : input_block_indices[tid]));
    I bx, by;
    {
        const I Hx = I(1) << lx;
        by = bid / Hx;
        bx = bid % Hx;
    }

    /* number of final virtual blocks in the current encompassing grid and next one */
    const I Ix = I(1) << px;

    /* subblock size in number of cells (a subblock at level p contains 2**(E-p) final blocks per direction) */
    const I cells_x = (Bx-1) << (Ex-px);
    const I cells_y = (By-1) << (Ey-py);

    /* Iterate on subblocks. Emptiness is tested with integers and bounds are computed with the same
     * formula as sample points (x0 + ix*dx) to avoid rounding issues (duplicated or missing blocks).
     * Bounding box axes follow scalar(x,y) argument order: a[0]/b[0] = x, a[1]/b[1] = y. */
    BoundingBox<T,2> domain;
    #pragma unroll 2
    for(I ky=0; ky<(I(1)<<sy); ++ky) {
        const I nby = (by << sy) + ky;
        const I iay = nby*cells_y;
        domain.a[1] = y0 + min(iay,           Ny)*dy;
        domain.b[1] = y0 + min(iay + cells_y, Ny)*dy;
        #pragma unroll 2
        for(I kx=0; kx<(I(1)<<sx); ++kx) {
            const I nbx = (bx << sx) + kx;
            const I iax = nbx*cells_x;
            const I subblock_index = (is_last_level ? nby*Gx + nbx : nby*Ix + nbx);
            domain.a[0] = x0 + min(iax,           Nx)*dx;
            domain.b[0] = x0 + min(iax + cells_x, Nx)*dx;
            const bool empty = (iax >= Nx) || (iay >= Ny);
            const bool active = (thread_is_idle || empty ? false : may_contain_zeros<T, I, 2, F...>(domain, fargs...));
            const I offset = prefix_sum<I, warps_per_block, warpsize>(
                    I(active), laneid, warpid, warp_offset, active_blocks_count);
            if (active && offset < allocated_output_index_count) {
                output_block_indices[offset] = subblock_index;
            }
        }
    }
}


template <typename T, typename I, I Bx, I By,
    unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx*By)
__global__ void computeSimplices2D(
        const I allocated_vertex_count, const I allocated_edge_count,
        const T x0, const T y0,
        const T dx, const T dy,
        const I Nx, const I Ny, /* cells in the grid (number of points - 1 in each direction) */
        const I *const __restrict compacted_block_indices,
        const I *const __restrict active_block_count, /* number of active blocks computed on the device (can be null) */
        trove::array<T,2> *const __restrict output_vertices,
        trove::array<I,2> *const __restrict output_edges,
        I *const __restrict output_vertex_and_edge_count,
        const F... fargs) {

    static_assert(is_same<T,float>::value || is_same<T,double>::value,
            "T has to be float or double.");
    static_assert(is_same<I,unsigned int>::value || is_same<I,unsigned long long int>::value,
            "I has to be unsigned int or unsigned long long int.");
    using II = typename I2<I>::type;

    /* the grid may be larger than the number of active blocks when it is only known on the device */
    if ((active_block_count != nullptr) && (I(blockIdx.x) >= *active_block_count)) {
        return;
    }

    constexpr I B = Bx*By;
    constexpr I warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    const I Gx = number_of_blocks(Nx, Bx);

    /* shared memory is allocated to share data of various types accross threads
     *  1) B*sizeof(T) shared_scalars
     *  2) B*(sizeof(I)+sizeof(unsigned char)) shared_vertex_offsets and shared_vertex_count
     *  3) (warps_per_block+1)*sizeof(II) = 2*(Bx*By/warpsize)*sizeof(I) shared_offsets
     * User should ensure to allocate at least the max of those three values. */
    extern __shared__ __align__(16) unsigned char shared_mem_buffer[];

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

        /* the user function is never evaluated outside of [0,N] */
        s[0] = (inside_domain ? cupmb::scalar<T,I,2>(cpos, cidx, fargs...) : T(1));

        shared_scalars[tid] = s[0];
        __syncthreads();

        /* Out-of-block neighbors are not evaluated: block boundary threads never generate edges and vertex
         * generation only depends on edges 0 and 3 (s[1] and s[3]), which are masked by bx and by respectively.
         * Setting them to s[0] keeps the case id consistent. */
        s[1] = (bx ? s[0] : shared_scalars[tid+0*Bx+1]);
        s[2] = (bb ? s[0] : shared_scalars[tid+1*Bx+1]);
        s[3] = (by ? s[0] : shared_scalars[tid+1*Bx+0]);
        __syncthreads();

        #pragma unroll 4
        for (unsigned int k = 0u; k < 4u; ++k) {
            case_id += (s[k]<=T(0)) << k;
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
                /* NaN scalars are considered positive and never produce NaN vertices (see edge_interpolation_parameter) */
                vertex[0] = x + (k==0 ? edge_interpolation_parameter(s[0], s[1])*dx : T(0));
                vertex[1] = y + (k==1 ? edge_interpolation_parameter(s[0], s[3])*dy : T(0));
                assert(!isnan(vertex[0]));
                assert(!isnan(vertex[1]));
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
