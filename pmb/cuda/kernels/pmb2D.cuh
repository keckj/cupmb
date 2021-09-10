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
    return sin(2*pi<T>*x)*cos(2*pi<T>*y);
}


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


template <typename T, typename I, I Bx, I By, I Nx, I Ny, unsigned int warpsize=32>
__launch_bounds__(Bx*By)
__global__ void computeBlockMask2D(
        const T isolevel,
        const T x0, const T dx,
        const T y0, const T dy,
        unsigned char *const __restrict block_mask) {

    constexpr unsigned int B = Bx*By;
    constexpr unsigned int warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = threadIdx.y*Bx + threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const I bid = blockIdx.y*Gx + blockIdx.x;
    const I xidx = blockIdx.x*(Bx-1) + threadIdx.x;
    const I yidx = blockIdx.y*(By-1) + threadIdx.y;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;
    const T v = ( ((xidx > Nx) || (yidx > Ny)) ? NaN : scalar(x,y) - isolevel);

    compute_block_mask<T, I, warps_per_block, warpsize>(v, bid, laneid, warpid, minbuf, maxbuf, block_mask);
}


template <typename T, typename I, I Bx, I By, I Nx, I Ny, unsigned int warpsize=32, bool use_constant_memory=true>
__launch_bounds__(Bx*By)
__global__ void computeSimplices2D(
        const I allocated_vertex_count, const I allocated_edge_count,
        const T isolevel, const T x0, const T dx, const T y0, const T dy,
        const I *const __restrict compacted_block_indices,
              T *const __restrict output_vertices,
              I *const __restrict output_edges,
              I *const __restrict output_vertex_and_edge_count) {

    static_assert(is_same<T,float>::value || is_same<T,double>::value,
            "T has to be float or double.");
    static_assert(is_same<I,unsigned int>::value || is_same<I,unsigned long long int>::value,
            "I has to be unsigned int or unsigned long long int.");
    using II = typename I2<I>::type;

    constexpr unsigned int B = Bx*By;
    constexpr unsigned int warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);

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

    const bool bx = (tx == Bx-1);
    const bool by = (ty == By-1);
    const bool bb = bx || by;

    const I bid = compacted_block_indices[blockIdx.x];
    const I bidx = bid%Gx;
    const I bidy = bid/Gx;
    const I xidx = bidx*(Bx-1) + tx;
    const I yidx = bidy*(By-1) + ty;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;

    /* load scalars and compute case id */
    T s[4];
    bool b[4];
    unsigned char case_id = 0u;
    {
        s[0] = ( ((xidx > Nx) || (yidx > Ny)) ? NaN : scalar(x,y) - isolevel);
        shared_scalars[tid] = s[0];
        __syncthreads();
        s[1] = (bx ? NaN : shared_scalars[tid+0*Bx+1]);
        s[2] = (bb ? NaN : shared_scalars[tid+1*Bx+1]);
        s[3] = (by ? NaN : shared_scalars[tid+1*Bx+0]);
        __syncthreads();

        #pragma unroll 4
        for (unsigned int k = 0u; k < 4u; ++k) {
            b[k] = s[k]<=T(0);
            case_id += b[k] << k;
        }
    }

    /* load or compute edge table, vertex count, edge count and vertex generation */
    char4 edges{};
    const char *const __restrict edges_ptr = reinterpret_cast<const char*>(&edges);

    uchar2 genvertex = {0};
    const unsigned char *const __restrict genvertex_ptr = reinterpret_cast<const unsigned char*>(&genvertex);

    uchar2 counts = {0};
    unsigned char& nvertices = counts.x;
    unsigned char& nedges    = counts.y;

    if constexpr(use_constant_memory) {
        edges = pmb2d_case_table[case_id];
        #pragma unroll 4
        for (unsigned int k=0u; k<4u; k++) {
            if (edges_ptr[k] == -1) { break; }
            genvertex.x |= (edges_ptr[k]==0);
            genvertex.y |= (edges_ptr[k]==3);
            nedges += (k==0 || k==2);
        }
    }
    else {
        nedges = (((b[0] && b[2] && !b[1] && !b[3]) || (b[1] && b[3] && !b[0] && !b[2])) << 1) + ((b[0] && !b[2]) || (b[1] && !b[3]) || (b[2] && !b[0]) || (b[3] && !b[1]));
        genvertex.x = ((b[0] && !b[1]) || (b[1] && !b[0]));
        genvertex.y = ((b[0] && !b[3]) || (b[3] && !b[0]));
        edges.x = (((b[0] && b[1] && !b[3]) || (b[3] && !b[0] && !b[1])) << 1) + ((b[0] && b[1]) || (!b[0] && !b[1]));
        edges.y = (((b[1] && b[2]) || (b[3] && !b[2]) || (!b[1] && !b[2]) || (!b[0] && !b[1] && !b[3])) << 1) + ((b[2] && b[3]) || (!b[2] && !b[3]) || (b[0] && !b[1] && !b[3]) || (b[1] && !b[0] && !b[2]));
        edges.z = ((!b[3]) << 1) + 1;
        edges.w = 2;
    }
    nvertices = genvertex.x + genvertex.y;

    /* prefix scan output vertex id and output edge id */
    II offsets;
    I& vertex_offset = offsets.x;
    I&   edge_offset = offsets.y;
    {
        II icounts;
        icounts.x = counts.x;
        icounts.y = counts.y;
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
            if (genvertex_ptr[k]) {
                output_vertices[2*vertex_offset+0] = x + (k==0 ? (s[0]/(s[0]-s[1]))*dx : T(0));
                output_vertices[2*vertex_offset+1] = y + (k==1 ? (s[0]/(s[0]-s[3]))*dy : T(0));
                ++vertex_offset;
            }
        }
    }

    /* compute edge vertex indices (at most two edges, representing four indices) and store them to global memory */
    if (!bb) {
        nvertices = 0u;
        I vidx;
        #pragma unroll 2
        for (unsigned int k = 0u; (k<nedges) && (edge_offset < allocated_edge_count); ++k) {
            #pragma unroll 2
            for(unsigned int i = 0u; i<2; ++i) {
                const char e = edges_ptr[2*k+i];
                const unsigned int neighbor_tid = tid + (e==2)*Bx + (e==1);
                vidx = shared_vertex_offset[neighbor_tid] + (e==1 || e==3)*(shared_vertex_count[neighbor_tid]==2);
                output_edges[2*edge_offset+i] = min(vidx, allocated_vertex_count-1);
            }
            ++edge_offset;
        }
    }
}
