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
__device__ __forceinline__ T scalar(const T x, const T y, const T z) {
    return cos(2*pi<T>*x)*sin(2*pi<T>*y)*cos(2*pi<T>*z);
}


/* table that stores triangles ids to be generated */
/* each edge is reprented by 4 bit in a 64 bit ulonglong (only 60 bits are used) */
/* Value 15 (4 MSB) corresponds to the number of triangles (at most 5 triangles, or 15 vertex indices) */
__constant__ unsigned long long int pmb3d_case_table[256] = {
    0x0fffffffffffffffull, 0x1ffffffffffff380ull, 0x1ffffffffffff910ull, 0x2fffffffff189381ull, 0x1ffffffffffffa21ull, 0x2fffffffffa21380ull, 0x2fffffffff920a29ull, 0x3ffffff89a8a2382ull
    0x1ffffffffffff2b3ull, 0x2fffffffff0b82b0ull, 0x2fffffffffb32091ull, 0x3ffffffb89b912b1ull, 0x2fffffffff3ab1a3ull, 0x3ffffffab8a801a0ull, 0x3ffffff9ab9b3093ull, 0x2fffffffffb8aa89ull
    0x1ffffffffffff874ull, 0x2fffffffff437034ull, 0x2fffffffff748910ull, 0x3ffffff137174914ull, 0x2fffffffff748a21ull, 0x3ffffffa21403743ull, 0x3ffffff748209a29ull, 0x4fff4973727929a2ull
    0x2fffffffff2b3748ull, 0x3ffffff40242b74bull, 0x3ffffffb32748109ull, 0x4fff1292b9b49b74ull, 0x3ffffff487ab31a3ull, 0x4fff4b7401b41ab1ull, 0x4fff30bab9b09874ull, 0x3ffffffab99b4b74ull
    0x1ffffffffffff459ull, 0x2fffffffff380459ull, 0x2fffffffff051450ull, 0x3ffffff513538458ull, 0x2fffffffff459a21ull, 0x3ffffff594a21803ull, 0x3ffffff204245a25ull, 0x4fff8434535235a2ull
    0x2fffffffffb32459ull, 0x3ffffff594b802b0ull, 0x3ffffffb32510450ull, 0x4fff584b82852512ull, 0x3ffffff45931ab3aull, 0x4fffab81a8180594ull, 0x4fff30bab5b05045ull, 0x3ffffffb8aa85845ull
    0x2fffffffff975879ull, 0x3ffffff375359039ull, 0x3ffffff751710870ull, 0x2fffffffff753351ull, 0x3ffffff21a759879ull, 0x4fff37503505921aull, 0x4fff25a758528208ull, 0x3ffffff7533525a2ull
    0x3ffffff2b3987597ull, 0x4fffb72029279759ull, 0x4fff751871810b32ull, 0x3ffffff51771b12bull, 0x4fffb3a31a758859ull, 0x50aba010b7905075ull, 0x507570805a30b0abull, 0x2fffffffff5b75abull
    0x1ffffffffffff56aull, 0x2fffffffff6a5380ull, 0x2fffffffff6a5109ull, 0x3ffffff6a5891381ull, 0x2fffffffff162561ull, 0x3ffffff803621561ull, 0x3ffffff620609569ull, 0x4fff823625285895ull
    0x2fffffffff56ab32ull, 0x3ffffff56a02b80bull, 0x3ffffff6a5b32910ull, 0x4fffb892b92916a5ull, 0x3ffffff315356b36ull, 0x4fff6b51505b0b80ull, 0x4fff9505606306b3ull, 0x3ffffff89bb96956ull
    0x2fffffffff8746a5ull, 0x3ffffffa56374034ull, 0x3ffffff7486a5091ull, 0x4fff49737179156aull, 0x3ffffff874156216ull, 0x4fff743403625521ull, 0x4fff620560509748ull, 0x5962695923497937ull
    0x3ffffff56a4872b3ull, 0x4fffb720242746a5ull, 0x4fff6a5b32874910ull, 0x56a54b7b492b9129ull, 0x4fff6b51535b3748ull, 0x5b404b7b016b5b15ull, 0x574836b630560950ull, 0x4fff9b7974b96956ull
    0x2fffffffffa4694aull, 0x3ffffff380a946a4ull, 0x3ffffff04606a10aull, 0x4fffa16468618138ull, 0x3ffffff462421941ull, 0x4fff462942921803ull, 0x2fffffffff624420ull, 0x3ffffff624428238ull
    0x3ffffff32b46a94aull, 0x4fff6a4a94b82280ull, 0x4fffa164606102b3ull, 0x51b8b12184a16146ull, 0x4fff36b319639469ull, 0x514641916b0181b8ull, 0x3ffffff4600636b3ull, 0x2fffffffff86b846ull
    0x3ffffffa98a876a7ull, 0x4fffa76a907a0370ull, 0x4fff0818717a176aull, 0x3ffffff37117a76aull, 0x4fff768981861621ull, 0x5937390976192962ull, 0x3ffffff206607087ull, 0x2fffffffff276237ull
    0x4fff76898a86ab32ull, 0x57a9a76790b72702ull, 0x5b32a767a1871081ull, 0x4fff17616a71b12bull, 0x563136b619768698ull, 0x2fffffffff76b190ull, 0x4fff06b0b3607087ull, 0x1ffffffffffff6b7ull
    0x1ffffffffffffb67ull, 0x2fffffffff67b803ull, 0x2fffffffff67b910ull, 0x3ffffff67b138918ull, 0x2fffffffff7b621aull, 0x3ffffff7b6803a21ull, 0x3ffffff7b69a2092ull, 0x4fff89a38a3a27b6ull
    0x2fffffffff726327ull, 0x3ffffff026067807ull, 0x3ffffff910732672ull, 0x4fff678891681261ull, 0x3ffffff73171a67aull, 0x4fff801781a7167aull, 0x4fff7a69a0a70730ull, 0x3ffffff9a88a7a67ull
    0x2fffffffff68b486ull, 0x3ffffff640603b63ull, 0x3ffffff109648b68ull, 0x4fff63b139369649ull, 0x3ffffff1a28b6486ull, 0x4fff640b60b03a21ull, 0x4fff9a2920b648b4ull, 0x536463b34923a39aull
    0x3ffffff264248328ull, 0x2fffffffff264240ull, 0x4fff834642432091ull, 0x3ffffff642241491ull, 0x4fff1a6648168318ull, 0x3ffffff40660a01aull, 0x539a9303a6834364ull, 0x2fffffffff4a649aull
    0x2fffffffffb67594ull, 0x3ffffff67b594380ull, 0x3ffffffb67045105ull, 0x4fff51345343867bull, 0x3ffffffb6721a459ull, 0x4fff594380a217b6ull, 0x4fff204a24a45b67ull, 0x567b25a523453843ull
    0x3ffffff945267327ull, 0x4fff786260680459ull, 0x4fff045051673263ull, 0x5851584812786826ull, 0x4fff73167161a459ull, 0x5459078701671a61ull, 0x5a737a6a305a4a04ull, 0x4fffa84a458a7a67ull
    0x3ffffff98b9b6596ull, 0x4fff590650360b63ull, 0x4fffb65510b508b0ull, 0x3ffffff1355363b6ull, 0x4fff65b8b9b59a21ull, 0x5a21965690b603b0ull, 0x552025a50865b58bull, 0x4fff35a3a25363b6ull
    0x4fff283265825985ull, 0x3ffffff260069659ull, 0x5826283865081851ull, 0x2fffffffff612651ull, 0x5698965683a61631ull, 0x4fff06505960a01aull, 0x2fffffffffa65830ull, 0x1ffffffffffff65aull
    0x2fffffffffb57a5bull, 0x3ffffff03857ba5bull, 0x3ffffff091ba57b5ull, 0x4fff1381897ba57aull, 0x3ffffff15717b21bull, 0x4fffb27571721380ull, 0x4fff7b2209729579ull, 0x5289823295b27257ull
    0x3ffffff573532a52ull, 0x4fff52a578258028ull, 0x4fff2a37353a5109ull, 0x525752a278129289ull, 0x2fffffffff573531ull, 0x3ffffff571170780ull, 0x3ffffff735539309ull, 0x2fffffffff795789ull
    0x3ffffff8ba8a5485ull, 0x4fff03bba50b5405ull, 0x4fff54aba8a48910ull, 0x541314943b54a4baull, 0x4fff8548b2582152ull, 0x5b151b2b543b0b40ull, 0x558b8545b2950520ull, 0x2fffffffff3b2549ull
    0x4fff483543253a52ull, 0x3ffffff0244252a5ull, 0x5910854583a532a3ull, 0x4fff2492914252a5ull, 0x3ffffff153358548ull, 0x2fffffffff501540ull, 0x4fff530509358548ull, 0x1ffffffffffff549ull
    0x3ffffffba9b947b4ull, 0x4fffba97b9794380ull, 0x4fffb470414b1ba1ull, 0x54bab474a1843413ull, 0x4fff219b294b97b4ull, 0x53801b2b197b9479ull, 0x3ffffff04224b47bull, 0x4fff42343824b47bull
    0x4fff947732972a92ull, 0x570207872a4797a9ull, 0x5a040a1a472a3a73ull, 0x2fffffffff4782a1ull, 0x3ffffff317714194ull, 0x4fff178180714194ull, 0x2fffffffff347304ull, 0x1ffffffffffff784ull
    0x2fffffffff8ba8a9ull, 0x3ffffffa9bb93903ull, 0x3ffffffba88a0a10ull, 0x2fffffffffa3ba13ull, 0x3ffffff8b99b1b21ull, 0x4fff9b2921b93903ull, 0x2fffffffffb08b20ull, 0x1ffffffffffffb23ull
    0x3ffffff98aa82832ull, 0x2fffffffff2902a9ull, 0x4fff8a1810a82832ull, 0x1ffffffffffff2a1ull, 0x2fffffffff819831ull, 0x1ffffffffffff190ull, 0x1ffffffffffff830ull, 0x0fffffffffffffffull
};

class Triangle {
    public:
        __device__ __forceinline__ Triangle() : _value() {}
        __device__ __forceinline__ Triangle(unsigned long long int value): _value(value) {}
        __device__ __forceinline__ set_value(unsigned long long int value) { this->_value = value; }
        __device__ __forceinline__ unsigned char operator[](unsigned int i) const {
            return (unsigned char)( (this->_val >> (4*i)) & 0xff );
        }
    private:
        unsigned long long int _value;
}


template <typename T, typename I, I Bx, I By, I Bz, I Nx, I Ny, I Nz, unsigned int warpsize=32>
__launch_bounds__(Bx*By)
__global__ void computeBlockMask2D(
        const T isolevel,
        const T x0, const T dx,
        const T y0, const T dy,
        const T z0, const T dz,
        unsigned char *const __restrict block_mask) {

    constexpr unsigned int B = Bx*By*Bz;
    constexpr unsigned int warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);
    constexpr I Gy = (Ny+By-2)/(By-1);

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = (threadIdx.z*By + threadIdx.y)*Bx + threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const I bid = (blockIdx.z*Gy + blockIdx.y)*Gx + blockIdx.x;
    const I xidx = blockIdx.x*(Bx-1) + threadIdx.x;
    const I yidx = blockIdx.y*(By-1) + threadIdx.y;
    const I zidx = blockIdx.z*(Bz-1) + threadIdx.z;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;
    const T z = z0 + zidx*dz;
    const T v = (((xidx > Nx) || (yidx > Ny) || (zidx > Nz)) ? NaN : scalar(x,y,z) - isolevel);

    compute_block_mask<T, I, warps_per_block, warpsize>(v, bid, laneid, warpid, minbuf, maxbuf, block_mask);
}


template <typename T, typename I, I Bx, I By, I Bz, I Nx, I Ny, I Nz,
    unsigned int warpsize=32, bool use_constant_memory=true>
__launch_bounds__(Bx*By*Bz)
__global__ void computeSimplices3D(
        const I allocated_vertex_count, const I allocated_triangle_count,
        const T isolevel, const T x0, const T dx, const T y0, const T dy,
        const I *const __restrict compacted_block_indices,
              T *const __restrict output_vertices,
              I *const __restrict output_triangles,
              I *const __restrict output_vertex_and_triangle_count) {

    static_assert(is_same<T,float>::value || is_same<T,double>::value,
            "T has to be float or double.");
    static_assert(is_same<I,unsigned int>::value || is_same<I,unsigned long long int>::value,
            "I has to be unsigned int or unsigned long long int.");
    using II = typename I2<I>::type;

    constexpr unsigned int B = Bx*By*Bz;
    constexpr unsigned int warps_per_block = B/warpsize;
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);
    constexpr I Gy = (Ny+By-2)/(By-1);

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
    const unsigned int tz = threadIdx.y;
    const unsigned int tid = (tz*By+ty)*Bx + tx;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const bool bxx = (tx == Bx-1);
    const bool byy = (ty == By-1);
    const bool bzz = (tz == Bz-1);
    const bool bxy = bxx || byy;
    const bool bxz = bxx || bzz;
    const bool byz = byy || bzz;
    const bool bbb = bxx || byy || bzz;

    const I bid = compacted_block_indices[blockIdx.x];
    const I bidx = bid % Gx;
    const I bidy = (bid % (Gx*Gy))/Gx;
    const I bidz = bid/(Gx*Gz);
    const I xidx = bidx*(Bx-1) + tx;
    const I yidx = bidy*(By-1) + ty;
    const I zidx = bidz*(Bz-1) + tz;

    const T NaN(nan(""));
    const T x = x0 + xidx*dx;
    const T y = y0 + yidx*dy;
    const T z = z0 + zidx*dz;

    /* load scalars and compute case id */
    T s[8];
    bool b[8];
    unsigned char case_id = 0u;
    {
        s[0] = ( ((xidx > Nx) || (yidx > Ny) || (zidx > Nz)) ? NaN : scalar(x,y,z) - isolevel);

        shared_scalars[tid] = s[0];
        __syncthreads();

        s[1] = (bxx ? NaN : shared_scalars[tid+0*Bx*By+0*Bx+1]);
        s[2] = (bxy ? NaN : shared_scalars[tid+0*Bx*By+1*Bx+1]);
        s[3] = (byy ? NaN : shared_scalars[tid+0*Bx*By+1*Bx+0]);

        s[4] = (bzz ? NaN : shared_scalars[tid+1*Bx*By+0*Bx+0]);
        s[5] = (bxz ? NaN : shared_scalars[tid+1*Bx*By+0*Bx+1]);
        s[6] = (bbb ? NaN : shared_scalars[tid+1*Bx*By+1*Bx+1]);
        s[7] = (byz ? NaN : shared_scalars[tid+1*Bx*By+1*Bx+0]);
        __syncthreads();

        #pragma unroll 8
        for (unsigned int k = 0u; k < 8u; ++k) {
            b[k] = s[k]<=T(0);
            case_id += b[k] << k;
        }
    }

    /* load or compute edge table, vertex count, edge count and vertex generation */
    Triangle triangles;
    uchar3 genvertex = {0};
    const unsigned char *const __restrict genvertex_ptr = reinterpret_cast<const unsigned char*>(&genvertex);

    uchar2 counts = {0};
    unsigned char& nvertices  = counts.x;
    unsigned char& ntriangles = counts.y;

    if constexpr(use_constant_memory) {
        triangles.set_value(pmb3d_case_table[case_id]);
    }
    else {
        triangles.set_value(0ull);
    }

    ntriangles = triangles[15];
    for (unsigned int k=0u; k<3*ntriangles; k++) {
        genvertex.x |= (triangles_ptr[k]==0);
        genvertex.y |= (triangles_ptr[k]==3);
        genvertex.z |= (triangles_ptr[k]==8);
    }
    nvertices = genvertex.x + genvertex.y + genvertex.z;

    /* prefix scan output vertex id and output edge id */
    II offsets;
    I& vertex_offset = offsets.x;
    I&   triangle_offset = offsets.y;
    {
        II icounts;
        icounts.x = counts.x;
        icounts.y = counts.y;
        offsets = prefix_sum<II, warps_per_block, warpsize>(
                icounts, laneid, warpid, shared_offsets,
                reinterpret_cast<II*>(output_vertex_and_triangle_count));
        shared_vertex_offset[tid] = vertex_offset;
        shared_vertex_count[tid] = nvertices;
        __syncthreads();
    }

    /* compute thread-local vertices (at most three out of twelve) and store them in global memory */
    {
        #pragma unroll 3
        for (unsigned int k = 0u; (k<3u) && (vertex_offset < allocated_vertex_count); ++k) {
            if (genvertex_ptr[k]) {
                output_vertices[2*vertex_offset+0] = x + (k==0 ? (s[0]/(s[0]-s[1]))*dx : T(0));
                output_vertices[2*vertex_offset+1] = y + (k==1 ? (s[0]/(s[0]-s[3]))*dy : T(0));
                output_vertices[2*vertex_offset+2] = z + (k==2 ? (s[0]/(s[0]-s[4]))*dz : T(0));
                ++vertex_offset;
            }
        }
    }

    /* compute edge vertex indices (at most two triangles, representing four indices) and store them to global memory */
    if (!bbb) {
        I vidx;
        #pragma unroll 5
        for (unsigned int k = 0u; (k<ntriangles) && (triangle_offset < allocated_triangle_count); ++k) {
            #pragma unroll 3
            for(unsigned int i = 0u; i<3; ++i) {
                const unsigned char e = triangles[3*k+i];
                const unsigned char oz = (4 <= e <= 7);
                const unsigned char oy = (4 <= e <= 7);

                const unsigned int neighbor_tid = tid + (oz*By+oy)*Bx+ox;
                vidx = shared_vertex_offset[neighbor_tid] + (e==1 || e==3)*(shared_vertex_count[neighbor_tid]==2);
                output_triangles[2*triangle_offset+i] = min(vidx, allocated_vertex_count-1);
            }
            ++triangle_offset;
        }
    }
}
