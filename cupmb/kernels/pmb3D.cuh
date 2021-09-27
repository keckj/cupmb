/*
 * Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
 *
 * Project CUPMB and package cupmb are proprietary and confidential
 * and can not be copied, reproduced, modified and/or distributed without the express permission
 * of all copyright holders (Jean-Baptiste Keck).
 *
 * Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
 */

#include "pmb.cuh"

namespace cupmb {

/* Table that stores vertices and triangles to be generated (at most 5 triangles -- 15 vertex indices/case).  */
/* Each of the 12 cube edge is reprented by 4 bit in a 64 bit ulonglong (60 bits total, from bit 4 to bit 63) */
/* Cube edges are indexed from 0 to 11 included, missing/unused edges are represented by the value 15 = 0xf.  */
/* Bit 0 corresponds to generate vertex on x edge, bit 1 to generate vertex on y edge and bit 2 to generate   */
/* vertex on z edge. Bit 3 is not used and is set to zero. The number of vertices to generate locally can be  */
/* deduced by computing the sum of vertex generation bits. The number of triangles to generate can be deduced */
/* efficiently by counting the number of leading (most significant) bits that are set to 1, divided by 12.    */
__constant__ unsigned long long pmb3d_case_table[256] = {
    0xfffffffffffffff0ull, 0xffffffffffff3807ull, 0xffffffffffff9101ull, 0xfffffffff1893816ull, 0xffffffffffffa210ull, 0xfffffffffa213807ull, 0xfffffffff920a291ull, 0xffffff89a8a23826ull,
    0xffffffffffff2b32ull, 0xfffffffff0b82b05ull, 0xfffffffffb320913ull, 0xffffffb89b912b14ull, 0xfffffffff3ab1a32ull, 0xffffffab8a801a05ull, 0xffffff9ab9b30933ull, 0xfffffffffb8aa894ull,
    0xffffffffffff8744ull, 0xfffffffff4370343ull, 0xfffffffff7489105ull, 0xffffff1371749142ull, 0xfffffffff748a214ull, 0xffffffa214037433ull, 0xffffff748209a295ull, 0xfff4973727929a22ull,
    0xfffffffff2b37486ull, 0xffffff40242b74b1ull, 0xffffffb327481097ull, 0xfff1292b9b49b740ull, 0xffffff487ab31a36ull, 0xfff4b7401b41ab11ull, 0xfff30bab9b098747ull, 0xffffffab99b4b740ull,
    0xffffffffffff4590ull, 0xfffffffff3804597ull, 0xfffffffff0514501ull, 0xffffff5135384586ull, 0xfffffffff459a210ull, 0xffffff594a218037ull, 0xffffff204245a251ull, 0xfff8434535235a26ull,
    0xfffffffffb324592ull, 0xffffff594b802b05ull, 0xffffffb325104503ull, 0xfff584b828525124ull, 0xffffff45931ab3a2ull, 0xfffab81a81805945ull, 0xfff30bab5b050453ull, 0xffffffb8aa858454ull,
    0xfffffffff9758794ull, 0xffffff3753590393ull, 0xffffff7517108705ull, 0xfffffffff7533512ull, 0xffffff21a7598794ull, 0xfff37503505921a3ull, 0xfff25a7585282085ull, 0xffffff7533525a22ull,
    0xffffff2b39875976ull, 0xfffb720292797591ull, 0xfff751871810b327ull, 0xffffff51771b12b0ull, 0xfffb3a31a7588596ull, 0x0aba010b79050751ull, 0x07570805a30b0ab7ull, 0xfffffffff5b75ab0ull,
    0xffffffffffff56a0ull, 0xfffffffff6a53807ull, 0xfffffffff6a51091ull, 0xffffff6a58913816ull, 0xfffffffff1625610ull, 0xffffff8036215617ull, 0xffffff6206095691ull, 0xfff8236252858956ull,
    0xfffffffff56ab322ull, 0xffffff56a02b80b5ull, 0xffffff6a5b329103ull, 0xfffb892b92916a54ull, 0xffffff315356b362ull, 0xfff6b51505b0b805ull, 0xfff9505606306b33ull, 0xffffff89bb969564ull,
    0xfffffffff8746a54ull, 0xffffffa563740343ull, 0xffffff7486a50915ull, 0xfff49737179156a2ull, 0xffffff8741562164ull, 0xfff7434036255213ull, 0xfff6205605097485ull, 0x9626959234979372ull,
    0xffffff56a4872b36ull, 0xfffb720242746a51ull, 0xfff6a5b328749107ull, 0x6a54b7b492b91290ull, 0xfff6b51535b37486ull, 0xb404b7b016b5b151ull, 0x74836b6305609507ull, 0xfff9b7974b969560ull,
    0xfffffffffa4694a0ull, 0xffffff380a946a47ull, 0xffffff04606a10a1ull, 0xfffa164686181386ull, 0xffffff4624219410ull, 0xfff4629429218037ull, 0xfffffffff6244201ull, 0xffffff6244282386ull,
    0xffffff32b46a94a2ull, 0xfff6a4a94b822805ull, 0xfffa164606102b33ull, 0x1b8b12184a161464ull, 0xfff36b3196394692ull, 0x14641916b0181b85ull, 0xffffff4600636b33ull, 0xfffffffff86b8464ull,
    0xffffffa98a876a74ull, 0xfffa76a907a03703ull, 0xfff0818717a176a5ull, 0xffffff37117a76a2ull, 0xfff7689818616214ull, 0x9373909761929623ull, 0xffffff2066070875ull, 0xfffffffff2762372ull,
    0xfff76898a86ab326ull, 0x7a9a76790b727021ull, 0xb32a767a18710817ull, 0xfff17616a71b12b0ull, 0x63136b6197686986ull, 0xfffffffff76b1901ull, 0xfff06b0b36070877ull, 0xffffffffffff6b70ull,
    0xffffffffffffb670ull, 0xfffffffff67b8037ull, 0xfffffffff67b9101ull, 0xffffff67b1389186ull, 0xfffffffff7b621a0ull, 0xffffff7b6803a217ull, 0xffffff7b69a20921ull, 0xfff89a38a3a27b66ull,
    0xfffffffff7263272ull, 0xffffff0260678075ull, 0xffffff9107326723ull, 0xfff6788916812614ull, 0xffffff73171a67a2ull, 0xfff801781a7167a5ull, 0xfff7a69a0a707303ull, 0xffffff9a88a7a674ull,
    0xfffffffff68b4864ull, 0xffffff640603b633ull, 0xffffff109648b685ull, 0xfff63b1393696492ull, 0xffffff1a28b64864ull, 0xfff640b60b03a213ull, 0xfff9a2920b648b45ull, 0x36463b34923a39a2ull,
    0xffffff2642483286ull, 0xfffffffff2642401ull, 0xfff8346424320917ull, 0xffffff6422414910ull, 0xfff1a66481683186ull, 0xffffff40660a01a1ull, 0x39a9303a68343647ull, 0xfffffffff4a649a0ull,
    0xfffffffffb675940ull, 0xffffff67b5943807ull, 0xffffffb670451051ull, 0xfff51345343867b6ull, 0xffffffb6721a4590ull, 0xfff594380a217b67ull, 0xfff204a24a45b671ull, 0x67b25a5234538436ull,
    0xffffff9452673272ull, 0xfff7862606804595ull, 0xfff0450516732633ull, 0x8515848127868264ull, 0xfff73167161a4592ull, 0x459078701671a615ull, 0xa737a6a305a4a043ull, 0xfffa84a458a7a674ull,
    0xffffff98b9b65964ull, 0xfff590650360b633ull, 0xfffb65510b508b05ull, 0xffffff1355363b62ull, 0xfff65b8b9b59a214ull, 0xa21965690b603b03ull, 0x52025a50865b58b5ull, 0xfff35a3a25363b62ull,
    0xfff2832658259856ull, 0xffffff2600696591ull, 0x8262838650818517ull, 0xfffffffff6126510ull, 0x698965683a616316ull, 0xfff06505960a01a1ull, 0xfffffffffa658307ull, 0xffffffffffff65a0ull,
    0xfffffffffb57a5b0ull, 0xffffff03857ba5b7ull, 0xffffff091ba57b51ull, 0xfff1381897ba57a6ull, 0xffffff15717b21b0ull, 0xfffb275717213807ull, 0xfff7b22097295791ull, 0x289823295b272576ull,
    0xffffff573532a522ull, 0xfff52a5782580285ull, 0xfff2a37353a51093ull, 0x25752a2781292894ull, 0xfffffffff5735312ull, 0xffffff5711707805ull, 0xffffff7355393093ull, 0xfffffffff7957894ull,
    0xffffff8ba8a54854ull, 0xfff03bba50b54053ull, 0xfff54aba8a489105ull, 0x41314943b54a4ba2ull, 0xfff8548b25821524ull, 0xb151b2b543b0b403ull, 0x58b8545b29505205ull, 0xfffffffff3b25492ull,
    0xfff483543253a526ull, 0xffffff0244252a51ull, 0x910854583a532a37ull, 0xfff2492914252a50ull, 0xffffff1533585486ull, 0xfffffffff5015401ull, 0xfff5305093585487ull, 0xffffffffffff5490ull,
    0xffffffba9b947b40ull, 0xfffba97b97943807ull, 0xfffb470414b1ba11ull, 0x4bab474a18434136ull, 0xfff219b294b97b40ull, 0x3801b2b197b94797ull, 0xffffff04224b47b1ull, 0xfff42343824b47b6ull,
    0xfff947732972a922ull, 0x70207872a4797a95ull, 0xa040a1a472a3a733ull, 0xfffffffff4782a14ull, 0xffffff3177141942ull, 0xfff1781807141945ull, 0xfffffffff3473043ull, 0xffffffffffff7844ull,
    0xfffffffff8ba8a94ull, 0xffffffa9bb939033ull, 0xffffffba88a0a105ull, 0xfffffffffa3ba132ull, 0xffffff8b99b1b214ull, 0xfff9b2921b939033ull, 0xfffffffffb08b205ull, 0xffffffffffffb232ull,
    0xffffff98aa828326ull, 0xfffffffff2902a91ull, 0xfff8a1810a828327ull, 0xffffffffffff2a10ull, 0xfffffffff8198316ull, 0xffffffffffff1901ull, 0xffffffffffff8307ull, 0xfffffffffffffff0ull,
};


__device__ __constant__ char4 pmb3d_offset_table[12] {
    char4{0, 0, 0, 0}, /* e0 */
    char4{1, 0, 0, 1}, /* e1 */
    char4{0, 1, 0, 0}, /* e2 */
    char4{0, 0, 0, 1}, /* e3 */

    char4{0, 0, 1, 0}, /* e4 */
    char4{1, 0, 1, 1}, /* e5 */
    char4{0, 1, 1, 0}, /* e6 */
    char4{0, 0, 1, 1}, /* e7 */

    char4{0, 0, 0, 2}, /* e8 */
    char4{1, 0, 0, 2}, /* e9 */
    char4{1, 1, 0, 2}, /* e10 */
    char4{0, 1, 0, 2}, /* e11 */
};


class Triangle {
    public:
        __device__ __forceinline__ Triangle() : _value() {}
        __device__ __forceinline__ Triangle(unsigned long long value): _value(value) {}
        __device__ __forceinline__ void set_value(unsigned long long value) { this->_value = value; }
        __device__ __forceinline__ unsigned char operator[](unsigned int i) const {
            return (unsigned char)( (this->_value >> ((i+1) << 2)) & 0xf);
        }
        __device__ __forceinline__ void determine(unsigned char *const __restrict genvertex,
                unsigned char& nvertices, unsigned char& ntriangles,
                const bool bxx, const bool byy, const bool bzz,
                const bool nxx, const bool nyy, const bool nzz,
                const bool inside_domain) const {
            /* vertices cannot be generated outside domain or at block/grid boundary */
            genvertex[0] = inside_domain && !bxx && !nxx;
            genvertex[1] = inside_domain && !byy && !nyy;
            genvertex[2] = inside_domain && !bzz && !nzz;
            const bool can_generate_triangles = genvertex[0] && genvertex[1] && genvertex[2];
            genvertex[0] &= (this->_value >> 0) & 0x1;
            genvertex[1] &= (this->_value >> 1) & 0x1;
            genvertex[2] &= (this->_value >> 2) & 0x1;
            /* bit 3 left unused */
            nvertices  = genvertex[0] + genvertex[1] + genvertex[2];
            ntriangles = (can_generate_triangles ? 5 - __clzll(~this->_value)/12 : 0);
        }
    private:
        unsigned long long _value;
};


template <typename T, typename I, I Bx, I By, I Bz, I Nx, I Ny, I Nz, unsigned int warpsize=32, typename... F>
__launch_bounds__(Bx*By*Bz)
__global__ void computeBlockMask3D(
        const T isolevel,
        const T x0, const T y0, const T z0,
        const T dx, const T dy, const T dz,
        unsigned char *const __restrict block_mask,
        F... fargs) {

    static_assert(is_same<T, float>::value || is_same<T, double>::value,
            "T has to be float or double.");
    static_assert(is_same<I, unsigned int>::value || is_same<I, unsigned long long>::value,
            "I has to be unsigned int or unsigned long long.");

    constexpr I B = Bx*By*Bz;
    constexpr I warps_per_block = B/warpsize;
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);
    constexpr I Gy = (Ny+By-2)/(By-1);
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");

    __shared__ T minbuf[warps_per_block], maxbuf[warps_per_block];

    const unsigned int tid = (threadIdx.z*By + threadIdx.y)*Bx + threadIdx.x;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    const I bid = (blockIdx.z*Gy + blockIdx.y)*Gx + blockIdx.x;
    const trove::array<I,3> idx = trove::make_array<I>(blockIdx.x*(Bx-1) + threadIdx.x,
                                                       blockIdx.y*(By-1) + threadIdx.y,
                                                       blockIdx.z*(Bz-1) + threadIdx.z);
    const I& ix = trove::get<0>(idx);
    const I& iy = trove::get<1>(idx);
    const I& iz = trove::get<2>(idx);

    const trove::array<T,3> pos = trove::make_array<T>(x0 + ix*dx,
                                                       y0 + iy*dy,
                                                       z0 + iz*dz);

    const bool inside_domain = (ix <= Nx) && (iy <= Ny) && (iz <= Nz);
    const T v = (inside_domain ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(pos, idx, fargs...) - isolevel : nan(""));

    compute_block_mask<T, I, warps_per_block, warpsize>(v, bid, laneid, warpid, minbuf, maxbuf, block_mask);
}


template <typename T, typename I, I Bx, I By, I Bz, I Nx, I Ny, I Nz,
    unsigned int warpsize=32, bool use_constant_memory=true, typename... F>
__launch_bounds__(Bx*By*Bz)
__global__ void computeSimplices3D(
        const I allocated_vertex_count, const I allocated_triangle_count,
        const T isolevel,
        const T x0, const T y0, const T z0,
        const T dx, const T dy, const T dz,
        const I *const __restrict compacted_block_indices,
        trove::array<T,3> *const __restrict output_vertices,
        trove::array<I,3> *const __restrict output_triangles,
        I *const __restrict output_vertex_and_triangle_count,
        const F... fargs) {

    static_assert(is_same<T, float>::value || is_same<T, double>::value,
            "T has to be float or double.");
    static_assert(is_same<I, unsigned int>::value || is_same<I, unsigned long long>::value,
            "I has to be unsigned int or unsigned long long.");
    using II = typename I2<I>::type;

    constexpr I B = Bx*By*Bz;
    constexpr I warps_per_block = B/warpsize;
    constexpr I Gx = (Nx+Bx-2)/(Bx-1);
    constexpr I Gy = (Ny+By-2)/(By-1);
    static_assert(B%warpsize==0, "B%warpsize != 0");
    static_assert(B/warpsize<=warpsize, "B/warpsize > warpsize (in practice B <= warpsize**2 = 32**2 = 1024)");

    /* Shared memory is allocated to share data of various types accross threads
     *  1) B*sizeof(T) shared_scalars
     *  2) B*(sizeof(I)+sizeof(unsigned char)) shared_vertex_offsets and shared_local_offsets
     *  3) (warps_per_block+1)*sizeof(II) = 2*(Bx*By*Bz/warpsize)*sizeof(I) shared_offsets
     * Kernel caller should ensure to allocate at least the max of those three values. */
    extern __shared__ unsigned char shared_mem_buffer[];

    T  *const __restrict shared_scalars       = reinterpret_cast<T*>(shared_mem_buffer);
    II *const __restrict shared_offsets       = reinterpret_cast<II*>(shared_mem_buffer);
    I  *const __restrict shared_vertex_offset = reinterpret_cast<I*>(shared_mem_buffer);
    unsigned char* const __restrict shared_local_offsets = reinterpret_cast<unsigned char*>(shared_vertex_offset + B);

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;
    const unsigned int tid = (tz*By+ty)*Bx + tx;
    const unsigned int laneid = tid % warpsize;
    const unsigned int warpid = tid / warpsize;

    /* compute vertex position */
    trove::array<I,3> idx;
    trove::array<T,3> pos;
    I &ix(trove::get<0>(idx)), &iy(trove::get<1>(idx)), &iz(trove::get<2>(idx));
    T  &x(trove::get<0>(pos)),  &y(trove::get<1>(pos)),  &z(trove::get<2>(pos));
    {
        I bid = compacted_block_indices[blockIdx.x];

        const I bidz = bid/(Gx*Gy);
        bid -= bidz*Gx*Gy;
        const I bidy = bid/Gx;
        bid -= bidy*Gx;
        const I bidx = bid;

        ix = bidx*(Bx-1) + tx;
        iy = bidy*(By-1) + ty;
        iz = bidz*(Bz-1) + tz;

        x = x0 + ix*dx;
        y = y0 + iy*dy;
        z = z0 + iz*dz;
    }

    /* domain boundary test */
    const bool nxx = (ix == Nx);
    const bool nyy = (iy == Ny);
    const bool nzz = (iz == Nz);
    const bool inside_domain = (ix <= Nx) && (iy <= Ny) && (iz <= Nz);

    /* thread block boundary test */
    const bool bxx = (tx == Bx-1);
    const bool byy = (ty == By-1);
    const bool bzz = (tz == Bz-1);

    /* load scalars and compute case id */
    T s[8];
    unsigned char case_id = 0u;
    {
        const trove::array<I,3>& cidx = idx;
        const trove::array<T,3>& cpos = pos;
        const bool bxy = bxx || byy;
        const bool bxz = bxx || bzz;
        const bool byz = byy || bzz;
        const bool bbb = bxx || byy || bzz;

        s[0] = cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel;

        shared_scalars[tid] = s[0];
        __syncthreads();

        /* Thread divergence at block boundaries here: caseid at block boundary is only required to determine the number
         * of vertices to generate. This information could be deduced directly from lower x/y/z-threads neighbors caseid.
         * As we already use 63 out of 64 bits in our case lookup table it is easier to just call the user function and
         * pretend that this do not decreases kernel performance when compared to additional local vertex requirement
         * computations, __shared__ memory writes, __syncthreads and finally __shared__ memory reads. */
        x += dx; ix += 1; s[1] = (bxx ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+0*Bx*By+0*Bx+1]);
        y += dy; iy += 1; s[2] = (bxy ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+0*Bx*By+1*Bx+1]);
        x -= dx; ix -= 1; s[3] = (byy ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+0*Bx*By+1*Bx+0]);
        z += dz; iz += 1; s[7] = (byz ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+1*Bx*By+1*Bx+0]);
        x += dx; ix += 1; s[6] = (bbb ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+1*Bx*By+1*Bx+1]);
        y -= dy; iy -= 1; s[5] = (bxz ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+1*Bx*By+0*Bx+1]);
        x -= dx; ix -= 1; s[4] = (bzz ? cupmb::scalar<T,I,3,Nx,Ny,Nz>(cpos, cidx, fargs...) - isolevel : shared_scalars[tid+1*Bx*By+0*Bx+0]);
        z -= dz; iz -= 1;
        __syncthreads();

        #pragma unroll 8
        for (unsigned int k = 0u; k < 8u; ++k) {
            case_id += (s[k]<T(0)) << k;
        }
    }

    /* load triangle table and compute vertex count, triangle count and vertex generation mask */
    unsigned char nvertices, ntriangles, genvertex[3];
    Triangle triangles(pmb3d_case_table[case_id]);
    triangles.determine(genvertex, nvertices, ntriangles,
            bxx, byy, bzz, nxx, nyy, nzz, inside_domain);

    /* prefix scan vertex count and triangle count to obtain offsets in global memory */
    II offsets;
    I& vertex_offset   = offsets.x;
    I& triangle_offset = offsets.y;
    {
        II icounts;
        icounts.x = nvertices;
        icounts.y = ntriangles;
        offsets = prefix_sum<II, warps_per_block, warpsize>(
                icounts, laneid, warpid, shared_offsets,
                reinterpret_cast<II*>(output_vertex_and_triangle_count));
        shared_vertex_offset[tid] = vertex_offset;
        shared_local_offsets[tid] = 0 + (genvertex[0]<<2) + ((genvertex[0]+genvertex[1])<<4);
        /* x vertex offset -- y vertex offset -- z vertex offset -- 2 bits each, max offset = 2 */
        __syncthreads();
    }

    /* compute thread-local vertices (at most three out of twelve) and store them in global memory */
    #pragma unroll 3
    for (unsigned int k = 0u; (k<3u) && (vertex_offset < allocated_vertex_count); ++k) {
        T vertex[3];
        if (genvertex[k]) {
            vertex[0] = x + (k==0 ? (s[0]/(s[0]-s[1]))*dx : T(0));
            vertex[1] = y + (k==1 ? (s[0]/(s[0]-s[3]))*dy : T(0));
            vertex[2] = z + (k==2 ? (s[0]/(s[0]-s[4]))*dz : T(0));
            output_vertices[vertex_offset] = trove::make_array<T>(vertex[0], vertex[1], vertex[2]);
            ++vertex_offset;
        }
    }

    /* compute edge vertex indices (at most 5 triangles, representing 15 indices) and store them to global memory */
    #pragma unroll 5
    for (unsigned int k = 0u; (k<ntriangles) && (triangle_offset < allocated_triangle_count); ++k) {
        T simplex[3];
        #pragma unroll 3
        for(unsigned int i = 0u; i<3u; ++i) {
            const unsigned char e = triangles[3*k+i];
            unsigned char neighbor_vid;
            bool ox, oy, oz;
            if constexpr (use_constant_memory) {
                const char4 ot = pmb3d_offset_table[e];
                ox = ot.x;
                oy = ot.y;
                oz = ot.z;
                neighbor_vid = ot.w;
            }
            else {
                ox = (e==1) || (e==5) || (e==9)  || (e==10);
                oy = (e==2) || (e==6) || (e==10) || (e==11);
                oz = (e>=4) && (e<8);
                neighbor_vid = (e<8 ? e%2==1 : 2);
            }
            const unsigned int  neighbor_tid = tid + (oz*By+oy)*Bx+ox;
            const unsigned char voffset = (shared_local_offsets[neighbor_tid]>>(neighbor_vid<<1)) & 0x3;
            const I vertex_id = shared_vertex_offset[neighbor_tid] + voffset;
            if (vertex_id >= output_vertex_and_triangle_count[0]) {
                printf("id=%u, bxx=%u, byy=%u, bzz=%y, nxx=%u, nyy=%u, nzz=%u, %u/%u, %u/%u, %u/%u, vertex_id = %u/%u, output_vertices=%u, output_triangles=%u\n",
                        inside_domain, bxx, byy, bzz, nxx, nyy, nzz,
                        ix, Nx, iy, Ny, iz, Nz,
                        vertex_id, allocated_vertex_count-1, output_vertex_and_triangle_count[0], output_vertex_and_triangle_count[1]);
            }
            simplex[i] = min(vertex_id, allocated_vertex_count-1);
        }
        output_triangles[triangle_offset] = trove::make_array<I>(simplex[0], simplex[1], simplex[2]);
        ++triangle_offset;
    }
}

} /* end of namespace cupmb */
