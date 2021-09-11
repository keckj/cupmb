import os, enum, tempfile

import numpy as np
import cupy as cp

# from cupmb import logger
from cupmb.utils.IO import cuda_kernels_dir


class BufferAllocationPolicy(enum.Enum):
    NO_EXTRA_ALLOCATION = 0
    ALLOCATE_EXACT_SIZE = 1
    ALLOCATE_UPPER_POW2 = 2


class KernelAutotuningRigor(enum.Enum):
    ESTIMATE = 0
    MEASURE = 1
    PATIENT = 2
    EXHAUSTIVE = 3


class ParallelMarchingBlocks:

    ftype_to_ctype = {
        np.float32: "float",
        np.float64: "double",
    }

    itype_to_ctype = {
        np.uint32: "unsigned int",
        np.uint64: "unsigned long long int",
    }

    def __init__(
        self,
        dim,
        fdef="",
        N=64,
        L=1.0,
        x0=0.0,
        isolevel=0.0,
        ftype=np.float32,
        itype=np.uint32,
        device=0,
        extra_build_options=None,
        allocation_policy=BufferAllocationPolicy.ALLOCATE_UPPER_POW2,
        autotuning_rigor=KernelAutotuningRigor.ESTIMATE,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self._source_tmpdir = tempfile.TemporaryDirectory()
        self._source_tmpfile = os.path.join(
            self._source_tmpdir.name, "cupmb_func_definition.cuh"
        )
        assert os.path.isdir(self._source_tmpdir.name), self._source_tmpdir.name
        assert not os.path.exists(self._source_tmpfile), self._source_tmpfile

        self._ready = False

        self._dim = None
        self._fdef = None
        self._ftype = None
        self._itype = None
        self._device = None
        self._extra_build_options = None
        self._N = None
        self._L = None
        self._x0 = None
        self._isolevel = None
        self._allocation_policy = None

        self._vertex_buffer = None
        self._index_buffer = None
        self._counting_buffer = None
        self._block_mask_buffer = None
        self._block_index_buffer = None

        self.dim = dim
        self.fdef = fdef
        self.ftype = ftype
        self.itype = itype
        self.device = device
        self.extra_build_options = extra_build_options
        self.N = N
        self.L = L
        self.x0 = x0
        self.isolevel = isolevel
        self.allocation_policy = allocation_policy

    def __call__(self, *fargs, stream=cp.cuda.Stream.null):
        self._prepare_kernels(stream, fargs)

    def _prepare_kernels(self, stream, fargs):
        if self._ready:
            return

        kernel_source_file = os.path.join(cuda_kernels_dir, f"pmb{self.dim}D.cuh")

        if not os.path.isfile(kernel_source_file):
            msg = f"Could not find kernel source file:///{kernel_source_file}."
            raise RuntimeError(msg)

        with open(kernel_source_file) as f:
            cuda_code = f.read()

        if not os.path.isfile(self._source_tmpfile):
            msg = "Scalar function has not been defined yet."
            raise RuntimeError(msg)

        device = cp.cuda.Device(self.device)
        with device:
            with stream:
                (Bopt, Topt) = self._autotune_kernels(stream, device, cuda_code, fargs)

    def _autotune_kernels(self, stream, device, cuda_code, fargs):
        itype = self.itype
        ftype = self.ftype
        assert self.dx.dtype == ftype, self.dx.dtype

        T = self.flt_ctype
        I = self.int_ctype

        dim = self.dim

        N = self.N
        Nstr = ", ".join(map(str, N))

        warp_size = device.attributes["WarpSize"]
        build_options = self.build_options

        Bs = self.candidate_block_size(device)
        Ts = self.candidate_thread_size(device)

        if len(Bs) == 0:
            msg = "Could not determine block size."
            raise RuntimeError(msg)

        if len(Ts) == 0:
            msg = "Could not determine thread size."
            raise RuntimeError(msg)

        Nthreads = Ts[0]
        for B in Bs:
            BB = np.prod(B, dtype=np.float64)
            G = (N + B - 2) // (B - 1)
            Nblocks = np.prod(G, dtype=itype)
            Ngrid = (Nblocks + Nthreads - 1) // Nthreads
            shared_memory_bytes = self._required_shared_memory_bytes(
                dim, warp_size, BB, ftype, itype
            )

            Bstr = ", ".join(map(str, B))
            kernel_names = (
                f"computeBlockMask{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u>",
            )
            kernel_names += (
                f"countActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            )
            kernel_names += (
                f"compactActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            )
            kernel_names += (
                f"computeSimplices{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u>",
            )

            module = cp.RawModule(
                code=cuda_code,
                name_expressions=kernel_names,
                backend="nvrtc",
                options=build_options,
            )

            (
                compute_block_mask,
                count_active_blocks,
                compact_active_blocks,
                compute_simplices,
            ) = tuple(map(module.get_function, kernel_names))

            if BB > compute_block_mask.max_threads_per_block:
                raise RuntimeError((BB, compute_block_mask.max_threads_per_block))
            if BB > compute_simplices.max_threads_per_block:
                raise RuntimeError((BB, compute_simplices.max_threads_per_block))
            if Nthreads > count_active_blocks.max_threads_per_block:
                raise RuntimeError(
                    (Nthreads, count_active_blocks.max_threads_per_block)
                )
            if Nthreads > compact_active_blocks.max_threads_per_block:
                raise RuntimeError(
                    (Nthreads, compact_active_blocks.max_threads_per_block)
                )

            self.counting_buffer = 2
            self.block_mask_buffer = int(Nblocks)

            args = (self.isolevel,) + tuple(self.x0) + tuple(self.dx)
            compute_block_mask(
                tuple(G),
                tuple(B),
                args + (self.block_mask_buffer,),
            )

            cp.cuda.runtime.memsetAsync(
                self.counting_buffer.data.ptr,
                0,
                self.counting_buffer.nbytes,
                stream.ptr,
            )
            count_active_blocks(
                (Ngrid,),
                (Nthreads,),
                (self.block_mask_buffer, self.counting_buffer),
            )
            Nactive = cp.asnumpy(self.counting_buffer)[0]

            self.block_index_buffer = int(Nactive)

            cp.cuda.runtime.memsetAsync(
                self.counting_buffer.data.ptr,
                0,
                self.counting_buffer.nbytes,
                stream.ptr,
            )
            compact_active_blocks(
                (Ngrid,),
                (Nthreads,),
                (
                    Nactive,
                    self.block_mask_buffer,
                    self.block_index_buffer,
                    self.counting_buffer,
                ),
            )
            Nactive_bis = cp.asnumpy(self.counting_buffer)[0]
            assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

            cp.cuda.runtime.memsetAsync(
                self.counting_buffer.data.ptr,
                0,
                self.counting_buffer.nbytes,
                stream.ptr,
            )
            compute_simplices(
                (Nactive,),
                tuple(B),
                (itype(0), itype(0))
                + args
                + (
                    self.block_index_buffer,
                    self.vertex_buffer,
                    self.index_buffer,
                    self.counting_buffer,
                ),
                shared_mem=shared_memory_bytes,
            )

        Bopt = B
        Topt = T
        return (Bopt, Topt)

    def set_dim(self, dim):
        assert isinstance(dim, int), type(dim)
        assert dim in {1, 2, 3}, dim
        self._ready &= dim == self._dim
        self._dim = dim

    def get_dim(self):
        return self._dim

    dim = property(get_dim, set_dim)

    def set_fdef(self, fdef):
        assert isinstance(fdef, str), type(fdef)
        if os.path.isfile(fdef):
            with open(fdef) as f:
                fdef = f.read()
        if fdef != self._fdef:
            if fdef:
                with open(self._source_tmpfile, "w") as f:
                    f.write(fdef)
            self._ready = False
            self._fdef = fdef

    def get_fdef(self):
        return self._fdef

    fdef = property(get_fdef, set_fdef)

    def set_device(self, device):
        assert isinstance(device, int), type(device)
        if device != self.device:
            self._block_mask_buffer = None
            self._block_index_buffer = None
            self._counting_buffer = None
            self._vertex_buffer = None
            self._index_buffer = None
            self._ready = False
        self._device = device

    def get_device(self):
        return self._device

    device = property(get_device, set_device)

    def set_ftype(self, ftype):
        assert ftype in {np.float32, np.float64}, ftype
        self._ready &= ftype == self._ftype
        self._ftype = ftype

    def get_ftype(self):
        return self._ftype

    ftype = property(get_ftype, set_ftype)

    def set_itype(self, itype):
        assert itype in {np.uint32, np.uint64}, itype
        self._ready &= itype == self._itype
        self._itype = itype

    def get_itype(self):
        return self._itype

    itype = property(get_itype, set_itype)

    def set_N(self, N):
        N = self._make_vector(N, dim=self.dim, dtype=self.itype, default=64)
        assert (N >= 2).all(), N
        self._ready &= (N == self._N).all()
        self._N = N

    def get_N(self):
        return self._N

    N = property(get_N, set_N)

    def set_L(self, L):
        L = self._make_vector(L, dim=self.dim, dtype=self.ftype, default=1.0)
        assert (L > 0).all(), L
        assert np.isfinite(L).all(), L
        self._ready &= True
        self._L = L

    def get_L(self):
        return self._L

    L = property(get_L, set_L)

    def set_x0(self, x0):
        x0 = self._make_vector(x0, dim=self.dim, dtype=self.ftype, default=0.0)
        assert np.isfinite(x0).all(), x0
        self._ready &= True
        self._x0 = x0

    def get_x0(self):
        return self._x0

    x0 = property(get_x0, set_x0)

    def set_isolevel(self, isolevel):
        if isolevel is None:
            isolevel = 0.0
        isolevel = self.ftype(isolevel)
        assert np.isfinite(isolevel), isolevel
        self._ready &= True
        self._isolevel = isolevel

    def get_isolevel(self):
        return self._isolevel

    isolevel = property(get_isolevel, set_isolevel)

    def set_extra_build_options(self, val):
        if val is None:
            val = tuple()
        assert isinstance(val, tuple), type(val)
        self._extra_build_options = val

    def get_extra_build_options(self):
        return self._extra_build_options

    extra_build_options = property(get_extra_build_options, set_extra_build_options)

    def set_counting_buffer(self, val):
        self._counting_buffer = self._new_cuda_buffer(
            val, dtype=self.itype, device=self.device, minsize=2
        )

    def get_counting_buffer(self):
        return self._counting_buffer

    counting_buffer = property(get_counting_buffer, set_counting_buffer)

    def set_block_index_buffer(self, val):
        self._block_index_buffer = self._new_cuda_buffer(
            val, dtype=self.itype, device=self.device
        )

    def get_block_index_buffer(self):
        return self._block_index_buffer

    block_index_buffer = property(get_block_index_buffer, set_block_index_buffer)

    def set_block_mask_buffer(self, val):
        self._block_mask_buffer = self._new_cuda_buffer(
            val, dtype=np.uint8, device=self.device
        )

    def get_block_mask_buffer(self):
        return self._block_mask_buffer

    block_mask_buffer = property(get_block_mask_buffer, set_block_mask_buffer)

    def set_vertex_buffer(self, val):
        self._vertex_buffer = self._new_cuda_buffer(
            val, dtype=self.ftype, device=self.device
        )

    def get_vertex_buffer(self):
        return self._vertex_buffer

    vertex_buffer = property(get_vertex_buffer, set_vertex_buffer)

    def set_index_buffer(self, val):
        self._index_buffer = self._new_cuda_buffer(
            val, dtype=self.itype, device=self.device
        )

    def get_index_buffer(self):
        return self._index_buffer

    index_buffer = property(get_index_buffer, set_index_buffer)

    def set_allocation_policy(self, val):
        assert isinstance(val, BufferAllocationPolicy), type(val)
        self._allocation_policy = val

    def get_allocation_policy(self):
        return self._allocation_policy

    allocation_policy = property(get_allocation_policy, set_allocation_policy)

    @property
    def dx(self):
        return self.L / self.N.astype(self.ftype)

    @property
    def flt_ctype(self):
        return self.ftype_to_ctype[self.ftype]

    @property
    def int_ctype(self):
        return self.itype_to_ctype[self.itype]

    @property
    def allocated_vertex_count(self):
        return self.itype(
            0 if (self._vertex_buffer is None) else self._vertex_buffer.shape[0]
        )

    @property
    def allocated_index_count(self):
        return self.itype(
            0 if (self._index_buffer is None) else self._index_buffer.shape[0]
        )

    @property
    def source_tmpdir(self):
        return self._source_tmpdir

    @property
    def source_tmpfile(self):
        return self._source_tmpfile

    @property
    def build_options(self):
        options = (
            "-std=c++17",
            "--restrict",
            f"-I{cuda_kernels_dir}",
            f"-I{self._source_tmpdir.name}",
        )
        options += self.extra_build_options
        return options

    @classmethod
    def _new_cuda_buffer(cls, new_buffer, dtype, device, minsize=None):
        if new_buffer is None:
            return None

        if isinstance(new_buffer, cp.ndarray):
            assert new_buffer.device == device, (new_buffer.device, device)
            assert new_buffer.dtype == dtype, (new_buffer.dtype, dtype)
            buf = new_buffer
        elif isinstance(new_buffer, int):
            with cp.cuda.Device(device):
                buf = cp.ndarray(shape=(new_buffer,), dtype=dtype)
        else:
            raise NotImplementedError(type(new_buffer))

        assert (minsize is None) or (buf.size >= minsize), (buf.size, minsize)
        return buf

    @classmethod
    def _make_vector(cls, a, dim, dtype, default):
        if a is None:
            a = np.full(shape=(dim,), dtype=dtype, fill_value=default)
        elif np.isscalar(a):
            a = np.full(shape=(dim,), dtype=dtype, fill_value=a)
        else:
            assert isinstance(a, (tuple, list, np.ndarray)), type(a)
            assert len(a) == dim, (len(a), dim)
            a = np.asarray(a, dtype=dtype)
        a.flags.writeable = False
        return a

    @classmethod
    def _required_shared_memory_bytes(cls, dim, warp_size, threads, ftype, itype):
        assert threads % warp_size == 0, threads
        assert threads // warp_size <= warp_size, (threads, warp_size)
        warps_per_block = threads // warp_size
        if dim == 3:
            m0 = threads * ftype(0).itemsize
            m1 = threads * (itype(0).itemsize + np.uint8(0).itemsize)
            m2 = 2 * (warps_per_block + 1) * itype(0).itemsize
            return max(m0, m1, m2)
        else:
            raise NotImplementedError(dim)

    def candidate_block_size(self, device):
        candidates = []

        dim = self.dim
        itype = self.itype
        ftype = self.ftype

        warp_size = device.attributes["WarpSize"]
        max_shared_memory_per_block = device.attributes["MaxSharedMemoryPerBlockOptin"]
        max_threads_per_block = device.attributes["MaxThreadsPerBlock"]

        Bmax = np.asarray(
            [
                device.attributes["MaxBlockDimX"],
                device.attributes["MaxBlockDimY"],
                device.attributes["MaxBlockDimZ"],
            ],
            dtype=itype,
        )[:dim]

        d = max(
            1 << int(np.ceil(np.log2(min(64, max_threads_per_block) ** (1.0 / dim)))), 2
        )
        B = np.full(shape=(dim,), dtype=itype, fill_value=d)

        k = 0
        threads = np.prod(B, dtype=np.int64)
        while threads <= max_threads_per_block:
            assert threads % warp_size == 0
            assert (B <= Bmax).all(), (B, Bmax)
            warps_per_block = threads // warp_size
            if warps_per_block > warp_size:
                break
            reqd = self._required_shared_memory_bytes(
                dim, warp_size, threads, ftype, itype
            )
            if reqd > max_shared_memory_per_block:
                break
            candidates.append(B.copy())
            if (2 * B > Bmax).all():
                break
            for i in range(dim):
                if 2 * B[k] <= Bmax[k]:
                    B[k] *= 2
                    k = (k + 1) % dim
                    break
                k = (k + 1) % dim
            threads = np.prod(B, dtype=np.int64)

        return candidates[::-1]

    def candidate_thread_size(self, device):
        candidates = []

        itype = self.itype
        ftype = self.ftype

        max_shared_memory_per_block = device.attributes["MaxSharedMemoryPerBlockOptin"]

        warp_size = device.attributes["WarpSize"]
        max_threads_per_block = min(
            device.attributes["MaxThreadsPerBlock"], device.attributes["MaxBlockDimX"]
        )

        threads = warp_size
        while threads <= max_threads_per_block:
            warps_per_block = threads // warp_size
            if warps_per_block > warp_size:
                break
            reqd = 2 * warps_per_block * ftype(0).itemsize
            if reqd > max_shared_memory_per_block:
                break
            candidates.append(itype(threads))
            threads *= 2

        return candidates[::-1]


if __name__ == "__main__":
    fdef = """
    template <typename T>
    constexpr T pi = T(3.14159265358979323846);

    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz) {
        return cos(2*pi<T>*x)*sin(2*pi<T>*y)*cos(2*pi<T>*z);
    }
    """
    pmb = ParallelMarchingBlocks(3, fdef)
    pmb.isolevel = 0.25
    pmb()
