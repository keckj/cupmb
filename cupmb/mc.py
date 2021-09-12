import os, enum, tempfile, operator

import numpy as np
import cupy as cp

from cupmb import logger as default_logger
from cupmb.utils.IO import cuda_kernels_dir
from cupmb.utils.units import time2str, bytes2str


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
        extra_scalar_params=None,
        logger=default_logger,
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
        self._autotuning_rigor = None
        self._extra_scalar_params = None

        # device buffers, can all be set individually
        self._vertex_buffer = None
        self._vertex_index_buffer = None
        self._counting_buffer = None
        self._block_mask_buffer = None
        self._block_index_buffer = None

        # kernel launcher (set during setup())
        self._kernel_launcher = None

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
        self.autotuning_rigor = autotuning_rigor
        self.extra_scalar_params = extra_scalar_params
        self.logger = logger

    def __call__(self, *fargs, stream=cp.cuda.Stream.null, to_numpy=False):
        device = cp.cuda.Device(self.device)
        with device:
            with stream:
                self.setup(device, stream, fargs)
                return self._kernel_launcher(stream, fargs, to_numpy)

    def setup(self, device, stream, fargs):
        if self._ready:
            return

        device_props = cp.cuda.runtime.getDeviceProperties(device.id)
        device_name = device_props["name"].decode("utf-8")
        device_bus_id = device.pci_bus_id
        msg = f"Initializing parallel marching cubes on device {device.id} [{device_name}] ({device_bus_id})."
        self.logger.info(msg)

        if not os.path.isfile(self.kernel_source_file):
            msg = f"Could not find kernel source file:///{self.kernel_source_file}."
            raise RuntimeError(msg)

        if not os.path.isfile(self._source_tmpfile):
            msg = "Scalar function has not been defined yet."
            raise RuntimeError(msg)

        (Bopt, Topt, use_cmem) = self._autotune_kernels(stream, device, fargs)

        self._build_kernel_launcher(device, Bopt, Topt, use_cmem)

        self._ready = True

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
            self._vertex_index_buffer = None
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

    def set_extra_scalar_params(self, extra_scalar_params):
        if extra_scalar_params is None:
            extra_scalar_params = tuple()
        assert isinstance(extra_scalar_params, (list, tuple)), type(extra_scalar_params)
        assert all(isinstance(x, str) for x in extra_scalar_params)
        extra_scalar_params = tuple(extra_scalar_params)
        self._ready &= extra_scalar_params == self._extra_scalar_params
        self._extra_scalar_params = extra_scalar_params

    def get_extra_scalar_params(self):
        return self._extra_scalar_params

    extra_scalar_params = property(get_extra_scalar_params, set_extra_scalar_params)

    def set_counting_buffer(self, val):
        self._counting_buffer = self._new_cuda_buffer(
            val,
            dtype=self.itype,
            device=self.device,
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
            val,
            dtype=self.ftype,
            device=self.device,
            factor=self.dim,
        )

    def get_vertex_buffer(self):
        return self._vertex_buffer

    vertex_buffer = property(get_vertex_buffer, set_vertex_buffer)

    def set_vertex_index_buffer(self, val):
        self._vertex_index_buffer = self._new_cuda_buffer(
            val,
            dtype=self.itype,
            device=self.device,
            factor=self.dim,
        )

    def get_vertex_index_buffer(self):
        return self._vertex_index_buffer

    vertex_index_buffer = property(get_vertex_index_buffer, set_vertex_index_buffer)

    def set_allocation_policy(self, val):
        assert isinstance(val, BufferAllocationPolicy), type(val)
        self._allocation_policy = val

    def get_allocation_policy(self):
        return self._allocation_policy

    allocation_policy = property(get_allocation_policy, set_allocation_policy)

    def set_autotuning_rigor(self, val):
        assert isinstance(val, KernelAutotuningRigor), type(val)
        self._autotuning_rigor = val

    def get_autotuning_rigor(self):
        return self._autotuning_rigor

    autotuning_rigor = property(get_autotuning_rigor, set_autotuning_rigor)

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
            0 if (self._vertex_buffer is None) else self._vertex_buffer.size // self.dim
        )

    @property
    def allocated_vertex_index_count(self):
        return self.itype(
            0
            if (self._vertex_index_buffer is None)
            else self._vertex_index_buffer.size // self.dim
        )

    @property
    def allocated_block_index_count(self):
        return self.itype(
            0 if (self._block_index_buffer is None) else self._block_index_buffer.size
        )

    @property
    def source_tmpdir(self):
        return self._source_tmpdir

    @property
    def source_tmpfile(self):
        return self._source_tmpfile

    @property
    def kernel_source_file(self):
        return os.path.join(cuda_kernels_dir, f"pmb{self.dim}D.cuh")

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
    def _new_cuda_buffer(cls, new_buffer, dtype, device, factor=None):
        if (new_buffer is None) or (new_buffer == 0):
            return None

        if isinstance(new_buffer, cp.ndarray):
            assert new_buffer.device.id == device, (new_buffer.device, device)
            assert new_buffer.dtype == dtype, (new_buffer.dtype, dtype)
            buf = new_buffer
        elif isinstance(new_buffer, (int, np.integer)):
            if factor is not None:
                new_buffer *= factor
            with cp.cuda.Device(device):
                buf = cp.ndarray(shape=(new_buffer,), dtype=dtype)
            if factor is not None:
                buf = buf.reshape(-1, factor)
        else:
            raise NotImplementedError(type(new_buffer))

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

    def _count_active_blocks_kernel_args(self):
        return (self.block_mask_buffer, self.counting_buffer)

    def _compact_active_blocks_kernel_args(self):
        return (
            self.allocated_block_index_count,
            self.block_mask_buffer,
            self.block_index_buffer,
            self.counting_buffer,
        )

    def _scalar_kernel_args(self):
        assert self.dx.dtype == self.ftype, self.dx.dtype
        return (self.isolevel,) + tuple(self.x0) + tuple(self.dx)

    def _compute_block_mask_kernel_args(self, fargs):
        if len(fargs) != len(self.extra_scalar_params):
            msg = f"Expected {len(self.extra_scalar_params)} extra parameters but got {len(fargs)}."
            raise RuntimeError(msg)
        return self._scalar_kernel_args() + (self.block_mask_buffer,) + fargs

    def _compute_simplices_kernel_args(self, fargs):
        if len(fargs) != len(self.extra_scalar_params):
            msg = f"Expected {len(self.extra_scalar_params)} extra parameters but got {len(fargs)}."
            raise RuntimeError(msg)
        args = (self.allocated_vertex_count, self.allocated_vertex_index_count)
        args += self._scalar_kernel_args()
        args += (
            self.block_index_buffer,
            self.vertex_buffer,
            self.vertex_index_buffer,
            self.counting_buffer,
        )
        args += fargs
        return args

    def _build_kernels(self, device, B, Nblocks, Nthreads, use_cmem):
        threads_per_blocks = np.prod(B, dtype=np.int64)

        dim = self.dim
        N = self.N

        T = self.flt_ctype
        I = self.int_ctype
        Bstr = ", ".join(map(str, B))
        Nstr = ", ".join(map(str, N))
        warp_size = device.attributes["WarpSize"]

        extra_templates = self.extra_scalar_params
        treplace = {"T": T, "I": I}
        extra_templates = tuple(
            treplace[e] if e in treplace else e for e in extra_templates
        )
        ftemplates = ", ".join(extra_templates)
        if ftemplates:
            ftemplates = ", " + ftemplates

        use_cmem_s = "true" if use_cmem else "false"

        kernel_names = (
            f"computeBlockMask{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{ftemplates}>",
        )
        kernel_names += (
            f"countActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
        )
        kernel_names += (
            f"compactActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
        )
        kernel_names += (
            f"computeSimplices{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u, {use_cmem_s}{ftemplates}>",
        )

        build_options = self.build_options

        kernel_source_file = self.kernel_source_file
        with open(kernel_source_file) as f:
            cuda_code = f.read()

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

        if threads_per_blocks > compute_block_mask.max_threads_per_block:
            raise RuntimeError(
                (threads_per_blocks, compute_block_mask.max_threads_per_block)
            )
        if threads_per_blocks > compute_simplices.max_threads_per_block:
            raise RuntimeError(
                (threads_per_blocks, compute_simplices.max_threads_per_block)
            )
        if Nthreads > count_active_blocks.max_threads_per_block:
            raise RuntimeError((Nthreads, count_active_blocks.max_threads_per_block))
        if Nthreads > compact_active_blocks.max_threads_per_block:
            raise RuntimeError((Nthreads, compact_active_blocks.max_threads_per_block))

        return (
            compute_block_mask,
            count_active_blocks,
            compact_active_blocks,
            compute_simplices,
        )

    def _generate_candidate_block_size(self, device):
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

        Gmax = np.asarray(
            [
                device.attributes["MaxGridDimX"],
                device.attributes["MaxGridDimY"],
                device.attributes["MaxGridDimZ"],
            ],
            dtype=itype,
        )[:dim]

        self.logger.debug(f"Max device threads per block is {max_threads_per_block}.")
        self.logger.debug(f"Max device block size is {tuple(Bmax)}.")
        self.logger.debug(f"Max device grid size is {tuple(Gmax)}.")
        self.logger.debug(
            f"Max device shared memory per block is {bytes2str(max_shared_memory_per_block)}."
        )

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

    def _generate_candidate_thread_size(self, device):
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
            reqd = max(
                2 * warps_per_block * ftype(0).itemsize,
                warps_per_block * itype(0).itemsize,
            )
            if reqd > max_shared_memory_per_block:
                break
            candidates.append(itype(threads))
            threads *= 2

        return candidates[::-1]

    def _reset_counter(self, stream):
        assert self.counting_buffer is not None
        cp.cuda.runtime.memsetAsync(
            self.counting_buffer.data.ptr,
            0,
            self.counting_buffer.nbytes,
            stream.ptr,
        )

    def _autotune_kernels(self, stream, device, fargs):
        self.logger.info(
            f"Autotuning kernels with new configuration by using {self.autotuning_rigor}."
        )

        Ts = self._generate_candidate_thread_size(device)
        if len(Ts) == 0:
            msg = "Could not determine any candidate thread size."
            raise RuntimeError(msg)

        Bs = self._generate_candidate_block_size(device)
        if len(Bs) == 0:
            msg = "Could not determine any candidate block size."
            raise RuntimeError(msg)

        rigor = self.autotuning_rigor
        if rigor == KernelAutotuningRigor.ESTIMATE:
            return (Bs[0], Ts[0], True)
        elif rigor == KernelAutotuningRigor.MEASURE:
            Nruns = 1
            max_candidates = 4
        elif rigor == KernelAutotuningRigor.PATIENT:
            Nruns = 4
            max_candidates = 16
        elif rigor == KernelAutotuningRigor.EXHAUSTIVE:
            Nruns = 16
            max_candidates = 32
        else:
            raise NotImplementedError(rigor)

        self.logger.debug(
            f"Autotuning parameters: max_candidates={max_candidates}, nruns={Nruns}"
        )

        (Bopt, use_cmem) = self._autotune_block_size(
            stream, device, fargs, Ts[0], Nruns, max_candidates
        )
        Topt = self._autotune_thread_size(
            stream, device, fargs, Bopt, Nruns, max_candidates
        )

        return (Bopt, Topt, use_cmem)

    def _autotune_block_size(
        self, stream, device, fargs, Nthreads, Nruns, max_candidates
    ):
        dim = self.dim
        N = self.N
        itype = self.itype
        ftype = self.ftype

        Bs = self._generate_candidate_block_size(device)
        if len(Bs) == 0:
            msg = "Could not determine block size."
            raise RuntimeError(msg)

        self.logger.debug(f"Estimating block size for N={N}, candidates are:")
        self.logger.debug("  " + ", ".join(map(str, map(tuple, Bs))))

        warp_size = device.attributes["WarpSize"]

        results = {}

        ncandidates = 0
        for use_cmem in (True, False):
            if ncandidates >= max_candidates:
                break
            for B in Bs:
                if ncandidates >= max_candidates:
                    self.logger.debug(
                        f"Max kernel candidates achieved ({max_candidates})."
                    )
                    break
                threads_per_blocks = np.prod(B, dtype=np.int64)
                G = (N + B - 2) // (B - 1)
                Nblocks = np.prod(G, dtype=itype)
                Ngrid = (Nblocks + Nthreads - 1) // Nthreads
                shared_memory_bytes = self._required_shared_memory_bytes(
                    dim, warp_size, threads_per_blocks, ftype, itype
                )

                self.logger.debug(
                    f"Benmarking with B={B}, G={G}, use_cmem={'1' if use_cmem else '0'}, "
                    f"shared_mem={bytes2str(shared_memory_bytes)}..."
                )

                (
                    compute_block_mask,
                    count_active_blocks,
                    compact_active_blocks,
                    compute_simplices,
                ) = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem=use_cmem)

                self.counting_buffer = 2
                self.block_mask_buffer = Nblocks

                evts = [cp.cuda.Event() for _ in range(Nruns + 1)]
                for i in range(Nruns):
                    evts[i].record()
                    compute_block_mask(
                        tuple(G), tuple(B), self._compute_block_mask_kernel_args(fargs)
                    )
                evts[-1].record()
                evts[-1].synchronize()
                ellapsed0_time_s = 1e-3 * np.asarray(
                    [
                        cp.cuda.get_elapsed_time(evts[i], evts[i + 1])
                        for i in range(Nruns)
                    ],
                    dtype=np.float64,
                )
                self.logger.debug(
                    f"  compute_block_mask: nruns={Nruns}, "
                    f"mean={time2str(ellapsed0_time_s.mean())}, "
                    f"min={time2str(ellapsed0_time_s.min())}, "
                    f"max={time2str(ellapsed0_time_s.max())}"
                )

                self._reset_counter(stream)
                count_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._count_active_blocks_kernel_args(),
                )
                Nactive = cp.asnumpy(self.counting_buffer)[0]

                self.block_index_buffer = Nactive

                self._reset_counter(stream)
                compact_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._compact_active_blocks_kernel_args(),
                )
                Nactive_bis = cp.asnumpy(self.counting_buffer)[0]
                assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

                self._reset_counter(stream)
                compute_simplices(
                    (Nactive,),
                    tuple(B),
                    self._compute_simplices_kernel_args(fargs),
                    shared_mem=shared_memory_bytes,
                )
                Ncount = cp.asnumpy(self.counting_buffer)
                self.vertex_buffer = Ncount[0]
                self.vertex_index_buffer = Ncount[1]

                start_evts = [cp.cuda.Event() for _ in range(Nruns)]
                end_evts = [cp.cuda.Event() for _ in range(Nruns)]
                for i in range(Nruns):
                    self._reset_counter(stream)
                    start_evts[i].record()
                    compute_simplices(
                        (Nactive,),
                        tuple(B),
                        self._compute_simplices_kernel_args(fargs),
                        shared_mem=shared_memory_bytes,
                    )
                    end_evts[i].record()
                end_evts[-1].synchronize()
                ellapsed1_time_s = 1e-3 * np.asarray(
                    [
                        cp.cuda.get_elapsed_time(start_evts[i], end_evts[i])
                        for i in range(Nruns)
                    ],
                    dtype=np.float64,
                )
                self.logger.debug(
                    f"   compute_simplices: nruns={Nruns}, "
                    f"mean={time2str(ellapsed1_time_s.mean())}, "
                    f"min={time2str(ellapsed1_time_s.min())}, "
                    f"max={time2str(ellapsed1_time_s.max())}"
                )
                Ncount_bis = cp.asnumpy(self.counting_buffer)
                assert (Ncount == Ncount_bis).all(), (Ncount, Ncount_bis)

                ellapsed = ellapsed0_time_s + ellapsed1_time_s
                results[(tuple(B), use_cmem)] = ellapsed.mean()
                ncandidates += 1

        if not results:
            msg = "Could not bench any kernels."
            raise RuntimeError(msg)

        (Bopt, use_cmem), _ = next(
            iter(sorted(results.items(), key=operator.itemgetter(1)))
        )
        self.logger.info(
            f"Best candidate block size is {Bopt} with use_cmem={'1' if use_cmem else '0'}."
        )
        return (np.asarray(Bopt, dtype=itype), use_cmem)

    def _autotune_thread_size(self, stream, device, fargs, B, Nruns, max_candidates):
        N = self.N
        itype = self.itype

        Ts = self._generate_candidate_thread_size(device)
        if len(Ts) == 0:
            msg = "Could not determine thread size."
            raise RuntimeError(msg)

        self.logger.debug(f"Estimating thread size for N={N}, candidates are:")
        self.logger.debug("  " + ", ".join(map(str, Ts)))

        G = (N + B - 2) // (B - 1)
        Nblocks = np.prod(G, dtype=itype)

        results = {}
        ncandidates = 0
        for Nthreads in Ts:
            if ncandidates >= max_candidates:
                self.logger.debug(f"Max kernel candidates achieved ({max_candidates}).")
                break

            Ngrid = (Nblocks + Nthreads - 1) // Nthreads

            self.logger.debug(f"Benmarking with Nthreads={Nthreads}, Ngrid={Ngrid}...")

            (
                compute_block_mask,
                count_active_blocks,
                compact_active_blocks,
                compute_simplices,
            ) = self._build_kernels(device, B, Nblocks, Nthreads, False)

            self.counting_buffer = 2
            self.block_mask_buffer = Nblocks

            compute_block_mask(
                tuple(G), tuple(B), self._compute_block_mask_kernel_args(fargs)
            )

            start_evts = [cp.cuda.Event() for _ in range(Nruns)]
            end_evts = [cp.cuda.Event() for _ in range(Nruns)]
            for i in range(Nruns):
                self._reset_counter(stream)
                start_evts[i].record()
                count_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._count_active_blocks_kernel_args(),
                )
                end_evts[i].record()
            end_evts[-1].synchronize()
            ellapsed0_time_s = 1e-3 * np.asarray(
                [
                    cp.cuda.get_elapsed_time(start_evts[i], end_evts[i])
                    for i in range(Nruns)
                ],
                dtype=np.float64,
            )
            self.logger.debug(
                f"    count_active_blocks: nruns={Nruns}, "
                f"mean={time2str(ellapsed0_time_s.mean())}, "
                f"min={time2str(ellapsed0_time_s.min())}, "
                f"max={time2str(ellapsed0_time_s.max())}"
            )
            Nactive = cp.asnumpy(self.counting_buffer)[0]

            self.block_index_buffer = Nactive

            start_evts = [cp.cuda.Event() for _ in range(Nruns)]
            end_evts = [cp.cuda.Event() for _ in range(Nruns)]
            for i in range(Nruns):
                self._reset_counter(stream)
                start_evts[i].record()
                compact_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._compact_active_blocks_kernel_args(),
                )
                end_evts[i].record()
            end_evts[-1].synchronize()
            ellapsed1_time_s = 1e-3 * np.asarray(
                [
                    cp.cuda.get_elapsed_time(start_evts[i], end_evts[i])
                    for i in range(Nruns)
                ],
                dtype=np.float64,
            )
            self.logger.debug(
                f"  compact_active_blocks: nruns={Nruns}, "
                f"mean={time2str(ellapsed1_time_s.mean())}, "
                f"min={time2str(ellapsed1_time_s.min())}, "
                f"max={time2str(ellapsed1_time_s.max())}"
            )
            Nactive_bis = cp.asnumpy(self.counting_buffer)[0]
            assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

            results[Nthreads] = ellapsed1_time_s.mean()
            ncandidates += 1

        if not results:
            msg = "Could not bench any kernels."
            raise RuntimeError(msg)

        Topt, _ = next(iter(sorted(results.items(), key=operator.itemgetter(1))))
        self.logger.info(f"Best candidate thread size is {Topt}.")
        return Topt

    def _build_kernel_launcher(self, device, B, Nthreads, use_cmem):
        dim = self.dim
        N = self.N
        itype = self.itype
        ftype = self.ftype

        self.logger.info(
            f"Building kernel launcher for N={N}, B={B}, Nthreads={Nthreads}, "
            f"use_cmem={'1' if use_cmem else '0'}..."
        )

        G = (N + B - 2) // (B - 1)
        Nblocks = np.prod(G, dtype=itype)
        Ngrid = (Nblocks + Nthreads - 1) // Nthreads

        warp_size = device.attributes["WarpSize"]
        threads_per_blocks = np.prod(B, dtype=np.int64)

        shared_mem = self._required_shared_memory_bytes(
            dim, warp_size, threads_per_blocks, ftype, itype
        )

        kernels = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem)

        def kernel_launcher(
            stream,
            fargs,
            to_numpy,
            s=self,
            kernels=kernels,
            shared_mem=shared_mem,
            G=tuple(G),
            B=tuple(B),
            Ngrid=(Ngrid,),
            Nthreads=(Nthreads,),
            Nblocks=Nblocks,
        ):
            self.logger.info(f"Kernel launcher called with parameters {fargs}.")

            s.counting_buffer = 2
            s.block_mask_buffer = Nblocks

            self.logger.debug(f"Computing block mask for {Nblocks} blocks.")
            kernels[0](G, B, s._compute_block_mask_kernel_args(fargs))

            is_first_call = s.allocated_block_index_count == 0
            if is_first_call:
                self.logger.debug(
                    "Counting active blocks because allocated block index size is zero."
                )
                s._reset_counter(stream)
                kernels[1](Ngrid, Nthreads, s._count_active_blocks_kernel_args())
            else:
                self.logger.debug(
                    f"Compacting active blocks (allocated_block_index={s.allocated_block_index_count})."
                )
                s._reset_counter(stream)
                kernels[2](Ngrid, Nthreads, s._compact_active_blocks_kernel_args())

            Nactive = cp.asnumpy(s.counting_buffer)[0]
            if Nactive == 0:
                self.logger.debug("No active blocks found, skipping block compaction.")
                return (None, None)

            if is_first_call or (s.allocated_block_index_count < Nactive):
                self.logger.info(
                    f"Reallocating block index buffer to handle {Nactive} active blocks."
                )
                s.block_index_buffer = Nactive

                self.logger.debug("Compacting active blocks with reallocated buffer.")
                s._reset_counter(stream)
                kernels[2](Ngrid, Nthreads, s._compact_active_blocks_kernel_args())

            self.logger.debug(
                "Generating simplices "
                f"(allocated_vertices={s.allocated_vertex_count}, "
                f"allocated_indices={s.allocated_vertex_index_count})"
            )
            s._reset_counter(stream)
            kernels[3](
                (Nactive,),
                B,
                s._compute_simplices_kernel_args(fargs),
                shared_mem=shared_mem,
            )

            Nvertices, Nindices = tuple(cp.asnumpy(s.counting_buffer))
            rellocate_vertices = s.allocated_vertex_count < Nvertices
            rellocate_indices = s.allocated_vertex_index_count < Nindices

            if rellocate_vertices:
                self.logger.info(
                    f"Reallocating vertex buffer to handle {Nvertices} vertices."
                )
                s.vertex_buffer = Nvertices

            if rellocate_indices:
                self.logger.info(
                    f"Reallocating vertex index buffer to handle {Nindices} indices."
                )
                s.vertex_index_buffer = Nindices

            if rellocate_vertices or rellocate_indices:
                self.logger.debug("Generating simplices with reallocated buffers.")
                s._reset_counter(stream)
                kernels[3](
                    (Nactive,),
                    B,
                    s._compute_simplices_kernel_args(fargs),
                    shared_mem=shared_mem,
                )

            buffers = (
                self.vertex_buffer[:Nvertices],
                self.vertex_index_buffer[:Nindices],
            )
            if to_numpy:
                buffers = tuple(map(cp.asnumpy, buffers))
            return buffers

        self._kernel_launcher = kernel_launcher
