import os, enum, tempfile, operator, collections

import numpy as np

from cupmb import logger as default_logger
from cupmb.backend import Backend, CudaBackend
from cupmb.utils.IO import cuda_kernels_dir
from cupmb.utils.units import time2str, bytes2str


class BufferType(enum.Enum):
    VERTICES = 0
    VERTEX_INDICES = 1
    BLOCK_INDICES = 2
    BLOCK_MASK = 3
    COUNTERS = 4


class AllocationPolicy(enum.Enum):
    LOCK_BUFFER = 0
    NO_EXTRA_ALLOCATION = 1
    ALLOCATE_EXACT_SIZE = 2
    ALLOCATE_UPPER_POW2 = 3


class AutotuningRigor(enum.Enum):
    ESTIMATE = 0
    MEASURE = 1
    PATIENT = 2
    EXHAUSTIVE = 3


class ParallelMarchingBlocks:

    __slots__ = [
        "_dim",
        "_fdef",
        "_ftype",
        "_itype",
        "_device_id",
        "_extra_build_options",
        "_N",
        "_L",
        "_x0",
        "_isolevel",
        "_autotuning_rigor",
        "_extra_scalar_params",
        "_logger",
        "_source_tmpdir",
        "_source_tmpfile",
        "_buffers",
        "_allocation_policies",
        "_kernel_launcher",
        "_ready",
        "_backend",
    ]

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
        device_id=0,
        extra_build_options=None,
        extra_scalar_params=None,
        autotuning_rigor=AutotuningRigor.ESTIMATE,
        allocation_policy=AllocationPolicy.ALLOCATE_UPPER_POW2,
        logger=default_logger,
        backend=Backend.CUPY,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._backend = CudaBackend(backend)

        self._source_tmpdir = tempfile.TemporaryDirectory()
        self._source_tmpfile = os.path.join(
            self._source_tmpdir.name, "cupmb_func_definition.cuh"
        )
        assert os.path.isdir(self._source_tmpdir.name), self._source_tmpdir.name
        assert not os.path.exists(self._source_tmpfile), self._source_tmpfile

        self._dim = None
        self._fdef = None
        self._ftype = None
        self._itype = None
        self._device_id = None
        self._extra_build_options = None
        self._N = None
        self._L = None
        self._x0 = None
        self._isolevel = None
        self._autotuning_rigor = None
        self._extra_scalar_params = None
        self._logger = None

        # kernel buffers
        buffers = collections.OrderedDict()
        allocation_policies = collections.OrderedDict()
        for e in BufferType:
            buffers[e] = None
            allocation_policies[e] = allocation_policy
        self._buffers = buffers
        self._allocation_policies = allocation_policies

        # kernel launcher and ready (variables set during setup)
        self._kernel_launcher = None
        self._ready = False

        self.logger = logger
        self.dim = dim
        self.fdef = fdef
        self.ftype = ftype
        self.itype = itype
        self.device_id = device_id
        self.extra_build_options = extra_build_options
        self.N = N
        self.L = L
        self.x0 = x0
        self.isolevel = isolevel
        self.autotuning_rigor = autotuning_rigor
        self.extra_scalar_params = extra_scalar_params

    def __call__(self, *fargs, stream=None, to_numpy=False):
        device = self.backend.Device(self.device_id)
        stream = self.backend.default_stream() if stream is None else stream
        assert (stream is None) or isinstance(stream, self.backend.Stream), type(stream)
        with device:
            self._setup(fargs, stream=stream)
            return self._kernel_launcher(stream, fargs, to_numpy)

    def setup(self, *fargs, stream=None):
        device = self.backend.Device(self.device_id)
        stream = self.backend.default_stream() if stream is None else stream
        assert (stream is None) or isinstance(stream, self.backend.Stream), type(stream)
        with device:
            self._setup(fargs, stream=stream)

    def _setup(self, fargs, stream):
        if self._ready:
            return

        device = self.backend.Device(self.device_id)

        device_id = device.id
        device_name = device.name
        device_bus_id = device.pci_bus_id

        msg = f"Initializing parallel marching cubes on device {device_id} [{device_name}] ({device_bus_id})."
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

    def get_buffer_properties(self, buffer_type):
        assert isinstance(buffer_type, BufferType), type(buffer_type)
        if buffer_type == BufferType.VERTICES:
            dtype = self.ftype
            factor = self.dim
        elif buffer_type == BufferType.VERTEX_INDICES:
            dtype = self.itype
            factor = self.dim
        elif buffer_type == BufferType.COUNTERS:
            dtype = self.itype
            factor = None
        elif buffer_type == BufferType.BLOCK_INDICES:
            dtype = self.itype
            factor = None
        elif buffer_type == BufferType.BLOCK_MASK:
            dtype = np.uint8
            factor = None
        else:
            raise NotImplementedError(buffer_type)
        return (dtype, factor)

    def set_buffer_allocation_policy(self, buffer_type, allocation_policy):
        assert isinstance(buffer_type, BufferType), type(buffer_type)
        assert isinstance(allocation_policy, AllocationPolicy), type(allocation_policy)
        self._allocation_policies[buffer_type] = allocation_policy

    def lock_buffer(self, buffer_type):
        self.set_buffer_allocation_policy(buffer_type, AllocationPolicy.LOCK_BUFFER)

    def replace_buffer(self, buffer_type, buf):
        self._update_cuda_buffer(buffer_type, buf, force=True)

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
        return self._allocated_count(BufferType.VERTICES)

    @property
    def allocated_vertex_index_count(self):
        return self._allocated_count(BufferType.VERTEX_INDICES)

    @property
    def allocated_block_index_count(self):
        return self._allocated_count(BufferType.BLOCK_INDICES)

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
    def backend(self):
        return self._backend

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

    def _set_dim(self, dim):
        assert isinstance(dim, int), type(dim)
        assert dim in {1, 2, 3}, dim
        self._ready &= dim == self._dim
        self._dim = dim

    def _get_dim(self):
        return self._dim

    dim = property(_get_dim, _set_dim)

    def _set_fdef(self, fdef):
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

    def _get_fdef(self):
        return self._fdef

    fdef = property(_get_fdef, _set_fdef)

    def _set_device_id(self, device_id):
        assert isinstance(device_id, int), type(device_id)
        if device_id != self.device_id:
            for e in BufferType:
                self.replace_buffer(e, None)
        self._device_id = device_id

    def _get_device_id(self):
        return self._device_id

    device_id = property(_get_device_id, _set_device_id)

    def _set_ftype(self, ftype):
        assert ftype in {np.float32, np.float64}, ftype
        self._ready &= ftype == self._ftype
        self._ftype = ftype

    def _get_ftype(self):
        return self._ftype

    ftype = property(_get_ftype, _set_ftype)

    def _set_itype(self, itype):
        assert itype in {np.uint32, np.uint64}, itype
        self._ready &= itype == self._itype
        self._itype = itype

    def _get_itype(self):
        return self._itype

    itype = property(_get_itype, _set_itype)

    def _set_N(self, N):
        N = self._make_vector(N, dim=self.dim, dtype=self.itype, default=64)
        assert (N >= 2).all(), N
        self._ready &= (N == self._N).all()
        self._N = N

    def _get_N(self):
        return self._N

    N = property(_get_N, _set_N)

    def _set_L(self, L):
        L = self._make_vector(L, dim=self.dim, dtype=self.ftype, default=1.0)
        assert (L > 0).all(), L
        assert np.isfinite(L).all(), L
        self._ready &= True
        self._L = L

    def _get_L(self):
        return self._L

    L = property(_get_L, _set_L)

    def _set_x0(self, x0):
        x0 = self._make_vector(x0, dim=self.dim, dtype=self.ftype, default=0.0)
        assert np.isfinite(x0).all(), x0
        self._ready &= True
        self._x0 = x0

    def _get_x0(self):
        return self._x0

    x0 = property(_get_x0, _set_x0)

    def _set_isolevel(self, isolevel):
        if isolevel is None:
            isolevel = 0.0
        isolevel = self.ftype(isolevel)
        assert np.isfinite(isolevel), isolevel
        self._ready &= True
        self._isolevel = isolevel

    def _get_isolevel(self):
        return self._isolevel

    isolevel = property(_get_isolevel, _set_isolevel)

    def _set_extra_build_options(self, val):
        if val is None:
            val = tuple()
        assert isinstance(val, tuple), type(val)
        self._extra_build_options = val

    def _get_extra_build_options(self):
        return self._extra_build_options

    extra_build_options = property(_get_extra_build_options, _set_extra_build_options)

    def _set_extra_scalar_params(self, extra_scalar_params):
        if extra_scalar_params is None:
            extra_scalar_params = tuple()
        assert isinstance(extra_scalar_params, (list, tuple)), type(extra_scalar_params)
        assert all(isinstance(x, str) for x in extra_scalar_params)
        extra_scalar_params = tuple(extra_scalar_params)
        self._ready &= extra_scalar_params == self._extra_scalar_params
        self._extra_scalar_params = extra_scalar_params

    def _get_extra_scalar_params(self):
        return self._extra_scalar_params

    extra_scalar_params = property(_get_extra_scalar_params, _set_extra_scalar_params)

    def _set_counting_buffer(self, val):
        self._update_cuda_buffer(BufferType.COUNTERS, val)

    def _get_counting_buffer(self):
        return self._buffers[BufferType.COUNTERS]

    counting_buffer = property(_get_counting_buffer, _set_counting_buffer)

    def _set_block_index_buffer(self, val):
        self._update_cuda_buffer(BufferType.BLOCK_INDICES, val)

    def _get_block_index_buffer(self):
        return self._buffers[BufferType.BLOCK_INDICES]

    block_index_buffer = property(_get_block_index_buffer, _set_block_index_buffer)

    def _set_block_mask_buffer(self, val):
        self._update_cuda_buffer(BufferType.BLOCK_MASK, val)

    def _get_block_mask_buffer(self):
        return self._buffers[BufferType.BLOCK_MASK]

    block_mask_buffer = property(_get_block_mask_buffer, _set_block_mask_buffer)

    def _set_vertex_buffer(self, val):
        self._update_cuda_buffer(BufferType.VERTICES, val)

    def _get_vertex_buffer(self):
        return self._buffers[BufferType.VERTICES]

    vertex_buffer = property(_get_vertex_buffer, _set_vertex_buffer)

    def _set_vertex_index_buffer(self, val):
        self._update_cuda_buffer(BufferType.VERTEX_INDICES, val)

    def _get_vertex_index_buffer(self):
        return self._buffers[BufferType.VERTEX_INDICES]

    vertex_index_buffer = property(_get_vertex_index_buffer, _set_vertex_index_buffer)

    def _set_autotuning_rigor(self, val):
        if val is None:
            val = default_logger
        assert isinstance(val, AutotuningRigor), type(val)
        self._ready &= (self._autotuning_rigor is not None) and (
            self._autotuning_rigor >= val
        )
        self._autotuning_rigor = val

    def _get_autotuning_rigor(self):
        return self._autotuning_rigor

    autotuning_rigor = property(_get_autotuning_rigor, _set_autotuning_rigor)

    def _set_logger(self, logger):
        if logger is None:
            logger = default_logger
        self._ready &= True
        self._logger = logger

    def _get_logger(self):
        return self._logger

    logger = property(_get_logger, _set_logger)

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

    def _allocated_count(self, buffer_type):
        itype = self.itype
        (_, factor) = self.get_buffer_properties(buffer_type)
        buf = self._buffers[buffer_type]
        if buf is None:
            return itype(0)
        elif factor is None:
            return itype(buf.size)
        else:
            return itype(buf.size // factor)

    def _update_cuda_buffer(self, buffer_type, a, force=False):
        backend = self.backend
        assert isinstance(buffer_type, BufferType), type(buffer_type)

        assert buffer_type in self._buffers, buffer_type
        assert buffer_type in self._allocation_policies, buffer_type

        actual_buffer = self._buffers[buffer_type]
        allocation_policy = self._allocation_policies[buffer_type]

        assert (actual_buffer is None) or isinstance(
            actual_buffer, backend.Array
        ), type(actual_buffer)
        assert isinstance(allocation_policy, AllocationPolicy), type(allocation_policy)

        dtype, factor = self.get_buffer_properties(buffer_type)

        def locked_buffer_error():
            if (not force) and (allocation_policy == AllocationPolicy.LOCK_BUFFER):
                msg = (
                    f"Attempting to modify buffer {buffer_type} with another buffer but its "
                    "policy was set to LOCK_BUFFER."
                )
                raise RuntimeError(msg)

        def extra_allocation_error(size):
            if (not force) and (
                allocation_policy == AllocationPolicy.NO_EXTRA_ALLOCATION
            ):
                msg = (
                    f"Attempting to reallocate {buffer_type} with size {size} but its policy "
                    "was set to NO_EXTRA_ALLOCATION."
                )
                raise RuntimeError(msg)

        # handle empty buffers
        if (a is None) or (isinstance(a, (int, np.integer)) and (a <= 0)):
            if force:
                self.logger.debug(f"Reseting cuda buffer {buffer_type}.")
                self._buffers[buffer_type] = None
            elif allocation_policy is AllocationPolicy.LOCK_BUFFER:
                locked_buffer_error()
            elif allocation_policy is AllocationPolicy.NO_EXTRA_ALLOCATION:
                pass
            else:
                self.logger.debug(f"Reseting cuda buffer {buffer_type}.")
                self._buffers[buffer_type] = None
            return

        if isinstance(a, backend.Array):
            self.logger.debug(
                f"Replacing cuda buffer {buffer_type} with {backend.Array.__name__}(shape={a.shape}, dtype={a.dtype})."
            )
            locked_buffer_error()
            buf = a
        elif isinstance(a, backend.Buffer):
            self.logger.debug(
                f"Replacing cuda buffer {buffer_type} with {backend.Buffer.__name__}(size={a.size})."
            )
            locked_buffer_error()
            size = a.size
            itemsize = dtype(0).itemsize
            assert size % itemsize == 0, (size, itemsize)
            size = backend.get_nbytes(a)
            buf = backend.array_from_pointer(
                ptr=a, shape=(size // itemsize), dtype=dtype
            )
        elif isinstance(a, (int, np.integer)):
            if factor is not None:
                a *= factor

            if force:
                self.logger.debug(
                    f"Forced reallocation of cuda buffer {buffer_type} to size {a}."
                )
                if allocation_policy is AllocationPolicy.ALLOCATE_UPPER_POW2:
                    a = self._upper_pow2(a)
                with backend.Device(self.device_id):
                    buf = backend.Array(shape=(a,), dtype=dtype)
            else:
                # check current buffer is large enough
                if (actual_buffer is not None) and (actual_buffer.size >= a):
                    return
                elif allocation_policy is AllocationPolicy.LOCK_BUFFER:
                    locked_buffer_error()
                elif allocation_policy is AllocationPolicy.NO_EXTRA_ALLOCATION:
                    extra_allocation_error(a)
                elif allocation_policy is AllocationPolicy.ALLOCATE_UPPER_POW2:
                    a = self._upper_pow2(a)
                elif allocation_policy is AllocationPolicy.ALLOCATE_EXACT_SIZE:
                    pass
                else:
                    raise NotImplementedError(allocation_policy)

                self.logger.debug(
                    f"Reallocation of cuda buffer {buffer_type} to size {a}."
                )
                with backend.Device(self.device_id):
                    buf = backend.Array(shape=(a,), dtype=dtype)
        else:
            raise NotImplementedError(type(a))

        assert buf.dtype == dtype, (buf.dtype, dtype)
        self._buffers[buffer_type] = buf

    def _upper_pow2(self, val):
        assert val >= 1, val
        power = 1
        while power < val:
            power <<= 1
        return power

    def _scalar_kernel_args(self):
        assert self.dx.dtype == self.ftype, self.dx.dtype
        return (self.isolevel,) + tuple(self.x0) + tuple(self.dx)

    def _compute_block_mask_kernel_args(self, fargs):
        if len(fargs) != len(self.extra_scalar_params):
            msg = f"Expected {len(self.extra_scalar_params)} extra parameters but got {len(fargs)}."
            raise RuntimeError(msg)
        return (
            self._scalar_kernel_args()
            + self.backend.get_data_pointers(self.block_mask_buffer)
            + fargs
        )

    def _count_active_blocks_kernel_args(self):
        return self.backend.get_data_pointers(
            self.block_mask_buffer, self.counting_buffer
        )

    def _compact_active_blocks_kernel_args(self):
        args = (self.allocated_block_index_count,)
        args += self.backend.get_data_pointers(
            self.block_mask_buffer,
            self.block_index_buffer,
            self.counting_buffer,
        )
        return args

    def _compute_simplices_kernel_args(self, fargs):
        if len(fargs) != len(self.extra_scalar_params):
            msg = f"Expected {len(self.extra_scalar_params)} extra parameters but got {len(fargs)}."
            raise RuntimeError(msg)
        args = (self.allocated_vertex_count, self.allocated_vertex_index_count)
        args += self._scalar_kernel_args()
        args += self.backend.get_data_pointers(
            self.block_index_buffer,
            self.vertex_buffer,
            self.vertex_index_buffer,
            self.counting_buffer,
        )
        args += fargs
        return args

    def _build_kernels(self, device, B, Nblocks, Nthreads, use_cmem):
        backend = self.backend
        threads_per_blocks = np.prod(B, dtype=np.int64)

        dim = self.dim
        N = self.N

        T = self.flt_ctype
        I = self.int_ctype
        Bstr = ", ".join(map(str, B))
        Nstr = ", ".join(map(str, N))
        warp_size = device.warp_size

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

        kernel_source_file = self.kernel_source_file
        build_options = self.build_options

        (
            compute_block_mask,
            count_active_blocks,
            compact_active_blocks,
            compute_simplices,
        ) = backend.build_module_and_extract_kernels(
            logger=self.logger,
            device_major=device.major,
            device_minor=device.minor,
            kernel_source_file=kernel_source_file,
            kernel_names=kernel_names,
            build_options=build_options,
        )

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

        warp_size = device.warp_size
        max_shared_memory_per_block = device.max_shared_memory_per_block_optin
        max_threads_per_block = device.max_threads_per_block

        Bmax = np.asarray(
            [
                device.max_block_dim_x,
                device.max_block_dim_y,
                device.max_block_dim_z,
            ],
            dtype=itype,
        )[:dim]

        Gmax = np.asarray(
            [
                device.max_grid_dim_x,
                device.max_grid_dim_y,
                device.max_grid_dim_z,
            ],
            dtype=itype,
        )[:dim]

        self.logger.debug(f"Max device threads per block is {max_threads_per_block}.")
        self.logger.debug(f"Max device block size is {tuple(Bmax)}.")
        self.logger.debug(f"Max device grid size is {tuple(Gmax)}.")
        self.logger.debug(
            f"Max device shared memory per block optin is {bytes2str(max_shared_memory_per_block)}."
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

        max_shared_memory_per_block = device.max_shared_memory_per_block_optin

        warp_size = device.warp_size
        max_threads_per_block = min(
            device.max_threads_per_block, device.max_block_dim_x
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
        self.backend.fill_zeros(self.counting_buffer, stream)

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
        if rigor == AutotuningRigor.ESTIMATE:
            return (Bs[0], Ts[0], True)
        elif rigor == AutotuningRigor.MEASURE:
            Nruns = 1
            max_candidates = 4
        elif rigor == AutotuningRigor.PATIENT:
            Nruns = 4
            max_candidates = 16
        elif rigor == AutotuningRigor.EXHAUSTIVE:
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
        backend = self.backend

        Bs = self._generate_candidate_block_size(device)
        if len(Bs) == 0:
            msg = "Could not determine block size."
            raise RuntimeError(msg)

        self.logger.debug(f"Estimating block size for N={N}, candidates are:")
        self.logger.debug("  " + ", ".join(map(str, map(tuple, Bs))))

        warp_size = device.warp_size

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

                ellapsed0_time_s = []
                for i in range(Nruns):
                    ellapsed0_time_s.append(
                        compute_block_mask(
                            tuple(G),
                            tuple(B),
                            self._compute_block_mask_kernel_args(fargs),
                            stream=stream,
                            time_kernel=True,
                        )
                    )
                ellapsed0_time_s = np.asarray(ellapsed0_time_s, dtype=np.float64)
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
                    stream=stream,
                )
                Nactive = backend.asnumpy(self.counting_buffer)[0]
                if Nactive == 0:
                    msg = "Cannot autotune if there is no active blocks."
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                self.block_index_buffer = Nactive

                self._reset_counter(stream)
                compact_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._compact_active_blocks_kernel_args(),
                    stream=stream,
                )
                Nactive_bis = backend.asnumpy(self.counting_buffer)[0]
                assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

                self._reset_counter(stream)
                compute_simplices(
                    (Nactive,),
                    tuple(B),
                    self._compute_simplices_kernel_args(fargs),
                    shared_mem=shared_memory_bytes,
                    stream=stream,
                )
                Ncount = backend.asnumpy(self.counting_buffer)
                self.vertex_buffer = Ncount[0]
                self.vertex_index_buffer = Ncount[1]

                ellapsed1_time_s = []
                for i in range(Nruns):
                    self._reset_counter(stream)
                    ellapsed1_time_s.append(
                        compute_simplices(
                            (Nactive,),
                            tuple(B),
                            self._compute_simplices_kernel_args(fargs),
                            shared_mem=shared_memory_bytes,
                            time_kernel=True,
                            stream=stream,
                        )
                    )
                ellapsed1_time_s = np.asarray(ellapsed1_time_s, dtype=np.float64)
                self.logger.debug(
                    f"   compute_simplices: nruns={Nruns}, "
                    f"mean={time2str(ellapsed1_time_s.mean())}, "
                    f"min={time2str(ellapsed1_time_s.min())}, "
                    f"max={time2str(ellapsed1_time_s.max())}"
                )
                Ncount_bis = backend.asnumpy(self.counting_buffer)
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
        backend = self.backend
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
                tuple(G),
                tuple(B),
                self._compute_block_mask_kernel_args(fargs),
                stream=stream,
            )

            ellapsed0_time_s = []
            for i in range(Nruns):
                self._reset_counter(stream)
                ellapsed0_time_s.append(
                    count_active_blocks(
                        (Ngrid,),
                        (Nthreads,),
                        self._count_active_blocks_kernel_args(),
                        time_kernel=True,
                        stream=stream,
                    )
                )
            ellapsed0_time_s = np.asarray(ellapsed0_time_s, dtype=np.float64)
            self.logger.debug(
                f"    count_active_blocks: nruns={Nruns}, "
                f"mean={time2str(ellapsed0_time_s.mean())}, "
                f"min={time2str(ellapsed0_time_s.min())}, "
                f"max={time2str(ellapsed0_time_s.max())}"
            )
            Nactive = backend.asnumpy(self.counting_buffer)[0]
            if Nactive == 0:
                msg = "Cannot autotune if there is no active blocks."
                self.logger.error(msg)
                raise RuntimeError(msg)

            self.block_index_buffer = Nactive

            ellapsed1_time_s = []
            for i in range(Nruns):
                self._reset_counter(stream)
                ellapsed1_time_s.append(
                    compact_active_blocks(
                        (Ngrid,),
                        (Nthreads,),
                        self._compact_active_blocks_kernel_args(),
                        time_kernel=True,
                        stream=stream,
                    )
                )
            ellapsed1_time_s = np.asarray(ellapsed1_time_s, dtype=np.float64)
            self.logger.debug(
                f"  compact_active_blocks: nruns={Nruns}, "
                f"mean={time2str(ellapsed1_time_s.mean())}, "
                f"min={time2str(ellapsed1_time_s.min())}, "
                f"max={time2str(ellapsed1_time_s.max())}"
            )
            Nactive_bis = backend.asnumpy(self.counting_buffer)[0]
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

        warp_size = device.warp_size
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
            s.logger.info(f"Kernel launcher called with parameters {fargs}.")

            s.counting_buffer = 2
            s.block_mask_buffer = Nblocks

            s.logger.debug(f"Computing block mask for {Nblocks} blocks.")
            kernels[0](G, B, s._compute_block_mask_kernel_args(fargs), stream=stream)

            is_first_call = s.allocated_block_index_count == 0
            if is_first_call:
                s.logger.debug(
                    "Counting active blocks because allocated block index size is zero."
                )
                s._reset_counter(stream)
                kernels[1](
                    Ngrid, Nthreads, s._count_active_blocks_kernel_args(), stream=stream
                )
            else:
                s.logger.debug(
                    f"Compacting active blocks (allocated_block_index={s.allocated_block_index_count})."
                )
                s._reset_counter(stream)
                kernels[2](
                    Ngrid,
                    Nthreads,
                    s._compact_active_blocks_kernel_args(),
                    stream=stream,
                )

            Nactive = s.backend.asnumpy(s.counting_buffer)[0]
            if Nactive == 0:
                s.logger.debug("No active blocks found, skipping block compaction.")
                return (None, None)

            if is_first_call or (s.allocated_block_index_count < Nactive):
                s.logger.info(
                    f"Reallocating block index buffer to handle {Nactive} active blocks."
                )
                s.block_index_buffer = Nactive

                s.logger.debug("Compacting active blocks with reallocated buffer.")
                s._reset_counter(stream)
                kernels[2](
                    Ngrid,
                    Nthreads,
                    s._compact_active_blocks_kernel_args(),
                    stream=stream,
                )

            s.logger.debug(
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
                stream=stream,
            )

            Nvertices, Nindices = tuple(s.backend.asnumpy(s.counting_buffer))
            rellocate_vertices = s.allocated_vertex_count < Nvertices
            rellocate_indices = s.allocated_vertex_index_count < Nindices

            if rellocate_vertices:
                s.logger.info(
                    f"Reallocating vertex buffer to handle {Nvertices} vertices."
                )
                s.vertex_buffer = Nvertices

            if rellocate_indices:
                s.logger.info(
                    f"Reallocating vertex index buffer to handle {Nindices} indices."
                )
                s.vertex_index_buffer = Nindices

            if rellocate_vertices or rellocate_indices:
                s.logger.debug("Generating simplices with reallocated buffers.")
                s._reset_counter(stream)
                kernels[3](
                    (Nactive,),
                    B,
                    s._compute_simplices_kernel_args(fargs),
                    shared_mem=shared_mem,
                    stream=stream,
                )

            buffers = (
                s.vertex_buffer[: dim * Nvertices].reshape(-1, dim),
                s.vertex_index_buffer[: dim * Nindices].reshape(-1, dim),
            )
            if to_numpy:
                buffers = tuple(map(s.backend.asnumpy, buffers))
            return buffers

        self._kernel_launcher = kernel_launcher
