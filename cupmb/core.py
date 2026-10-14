#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, time, logging, operator, collections, functools

import numpy as np
import jinja2

from cupmb import logger as default_logger, log_once
from cupmb.backend import Backend, CudaBackend
from cupmb.parameters import ExtraParameter
from cupmb.properties import VertexProperty
from cupmb.bbox import BoundingBox
from cupmb.enums import BufferType, AllocationPolicy, AutotuningRigor
from cupmb.utils.IO import cuda_kernels_dir
from cupmb.utils.kernel_cache import (
    kernel_cache_dir,
    kernel_cache_enabled,
    kernel_cache_key,
    load_kernel_cache_entry,
    store_kernel_cache_entry,
)
from cupmb.utils.types import dtype_to_ctype, ctype_to_dtype, normalize_scalar_type
from cupmb.utils.units import time2str, bytes2str, decimal_unit2str


class CudaBlockSizeError(ValueError):
    def __init__(self, kernel_name, threads_per_block, max_reported_kernel_thread_per_block):
        msg = (
            f"Attempted to use {threads_per_block} threads per block for kernel {kernel_name} "
            f"while reported kernel maximum is {max_reported_kernel_thread_per_block}."
        )
        super().__init__(msg)
        self.threads_per_block = threads_per_block
        self.max_reported_kernel_thread_per_block = max_reported_kernel_thread_per_block


class _NoActiveBlocksError(RuntimeError):
    """Raised during autotuning when the scalar field does not contain any active block."""


def _tuple_str(values):
    """Format a sequence of integers as a tuple: str(tuple) uses the repr of numpy scalars since numpy 2."""
    return "(" + ", ".join(str(int(v)) for v in values) + ")"


class ParallelMarchingBlocks:

    __slots__ = [
        "_dim",
        "_fdef",
        "_pdef",
        "_ftype",
        "_itype",
        "_device_id",
        "_export_scalar_field",
        "_extra_build_options",
        "_vertex_properties",
        "_N",
        "_bbox",
        "_update_bbox",
        "_autotuning_rigor",
        "_extra_params",
        "_logger",
        "_kernel_launcher",
        "_ready",
        "_backend",
        "_buffers",
        "_allocation_policies",
        "_default_allocation_policy",
        "_ignore_per_kernel_thread_limits",
        "_kernel_source_code",
        "_scalar_kernel_args_cache",
    ]

    def __init__(
        self,
        dim,
        fdef="",
        pdef="",
        N=64,
        L=1.0,
        x0=0.0,
        bbox=None,
        update_bbox=None,
        ftype=np.float32,
        itype=np.uint32,
        device_id=0,
        extra_build_options=None,
        extra_params=None,
        vertex_properties=None,
        export_scalar_field=False,
        enable_cuda_std_lib=False,
        initialize_context=True,
        ignore_per_kernel_thread_limits=True,
        autotuning_rigor=AutotuningRigor.ESTIMATE,
        default_allocation_policy=AllocationPolicy.ALLOCATE_UPPER_POW2,
        logger=default_logger,
        backend=Backend.CUPY,
        *args,
        **kwargs,
    ):

        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            logger.error(
                f"Failed to call {type(self).__mro__[1].__name__}.__init__ with args={args} and kwargs={kwargs}."
            )
            raise

        for name in self.__slots__:
            setattr(self, name, None)

        self._backend = CudaBackend(backend, initialize_context=initialize_context)
        self._ready = False

        self.logger = logger
        self.default_allocation_policy = default_allocation_policy

        self.ftype = ftype
        self.itype = itype
        self.dim = dim
        self.fdef = fdef
        self.pdef = pdef
        self.device_id = device_id
        self.extra_build_options = extra_build_options
        self.bbox = bbox
        if bbox is None:
            self.L = L
            self.x0 = x0
        self.N = N
        self.update_bbox = update_bbox
        self.autotuning_rigor = autotuning_rigor
        self.extra_params = extra_params
        self.vertex_properties = vertex_properties
        self.export_scalar_field = export_scalar_field
        self.ignore_per_kernel_thread_limits = ignore_per_kernel_thread_limits

        if enable_cuda_std_lib:
            cuda_include_dirs = self.backend.determine_cuda_include_dirs(self.logger)
            self.extra_build_options = (
                self.extra_build_options + ("-DCUPMB_ENABLE_CUDA_STD_LIB",) + tuple(f"-I{d}" for d in cuda_include_dirs)
            )

    def __call__(
        self,
        *param_args,
        stream=None,
        to_numpy=False,
        return_elapsed_time=False,
        **param_kwargs,
    ):
        device, stream, params = self.setup(*param_args, stream=stream, **param_kwargs)
        with device:
            return self._kernel_launcher(stream, params, to_numpy, return_elapsed_time)

    def setup(self, *param_args, stream=None, **param_kwargs):
        device = self.backend.Device(self.device_id)
        stream = self.backend.default_stream() if (stream is None) else stream
        params = self._prepare_extra_params(param_args, param_kwargs)
        self.update_bbox(self._bbox, *params)
        with device:
            self._setup(params, stream=stream)
        return (device, stream, params)

    def get_buffer_properties(self, buffer_key):
        assert buffer_key in self._buffers, buffer_key
        if isinstance(buffer_key, VertexProperty):
            dtype = self.ctype_to_dtype(buffer_key.dtype)
            shape = buffer_key.effective_shape(self.dim)
        elif isinstance(buffer_key, BufferType):
            if buffer_key == BufferType.SCALAR_FIELD:
                dtype = self.ftype
                shape = tuple()
            elif buffer_key == BufferType.VERTICES:
                dtype = self.ftype
                shape = (self.dim,)
            elif buffer_key == BufferType.SIMPLICES:
                dtype = self.itype
                shape = (self.dim,)
            elif buffer_key == BufferType.COUNTERS:
                dtype = self.itype
                shape = tuple()
            elif buffer_key == BufferType.BLOCK_INDICES_PING:
                dtype = self.itype
                shape = tuple()
            elif buffer_key == BufferType.BLOCK_INDICES_PONG:
                dtype = self.itype
                shape = tuple()
            elif buffer_key == BufferType.BLOCK_MASK:
                dtype = np.uint8
                shape = tuple()
            else:
                raise NotImplementedError(buffer_key)
        else:
            raise NotImplementedError(type(buffer_key))
        size = np.prod(shape, dtype=np.int64)
        return (dtype, shape, size)

    def set_buffer_allocation_policy(self, buffer_key, allocation_policy):
        assert buffer_key in self._buffers, buffer_key
        assert (allocation_policy is None) or isinstance(allocation_policy, AllocationPolicy), type(allocation_policy)
        default_allocation_policy = self.get_default_buffer_allocation_policy(buffer_key)
        allocation_policy = default_allocation_policy if (allocation_policy is None) else allocation_policy
        self.logger.debug(f"Setting allocation policy of buffer {buffer_key} to {allocation_policy}.")
        self._allocation_policies[buffer_key] = allocation_policy

    def get_default_buffer_allocation_policy(self, buffer_key):
        if buffer_key in {
            BufferType.SCALAR_FIELD,
            BufferType.BLOCK_MASK,
            BufferType.COUNTERS,
        }:
            return AllocationPolicy.ALLOCATE_EXACT_SIZE
        else:
            return self.default_allocation_policy

    def replace_buffer(self, buffer_key, buf):
        self._update_cuda_buffer(buffer_key, buf, force=True)

    def lock_buffer(self, buffer_key):
        self.set_buffer_allocation_policy(buffer_key, AllocationPolicy.LOCK_BUFFER)

    def lock_all_buffers(self):
        for buffer_key in tuple(BufferType) + tuple(self.vertex_properties):
            self.lock_buffer(buffer_key)

    def allocated_elements(self, buffer_key):
        buf = self._buffers[buffer_key]
        return self.itype(0 if (buf is None) else buf.shape[0])

    @property
    def Nscalars(self):
        return np.prod(self.N + 1, dtype=np.uint64)

    @property
    def has_extra_parameters(self):
        return len(self._extra_params) > 0

    @property
    def has_vertex_properties(self):
        return len(self._vertex_properties) > 0

    @property
    def xmin(self):
        return self.x0

    @property
    def xmax(self):
        return (self.x0 + self.L).astype(self.ftype)

    @property
    def dx(self):
        return (self.L / self.N).astype(self.ftype)

    @property
    def flt_ctype(self):
        return dtype_to_ctype[self.ftype]

    @property
    def int_ctype(self):
        return dtype_to_ctype[self.itype]

    @property
    def extra_templates(self):
        extra_templates = ", ".join(self._get_ctypes(*tuple(p.ctype for p in self.extra_params.values())))
        if extra_templates:
            extra_templates = ", " + extra_templates
        return extra_templates

    @property
    def allocated_scalars(self):
        return self.allocated_elements(BufferType.SCALAR_FIELD)

    @property
    def allocated_vertex_count(self):
        return self.allocated_elements(BufferType.VERTICES)

    @property
    def allocated_simplices_count(self):
        return self.allocated_elements(BufferType.SIMPLICES)

    @property
    def allocated_ping_block_index_count(self):
        return self.allocated_elements(BufferType.BLOCK_INDICES_PING)

    @property
    def allocated_pong_block_index_count(self):
        return self.allocated_elements(BufferType.BLOCK_INDICES_PONG)

    @property
    def kernel_source_template(self):
        template_file = os.path.join(cuda_kernels_dir, f"pmb{self.dim}D.cuh")

        if not os.path.isfile(template_file):
            msg = f"Could not find cupmb kernel template file:///{template_file}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        with open(template_file) as f:
            template = f.read()

        return jinja2.Template(template)

    @property
    def kernel_source_code(self):
        return self._kernel_source_code

    @property
    def backend(self):
        return self._backend

    @classmethod
    def required_build_options(cls):
        return (
            "-std=c++17",
            f"-I{cuda_kernels_dir}",
        )

    @property
    def build_options(self):
        bopts = self.required_build_options() + self.extra_build_options
        return bopts

    @property
    def vertex_properties_buffers(self):
        return tuple(map(self._buffers.__getitem__, self.vertex_properties))

    def _set_dim(self, dim):
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
            raise TypeError(f"Dimension should be an integer, got {type(dim).__name__}.")
        dim = int(dim)
        if dim not in {1, 2, 3}:
            msg = f"Dimension should be 1, 2 or 3, got dim={dim}."
            self.logger.error(msg)
            raise ValueError(msg)
        if dim != self._dim:
            # changing dimension resets the bounding box to the unit box, the grid
            # size is kept if it was isotropic and reset to its default value otherwise
            self._dim = dim
            self.bbox = None
            if self._N is not None:
                N = self._N
                self.N = N[0] if (N == N[0]).all() else None
            self._release_buffers("the dimension changed")
            self._ready = False

    def _get_dim(self):
        return self._dim

    dim = property(_get_dim, _set_dim)

    def _set_default_allocation_policy(self, default_allocation_policy):
        assert isinstance(default_allocation_policy, AllocationPolicy), type(default_allocation_policy)
        self._default_allocation_policy = default_allocation_policy
        self._ready &= True

    def _get_default_allocation_policy(self):
        return self._default_allocation_policy

    default_allocation_policy = property(_get_default_allocation_policy, _set_default_allocation_policy)

    def _set_fdef(self, fdef):
        assert isinstance(fdef, str), type(fdef)
        if os.path.isfile(fdef):
            with open(fdef) as f:
                fdef = f.read()
        if fdef != self._fdef:
            self._ready = False
            self._fdef = fdef

    def _get_fdef(self):
        return self._fdef

    fdef = property(_get_fdef, _set_fdef)

    def _set_pdef(self, pdef):
        if pdef is None:
            pdef = ""

        if pdef != "" and self.export_scalar_field:
            msg = "Domain decomposition feature from predicate is not compatible with export_scalar_field."
            self.logger.error(msg)
            raise RuntimeError(msg)

        assert isinstance(pdef, str), type(pdef)
        if os.path.isfile(pdef):
            with open(pdef) as f:
                pdef = f.read()

        if pdef != self._pdef:
            self._ready = False
            self._pdef = pdef

    def _get_pdef(self):
        return self._pdef

    pdef = property(_get_pdef, _set_pdef)

    @property
    def has_block_predicate(self):
        return self._pdef != ""

    def _set_vertex_properties(self, vertex_properties):
        if vertex_properties is None:
            vertex_properties = tuple()

        assert isinstance(vertex_properties, tuple), type(vertex_properties)
        assert all(map(lambda vp: isinstance(vp, VertexProperty), vertex_properties)), vertex_properties

        property_names = tuple(map(VertexProperty.name.fget, vertex_properties))
        assert len(property_names) == len(set(property_names)), "Some properties share the same name."

        # forget everything about old vertex properties
        if self._vertex_properties is not None:
            for vp in self._vertex_properties:
                self._delete_buffer(vp)

        # register new vertex properties buffers
        for vp in vertex_properties:
            self._create_buffer(vp)

        self._vertex_properties = vertex_properties
        self._ready = False

    def _get_vertex_properties(self):
        return self._vertex_properties

    vertex_properties = property(_get_vertex_properties, _set_vertex_properties)

    def _set_device_id(self, device_id):
        if isinstance(device_id, bool) or not isinstance(device_id, (int, np.integer)):
            raise TypeError(f"Device id should be an integer, got {type(device_id).__name__}.")
        device_id = int(device_id)
        if device_id != self.device_id:
            if self._buffers is None:
                self._reset_buffers()
            else:
                # keep registered buffers (vertex properties) and allocation policies
                self._release_buffers("the device changed")
            self._ready = False
            self._device_id = device_id

    def _get_device_id(self):
        return self._device_id

    device_id = property(_get_device_id, _set_device_id)

    def _set_ftype(self, ftype):
        ftype = normalize_scalar_type(ftype)
        if ftype not in {np.float32, np.float64}:
            msg = f"Floating point type should be float32 or float64, got {ftype.__name__}."
            self.logger.error(msg)
            raise TypeError(msg)
        if ftype != self._ftype:
            self._ftype = ftype
            if self._bbox is not None:
                # recast the bounding box in place (update_bbox may hold a reference to it)
                self._bbox._astype(ftype)
            self._release_buffers("the floating point type changed")
            self._ready = False

    def _get_ftype(self):
        return self._ftype

    ftype = property(_get_ftype, _set_ftype)

    def _set_itype(self, itype):
        itype = normalize_scalar_type(itype)
        if itype not in {np.uint32, np.uint64}:
            msg = f"Integer type should be uint32 or uint64, got {itype.__name__}."
            self.logger.error(msg)
            raise TypeError(msg)
        if itype != self._itype:
            self._itype = itype
            if self._N is not None:
                self._N = self._make_vector(self._N, dim=self.dim, dtype=itype, default=64)
            self._release_buffers("the integer type changed")
            self._ready = False

    def _get_itype(self):
        return self._itype

    itype = property(_get_itype, _set_itype)

    def _set_N(self, N):
        N = self._make_vector(N, dim=self.dim, dtype=self.itype, default=64)
        assert (N >= 2).all(), N
        self._ready &= np.array_equal(N, self._N) if (self._N is not None) else False
        self._N = N

    def _get_N(self):
        return self._N

    N = property(_get_N, _set_N)

    def _set_bbox(self, bbox):
        assert self.dim is not None
        assert self.ftype is not None
        if bbox is None:
            xmin = np.zeros(shape=(self.dim,), dtype=self.ftype)
            xmax = np.ones(shape=(self.dim,), dtype=self.ftype)
            bbox = BoundingBox(xmin=xmin, xmax=xmax)
        assert isinstance(bbox, BoundingBox) or callable(bbox), type(bbox)
        assert bbox.dtype == self.ftype, (bbox.dtype, self.ftype)
        assert bbox.dim == self.dim, (bbox.dim, self.dim)
        self._bbox = bbox
        self._ready &= True

    def _get_bbox(self):
        return self._bbox

    bbox = property(_get_bbox, _set_bbox)

    def _set_update_bbox(self, update_bbox):
        if update_bbox is None:

            def update_bbox(*args, **kwds):
                pass

        assert callable(update_bbox), type(update_bbox)
        self._update_bbox = update_bbox
        self._ready &= True

    def _get_update_bbox(self):
        return self._update_bbox

    update_bbox = property(_get_update_bbox, _set_update_bbox)

    def _set_L(self, L):
        L = self._make_vector(L, dim=self.dim, dtype=self.ftype, default=1.0)
        assert (L > 0).all(), L
        assert np.isfinite(L).all(), L
        self._ready &= True
        self._bbox._xmax[...] = self._bbox.xmin + L

    def _get_L(self):
        return self.bbox.length

    L = property(_get_L, _set_L)

    def _set_x0(self, x0):
        x0 = self._make_vector(x0, dim=self.dim, dtype=self.ftype, default=0.0)
        assert np.isfinite(x0).all(), x0
        self._ready &= True
        L = self.bbox.length
        self.bbox.xmin[...] = x0
        self.bbox.xmax[...] = x0 + L

    def _get_x0(self):
        return self.bbox.xmin

    x0 = property(_get_x0, _set_x0)

    def _set_extra_build_options(self, val):
        if val is None:
            val = tuple()
        assert isinstance(val, tuple), type(val)
        self._ready &= val == self._extra_build_options
        self._extra_build_options = val

    def _get_extra_build_options(self):
        return self._extra_build_options

    extra_build_options = property(_get_extra_build_options, _set_extra_build_options)

    def _set_extra_params(self, extra_params):
        if extra_params is None:
            extra_params = collections.OrderedDict()
        elif isinstance(extra_params, collections.OrderedDict):
            pass
        elif isinstance(extra_params, dict):
            raise ValueError(
                "Extra parameters should be passed ordered by using either a "
                "collections.OrderedDict, or an iterable of (key, value) pairs, instead of a dictionary."
            )
        elif isinstance(extra_params, (list, tuple)):
            assert all(len(_) == 2 for _ in extra_params)
            extra_params = collections.OrderedDict(extra_params)
        else:
            raise NotImplementedError(type(extra_params))

        for k, v in extra_params.items():
            if isinstance(v, ExtraParameter):
                pass
            elif isinstance(v, str):
                v = ExtraParameter(v)
            else:
                raise TypeError(type(v))
            extra_params[k] = v

        self._ready &= extra_params == self._extra_params
        self._extra_params = extra_params

    def _get_extra_params(self):
        return self._extra_params

    extra_params = property(_get_extra_params, _set_extra_params)

    def _set_counting_buffer(self, val):
        self._update_cuda_buffer(BufferType.COUNTERS, val)

    def _get_counting_buffer(self):
        return self._buffers[BufferType.COUNTERS]

    counting_buffer = property(_get_counting_buffer, _set_counting_buffer)

    def _set_block_index_ping_buffer(self, val):
        self._update_cuda_buffer(BufferType.BLOCK_INDICES_PING, val)

    def _get_block_index_ping_buffer(self):
        return self._buffers[BufferType.BLOCK_INDICES_PING]

    block_index_ping_buffer = property(_get_block_index_ping_buffer, _set_block_index_ping_buffer)

    def _set_block_index_pong_buffer(self, val):
        self._update_cuda_buffer(BufferType.BLOCK_INDICES_PONG, val)

    def _get_block_index_pong_buffer(self):
        return self._buffers[BufferType.BLOCK_INDICES_PONG]

    block_index_pong_buffer = property(_get_block_index_pong_buffer, _set_block_index_pong_buffer)

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

    def _set_simplex_buffer(self, val):
        self._update_cuda_buffer(BufferType.SIMPLICES, val)

    def _get_simplex_buffer(self):
        return self._buffers[BufferType.SIMPLICES]

    simplex_buffer = property(_get_simplex_buffer, _set_simplex_buffer)

    def _set_scalar_field_buffer(self, val):
        self._update_cuda_buffer(BufferType.SCALAR_FIELD, val)

    def _get_scalar_field_buffer(self):
        return self._buffers[BufferType.SCALAR_FIELD]

    scalar_field_buffer = property(_get_scalar_field_buffer, _set_scalar_field_buffer)

    def _set_autotuning_rigor(self, val):
        if val is None:
            val = AutotuningRigor.ESTIMATE
        elif isinstance(val, str):
            val = AutotuningRigor[val.strip().upper()]
        else:
            val = AutotuningRigor(val)
        # kernels have to be autotuned again only if the requested rigor is higher
        self._ready &= (self._autotuning_rigor is not None) and (self._autotuning_rigor >= val)
        self._autotuning_rigor = val

    def _get_autotuning_rigor(self):
        return self._autotuning_rigor

    autotuning_rigor = property(_get_autotuning_rigor, _set_autotuning_rigor)

    def _set_logger(self, logger):
        if logger is None:
            logger = default_logger
        self._logger = logger

    def _get_logger(self):
        return self._logger

    logger = property(_get_logger, _set_logger)

    def _set_export_scalar_field(self, export_scalar_field):
        if export_scalar_field is None:
            export_scalar_field = False
        assert isinstance(export_scalar_field, bool), type(export_scalar_field)

        if export_scalar_field and self.has_block_predicate:
            msg = "Domain decomposition feature from predicate is not compatible with export_scalar_field."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self._export_scalar_field = export_scalar_field

    def _get_export_scalar_field(self):
        return self._export_scalar_field

    export_scalar_field = property(_get_export_scalar_field, _set_export_scalar_field)

    def _set_ignore_per_kernel_thread_limits(self, ignore_per_kernel_thread_limits):
        if ignore_per_kernel_thread_limits is None:
            ignore_per_kernel_thread_limits = False
        assert isinstance(ignore_per_kernel_thread_limits, bool), type(ignore_per_kernel_thread_limits)
        self._ignore_per_kernel_thread_limits = ignore_per_kernel_thread_limits

    def _get_ignore_per_kernel_thread_limits(self):
        return self._ignore_per_kernel_thread_limits

    ignore_per_kernel_thread_limits = property(
        _get_ignore_per_kernel_thread_limits, _set_ignore_per_kernel_thread_limits
    )

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
        # shared scalars and prefix sum offsets (vertex and simplex counts) are required in all dimensions
        m0 = threads * ftype(0).itemsize
        m2 = 2 * (warps_per_block + 1) * itype(0).itemsize
        if dim == 1:
            return max(m0, m2)
        elif dim in (2, 3):
            # per thread vertex offsets and vertex counts (simplices reference vertices of neighbor threads)
            m1 = threads * (itype(0).itemsize + np.uint8(0).itemsize)
            return max(m0, m1, m2)
        else:
            raise NotImplementedError(dim)

    def _reset_buffers(self):
        self.logger.debug("Resetting all buffers.")
        self._buffers = collections.OrderedDict()
        self._allocation_policies = collections.OrderedDict()
        for e in BufferType:
            self._create_buffer(e)

    def _release_buffers(self, reason):
        """Release all buffers but keep registered buffers (vertex properties) and allocation policies."""
        if self._buffers is None:
            return
        user_policies = {AllocationPolicy.LOCK_BUFFER, AllocationPolicy.NO_EXTRA_ALLOCATION}
        for buffer_key, buf in self._buffers.items():
            if buf is None:
                continue
            if self._allocation_policies[buffer_key] in user_policies:
                self.logger.warning(
                    f"Buffer {buffer_key} has been released because {reason} but its allocation policy "
                    f"is {self._allocation_policies[buffer_key]}, it has to be replaced by using replace_buffer."
                )
            self._buffers[buffer_key] = None
        self.logger.debug(f"Released all buffers because {reason}.")

    def _reset_allocated_buffers(self):
        self.logger.debug("Resetting internally allocated buffers.")
        excluded = {AllocationPolicy.LOCK_BUFFER, AllocationPolicy.NO_EXTRA_ALLOCATION}
        for k in self._buffers.keys():
            if self._allocation_policies[k] in excluded:
                continue
            self.replace_buffer(k, None)

    def _create_buffer(self, buffer_key):
        assert isinstance(buffer_key, (BufferType, VertexProperty)), type(buffer_key)
        default_allocation_policy = self.get_default_buffer_allocation_policy(buffer_key)
        self.logger.debug(
            f"Registering buffer {buffer_key} with default allocation policy {default_allocation_policy}."
        )
        if (buffer_key in self._buffers) or (buffer_key in self._allocation_policies):
            msg = f"Buffer {buffer_key} already exists."
            self.logger.error(msg)
            raise RuntimeError(msg)
        self._buffers[buffer_key] = None
        self._allocation_policies[buffer_key] = default_allocation_policy

    def _delete_buffer(self, buffer_key):
        assert isinstance(buffer_key, (BufferType, VertexProperty)), type(buffer_key)
        self.logger.debug(f"Removing buffer {buffer_key} from registered buffers.")
        if isinstance(buffer_key, BufferType):
            msg = f"Buffer {buffer_key} cannot be removed."
            self.logger.error(msg)
            raise RuntimeError(msg)
        if not ((buffer_key in self._buffers) and (buffer_key in self._allocation_policies)):
            msg = f"Buffer {buffer_key} has not been registered yet."
            self.logger.error(msg)
            raise RuntimeError(msg)
        self._buffers.pop(buffer_key)
        self._allocation_policies.pop(buffer_key)

    def _get_dtypes(self, *types):
        replacements = {"T": self.ftype, "I": self.itype}
        dtypes = []
        for t in types:
            if t in replacements:
                dtypes.append(replacements[t])
            else:
                dtypes.append(t)
        return tuple(dtypes)

    def _get_ctypes(self, *types):
        replacements = {"T": self.flt_ctype, "I": self.int_ctype}
        ctypes = []
        for t in types:
            if t in replacements:
                ctypes.append(replacements[t])
            elif t[0] in replacements and t[1] in {"&", "*"}:
                ctypes.append(replacements[t[0]] + t[1])
            elif t in dtype_to_ctype:
                ctypes.append(dtype_to_ctype[t])
            else:
                ctypes.append(t)
        return tuple(ctypes)

    def _prepare_extra_params(self, param_args, param_kwargs):
        d_param_args = collections.deque(param_args)
        params = []
        for pname, pvalue in self.extra_params.items():
            if pname in param_kwargs:
                arg = param_kwargs.pop(pname)
            elif d_param_args:
                arg = d_param_args.popleft()
            elif pvalue.is_callable:
                arg = pvalue()
            else:
                msg = (
                    f"Could not determine extra parameter '{pname}' from param_args={param_args} "
                    f"param_kwargs={param_kwargs}"
                )
                self.logger.error(msg)
                raise RuntimeError(msg)

            params.append(self._convert_extra_param(pname, pvalue.ctype, arg))

        if param_kwargs:
            msg = (
                f"Got unexpected keyword argument(s) {', '.join(map(repr, param_kwargs))}, "
                f"expected extra parameters are ({', '.join(self.extra_params.keys())})."
            )
            self.logger.error(msg)
            raise TypeError(msg)

        if d_param_args:
            msg = (
                f"Got {len(param_args)} positional argument(s) but only {len(param_args) - len(d_param_args)} "
                f"could be matched to extra parameters ({', '.join(self.extra_params.keys())})."
            )
            self.logger.error(msg)
            raise TypeError(msg)

        return tuple(params)

    def _convert_extra_param(self, pname, ctype, arg):
        is_pointer = isinstance(ctype, str) and ctype.strip().endswith("*")
        if is_pointer:
            try:
                dtype = self.ctype_to_dtype(ctype.strip()[:-1])
            except KeyError:
                dtype = None  # user defined type, cannot check the array type
            if (dtype is not None) and hasattr(arg, "dtype") and arg.dtype != dtype:
                msg = (
                    f"Expected an array of type {dtype.__name__} for parameter '{pname}' "
                    f"but got an array of type {arg.dtype}."
                )
                self.logger.error(msg)
                raise ValueError(msg)
            (arg,) = self.backend.get_data_pointers(arg)
        else:
            try:
                dtype = self.ctype_to_dtype(ctype)
            except KeyError:
                dtype = None  # user defined type, passed as is
            if dtype is not None:
                arg = dtype(arg)
        return arg

    def _setup(self, params, stream):
        if self._ready:
            return

        device = self.backend.Device(self.device_id)

        device_id = device.id
        device_name = device.name
        device_bus_id = device.pci_bus_id

        msg = f"Initializing parallel marching cubes on device {device_id} [{device_name}] ({device_bus_id})."
        self.logger.info(msg)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Device global memory is {bytes2str(device.global_memory_bytes)} "
                f"({bytes2str(device.free_memory_bytes)} free)."
            )
            self.logger.debug(f"Max device threads per block is {device.max_threads_per_block}.")
            self.logger.debug(f"Max device block size is {tuple(device.max_block_dim)}.")
            self.logger.debug(f"Max device grid size is {tuple(device.max_grid_dim)}.")
            self.logger.debug(
                f"Max device shared memory per block optin is {bytes2str(device.max_shared_memory_per_block_optin)}."
            )

        self._generate_kernel_source_code_from_template()

        Bopt, Topt, use_cmem = self._autotune_kernels(stream, device, params)

        self._build_kernel_launcher(device, Bopt, Topt, use_cmem)
        self._ready = True

    def _generate_kernel_source_code_from_template(self):
        template = self.kernel_source_template

        fdef = self.fdef
        pdef = self.pdef

        if fdef is None:
            msg = "Scalar function has not been defined yet."
            self.logger.error(msg)
            raise RuntimeError(msg)

        vertex_properties_kernel_src = VertexProperty.generate_kernel(
            T=self.flt_ctype,
            I=self.int_ctype,
            dim=self.dim,
            vertex_properties=self.vertex_properties,
        )

        self._kernel_source_code = template.render(
            cupmb_func_definitions=fdef + os.linesep + pdef,
            cupmb_vertex_properties=vertex_properties_kernel_src,
        )

    def _update_cuda_buffer(self, buffer_key, a, force=False):
        if (not force) and isinstance(a, (int, np.integer)) and (a > 0):
            # fast path: current buffer is large enough
            buf = self._buffers[buffer_key]
            if (buf is not None) and (buf.shape[0] >= a):
                return

        backend = self.backend

        assert buffer_key in self._buffers, buffer_key
        assert buffer_key in self._allocation_policies, buffer_key

        actual_buffer = self._buffers[buffer_key]
        allocation_policy = self._allocation_policies[buffer_key]

        assert (actual_buffer is None) or isinstance(actual_buffer, backend.Array), type(actual_buffer)
        assert isinstance(allocation_policy, AllocationPolicy), type(allocation_policy)

        dtype, elem_shape, elem_size = self.get_buffer_properties(buffer_key)
        itemsize = dtype(0).itemsize
        elem_bytes = elem_size * itemsize

        def locked_buffer_error():
            if (not force) and (allocation_policy == AllocationPolicy.LOCK_BUFFER):
                msg = (
                    f"Attempting to modify buffer {buffer_key} with another buffer but its "
                    "policy was set to LOCK_BUFFER."
                )
                self.logger.error(msg)
                raise RuntimeError(msg)

        def extra_allocation_error(size):
            if (not force) and (allocation_policy == AllocationPolicy.NO_EXTRA_ALLOCATION):
                msg = (
                    f"Attempting to reallocate {buffer_key} with size {size} but its policy "
                    "was set to NO_EXTRA_ALLOCATION."
                )
                self.logger.error(msg)
                raise RuntimeError(msg)

        # handle empty buffers
        if (a is None) or (isinstance(a, (int, np.integer)) and (a <= 0)):
            if force:
                self.logger.debug(f"Reseting cuda buffer {buffer_key}.")
                self._buffers[buffer_key] = None
            elif allocation_policy is AllocationPolicy.LOCK_BUFFER:
                locked_buffer_error()
            else:
                self.logger.debug(f"Reseting cuda buffer {buffer_key}.")
                self._buffers[buffer_key] = None
            return

        if isinstance(a, backend.Array):
            self.logger.debug(
                f"Replacing cuda buffer {buffer_key} with {backend.Array.__name__}(size={a.size}, dtype={a.dtype})."
            )
            locked_buffer_error()
            if a.dtype != dtype:
                msg = f"Buffer {buffer_key} expects an array of type {np.dtype(dtype)} but got {a.dtype}."
                self.logger.error(msg)
                raise TypeError(msg)
            if not a.flags.c_contiguous:
                # ravel() would silently copy the array and the user buffer would never be written to
                msg = f"Buffer {buffer_key} cannot be replaced by a non C-contiguous array."
                self.logger.error(msg)
                raise ValueError(msg)
            buf = a.ravel()[: (a.size // elem_size) * elem_size].reshape((-1,) + elem_shape)
        elif isinstance(a, backend.Buffer):
            self.logger.debug(
                f"Replacing cuda buffer {buffer_key} with {backend.Buffer.__name__}(size={backend.get_nbytes(a)})."
            )
            locked_buffer_error()
            bsize = backend.get_nbytes(a)
            shape = (bsize // elem_bytes,) + elem_shape
            buf = backend.array_from_pointer(ptr=a, shape=shape, dtype=dtype)
        elif isinstance(a, (int, np.integer)):
            elem_count = int(a)
            del a

            if not force:
                # check current buffer is large enough (fast path, no device query)
                if self.allocated_elements(buffer_key) >= elem_count:
                    return
                elif allocation_policy is AllocationPolicy.LOCK_BUFFER:
                    locked_buffer_error()
                elif allocation_policy is AllocationPolicy.NO_EXTRA_ALLOCATION:
                    extra_allocation_error(elem_count)

            if allocation_policy is AllocationPolicy.ALLOCATE_UPPER_POW2:
                elem_count = self._upper_pow2(elem_count)

            nbytes = elem_count * elem_bytes
            shape = (elem_count,) + elem_shape

            device = self.backend.Device(self.device_id)
            mem_free = device.free_memory_bytes

            if nbytes >= 0.95 * mem_free:
                log_once(
                    "warning",
                    f"Device {device.name} has only {bytes2str(mem_free)} of memory left.",
                    self.logger,
                )

            if self.logger.isEnabledFor(logging.DEBUG):
                mem_pct = 100.0 * nbytes / float(device.global_memory_bytes)
                self.logger.debug(
                    f"{'Forced reallocation' if force else 'Reallocation'} of {elem_count} {buffer_key} "
                    f"elements ({dtype.__name__}), new shape is {shape}, new size is {bytes2str(nbytes)} "
                    f"({mem_pct:2.1f}% of device memory)."
                )

            with device:
                buf = backend.Array(shape=shape, dtype=dtype)
        else:
            raise NotImplementedError(type(a))

        assert buf.dtype == dtype, (buf.dtype, dtype)
        assert buf.shape[1:] == elem_shape, (buf.shape, elem_shape)
        self._buffers[buffer_key] = buf

    def _upper_pow2(self, val):
        assert val >= 1, val
        power = 1
        while power < val:
            power <<= 1
        return power

    def _scalar_kernel_args(self):
        # cached as long as the domain, the grid size and the floating point type do not change
        key = (self.ftype, self._bbox.xmin.tobytes(), self._bbox.xmax.tobytes(), self.N.tobytes())
        cache = self._scalar_kernel_args_cache
        if (cache is None) or (cache[0] != key):
            dx = self.dx
            assert dx.dtype == self.ftype, dx.dtype
            cache = (key, tuple(self.x0) + tuple(dx))
            self._scalar_kernel_args_cache = cache
        return cache[1]

    def _compute_block_mask_kernel_args(self, params):
        assert len(params) == len(self.extra_params), (
            len(params),
            len(self.extra_params),
        )
        return (
            self._scalar_kernel_args()
            + self.backend.get_data_pointers(self.block_mask_buffer, self.scalar_field_buffer)
            + params
        )

    def _count_active_blocks_kernel_args(self):
        return self.backend.get_data_pointers(self.block_mask_buffer, self.counting_buffer)

    def _compact_active_blocks_kernel_args(self, output_count=None):
        args = (self.allocated_ping_block_index_count,)
        args += self.backend.get_data_pointers(
            self.block_mask_buffer,
            self.block_index_ping_buffer,
            self.counting_buffer if (output_count is None) else output_count,
        )
        return args

    def _compact_active_blocks_from_predicate_kernel_args(
        self, step_idx, current_active_blocks, levels, splits, params, input_count=None, output_count=None
    ):
        """
        Without input_count, current_active_blocks is the number of input blocks. Otherwise, the number of
        input blocks is read on the device from input_count and current_active_blocks is an upper bound.
        """
        level = levels[step_idx].astype(self.itype)
        split = splits[step_idx].astype(self.itype)

        kernel_calls = levels.shape[0]
        output_is_ping = (kernel_calls + step_idx) % 2 == 0

        assert self.dx.dtype == self.ftype, self.dx.dtype
        assert self.x0.dtype == self.ftype, self.x0.dtype
        args = tuple(self.x0) + tuple(self.dx) + tuple(level) + tuple(split)
        args += (self.itype(current_active_blocks),)
        args += self.backend.get_data_pointers(input_count)
        args += (
            self.itype(
                self.allocated_ping_block_index_count if output_is_ping else self.allocated_pong_block_index_count
            ),
        )
        args += self.backend.get_data_pointers(
            self.block_index_pong_buffer if output_is_ping else self.block_index_ping_buffer,
            self.block_index_ping_buffer if output_is_ping else self.block_index_pong_buffer,
            self.counting_buffer if (output_count is None) else output_count,
        )
        args += params
        return args, output_is_ping

    def _compute_simplices_kernel_args(self, params, active_count=None, output_counts=None):
        """
        Without active_count, the grid size is the number of active blocks. Otherwise, the number of active blocks
        is read on the device from active_count and the grid size is an upper bound.
        """
        assert len(params) == len(self.extra_params), (
            len(params),
            len(self.extra_params),
        )
        args = (self.allocated_vertex_count, self.allocated_simplices_count)
        args += self._scalar_kernel_args()
        args += self.backend.get_data_pointers(
            self.block_index_ping_buffer,
            active_count,
            self.vertex_buffer,
            self.simplex_buffer,
            self.counting_buffer if (output_counts is None) else output_counts,
        )
        args += params
        return args

    def _compute_vertex_properties_kernel_args(self, params, max_vertices, vertex_count=None):
        assert len(params) == len(self.extra_params), (
            len(params),
            len(self.extra_params),
        )
        buffers = (
            self.counting_buffer if (vertex_count is None) else vertex_count,
            self.vertex_buffer,
        ) + self.vertex_properties_buffers
        args = (self.itype(max_vertices),)
        args += self.backend.get_data_pointers(*buffers)
        args += params
        return args

    def _build_kernels(self, device, B, Nblocks, Nthreads, use_cmem):
        threads_per_blocks = np.prod(B, dtype=np.int64)

        dim = self.dim
        N = self.N

        T = self.flt_ctype
        I = self.int_ctype
        Bstr = ", ".join(map(str, B))
        Nstr = ", ".join(map(str, N))
        warp_size = device.warp_size

        extra_templates = self.extra_templates

        if dim == 3:
            use_cmem_s = ", true" if use_cmem else ", false"
        else:
            assert use_cmem is None, use_cmem
            use_cmem_s = ""

        kernel_names = (
            f"cupmb::computeSimplices{dim}D<{T}, {I}, {Bstr}, {Nstr}, " f"{warp_size}u{use_cmem_s}{extra_templates}>",
        )

        if self.has_block_predicate:
            kernel_names += (
                f"cupmb::compactActiveBlocksFromPredicate{dim}D<{T}, {I}, {Nthreads}, "
                f"{Bstr}, {Nstr}, {warp_size}u{extra_templates}>",
            )
        else:
            kernel_names += (
                f"cupmb::computeBlockMask{dim}D<{T}, {I}, " f"{Bstr}, {Nstr}, {warp_size}u{extra_templates}>",
                f"cupmb::countActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
                f"cupmb::compactActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            )

        if self.has_vertex_properties:
            kernel_names += (f"cupmb::compute_vertex_properties<{T}, {I}, {dim}{extra_templates}>",)

        kernels = self.build_module_and_extract_kernels_from_source(
            self._kernel_source_code,
            kernel_names,
            device,
        )

        special_kernels = {}
        thread_based_kernels = {}
        block_based_kernels = {"compute_simplices": kernels[0]}

        if self.has_block_predicate:
            special_kernels["compact_active_blocks_from_predicate"] = kernels[1]
        else:
            block_based_kernels["compute_block_mask"] = kernels[1]
            thread_based_kernels["count_active_blocks"] = kernels[2]
            thread_based_kernels["compact_active_blocks"] = kernels[3]

        if self.has_vertex_properties:
            thread_based_kernels["compute_vertex_properties"] = kernels[-1]

        for kname, kernel in block_based_kernels.items():
            if threads_per_blocks > kernel.max_threads_per_block:
                self.logger.error(
                    f"Thread block is too large for kernel {kname}, "
                    "consider setting ignore_per_kernel_thread_limits to False."
                )
                raise CudaBlockSizeError(kname, threads_per_blocks, kernel.max_threads_per_block)

        for kname, kernel in thread_based_kernels.items():
            if kernel is None:
                continue
            if Nthreads > kernel.max_threads_per_block:
                self.logger.error(
                    f"Thread block is too large for kernel {kname}, "
                    "consider setting ignore_per_kernel_thread_limits to False."
                )
                raise CudaBlockSizeError(kname, Nthreads, kernel.max_threads_per_block)

        kernels = block_based_kernels
        kernels.update(thread_based_kernels)
        kernels.update(special_kernels)
        return kernels

    def build_module_and_extract_kernels_from_source(self, kernel_source_code, kernel_names, device=None):
        if device is None:
            device = self.backend.Device(self.device_id)
        build_args = dict(
            kernel_names=tuple(kernel_names),
            device_major=device.major,
            device_minor=device.minor,
            source_code=kernel_source_code,
            build_options=tuple(self.build_options),
        )

        if not kernel_cache_enabled():
            return self.backend.build_module_and_extract_kernels_from_source(logger=self.logger, **build_args)

        key = kernel_cache_key(
            backend=self.backend.__name__, compiler_version=self.backend.compiler_version(), **build_args
        )
        entry = load_kernel_cache_entry(key)
        if entry is not None:
            binary_path, mangled_kernel_names = entry
            try:
                kernels = self.backend.load_kernels(self.logger, binary_path, mangled_kernel_names)
                self.logger.debug(f"Loaded kernels from cache file://{binary_path}.")
                return kernels
            except Exception as e:
                self.logger.warning(f"Failed to load cached kernels from file://{binary_path}, rebuilding: {e}")

        kernels, binary, mangled_kernel_names = self.backend.build_kernels(logger=self.logger, **build_args)
        if binary is not None:
            try:
                binary_path = store_kernel_cache_entry(key, binary, mangled_kernel_names)
                self.logger.debug(f"Stored kernels in cache file://{binary_path}.")
            except OSError as e:
                log_once("warning", f"Failed to write the kernel cache in {kernel_cache_dir()}: {e}", self.logger)
        return kernels

    def build_module_and_extract_kernels_from_file(self, kernel_source_file, kernel_names, device=None):

        with open(kernel_source_file) as f:
            kernel_source_code = f.read()

        return self.build_module_and_extract_kernels_from_source(kernel_source_code, kernel_names, device)

    def _build_block_based_kernels(self, device, B):
        dim = self.dim
        N = self.N

        T = self.flt_ctype
        I = self.int_ctype
        Bstr = ", ".join(map(str, B))
        Nstr = ", ".join(map(str, N))
        warp_size = device.warp_size
        extra_templates = self.extra_templates
        use_cmem_s = ", false" if dim == 3 else ""

        kernel_names = (
            f"cupmb::computeSimplices{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{use_cmem_s}{extra_templates}>",
        )
        if not self.has_block_predicate:
            kernel_names += (f"cupmb::computeBlockMask{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{extra_templates}>",)

        kernels = self.build_module_and_extract_kernels_from_source(
            self.kernel_source_code,
            kernel_names,
            device,
        )

        kd = collections.OrderedDict()
        kd["compute_simplices"] = kernels[0]
        if not self.has_block_predicate:
            kd["compute_block_mask"] = kernels[1]
        return kd

    def _build_thread_based_kernels(self, device, Nthreads):
        dim = self.dim

        # here we use a fake block size
        N = self.N
        B = np.full_like(N, fill_value=4)
        G = list(map(int, (N + B - 2) // (B - 1)))
        Nblocks = np.prod(G, dtype=self.itype)

        Bstr = ", ".join(map(str, B))
        Nstr = ", ".join(map(str, N))

        T = self.flt_ctype
        I = self.int_ctype
        warp_size = device.warp_size
        extra_templates = self.extra_templates

        if self.has_block_predicate:
            kernel_names = (
                f"cupmb::compactActiveBlocksFromPredicate{dim}D<{T}, {I}, {Nthreads}, "
                f"{Bstr}, {Nstr}, {warp_size}u{extra_templates}>",
            )
        else:
            kernel_names = (
                f"cupmb::countActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
                f"cupmb::compactActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            )

        if self.has_vertex_properties:
            kernel_names += (f"cupmb::compute_vertex_properties<{T}, {I}, {dim}{extra_templates}>",)

        kernels = self.build_module_and_extract_kernels_from_source(
            self.kernel_source_code,
            kernel_names,
            device,
        )

        kd = collections.OrderedDict()
        if self.has_block_predicate:
            kd["compact_active_blocks_from_predicate"] = kernels[0]
        else:
            kd["count_active_blocks"] = kernels[0]
            kd["compact_active_blocks"] = kernels[1]

        if self.has_vertex_properties:
            kd["compute_vertex_properties"] = kernels[-1]

        return kd

    def _generate_candidate_block_size(self, device):
        candidates = []

        dim = self.dim
        itype = self.itype
        ftype = self.ftype

        warp_size = device.warp_size
        max_threads_per_block = device.max_threads_per_block
        max_shared_memory_per_block = device.max_shared_memory_per_block_optin

        Bmax = np.asarray(
            device.max_block_dim,
            dtype=itype,
        )[:dim]

        d = max(1 << int(np.ceil(np.log2(min(64, max_threads_per_block) ** (1.0 / dim)))), 2)
        B = np.full(shape=(dim,), dtype=itype, fill_value=d)

        k = 0
        threads = np.prod(B, dtype=np.int64)
        while threads <= max_threads_per_block:
            assert threads % warp_size == 0
            assert (B <= Bmax).all(), (B, Bmax)
            warps_per_block = threads // warp_size
            if warps_per_block > warp_size:
                break
            reqd = self._required_shared_memory_bytes(dim, warp_size, threads, ftype, itype)
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
        candidates = candidates[::-1]

        if not self._ignore_per_kernel_thread_limits:
            # We need to build kernels to determine effective max block size
            kernels = self._build_block_based_kernels(device, candidates[0])
            for kname, kernel in kernels.items():
                self.logger.debug(
                    f"Compiler reports a maximum of {kernel.max_threads_per_block} "
                    f"threads per block for kernel {kname}."
                )
                max_threads_per_block = min(max_threads_per_block, kernel.max_threads_per_block)
            self.logger.debug(f"Max threads per block for block based kernels has been set to {max_threads_per_block}.")

        candidates = tuple(
            filter(
                lambda b: np.prod(b, dtype=np.uint64) <= max_threads_per_block,
                candidates,
            )
        )

        if len(candidates) == 0:
            msg = "Could not determine any candidate block size."
            self.logger.error(msg)
            raise RuntimeError(msg)

        return candidates

    def _generate_candidate_thread_size(self, device):
        candidates = []

        itype = self.itype
        ftype = self.ftype

        max_shared_memory_per_block = device.max_shared_memory_per_block_optin

        warp_size = device.warp_size
        max_threads_per_block = min(device.max_threads_per_block, device.max_block_dim_x)

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
        candidates = candidates[::-1]

        if not self._ignore_per_kernel_thread_limits:
            # We need to build kernels to determine effective max thread size
            kernels = self._build_thread_based_kernels(device, candidates[0])
            for kname, kernel in kernels.items():
                self.logger.debug(
                    f"Compiler reports a maximum of {kernel.max_threads_per_block} "
                    f"threads per block for kernel {kname}."
                )
                max_threads_per_block = min(max_threads_per_block, kernel.max_threads_per_block)
            self.logger.debug(
                "Max threads per block for thread based (one-dimensional) kernels has been set to "
                f"{max_threads_per_block}."
            )

        candidates = tuple(
            filter(
                lambda b: np.prod(b, dtype=np.uint64) <= max_threads_per_block,
                candidates,
            )
        )

        if len(candidates) == 0:
            msg = "Could not determine any candidate thread size."
            self.logger.error(msg)
            raise RuntimeError(msg)

        return candidates

    def _reset_counters(self, stream):
        self.backend.fill_zeros(self.counting_buffer, stream)

    def _allocate_scalar_field(self):
        if not self._export_scalar_field:
            return
        if self.allocated_scalars != self.Nscalars:
            self.scalar_field_buffer = self.Nscalars

    def _compact_active_blocks_from_predicate(
        self, kernel, stream, Nthreads, levels, splits, params, time_kernel=False
    ):
        """
        Multiscale compaction of active blocks by using the user defined block predicate.

        Index buffers are reallocated on the fly when required (the kernel is then launched
        a second time for the current step). On exit, active block indices are stored in the
        ping block index buffer (unless there are no active blocks).

        Returns the number of active blocks and the cumulated kernel time (0.0 if time_kernel is False).
        """
        backend = self.backend
        logger = self.logger
        debug = logger.isEnabledFor(logging.DEBUG)

        Nthreads = int(Nthreads)
        elapsed = 0.0
        Nactive = 0
        output_is_ping = True
        current_active_blocks = 1 << int(levels[0].sum())

        for k in range(splits.shape[0]):
            grid_size = ((current_active_blocks + Nthreads - 1) // Nthreads,)
            for reallocated in (False, True):
                (
                    args,
                    output_is_ping,
                ) = self._compact_active_blocks_from_predicate_kernel_args(
                    k, current_active_blocks, levels, splits, params
                )
                if debug:
                    logger.debug(
                        f"Compacting active blocks at level={levels[k]}, split={splits[k]}, "
                        f"target_blocks={2**levels[k+1]}, active_blocks={current_active_blocks}"
                        + (" with reallocated buffers." if reallocated else ".")
                    )
                self._reset_counters(stream)
                tk = kernel(grid_size, (Nthreads,), args, stream=stream, time_kernel=time_kernel)
                Nactive = int(backend.asnumpy(self.counting_buffer[:2], stream)[0])

                allocated_indices = (
                    self.allocated_ping_block_index_count if output_is_ping else self.allocated_pong_block_index_count
                )
                if Nactive <= allocated_indices:
                    break
                elif reallocated:
                    msg = f"Block index buffer is still too small after reallocation ({allocated_indices} < {Nactive})."
                    logger.error(msg)
                    raise RuntimeError(msg)

                if output_is_ping:
                    self.block_index_ping_buffer = Nactive
                else:
                    self.block_index_pong_buffer = Nactive

            if time_kernel:
                elapsed += tk

            if Nactive == 0:
                break

            current_active_blocks = Nactive

        assert Nactive == 0 or output_is_ping, (Nactive, output_is_ping)
        return Nactive, elapsed

    def _autotune_kernels(self, stream, device, params):
        logger = self.logger

        logger.info(f"Autotuning kernels with new configuration by using autotuning rigor {self.autotuning_rigor}.")

        Ts = self._generate_candidate_thread_size(device)
        Bs = self._generate_candidate_block_size(device)

        # Heuristics to get best sizes without measuring:
        #   1) use smallest block with predicate, else second biggest one
        #   2) use third highest thread size for thread based kernels
        B0 = Bs[-1 if self.has_block_predicate else 1 % len(Bs)]
        T0 = Ts[min(2, len(Ts) - 1)]
        use_cmem0 = False if self.dim == 3 else None

        rigor = self.autotuning_rigor
        if rigor == AutotuningRigor.ESTIMATE:
            return (B0, T0, use_cmem0)
        elif rigor == AutotuningRigor.MEASURE:
            Nruns = 1
            max_candidates = 4
        elif rigor == AutotuningRigor.PATIENT:
            Nruns = 1
            max_candidates = 16
        elif rigor == AutotuningRigor.EXHAUSTIVE:
            Nruns = 1
            max_candidates = 64
        else:
            raise NotImplementedError(rigor)

        logger.debug(f"Autotuning parameters: max_candidates={max_candidates}, nruns={Nruns}")

        self._allocate_scalar_field()

        try:
            Bopt, use_cmem = self._autotune_block_size(stream, device, params, T0, Nruns, max_candidates)
            self._reset_allocated_buffers()

            Topt = self._autotune_thread_size(stream, device, params, Bopt, use_cmem, Nruns, max_candidates)
            self._reset_allocated_buffers()
        except _NoActiveBlocksError:
            logger.warning(
                f"Cannot autotune kernels with rigor {rigor} because there is no active block "
                f"for current parameters, falling back to {AutotuningRigor.ESTIMATE}."
            )
            self._reset_allocated_buffers()
            return (B0, T0, use_cmem0)

        return (Bopt, Topt, use_cmem)

    def _autotune_block_size(self, stream, device, params, Nthreads, Nruns, max_candidates):
        dim = self.dim
        N = self.N
        itype = self.itype
        ftype = self.ftype
        backend = self.backend
        logger = self.logger

        Bs = self._generate_candidate_block_size(device)

        # use smaller block size first with predicate
        if self.has_block_predicate:
            Bs = Bs[::-1]

        logger.debug(f"Estimating block size for N={N}, candidates are:")
        logger.debug("  " + ", ".join(map(_tuple_str, Bs)))

        warp_size = device.warp_size

        results = {}

        ncandidates = 0
        allocation_error = False

        use_cmem_choices = (True, False) if dim == 3 else (None,)

        for use_cmem in use_cmem_choices:
            if ncandidates >= max_candidates or allocation_error:
                break

            for B in Bs:
                if allocation_error:
                    logger.warning("Device allocation error, ignoring other kernel candidates.")
                    break

                if ncandidates >= max_candidates:
                    logger.debug(f"Max kernel candidates achieved ({max_candidates}).")
                    break

                threads_per_blocks = np.prod(B, dtype=np.int64)
                G = (N + B - 2) // (B - 1)
                Nblocks = np.prod(G, dtype=itype)
                Ngrid = itype(Nblocks + Nthreads - 1) // Nthreads
                shared_memory_bytes = self._required_shared_memory_bytes(
                    dim, warp_size, threads_per_blocks, ftype, itype
                )

                logger.debug(
                    f"Benchmarking with B={B}, G={G}, use_cmem={'1' if use_cmem else '0'}, "
                    f"shared_mem={bytes2str(shared_memory_bytes)}..."
                )

                try:
                    kernels = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem=use_cmem)
                except CudaBlockSizeError as e:
                    logger.debug(f"  skipping candidate B={_tuple_str(B)}: {e}")
                    continue

                try:
                    self.counting_buffer = 2

                    if self.has_block_predicate:
                        levels, splits = self._determine_multiscale_parameters(
                            G,
                            device.multiprocessor_count * device.max_threads_per_multiprocessor,
                            dim,
                        )

                        elapsed0_time_s = []
                        for i in range(Nruns):
                            Nactive, tk = self._compact_active_blocks_from_predicate(
                                kernels["compact_active_blocks_from_predicate"],
                                stream,
                                Nthreads,
                                levels,
                                splits,
                                params,
                                time_kernel=True,
                            )
                            elapsed0_time_s.append(tk)
                        elapsed0_time_s = np.asarray(elapsed0_time_s, dtype=np.float64)
                        logger.debug(
                            f"  compact_active_blocks_from_predicate: nruns={Nruns}, "
                            f"mean={time2str(elapsed0_time_s.mean())}, "
                            f"min={time2str(elapsed0_time_s.min())}, "
                            f"max={time2str(elapsed0_time_s.max())}"
                        )
                    else:
                        self.block_mask_buffer = Nblocks

                        elapsed0_time_s = []
                        for i in range(Nruns):
                            elapsed0_time_s.append(
                                kernels["compute_block_mask"](
                                    tuple(G),
                                    tuple(B),
                                    self._compute_block_mask_kernel_args(params),
                                    stream=stream,
                                    time_kernel=True,
                                )
                            )
                        elapsed0_time_s = np.asarray(elapsed0_time_s, dtype=np.float64)

                        self._reset_counters(stream)
                        kernels["count_active_blocks"](
                            (Ngrid,),
                            (Nthreads,),
                            self._count_active_blocks_kernel_args(),
                            stream=stream,
                        )
                        Nactive = int(backend.asnumpy(self.counting_buffer[:2], stream)[0])

                        self.block_index_ping_buffer = Nactive

                        self._reset_counters(stream)
                        kernels["compact_active_blocks"](
                            (Ngrid,),
                            (Nthreads,),
                            self._compact_active_blocks_kernel_args(),
                            stream=stream,
                        )
                        Nactive_bis = int(backend.asnumpy(self.counting_buffer[:2], stream)[0])
                        assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

                    if Nactive == 0:
                        raise _NoActiveBlocksError("Cannot autotune if there is no active blocks.")

                    self._reset_counters(stream)
                    kernels["compute_simplices"](
                        (Nactive,),
                        tuple(B),
                        self._compute_simplices_kernel_args(params),
                        shared_mem=shared_memory_bytes,
                        stream=stream,
                    )
                    Ncount = backend.asnumpy(self.counting_buffer[:2], stream)

                    self.vertex_buffer = Ncount[0]
                    self.simplex_buffer = Ncount[1]
                except backend.AllocationError:
                    allocation_error = True
                    continue

                elapsed1_time_s = []
                for i in range(Nruns):
                    self._reset_counters(stream)
                    elapsed1_time_s.append(
                        kernels["compute_simplices"](
                            (Nactive,),
                            tuple(B),
                            self._compute_simplices_kernel_args(params),
                            shared_mem=shared_memory_bytes,
                            time_kernel=True,
                            stream=stream,
                        )
                    )
                elapsed1_time_s = np.asarray(elapsed1_time_s, dtype=np.float64)
                logger.debug(
                    f"  compute_simplices: nruns={Nruns}, "
                    f"mean={time2str(elapsed1_time_s.mean())}, "
                    f"min={time2str(elapsed1_time_s.min())}, "
                    f"max={time2str(elapsed1_time_s.max())}"
                )
                Ncount_bis = backend.asnumpy(self.counting_buffer[:2], stream)
                assert (Ncount == Ncount_bis).all(), (Ncount, Ncount_bis)

                elapsed = elapsed0_time_s + elapsed1_time_s
                results[(tuple(B), use_cmem)] = elapsed.mean()
                ncandidates += 1

        if ncandidates == 0:
            msg = "Could not benchmark any kernel, cannot continue."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"Successfully benchmarked {ncandidates} kernel candidates to determine block size:")
        blength = max(len(_tuple_str(b)) for b, _ in results.keys())
        for (B, cmem), v in sorted(results.items(), key=operator.itemgetter(1)):
            logger.info(f"  B={_tuple_str(B):{blength}s}, use_cmem={'1' if cmem else '0'}: {time2str(v)}")

        (Bopt, use_cmem), _ = next(iter(sorted(results.items(), key=operator.itemgetter(1))))
        logger.info(f"Best candidate block size is {_tuple_str(Bopt)} with use_cmem={'1' if use_cmem else '0'}.")
        return (np.asarray(Bopt, dtype=itype), use_cmem)

    def _autotune_thread_size(self, stream, device, params, B, use_cmem, Nruns, max_candidates):
        backend = self.backend
        N = self.N
        dim = self.dim
        itype = self.itype
        logger = self.logger

        Ts = self._generate_candidate_thread_size(device)

        logger.debug(f"Estimating thread size for N={N}, candidates are:")
        logger.debug("  " + ", ".join(map(str, Ts)))

        G = (N + B - 2) // (B - 1)
        Nblocks = np.prod(G, dtype=itype)

        results = {}
        ncandidates = 0
        allocation_error = False
        for Nthreads in Ts:
            if allocation_error:
                logger.warning("Device allocation error, ignoring other kernel candidates.")
                break

            if ncandidates >= max_candidates:
                logger.debug(f"Max kernel candidates achieved ({max_candidates}).")
                break

            logger.debug(f"Benchmarking with Nthreads={Nthreads}...")

            try:
                kernels = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem)
            except CudaBlockSizeError as e:
                logger.debug(f"  skipping candidate Nthreads={Nthreads}: {e}")
                continue

            try:
                self.counting_buffer = 2

                if self.has_block_predicate:
                    levels, splits = self._determine_multiscale_parameters(
                        G,
                        device.multiprocessor_count * device.max_threads_per_multiprocessor,
                        dim,
                    )

                    elapsed0_time_s = []
                    for i in range(Nruns):
                        Nactive, tk = self._compact_active_blocks_from_predicate(
                            kernels["compact_active_blocks_from_predicate"],
                            stream,
                            Nthreads,
                            levels,
                            splits,
                            params,
                            time_kernel=True,
                        )
                        elapsed0_time_s.append(tk)
                    elapsed0_time_s = np.asarray(elapsed0_time_s, dtype=np.float64)
                    logger.debug(
                        f"  compact_active_blocks_from_predicate: nruns={Nruns}, "
                        f"mean={time2str(elapsed0_time_s.mean())}, "
                        f"min={time2str(elapsed0_time_s.min())}, "
                        f"max={time2str(elapsed0_time_s.max())}"
                    )
                    elapsed = elapsed0_time_s.mean()
                else:
                    Ngrid = itype(Nblocks + Nthreads - 1) // Nthreads
                    self.block_mask_buffer = Nblocks

                    kernels["compute_block_mask"](
                        tuple(G),
                        tuple(B),
                        self._compute_block_mask_kernel_args(params),
                        stream=stream,
                    )

                    elapsed0_time_s = []
                    for i in range(Nruns):
                        self._reset_counters(stream)
                        elapsed0_time_s.append(
                            kernels["count_active_blocks"](
                                (Ngrid,),
                                (Nthreads,),
                                self._count_active_blocks_kernel_args(),
                                time_kernel=True,
                                stream=stream,
                            )
                        )
                    elapsed0_time_s = np.asarray(elapsed0_time_s, dtype=np.float64)
                    logger.debug(
                        f"  count_active_blocks: nruns={Nruns}, "
                        f"mean={time2str(elapsed0_time_s.mean())}, "
                        f"min={time2str(elapsed0_time_s.min())}, "
                        f"max={time2str(elapsed0_time_s.max())}"
                    )
                    Nactive = int(backend.asnumpy(self.counting_buffer[:2], stream)[0])

                    self.block_index_ping_buffer = Nactive

                    elapsed1_time_s = []
                    for i in range(Nruns):
                        self._reset_counters(stream)
                        elapsed1_time_s.append(
                            kernels["compact_active_blocks"](
                                (Ngrid,),
                                (Nthreads,),
                                self._compact_active_blocks_kernel_args(),
                                time_kernel=True,
                                stream=stream,
                            )
                        )
                    elapsed1_time_s = np.asarray(elapsed1_time_s, dtype=np.float64)
                    logger.debug(
                        f"  compact_active_blocks: nruns={Nruns}, "
                        f"mean={time2str(elapsed1_time_s.mean())}, "
                        f"min={time2str(elapsed1_time_s.min())}, "
                        f"max={time2str(elapsed1_time_s.max())}"
                    )
                    Nactive_bis = int(backend.asnumpy(self.counting_buffer[:2], stream)[0])
                    assert Nactive == Nactive_bis, (Nactive, Nactive_bis)
                    elapsed = elapsed0_time_s.mean() + elapsed1_time_s.mean()
            except backend.AllocationError:
                allocation_error = True
                continue

            if Nactive == 0:
                raise _NoActiveBlocksError("Cannot autotune if there is no active blocks.")

            results[Nthreads] = elapsed
            ncandidates += 1

        if not results:
            msg = "Could not benchmark any kernel."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"Successfully benchmarked {ncandidates} kernel candidates to determine thread size:")
        for T, v in sorted(results.items(), key=operator.itemgetter(1)):
            logger.info(f"  T={int(T):4d}: {time2str(v)}")

        Topt, _ = next(iter(sorted(results.items(), key=operator.itemgetter(1))))
        logger.info(f"Best candidate thread size is {Topt}.")
        return Topt

    def _determine_multiscale_parameters(self, G, initial_domain_count, levels_per_round):
        dim = G.size

        end_level = np.ceil(np.log2(G)).astype(np.int32)
        end_levels = end_level.sum()

        start_levels = min(end_levels, int(np.floor(np.log2(initial_domain_count))))

        Nlevels = end_levels - start_levels
        Nrounds = (Nlevels + levels_per_round - 1) // levels_per_round

        start_level = np.zeros(dim, dtype=np.int32)
        while start_level.sum() != start_levels:
            block_size = self.L / (2**start_level)
            can_update = set(np.where(start_level < end_level)[0])
            for k in np.argsort(-block_size, kind="stable"):
                if k in can_update:
                    start_level[k] += 1
                    break
            else:
                raise RuntimeError

        level = start_level.copy()
        splits = np.zeros(dim, dtype=np.int32)
        levels = np.zeros(
            (
                Nrounds + 2,
                dim,
            ),
            dtype=np.int32,
        )
        levels[0] = level
        levels[1] = level
        l = 2
        while (level != end_level).any():
            block_size = self.L / (2**level)
            can_update = set(np.where(level < end_level)[0])
            for k in np.argsort(-block_size, kind="stable"):
                if k in can_update:
                    level[k] += 1
                    splits[k] += 1
                    if splits.sum() == levels_per_round:
                        levels[l] = level
                        splits[...] = 0
                        l += 1
                    break
            else:
                raise RuntimeError
        if any(splits):
            levels[l] = level

        splits = np.diff(levels, axis=0)

        return levels, splits

    def _build_kernel_launcher(self, device, B, Nthreads, use_cmem):
        dim = self.dim
        N = self.N
        itype = self.itype
        ftype = self.ftype

        self.logger.info(
            f"Building kernel launcher for N={N}, B={B}, Nthreads={Nthreads}"
            + ("" if use_cmem is None else f", use_cmem={'1' if use_cmem else '0'}...")
        )

        G = (N + B - 2) // (B - 1)
        Nblocks = np.prod(G, dtype=itype)
        Ngrid = itype(Nblocks + Nthreads - 1) // Nthreads

        if self.has_block_predicate:
            max_resident_threads = device.multiprocessor_count * device.max_threads_per_multiprocessor
            levels, splits = self._determine_multiscale_parameters(
                G,
                max_resident_threads,
                dim,
            )
        else:
            levels, splits = None, None

        warp_size = device.warp_size
        threads_per_blocks = np.prod(B, dtype=np.int64)

        shared_mem = self._required_shared_memory_bytes(dim, warp_size, threads_per_blocks, ftype, itype)

        kernels = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem)

        def kernel_launcher(
            stream,
            params,
            to_numpy,
            return_elapsed_time,
            s=self,
            kernels=kernels,
            shared_mem=shared_mem,
            G=tuple(G),
            B=tuple(B),
            Nthreads=(Nthreads,),
            Ngrid=(Ngrid,),
            Nblocks=Nblocks,
            levels=levels,
            splits=splits,
            counter_views=[None, None],
        ):
            tstart = time.perf_counter()

            backend = s.backend
            logger = s.logger
            debug = logger.isEnabledFor(logging.DEBUG)

            if debug:
                pretty_params = ", ".join(f"{pname}={pvalue}" for (pname, pvalue) in zip(s.extra_params.keys(), params))
                logger.debug(
                    f"Kernel launcher called with parameters {pretty_params}."
                    if pretty_params
                    else "Kernel launcher called."
                )

            def get_scalar_field_buffer(x):
                if not s.export_scalar_field:
                    return tuple()
                x = x[: s.Nscalars]
                if to_numpy:
                    x = backend.asnumpy(x, stream)
                # scalar field is stored x-fastest: S[(iz,)iy,ix] (C-contiguous, shape N[::-1]+1)
                x = x.reshape(tuple(map(int, s.N[::-1] + 1)))
                return (x,)

            def finalize(buffers):
                backend.synchronize_stream(stream)
                elapsed = time.perf_counter() - tstart
                if debug:
                    logger.debug(f"Kernel launcher finished in {time2str(elapsed)}.")
                if return_elapsed_time:
                    return buffers, elapsed
                else:
                    return buffers

            def empty_result():
                buffers = (None, None)
                buffers += get_scalar_field_buffer(s.scalar_field_buffer)
                buffers += (None,) * len(s.vertex_properties)
                return finalize(buffers)

            def launch_step_by_step():
                """
                Launch the kernels step by step, counts are read back after each step to allocate the buffers.
                Returns the vertex and simplex counts, or None when there is no active block.
                """
                s.counting_buffer = 2

                if s.has_block_predicate:
                    if debug:
                        logger.debug(
                            f"Compacting {np.prod(G, dtype=np.uint64)} blocks by using predicate, "
                            f"target block grid size is {G}."
                        )
                    Nactive, _ = s._compact_active_blocks_from_predicate(
                        kernels["compact_active_blocks_from_predicate"],
                        stream,
                        Nthreads[0],
                        levels,
                        splits,
                        params,
                    )
                    if Nactive == 0:
                        logger.debug("No active blocks found by predicate.")
                        return None
                else:
                    s.block_mask_buffer = Nblocks
                    s._allocate_scalar_field()

                    if debug:
                        logger.debug(f"Computing block mask for {Nblocks} blocks.")
                    kernels["compute_block_mask"](
                        G,
                        B,
                        s._compute_block_mask_kernel_args(params),
                        stream=stream,
                    )

                    is_first_call = s.allocated_ping_block_index_count == 0
                    if is_first_call:
                        logger.debug("Counting active blocks because allocated block index size is zero.")
                        s._reset_counters(stream)
                        kernels["count_active_blocks"](
                            Ngrid,
                            Nthreads,
                            s._count_active_blocks_kernel_args(),
                            stream=stream,
                        )
                    else:
                        if debug:
                            logger.debug(
                                "Compacting active blocks "
                                f"(allocated_block_index={s.allocated_ping_block_index_count})."
                            )
                        s._reset_counters(stream)
                        kernels["compact_active_blocks"](
                            Ngrid,
                            Nthreads,
                            s._compact_active_blocks_kernel_args(),
                            stream=stream,
                        )

                    Nactive = int(backend.asnumpy(s.counting_buffer[:2], stream)[0])
                    if Nactive == 0:
                        logger.debug("No active blocks found, skipping block compaction.")
                        return None

                    if is_first_call or (s.allocated_ping_block_index_count < Nactive):
                        logger.info(f"Reallocating block index buffer to handle {Nactive} active blocks.")
                        s.block_index_ping_buffer = Nactive

                        logger.debug("Compacting active blocks with reallocated buffer.")
                        s._reset_counters(stream)
                        kernels["compact_active_blocks"](
                            Ngrid,
                            Nthreads,
                            s._compact_active_blocks_kernel_args(),
                            stream=stream,
                        )

                if debug:
                    logger.debug(
                        "Generating simplices "
                        f"(allocated_vertices={s.allocated_vertex_count}, "
                        f"allocated_indices={s.allocated_simplices_count})"
                    )
                s._reset_counters(stream)
                kernels["compute_simplices"](
                    (Nactive,),
                    B,
                    s._compute_simplices_kernel_args(params),
                    shared_mem=shared_mem,
                    stream=stream,
                )

                Nvertices, Nsimplices = map(int, backend.asnumpy(s.counting_buffer[:2], stream))
                reallocate_vertices = s.allocated_vertex_count < Nvertices
                reallocate_indices = s.allocated_simplices_count < Nsimplices

                if debug:
                    fmt = functools.partial(decimal_unit2str, unit="")
                    logger.debug(
                        f"Generated {fmt(Nvertices)} vertices ({fmt(s.allocated_vertex_count)} allocated), "
                        + ("reallocation is required " if reallocate_vertices else "vertex buffer was large enough ")
                        + (
                            f"({100.0*Nvertices/s.allocated_vertex_count:2.1f}% filling)."
                            if s.allocated_vertex_count
                            else "(buffer was not allocated)."
                        )
                    )
                    logger.debug(
                        f"Generated {fmt(Nsimplices)} simplices ({fmt(s.allocated_simplices_count)} allocated), "
                        + ("reallocation is required " if reallocate_indices else "index buffer was large enough ")
                        + (
                            f"({100.0*Nsimplices/s.allocated_simplices_count:2.1f}% filling)."
                            if s.allocated_simplices_count
                            else "(buffer was not allocated)."
                        )
                    )

                if reallocate_vertices:
                    logger.info(f"Reallocating vertex buffer to handle {Nvertices} vertices.")
                    s.vertex_buffer = Nvertices

                if reallocate_indices:
                    logger.info(f"Reallocating simplex buffer to handle {Nsimplices} indices.")
                    s.simplex_buffer = Nsimplices

                if reallocate_vertices or reallocate_indices:
                    logger.debug("Generating simplices with reallocated buffers.")
                    s._reset_counters(stream)
                    kernels["compute_simplices"](
                        (Nactive,),
                        B,
                        s._compute_simplices_kernel_args(params),
                        shared_mem=shared_mem,
                        stream=stream,
                    )

                if s.has_vertex_properties and (Nvertices > 0):
                    for vp in s.vertex_properties:
                        s._update_cuda_buffer(vp, Nvertices)

                    logger.debug("Computing vertex properties.")
                    kernels["compute_vertex_properties"](
                        ((Nvertices + Nthreads[0] - 1) // Nthreads[0],),
                        Nthreads,
                        s._compute_vertex_properties_kernel_args(params, Nvertices),
                        stream=stream,
                    )

                return Nvertices, Nsimplices

            def launch_device_driven():
                """
                Launch all the kernels without intermediate host synchronizations: kernels read their number of
                input blocks (or vertices) from device counters and grids are sized from the buffer capacities.
                Returns the vertex and simplex counts after a single device to host copy, or None when a buffer is
                not allocated yet or was too small (kernels are then launched again step by step).
                """
                vertex_capacity = s.allocated_vertex_count
                if s.has_vertex_properties:
                    vertex_capacity = min([vertex_capacity] + [s.allocated_elements(vp) for vp in s.vertex_properties])
                simplex_capacity = s.allocated_simplices_count
                ping_capacity = s.allocated_ping_block_index_count
                nsteps = splits.shape[0] if s.has_block_predicate else 1
                pong_capacity = s.allocated_pong_block_index_count if (nsteps > 1) else ping_capacity
                if min(ping_capacity, pong_capacity, vertex_capacity, simplex_capacity) == 0:
                    return None

                # counters: active blocks after each step, vertices and simplices (views are reused between calls)
                s.counting_buffer = nsteps + 2
                counters = s.counting_buffer
                if counter_views[0] is not counters:
                    counter_views[:] = [
                        counters,
                        (
                            tuple(counters[k : k + 1] for k in range(nsteps)),
                            counters[nsteps : nsteps + 1],
                            counters[nsteps : nsteps + 2],
                            counters[: nsteps + 2],
                        ),
                    ]
                active_counts, vertex_count, output_counts, all_counts = counter_views[1]
                s._reset_counters(stream)

                block_capacities = []
                if s.has_block_predicate:
                    max_blocks = 1 << int(levels[0].sum())
                    for k in range(nsteps):
                        args, output_is_ping = s._compact_active_blocks_from_predicate_kernel_args(
                            k,
                            max_blocks,
                            levels,
                            splits,
                            params,
                            input_count=(None if (k == 0) else active_counts[k - 1]),
                            output_count=active_counts[k],
                        )
                        kernels["compact_active_blocks_from_predicate"](
                            ((max_blocks + Nthreads[0] - 1) // Nthreads[0],),
                            Nthreads,
                            args,
                            stream=stream,
                        )
                        capacity = ping_capacity if output_is_ping else pong_capacity
                        block_capacities.append(capacity)
                        max_blocks = min(capacity, 1 << int(levels[k + 1].sum()))
                else:
                    s.block_mask_buffer = Nblocks
                    s._allocate_scalar_field()
                    kernels["compute_block_mask"](
                        G,
                        B,
                        s._compute_block_mask_kernel_args(params),
                        stream=stream,
                    )
                    kernels["compact_active_blocks"](
                        Ngrid,
                        Nthreads,
                        s._compact_active_blocks_kernel_args(output_count=active_counts[0]),
                        stream=stream,
                    )
                    block_capacities.append(ping_capacity)
                    max_blocks = min(ping_capacity, int(Nblocks))

                kernels["compute_simplices"](
                    (max_blocks,),
                    B,
                    s._compute_simplices_kernel_args(
                        params,
                        active_count=active_counts[-1],
                        output_counts=output_counts,
                    ),
                    shared_mem=shared_mem,
                    stream=stream,
                )

                if s.has_vertex_properties:
                    kernels["compute_vertex_properties"](
                        ((vertex_capacity + Nthreads[0] - 1) // Nthreads[0],),
                        Nthreads,
                        s._compute_vertex_properties_kernel_args(params, vertex_capacity, vertex_count=vertex_count),
                        stream=stream,
                    )

                counts = tuple(map(int, backend.asnumpy(all_counts, stream)))
                Nactive, (Nvertices, Nsimplices) = counts[:nsteps], counts[nsteps:]
                too_small = (
                    any(n > c for (n, c) in zip(Nactive, block_capacities))
                    or (Nvertices > vertex_capacity)
                    or (Nsimplices > simplex_capacity)
                )
                if debug:
                    logger.debug(
                        f"Device driven launch generated {Nvertices} vertices and {Nsimplices} simplices "
                        f"from {Nactive[-1]} active blocks" + (", buffers were too small." if too_small else ".")
                    )
                return None if too_small else (Nvertices, Nsimplices)

            counts = launch_device_driven()
            if counts is None:
                counts = launch_step_by_step()
                if counts is None:
                    return empty_result()
            Nvertices, Nsimplices = counts

            def get_vertex_buffer(x):
                if Nvertices == 0:
                    return None
                x = x[:Nvertices]
                if to_numpy:
                    return backend.asnumpy(x, stream)
                return x

            def get_simplex_buffer(x):
                if Nsimplices == 0:
                    return None
                x = x[:Nsimplices]
                if to_numpy:
                    return backend.asnumpy(x, stream)
                return x

            buffers = (
                get_vertex_buffer(s.vertex_buffer),
                get_simplex_buffer(s.simplex_buffer),
            )
            buffers += get_scalar_field_buffer(s.scalar_field_buffer)
            buffers += tuple(map(get_vertex_buffer, s.vertex_properties_buffers))

            return finalize(buffers)

        self._kernel_launcher = kernel_launcher

    def ctype_to_dtype(self, ctype):
        if not isinstance(ctype, str):
            return normalize_scalar_type(ctype)
        ctype = ctype.strip()
        if ctype.startswith("const "):
            ctype = ctype[len("const ") :].strip()
        if ctype == "T":
            dtype = self.ftype
        elif ctype == "I":
            dtype = self.itype
        elif ctype in ctype_to_dtype:
            dtype = ctype_to_dtype[ctype]
        else:
            raise KeyError(ctype)
        return dtype
