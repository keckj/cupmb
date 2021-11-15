#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, time, tempfile, operator, collections, functools

import numpy as np

from cupmb import logger as default_logger
from cupmb.backend import Backend, CudaBackend
from cupmb.parameters import ExtraParameter
from cupmb.properties import VertexProperty
from cupmb.bbox import BoundingBox
from cupmb.enums import BufferType, AllocationPolicy, AutotuningRigor
from cupmb.utils.IO import cuda_kernels_dir
from cupmb.utils.types import dtype_to_ctype, ctype_to_dtype
from cupmb.utils.units import time2str, bytes2str, decimal_unit2str


class CudaBlockSizeError(ValueError):
    def __init__(
        self, kernel_name, threads_per_block, max_reported_kernel_thread_per_block
    ):
        msg = (
            f"Attempted to use {threads_per_block} threads per block for kernel {kernel_name} "
            f"while reported kernel maximum is {max_reported_kernel_thread_per_block}."
        )
        super().__init__(msg)
        self.threads_per_block = threads_per_block
        self.max_reported_kernel_thread_per_block = max_reported_kernel_thread_per_block


class ParallelMarchingBlocks:

    __slots__ = [
        "_dim",
        "_fdef",
        "_ftype",
        "_itype",
        "_device_id",
        "_export_scalar_field",
        "_extra_build_options",
        "_vertex_properties",
        "_N",
        "_bbox",
        "_update_bbox",
        "_isolevel",
        "_autotuning_rigor",
        "_extra_params",
        "_logger",
        "_source_tmpdir",
        "_function_def_tmpfile",
        "_vertex_props_tmpfile",
        "_kernel_launcher",
        "_ready",
        "_backend",
        "_buffers",
        "_allocation_policies",
        "_default_allocation_policy",
        "_ignore_per_kernel_thread_limits",
    ]

    def __init__(
        self,
        dim,
        fdef="",
        N=64,
        L=1.0,
        x0=0.0,
        bbox=None,
        update_bbox=None,
        isolevel=0.0,
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

        self._source_tmpdir = tempfile.TemporaryDirectory()
        self._function_def_tmpfile = os.path.join(
            self._source_tmpdir.name, "cupmb_func_definition.cuh"
        )
        self._vertex_props_tmpfile = os.path.join(
            self._source_tmpdir.name, "cupmb_vertex_properties.cuh"
        )
        assert os.path.isdir(self._source_tmpdir.name), self._source_tmpdir.name
        assert not os.path.exists(
            self._function_def_tmpfile
        ), self._function_def_tmpfile
        assert not os.path.exists(
            self._vertex_props_tmpfile
        ), self._vertex_props_tmpfile

        self.logger = logger
        self.default_allocation_policy = default_allocation_policy

        self.ftype = ftype
        self.itype = itype
        self.dim = dim
        self.fdef = fdef
        self.device_id = device_id
        self.extra_build_options = extra_build_options
        self.bbox = bbox
        if bbox is None:
            self.L = L
            self.x0 = x0
        self.N = N
        self.update_bbox = update_bbox
        self.isolevel = isolevel
        self.autotuning_rigor = autotuning_rigor
        self.extra_params = extra_params
        self.vertex_properties = vertex_properties
        self.export_scalar_field = export_scalar_field
        self.ignore_per_kernel_thread_limits = ignore_per_kernel_thread_limits

        if enable_cuda_std_lib:
            cuda_include_dir = self.backend.determine_cuda_include_dir(self.logger)
            self.extra_build_options = self.extra_build_options + (
                "-DCUPMB_ENABLE_CUDA_STD_LIB",
                f"-I{cuda_include_dir}",
            )

    def __call__(self, *param_args, stream=None, to_numpy=False, **param_kwargs):
        (device, stream, params) = self.setup(
            *param_args, stream=stream, **param_kwargs
        )
        with device:
            return self._kernel_launcher(stream, params, to_numpy)

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
            (dtype,) = self._get_dtypes(buffer_key.dtype)
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
            elif buffer_key == BufferType.BLOCK_INDICES:
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
        assert (allocation_policy is None) or isinstance(
            allocation_policy, AllocationPolicy
        ), type(allocation_policy)
        default_allocation_policy = self.get_default_buffer_allocation_policy(
            buffer_key
        )
        allocation_policy = (
            default_allocation_policy
            if (allocation_policy is None)
            else allocation_policy
        )
        self.logger.debug(
            f"Setting allocation policy of buffer {buffer_key} to {allocation_policy}."
        )
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
        for buffer_key in BufferType:
            self.lock_buffer(buffer_key)

    def allocated_elements(self, buffer_key):
        assert buffer_key in self._buffers, buffer_key
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
        extra_templates = ", ".join(
            self._get_ctypes(*tuple(p.ctype for p in self.extra_params.values()))
        )
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
    def allocated_block_index_count(self):
        return self.allocated_elements(BufferType.BLOCK_INDICES)

    @property
    def source_tmpdir(self):
        return self._source_tmpdir

    @property
    def function_def_tmpfile(self):
        return self._function_def_tmpfile

    @property
    def kernel_source_file(self):
        return os.path.join(cuda_kernels_dir, f"pmb{self.dim}D.cuh")

    @property
    def backend(self):
        return self._backend

    @classmethod
    def required_build_options(cls):
        return (
            "-std=c++17",
            "--restrict",
            f"-I{cuda_kernels_dir}",
        )

    @property
    def build_options(self):
        return (
            self.required_build_options()
            + (f"-I{self._source_tmpdir.name}",)
            + self.extra_build_options
        )

    @property
    def vertex_properties_buffers(self):
        return tuple(map(self._buffers.__getitem__, self.vertex_properties))

    def _set_dim(self, dim):
        assert isinstance(dim, int), type(dim)
        assert dim in {1, 2, 3}, dim
        if dim != self._dim:
            xmin = np.zeros(shape=(dim,), dtype=self.ftype)
            xmax = np.ones(shape=(dim,), dtype=self.ftype)
            self._dim = dim
            self.bbox = BoundingBox(xmin=xmin, xmax=xmax)
            self._ready = False

    def _get_dim(self):
        return self._dim

    dim = property(_get_dim, _set_dim)

    def _set_default_allocation_policy(self, default_allocation_policy):
        assert isinstance(default_allocation_policy, AllocationPolicy), type(
            default_allocation_policy
        )
        self._default_allocation_policy = default_allocation_policy
        self._ready &= True

    def _get_default_allocation_policy(self):
        return self._default_allocation_policy

    default_allocation_policy = property(
        _get_default_allocation_policy, _set_default_allocation_policy
    )

    def _set_fdef(self, fdef):
        assert isinstance(fdef, str), type(fdef)
        if os.path.isfile(fdef):
            with open(fdef) as f:
                fdef = f.read()
        if fdef != self._fdef:
            if fdef:
                with open(self._function_def_tmpfile, "w") as f:
                    f.write(fdef)
            self._ready = False
            self._fdef = fdef

    def _get_fdef(self):
        return self._fdef

    fdef = property(_get_fdef, _set_fdef)

    def _set_vertex_properties(self, vertex_properties):
        if vertex_properties is None:
            vertex_properties = tuple()

        assert isinstance(vertex_properties, tuple), type(vertex_properties)
        assert all(
            map(lambda vp: isinstance(vp, VertexProperty), vertex_properties)
        ), vertex_properties

        property_names = tuple(map(VertexProperty.name.fget, vertex_properties))
        assert len(property_names) == len(
            set(property_names)
        ), "Some properties share the same name."

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
        assert isinstance(device_id, int), type(device_id)
        if device_id != self.device_id:
            self._reset_buffers()
            self._ready = False
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
        self._logger = logger

    def _get_logger(self):
        return self._logger

    logger = property(_get_logger, _set_logger)

    def _set_export_scalar_field(self, export_scalar_field):
        if export_scalar_field is None:
            export_scalar_field = False
        assert isinstance(export_scalar_field, bool), type(export_scalar_field)
        self._export_scalar_field = export_scalar_field

    def _get_export_scalar_field(self):
        return self._export_scalar_field

    export_scalar_field = property(_get_export_scalar_field, _set_export_scalar_field)

    def _set_ignore_per_kernel_thread_limits(self, ignore_per_kernel_thread_limits):
        if ignore_per_kernel_thread_limits is None:
            ignore_per_kernel_thread_limits = False
        assert isinstance(ignore_per_kernel_thread_limits, bool), type(
            ignore_per_kernel_thread_limits
        )
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
        if dim in (2, 3):
            m0 = threads * ftype(0).itemsize
            m1 = threads * (itype(0).itemsize + np.uint8(0).itemsize)
            m2 = 2 * (warps_per_block + 1) * itype(0).itemsize
            return max(m0, m1, m2)
        else:
            raise NotImplementedError(dim)

    def _reset_buffers(self):
        self.logger.debug("Resetting all buffers.")
        self._buffers = collections.OrderedDict()
        self._allocation_policies = collections.OrderedDict()
        for e in BufferType:
            self._create_buffer(e)

    def _reset_allocated_buffers(self):
        self.logger.debug("Resetting internally allocated buffers.")
        excluded = {AllocationPolicy.LOCK_BUFFER, AllocationPolicy.NO_EXTRA_ALLOCATION}
        for k in self._buffers.keys():
            if self._allocation_policies[k] in excluded:
                continue
            self.replace_buffer(k, None)

    def _create_buffer(self, buffer_key):
        assert isinstance(buffer_key, (BufferType, VertexProperty)), type(buffer_key)
        default_allocation_policy = self.get_default_buffer_allocation_policy(
            buffer_key
        )
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
        self.logger.debug("Removing buffer {buffer_key} from registered buffers.")
        if isinstance(buffer_key, BufferType):
            msg = f"Buffer {buffer_key} cannot be removed."
            self.logger.error(msg)
            raise RuntimeError(msg)
        if not (
            (buffer_key in self._buffers) and (buffer_key in self._allocation_policies)
        ):
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
        if "isolevel" in param_kwargs:
            self.isolevel = param_kwargs.pop("isolevel")

        d_param_args = collections.deque(param_args)
        params = []
        for (pname, pvalue) in self.extra_params.items():
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

            if pvalue.ctype == "T":
                arg = self.ftype(arg)
            elif pvalue.ctype == "I":
                arg = self.itype(arg)
            elif pvalue.ctype.endswith("*"):
                ctype = pvalue.ctype[:-1]
                dtype = self.ctype_to_dtype(ctype)
                if hasattr(arg, "dtype") and arg.dtype != dtype:
                    msg = (
                        f"Expected an array of type {dtype.__name__} for parameter '{pname}' "
                        f"but got an array of type {arg.dtype}."
                    )
                    raise ValueError(msg)
                (arg,) = self.backend.get_data_pointers(arg)

            params.append(arg)

        return tuple(params)

    def _setup(self, params, stream):
        if self._ready:
            return

        vertex_properties_kernel_src = VertexProperty.generate_kernel(
            T=self.flt_ctype,
            I=self.int_ctype,
            dim=self.dim,
            vertex_properties=self.vertex_properties,
        )

        with open(self._vertex_props_tmpfile, "w") as f:
            f.write(vertex_properties_kernel_src)

        device = self.backend.Device(self.device_id)

        device_id = device.id
        device_name = device.name
        device_bus_id = device.pci_bus_id

        msg = f"Initializing parallel marching cubes on device {device_id} [{device_name}] ({device_bus_id})."
        self.logger.info(msg)

        self.logger.debug(
            f"Device global memory is {bytes2str(device.global_memory_bytes)} "
            f"({bytes2str(device.free_memory_bytes)} free)."
        )
        self.logger.debug(
            f"Max device threads per block is {device.max_threads_per_block}."
        )
        self.logger.debug(f"Max device block size is {tuple(device.max_block_dim)}.")
        self.logger.debug(f"Max device grid size is {tuple(device.max_grid_dim)}.")
        self.logger.debug(
            f"Max device shared memory per block optin is {bytes2str(device.max_shared_memory_per_block_optin)}."
        )

        if not os.path.isfile(self.kernel_source_file):
            msg = f"Could not find kernel source file:///{self.kernel_source_file}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if not os.path.isfile(self._function_def_tmpfile):
            msg = "Scalar function has not been defined yet."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if not os.path.isfile(self._vertex_props_tmpfile):
            msg = "Vertex properties kernel has not been written to disk."
            self.logger.error(msg)
            raise RuntimeError(msg)

        (Bopt, Topt, use_cmem) = self._autotune_kernels(stream, device, params)

        self._build_kernel_launcher(device, Bopt, Topt, use_cmem)
        self._ready = True

    def _update_cuda_buffer(self, buffer_key, a, force=False):
        backend = self.backend

        assert buffer_key in self._buffers, buffer_key
        assert buffer_key in self._allocation_policies, buffer_key

        actual_buffer = self._buffers[buffer_key]
        allocation_policy = self._allocation_policies[buffer_key]

        assert (actual_buffer is None) or isinstance(
            actual_buffer, backend.Array
        ), type(actual_buffer)
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
            if (not force) and (
                allocation_policy == AllocationPolicy.NO_EXTRA_ALLOCATION
            ):
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
            assert a.dtype == dtype, (a.dtype, dtype)
            buf = a.ravel()[: (a.size // elem_size) * elem_size].reshape(
                (-1,) + elem_shape
            )
        elif isinstance(a, backend.Buffer):
            self.logger.debug(
                f"Replacing cuda buffer {buffer_key} with {backend.Buffer.__name__}(size={a.size})."
            )
            locked_buffer_error()
            bsize = backend.get_nbytes(a)
            shape = (bsize // elem_bytes,) + elem_shape
            buf = backend.array_from_pointer(ptr=a, shape=shape, dtype=dtype)
        elif isinstance(a, (int, np.integer)):
            elem_count = int(a)
            del a
            if allocation_policy is AllocationPolicy.ALLOCATE_UPPER_POW2:
                elem_count = self._upper_pow2(elem_count)

            nbytes = elem_count * elem_bytes
            shape = (elem_count,) + elem_shape

            device = self.backend.Device(self.device_id)
            mem_free = device.free_memory_bytes
            mem_pct = 100.0 * nbytes / float(device.global_memory_bytes)

            if nbytes >= 0.9 * mem_free:
                self.logger.warning(
                    f"Device {device.name} has only {bytes2str(mem_free)} of memory left."
                )

            if force:
                self.logger.debug(
                    f"Forced reallocation of {elem_count} {buffer_key} elements ({dtype.__name__}, new shape "
                    f"is {shape}, new size is {bytes2str(nbytes)} ({mem_pct:2.1f}% of device memory)."
                )

                with backend.Device(self.device_id):
                    buf = backend.Array(shape=shape, dtype=dtype)
            else:
                # check current buffer is large enough
                if self.allocated_elements(buffer_key) >= elem_count:
                    return
                elif allocation_policy is AllocationPolicy.LOCK_BUFFER:
                    locked_buffer_error()
                elif allocation_policy is AllocationPolicy.NO_EXTRA_ALLOCATION:
                    extra_allocation_error(elem_count)

                self.logger.debug(
                    f"Reallocation of {elem_count} {buffer_key} elements ({dtype.__name__}), new shape "
                    f"is {shape}, new size is {bytes2str(nbytes)} ({mem_pct:2.1f}% of device memory)."
                )

                with backend.Device(self.device_id):
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
        assert self.dx.dtype == self.ftype, self.dx.dtype
        return (self.isolevel,) + tuple(self.x0) + tuple(self.dx)

    def _compute_block_mask_kernel_args(self, params):
        assert len(params) == len(self.extra_params), (
            len(params),
            len(self.extra_params),
        )
        return (
            self._scalar_kernel_args()
            + self.backend.get_data_pointers(
                self.block_mask_buffer, self.scalar_field_buffer
            )
            + params
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

    def _compute_simplices_kernel_args(self, params):
        assert len(params) == len(self.extra_params), (
            len(params),
            len(self.extra_params),
        )
        args = (self.allocated_vertex_count, self.allocated_simplices_count)
        args += self._scalar_kernel_args()
        args += self.backend.get_data_pointers(
            self.block_index_buffer,
            self.vertex_buffer,
            self.simplex_buffer,
            self.counting_buffer,
        )
        args += params
        return args

    def _compute_vertex_properties_kernel_args(self, params):
        assert len(params) == len(self.extra_params), (
            len(params),
            len(self.extra_params),
        )
        buffers = (
            self.counting_buffer,
            self.vertex_buffer,
        ) + self.vertex_properties_buffers
        args = self.backend.get_data_pointers(*buffers)
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
            f"cupmb::computeBlockMask{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{extra_templates}>",
            f"cupmb::countActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            f"cupmb::compactActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            f"cupmb::computeSimplices{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{use_cmem_s}{extra_templates}>",
        )
        if self.has_vertex_properties:
            kernel_names += (
                f"cupmb::compute_vertex_properties<{T}, {I}, {dim}{extra_templates}>",
            )

        kernels = self.build_module_and_extract_kernels(
            self.kernel_source_file,
            kernel_names,
            device,
        )

        (
            compute_block_mask,
            count_active_blocks,
            compact_active_blocks,
            compute_simplices,
        ) = kernels[:4]

        if self.has_vertex_properties:
            compute_vertex_properties = kernels[4]
        else:
            compute_vertex_properties = None

        for kname, kernel in zip(
            ("compute_block_mask", "compute_simplices"),
            (compute_block_mask, compute_simplices),
        ):
            if threads_per_blocks > kernel.max_threads_per_block:
                self.logger.error(
                    f"Thread block is too large for kernel {kname}, "
                    "consider setting ignore_per_kernel_thread_limits to False."
                )
                raise CudaBlockSizeError(
                    kname, threads_per_blocks, kernel.max_threads_per_block
                )

        for kname, kernel in zip(
            (
                "count_active_blocks",
                "compact_active_blocks",
                "compute_vertex_properties",
            ),
            (count_active_blocks, compact_active_blocks, compute_vertex_properties),
        ):
            if kernel is None:
                continue
            if Nthreads > kernel.max_threads_per_block:
                self.logger.error(
                    f"Thread block is too large for kernel {kname}, "
                    "consider setting ignore_per_kernel_thread_limits to False."
                )
                raise CudaBlockSizeError(kname, Nthreads, kernel.max_threads_per_block)

        return (
            compute_block_mask,
            count_active_blocks,
            compact_active_blocks,
            compute_simplices,
            compute_vertex_properties,
        )

    def build_module_and_extract_kernels(
        self, kernel_source_file, kernel_names, device=None
    ):
        if device is None:
            device = self.backend.Device(self.device_id)
        kernels = self.backend.build_module_and_extract_kernels(
            kernel_names=kernel_names,
            logger=self.logger,
            device_major=device.major,
            device_minor=device.minor,
            kernel_source_file=kernel_source_file,
            build_options=self.build_options,
        )
        return kernels

    def build_module_and_extract_kernels_from_string(
        self, kernel_source_code, kernel_names, device=None
    ):

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_source_file = os.path.join(tmpdir, "source.cu")
            with open(kernel_source_file, "w") as f:
                f.write(kernel_source_code)
            return self.build_module_and_extract_kernels(
                kernel_source_file, kernel_names, device
            )

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
            f"cupmb::computeBlockMask{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{extra_templates}>",
            f"cupmb::computeSimplices{dim}D<{T}, {I}, {Bstr}, {Nstr}, {warp_size}u{use_cmem_s}{extra_templates}>",
        )

        kernels = self.build_module_and_extract_kernels(
            self.kernel_source_file,
            kernel_names,
            device,
        )

        kd = collections.OrderedDict()
        kd["compute_block_mask"] = kernels[0]
        kd["compute_simplices"] = kernels[1]
        return kd

    def _build_thread_based_kernels(self, device, Nthreads):
        dim = self.dim
        Nblocks = np.prod(self.N, dtype=self.itype)

        T = self.flt_ctype
        I = self.int_ctype
        warp_size = device.warp_size
        extra_templates = self.extra_templates

        kernel_names = (
            f"cupmb::countActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
            f"cupmb::compactActiveBlocks<{I}, {Nblocks}, {Nthreads}, {warp_size}u>",
        )
        if self.has_vertex_properties:
            kernel_names += (
                f"cupmb::compute_vertex_properties<{T}, {I}, {dim}{extra_templates}>",
            )

        kernels = self.build_module_and_extract_kernels(
            self.kernel_source_file,
            kernel_names,
            device,
        )

        kd = collections.OrderedDict()
        kd["count_active_blocks"] = kernels[0]
        kd["compact_active_blocks"] = kernels[1]
        if self.has_vertex_properties:
            kd["compute_vertex_properties"] = kernels[2]
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
        candidates = candidates[::-1]

        if not self._ignore_per_kernel_thread_limits:
            # We need to build kernels to determine effective max block size
            kernels = self._build_block_based_kernels(device, candidates[0])
            for (kname, kernel) in kernels.items():
                self.logger.debug(
                    f"Compiler reports a maximum of {kernel.max_threads_per_block} "
                    f"threads per block for kernel {kname}."
                )
                max_threads_per_block = min(
                    max_threads_per_block, kernel.max_threads_per_block
                )
            self.logger.debug(
                f"Max threads per block for one-dimensional kernels has been set to {max_threads_per_block}."
            )

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
        candidates = candidates[::-1]

        if not self._ignore_per_kernel_thread_limits:
            # We need to build kernels to determine effective max thread size
            kernels = self._build_thread_based_kernels(device, candidates[0])
            for (kname, kernel) in kernels.items():
                self.logger.debug(
                    f"Compiler reports a maximum of {kernel.max_threads_per_block} "
                    f"threads per block for kernel {kname}."
                )
                max_threads_per_block = min(
                    max_threads_per_block, kernel.max_threads_per_block
                )
            self.logger.debug(
                f"Max threads per block for block based kernels has been set to {max_threads_per_block}."
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

    def _autotune_kernels(self, stream, device, params):
        logger = self.logger

        logger.info(
            f"Autotuning kernels with new configuration by using autotuning rigor {self.autotuning_rigor}."
        )

        Ts = self._generate_candidate_thread_size(device)
        Bs = self._generate_candidate_block_size(device)

        rigor = self.autotuning_rigor
        if rigor == AutotuningRigor.ESTIMATE:
            return (Bs[0], Ts[0], True if self.dim == 3 else None)
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

        logger.debug(
            f"Autotuning parameters: max_candidates={max_candidates}, nruns={Nruns}"
        )

        self._allocate_scalar_field()

        (Bopt, use_cmem) = self._autotune_block_size(
            stream, device, params, Ts[0], Nruns, max_candidates
        )
        self._reset_allocated_buffers()

        Topt = self._autotune_thread_size(
            stream, device, params, Bopt, Nruns, max_candidates
        )
        self._reset_allocated_buffers()

        return (Bopt, Topt, use_cmem)

    def _autotune_block_size(
        self, stream, device, params, Nthreads, Nruns, max_candidates
    ):
        dim = self.dim
        N = self.N
        itype = self.itype
        ftype = self.ftype
        backend = self.backend
        logger = self.logger

        Bs = self._generate_candidate_block_size(device)

        logger.debug(f"Estimating block size for N={N}, candidates are:")
        logger.debug("  " + ", ".join(map(str, map(tuple, Bs))))

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
                    logger.warning(
                        "Device allocation error, ignoring other kernel candidates."
                    )
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
                    f"Benmarking with B={B}, G={G}, use_cmem={'1' if use_cmem else '0'}, "
                    f"shared_mem={bytes2str(shared_memory_bytes)}..."
                )

                (
                    compute_block_mask,
                    count_active_blocks,
                    compact_active_blocks,
                    compute_simplices,
                    compute_vertex_properties,
                ) = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem=use_cmem)

                self.counting_buffer = 2

                try:
                    self.block_mask_buffer = Nblocks
                except backend.AllocationError:
                    allocation_error = True
                    continue

                ellapsed0_time_s = []
                for i in range(Nruns):
                    ellapsed0_time_s.append(
                        compute_block_mask(
                            tuple(G),
                            tuple(B),
                            self._compute_block_mask_kernel_args(params),
                            stream=stream,
                            time_kernel=True,
                        )
                    )
                ellapsed0_time_s = np.asarray(ellapsed0_time_s, dtype=np.float64)
                logger.debug(
                    f"  compute_block_mask: nruns={Nruns}, "
                    f"mean={time2str(ellapsed0_time_s.mean())}, "
                    f"min={time2str(ellapsed0_time_s.min())}, "
                    f"max={time2str(ellapsed0_time_s.max())}"
                )

                self._reset_counters(stream)
                count_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._count_active_blocks_kernel_args(),
                    stream=stream,
                )
                Nactive = backend.asnumpy(self.counting_buffer)[0]
                if Nactive == 0:
                    msg = "Cannot autotune if there is no active blocks."
                    logger.error(msg)
                    raise RuntimeError(msg)

                try:
                    self.block_index_buffer = Nactive
                except backend.AllocationError:
                    allocation_error = True
                    continue

                self._reset_counters(stream)
                compact_active_blocks(
                    (Ngrid,),
                    (Nthreads,),
                    self._compact_active_blocks_kernel_args(),
                    stream=stream,
                )
                Nactive_bis = backend.asnumpy(self.counting_buffer)[0]
                assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

                self._reset_counters(stream)
                compute_simplices(
                    (Nactive,),
                    tuple(B),
                    self._compute_simplices_kernel_args(params),
                    shared_mem=shared_memory_bytes,
                    stream=stream,
                )
                Ncount = backend.asnumpy(self.counting_buffer)

                try:
                    self.vertex_buffer = Ncount[0]
                    self.simplex_buffer = Ncount[1]
                except backend.AllocationError:
                    allocation_error = True
                    continue

                ellapsed1_time_s = []
                for i in range(Nruns):
                    self._reset_counters(stream)
                    ellapsed1_time_s.append(
                        compute_simplices(
                            (Nactive,),
                            tuple(B),
                            self._compute_simplices_kernel_args(params),
                            shared_mem=shared_memory_bytes,
                            time_kernel=True,
                            stream=stream,
                        )
                    )
                ellapsed1_time_s = np.asarray(ellapsed1_time_s, dtype=np.float64)
                logger.debug(
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

        if ncandidates == 0:
            msg = "Could not benchmark any kernel, cannot continue."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(f"Successfully benchmarked {ncandidates} kernel candidates.")

        (Bopt, use_cmem), _ = next(
            iter(sorted(results.items(), key=operator.itemgetter(1)))
        )
        logger.info(
            f"Best candidate block size is {Bopt} with use_cmem={'1' if use_cmem else '0'}."
        )
        return (np.asarray(Bopt, dtype=itype), use_cmem)

    def _autotune_thread_size(self, stream, device, params, B, Nruns, max_candidates):
        backend = self.backend
        N = self.N
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
                logger.warning(
                    "Device allocation error, ignoring other kernel candidates."
                )
                break

            if ncandidates >= max_candidates:
                logger.debug(f"Max kernel candidates achieved ({max_candidates}).")
                break

            Ngrid = itype(Nblocks + Nthreads - 1) // Nthreads

            logger.debug(f"Benmarking with Nthreads={Nthreads}, Ngrid={Ngrid}...")

            (
                compute_block_mask,
                count_active_blocks,
                compact_active_blocks,
                compute_simplices,
                compute_vertex_properties,
            ) = self._build_kernels(device, B, Nblocks, Nthreads, False)

            try:
                self.counting_buffer = 2
                self.block_mask_buffer = Nblocks
            except backend.AllocationError:
                allocation_error = True
                continue

            compute_block_mask(
                tuple(G),
                tuple(B),
                self._compute_block_mask_kernel_args(params),
                stream=stream,
            )

            ellapsed0_time_s = []
            for i in range(Nruns):
                self._reset_counters(stream)
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
            logger.debug(
                f"    count_active_blocks: nruns={Nruns}, "
                f"mean={time2str(ellapsed0_time_s.mean())}, "
                f"min={time2str(ellapsed0_time_s.min())}, "
                f"max={time2str(ellapsed0_time_s.max())}"
            )
            Nactive = backend.asnumpy(self.counting_buffer)[0]
            if Nactive == 0:
                msg = "Cannot autotune if there is no active blocks."
                logger.error(msg)
                raise RuntimeError(msg)

            try:
                self.block_index_buffer = Nactive
            except backend.AllocationError:
                allocation_error = True
                continue

            ellapsed1_time_s = []
            for i in range(Nruns):
                self._reset_counters(stream)
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
            logger.debug(
                f"  compact_active_blocks: nruns={Nruns}, "
                f"mean={time2str(ellapsed1_time_s.mean())}, "
                f"min={time2str(ellapsed1_time_s.min())}, "
                f"max={time2str(ellapsed1_time_s.max())}"
            )
            Nactive_bis = backend.asnumpy(self.counting_buffer)[0]
            assert Nactive == Nactive_bis, (Nactive, Nactive_bis)

            results[Nthreads] = ellapsed1_time_s.mean()
            ncandidates += 1

        logger.debug(f"Successfully benchmarked {ncandidates} kernel candidates.")

        if not results:
            msg = "Could not benchmark any kernel."
            logger.error(msg)
            raise RuntimeError(msg)

        Topt, _ = next(iter(sorted(results.items(), key=operator.itemgetter(1))))
        logger.info(f"Best candidate thread size is {Topt}.")
        return Topt

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

        warp_size = device.warp_size
        threads_per_blocks = np.prod(B, dtype=np.int64)

        shared_mem = self._required_shared_memory_bytes(
            dim, warp_size, threads_per_blocks, ftype, itype
        )

        kernels = self._build_kernels(device, B, Nblocks, Nthreads, use_cmem)

        def kernel_launcher(
            stream,
            params,
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
            tstart = time.perf_counter()

            pretty_params = ", ".join(
                f"{pname}={pvalue}"
                for (pname, pvalue) in zip(s.extra_params.keys(), params)
            )
            s.logger.info(f"Kernel launcher called with parameters {pretty_params}.")

            s.counting_buffer = 2
            s.block_mask_buffer = Nblocks
            s._allocate_scalar_field()

            s.logger.debug(f"Computing block mask for {Nblocks} blocks.")
            kernels[0](G, B, s._compute_block_mask_kernel_args(params), stream=stream)

            is_first_call = s.allocated_block_index_count == 0
            if is_first_call:
                s.logger.debug(
                    "Counting active blocks because allocated block index size is zero."
                )
                s._reset_counters(stream)
                kernels[1](
                    Ngrid, Nthreads, s._count_active_blocks_kernel_args(), stream=stream
                )
            else:
                s.logger.debug(
                    f"Compacting active blocks (allocated_block_index={s.allocated_block_index_count})."
                )
                s._reset_counters(stream)
                kernels[2](
                    Ngrid,
                    Nthreads,
                    s._compact_active_blocks_kernel_args(),
                    stream=stream,
                )

            Nactive = s.backend.asnumpy(s.counting_buffer)[0]
            if Nactive == 0:
                s.logger.debug("No active blocks found, skipping block compaction.")
                return (None,) * (
                    2 + self._export_scalar_field + len(self.vertex_properties)
                )

            if is_first_call or (s.allocated_block_index_count < Nactive):
                s.logger.info(
                    f"Reallocating block index buffer to handle {Nactive} active blocks."
                )
                s.block_index_buffer = Nactive

                s.logger.debug("Compacting active blocks with reallocated buffer.")
                s._reset_counters(stream)
                kernels[2](
                    Ngrid,
                    Nthreads,
                    s._compact_active_blocks_kernel_args(),
                    stream=stream,
                )

            s.logger.debug(
                "Generating simplices "
                f"(allocated_vertices={s.allocated_vertex_count}, "
                f"allocated_indices={s.allocated_simplices_count})"
            )
            s._reset_counters(stream)
            kernels[3](
                (Nactive,),
                B,
                s._compute_simplices_kernel_args(params),
                shared_mem=shared_mem,
                stream=stream,
            )

            Nvertices, Nsimplices = tuple(s.backend.asnumpy(s.counting_buffer))
            reallocate_vertices = s.allocated_vertex_count < Nvertices
            reallocate_indices = s.allocated_simplices_count < Nsimplices

            fmt = functools.partial(decimal_unit2str, unit="")

            s.logger.debug(
                f"Generated {fmt(Nvertices)} vertices ({fmt(s.allocated_vertex_count)} allocated), "
                + (
                    "reallocation is required "
                    if reallocate_vertices
                    else "vertex buffer was large enough "
                )
                + (
                    f"({100.0*Nvertices/s.allocated_vertex_count:2.1f}% filling)."
                    if s.allocated_vertex_count
                    else "(buffer was not allocated)."
                )
            )
            s.logger.debug(
                f"Generated {fmt(Nsimplices)} simplices ({fmt(s.allocated_simplices_count)} allocated), "
                + (
                    "reallocation is required "
                    if reallocate_indices
                    else "index buffer was large enough "
                )
                + (
                    f"({100.0*Nsimplices/s.allocated_simplices_count:2.1f}% filling)."
                    if s.allocated_simplices_count
                    else "(buffer was not allocated)."
                )
            )

            if reallocate_vertices:
                s.logger.info(
                    f"Reallocating vertex buffer to handle {Nvertices} vertices."
                )
                s.vertex_buffer = Nvertices

            if reallocate_indices:
                s.logger.info(
                    f"Reallocating simplex buffer to handle {Nsimplices} indices."
                )
                s.simplex_buffer = Nsimplices

            if reallocate_vertices or reallocate_indices:
                s.logger.debug("Generating simplices with reallocated buffers.")
                s._reset_counters(stream)
                kernels[3](
                    (Nactive,),
                    B,
                    s._compute_simplices_kernel_args(params),
                    shared_mem=shared_mem,
                    stream=stream,
                )

            if s.has_vertex_properties:
                for vp in s.vertex_properties:
                    s._update_cuda_buffer(vp, Nvertices)

                s.logger.debug("Computing vertex properties.")
                kernels[4](
                    ((Nvertices + Nthreads[0] - 1) // Nthreads[0],),
                    Nthreads,
                    s._compute_vertex_properties_kernel_args(params),
                    stream=stream,
                )

            def get_vertex_buffer(x):
                if Nvertices == 0:
                    return None
                x = x[:Nvertices]
                if to_numpy:
                    return s.backend.asnumpy(x)
                return x

            def get_simplex_buffer(x):
                if Nsimplices == 0:
                    return None
                x = x[:Nsimplices]
                if to_numpy:
                    return s.backend.asnumpy(x)
                return x

            def get_scalar_field_buffer(x):
                if not s.export_scalar_field:
                    return tuple()
                x = x[: s.Nscalars]
                if to_numpy:
                    x = s.backend.asnumpy(x)
                x = x.reshape(*tuple(map(int, s.N + 1)))
                return (x,)

            buffers = (
                get_vertex_buffer(s.vertex_buffer),
                get_simplex_buffer(s.simplex_buffer),
            )
            buffers += get_scalar_field_buffer(self.scalar_field_buffer)
            buffers += tuple(map(get_vertex_buffer, s.vertex_properties_buffers))

            s.backend.synchronize_stream(stream)

            tend = time.perf_counter()
            ellapsed = tend - tstart
            s.logger.debug(f"Kernel launcher finished in {time2str(ellapsed)}.")

            return buffers

        self._kernel_launcher = kernel_launcher

    def ctype_to_dtype(self, ctype):
        if ctype in ctype_to_dtype:
            dtype = ctype_to_dtype[ctype]
        elif ctype == "T":
            dtype = self.ftype
        elif ctype == "I":
            dtype = self.itype
        else:
            raise KeyError(ctype)
        return dtype
