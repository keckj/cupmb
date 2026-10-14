#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import jinja2
import numpy as np

from cupmb.utils.types import dtype_to_ctype


class VertexProperty:
    _kernel_template = jinja2.Template(
        """
#include <trove/block.h>

namespace cupmb {

template <typename T, typename I, I dim, typename... Args>
__global__ void compute_vertex_properties(
    const I* const __restrict nvertices_ptr,
    const T* const __restrict vertices,
{%- for vp in vertex_properties %}
    {{vp.ctype(T,I)}}* const __restrict vertex_property_{{loop.index}},
{%- endfor %}
    const Args... fargs)
{
    const I global_index = threadIdx.x + blockIdx.x * blockDim.x;
    const I nvertices = __ldg(nvertices_ptr);

    for(I index = global_index; index < nvertices; index += gridDim.x * blockDim.x) {
        if (trove::warp_converged()) {
            const trove::array<T,dim> vertex = trove::load_array_warp_contiguous<dim>(vertices, index);
{%- for vp in vertex_properties %}
            trove::store_array_warp_contiguous<{{vp.size(dim)}}>(vertex_property_{{loop.index}},
                index, {{vp.fname}}<T,I,dim>(vertex, fargs...));
{%- endfor %}
        } else {
            const trove::array<T,dim> vertex = trove::load_array<dim>(vertices, index);
{%- for vp in vertex_properties %}
            trove::store_array<{{vp.size(dim)}}>(vertex_property_{{loop.index}},
                index, {{vp.fname}}<T,I,dim>(vertex, fargs...));
{%- endfor %}
        }
    }
}

}
"""
    )

    @classmethod
    def generate_kernel(cls, T, I, dim, vertex_properties):
        assert T in {"float", "double"}, T
        assert I in {"unsigned int", "unsigned long long"}, I
        assert isinstance(dim, int) and dim >= 1, dim
        assert isinstance(vertex_properties, tuple), type(vertex_properties)
        assert all(map(lambda vp: isinstance(vp, cls), vertex_properties)), cls

        if len(vertex_properties) > 0:
            kernel_src = cls._kernel_template.render(
                T=T, I=I, dim=dim, vertex_properties=vertex_properties
            )
        else:
            kernel_src = ""

        return kernel_src

    def __init__(self, name, fname, dtype, shape):
        assert isinstance(name, str), type(name)
        assert isinstance(fname, str), type(fname)
        if not isinstance(dtype, str):
            # accept numpy scalar types, np.dtype instances and dtype-likes
            dtype = np.dtype(dtype).type
        assert isinstance(dtype, str) or issubclass(dtype, np.number), dtype
        assert len(name) > 0, name
        assert len(fname) > 0, fname

        if shape is None:
            shape = ()
        assert isinstance(shape, tuple), type(shape)

        self._name = name
        self._fname = fname
        self._dtype = dtype
        self._shape = shape

    def __str__(self):
        return self.name.upper()

    def ctype(self, T, I):
        dtype = self.dtype
        if dtype == "T":
            return T
        elif dtype == "I":
            return I
        elif isinstance(dtype, str):
            return dtype
        elif issubclass(dtype, np.number):
            return dtype_to_ctype[dtype]
        else:
            raise NotImplementedError(type(dtype))

    def effective_shape(self, dim):
        shape = []
        for s in self.shape:
            if isinstance(s, str):
                if s == "dim":
                    s = dim
                else:
                    raise NotImplementedError(s)
            shape.append(s)
        return tuple(shape)

    def size(self, dim):
        return np.prod(self.effective_shape(dim), dtype=np.int64)

    @property
    def name(self):
        return self._name

    @property
    def fname(self):
        return "cupmb::" + self._fname

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape


if __name__ == "__main__":
    p0 = VertexProperty("N", fname="compute_normal", dtype="T", shape=("dim",))
    p1 = VertexProperty(
        "P", fname="compute_property", dtype=np.uint8, shape=("dim", 4, "dim")
    )

    kernel = VertexProperty.generate_kernel(
        T="float", I="unsigned int", dim=3, vertex_properties=(p0, p1)
    )
    print(kernel)
