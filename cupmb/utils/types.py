#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#

import numpy as np

dtype_to_ctype = {
    np.bool_: "bool",
    np.int8: "char",
    np.int16: "short",
    np.int32: "int",
    np.int64: "long long",
    np.uint8: "unsigned char",
    np.uint16: "unsigned short",
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long",
    np.float16: "half",
    np.float32: "float",
    np.float64: "double",
    np.complex64: "complex<float>",
    np.complex128: "complex<double>",
}

ctype_to_dtype = {v: k for (k, v) in dtype_to_ctype.items()}
ctype_to_dtype.update(
    {
        "signed char": np.int8,
        "long": np.int64,
        "unsigned long": np.uint64,
        "size_t": np.uint64,
        "int8_t": np.int8,
        "int16_t": np.int16,
        "int32_t": np.int32,
        "int64_t": np.int64,
        "uint8_t": np.uint8,
        "uint16_t": np.uint16,
        "uint32_t": np.uint32,
        "uint64_t": np.uint64,
    }
)


def normalize_scalar_type(t):
    """
    Convert a dtype-like (numpy scalar type, np.dtype instance, numpy type string such as 'float32'
    or C type string such as 'float' or 'unsigned int') to a numpy scalar type.
    """
    if isinstance(t, str) and t.strip() in ctype_to_dtype:
        return ctype_to_dtype[t.strip()]
    return np.dtype(t).type
