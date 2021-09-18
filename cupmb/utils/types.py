#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
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
