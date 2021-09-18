#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#

from cupmb.enums import Backend


class CudaBackend:
    def __new__(cls, backend):
        if backend is Backend.CUPY:
            from cupmb.backend.cupy_backend import CupyBackend

            return CupyBackend
        elif backend is Backend.PYCUDA:
            from cupmb.backend.pycuda_backend import PyCudaBackend

            return PyCudaBackend
        else:
            raise NotImplementedError(backend)
