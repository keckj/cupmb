import enum


class Backend(enum.Enum):
    CUPY = 0
    PYCUDA = 1


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
