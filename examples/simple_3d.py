import os
import trimesh

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, logger

if __name__ == "__main__":

    fdef = r"""
    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz) {
        return (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5) - 0.5*0.5;
    }
    """

    from cupmb import Backend

    pmb = ParallelMarchingBlocks(3, fdef, backend=Backend.PYCUDA)

    (V, I) = pmb(to_numpy=True)
    print(pmb.block_mask_buffer)

    if V is None:
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        raise RuntimeError(msg)

    mesh = trimesh.Trimesh(vertices=V, faces=I)
    mesh.show()
