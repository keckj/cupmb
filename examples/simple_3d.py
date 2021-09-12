import os
import trimesh

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks

if __name__ == "__main__":

    fdef = """
    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz) {
        return (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5) - 0.5*0.5;
    }
    """

    pmb = ParallelMarchingBlocks(3, fdef)

    (V, I) = pmb(to_numpy=True)

    mesh = trimesh.Trimesh(vertices=V, faces=I)
    mesh.show()
