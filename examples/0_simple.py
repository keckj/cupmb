#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, sys
import trimesh

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, Backend, AutotuningRigor, logger

if __name__ == "__main__":

    fdef = r"""
    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz) {
        return x*x + y*y + z*z - T(0.5)*T(0.5);
    }
    """

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        x0=-0.5,
        backend=Backend.CUPY,
        N=4096,
        autotuning_rigor=AutotuningRigor.EXHAUSTIVE,
    )

    (V, I) = pmb(to_numpy=True)

    if V is None:
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit()

    mesh = trimesh.Trimesh(vertices=V, faces=I)
    mesh.show()
