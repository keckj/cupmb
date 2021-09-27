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

from cupmb import ParallelMarchingBlocks, Backend, logger

if __name__ == "__main__":

    fdef = r"""
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i) {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        const T r(0.55);
        return x*x + y*y + z*z - r*r;
    }
}
    """

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        x0=-0.5,
        N=32,
        backend=Backend.CUPY,
    )

    (V, I) = pmb(to_numpy=True)

    if V is None:
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit()

    mesh = trimesh.Trimesh(vertices=V, faces=I)
    mesh.show()
