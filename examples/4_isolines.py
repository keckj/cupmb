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
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, Backend, logger

if __name__ == "__main__":

    fdef = """
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i,
            const T& t) {
        const auto& [x, y] = p;
        return sin(x*x - y*y) * cos(2*x+x*y/4);
    }
}
    """

    pmb = ParallelMarchingBlocks(
        2,
        fdef,
        L=3,
        backend=Backend.CUPY,
        extra_params={"t": "T"},
        export_scalar_field=True,
        enable_cuda_std_lib=True,
    )

    (V, I, S) = pmb(t=0, to_numpy=True)

    if V is None:
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit(1)

    x0 = pmb.x0
    N = pmb.N
    L = pmb.L
    dx = pmb.dx

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    ax.set_title(f"Marching squares: Nx={N[0]}, Ny={N[1]}")
    ax.imshow(
        S,
        vmin=S.min(),
        vmax=S.max(),
        extent=[
            x0[0] - dx[0] / 2,
            x0[0] + L[0] + dx[0] / 2,
            x0[1] - dx[1] / 2,
            x0[1] + L[1] + dx[1] / 2,
        ],
        interpolation="none",
    )
    ax.scatter(V[:, 0], V[:, 1])
    plt.show()
