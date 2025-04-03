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

import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, Backend, logger


def as_line_collection(V, I):
    lines = [[tuple(V[j]) for j in i] for i in I]
    return matplotlib.collections.LineCollection(lines, lw=2, color="r", zorder=3)


def animate_plot(pmb, fig, ax, level):
    (V, I, S) = pmb(t=0, level=level, to_numpy=True)
    if V is None:
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit(1)
    lc = as_line_collection(V, I)

    x0 = pmb.x0
    N = pmb.N
    L = pmb.L
    dx = pmb.dx

    x = x0[0] + dx[0] * np.arange(N[0] + 1)
    y = x0[1] + dx[1] * np.arange(N[1] + 1)
    X, Y = np.meshgrid(x, y)

    ax.set_title(
        f"Marching squares with Nx={N[0]} and Ny={N[1]} (black = matplotlib.contour, red = cupmb)"
    )
    img = ax.imshow(
        S,
        vmin=S.min(),
        vmax=S.max(),
        extent=[
            x0[0] - dx[0] / 2,
            x0[0] + dx[0] / 2 + L[0],
            x0[1] - dx[1] / 2,
            x0[1] + dx[1] / 2 + L[1],
        ],
        interpolation="none",
        origin="lower",
        zorder=1,
    )
    contours = ax.contour(X, Y, S, levels=[0], colors=["k"], linewidths=[5], zorder=2)
    ax.add_collection(lc)
    ax.set_xlim(x0[0], x0[0] + L[0])
    ax.set_ylim(x0[1], x0[1] + L[1])
    fig.show()

    it = 1
    dt = 0.01
    while plt.fignum_exists(fig.number):
        (V, I, S) = pmb(t=it * dt, level=level, to_numpy=True)

        img.set_data(S)

        ax.collections.remove(lc)
        lc = as_line_collection(V, I)
        ax.add_collection(lc)

        for collection in contours.collections:
            ax.collections.remove(collection)
        contours = ax.contour(
            X, Y, S, levels=[0], colors=["k"], linewidths=[5], zorder=2
        )

        fig.canvas.draw()
        fig.canvas.flush_events()
        it += 1


if __name__ == "__main__":

    fdef = r"""
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i,
            const T& level, const T& t) {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        return sin(x*x - y*y + t) * cos(2*x+x*y/4 - t) - level;
    }
}
    """

    level = 0.2

    pmb = ParallelMarchingBlocks(
        2,
        fdef,
        L=3,
        N=128,
        backend=Backend.CUPY,
        extra_params=[("level", "T"), ("t", "T")],
        export_scalar_field=True,
        enable_cuda_std_lib=True,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    animate_plot(pmb, fig, ax, level)
