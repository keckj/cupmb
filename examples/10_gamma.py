#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#

# One-dimensional example: root finding on a very fine grid with a block predicate.
#
# The gamma function has no zero, we look instead for the solutions of Gamma(x) = exp(k*pi)
# for x > 0 and any integer k, that is the zeros of f(x) = sin(lgamma(x)). Among them x=1 and
# x=2 (Gamma=1) are exact. The other ones are computed on the host by bisection to check the results.
#
# The default grid has 2^32 cells on [2^-8, 10^6] (64 bit indices), it contains about 4 million roots.
# The roots are sparse on such a grid: the predicate, evaluated with the affine arithmetic
# library cusvna (https://github.com/keckj/cusvna, lgamma is only defined for x > 0), discards almost
# all the blocks of the binary tree before any scalar is evaluated.
#
import os
import math
import argparse

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

import numpy as np

from cupmb import ParallelMarchingBlocks, AllocationPolicy, logger
from cupmb.utils.units import time2str

# Scalar field definition (fdef): a templated function S f<T>(S x) where S is either T (float/double)
# or an affine form, lgamma and sin are found by argument dependent lookup for affine forms.
fdef = r"""
namespace cupmb {
    template <typename T, typename S>
    __device__ __forceinline__ S f(const S& x) {
        return sin(lgamma(x));
    }

    template <typename T, typename I, I dim>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i) {
        return f<T>(trove::get<0>(p));
    }
}
"""

# Predicate definition (pdef): returns false only if the block [a, b] certainly does not contain any zero of f.
pdef = r"""
#include <cusvna/af1.cuh>

namespace cupmb {
    template <typename T, typename I, I dim>
    __device__ __forceinline__ bool may_contain_zeros(const BoundingBox<T,dim>& domain) {
        using Interval = cusvna::Interval<T>;
        using AF = cusvna::AF1<T,1>;  /* 1 input noise symbol: x */
        const AF x = AF::template from_interval<0>(Interval(domain.a[0], domain.b[0]));
        return possibly_zero(f<T>(x));
    }
}
"""

XMIN = 1.4616321449683623  # minimum of the gamma function on x > 0, lgamma(XMIN) > -pi


def bisect(g, a, b, iterations=200):
    """Zero of a monotonic function g on [a, b] with g(a) and g(b) of opposite signs."""
    ga = g(a)
    for _ in range(iterations):
        m = 0.5 * (a + b)
        if m in (a, b):
            break
        if (g(m) > 0) == (ga > 0):
            a, ga = m, g(m)
        else:
            b = m
    return 0.5 * (a + b)


XSTIRLING = 10.0  # the Stirling series below is used on [XSTIRLING, +inf)


def lgamma_stirling(x):
    """Vectorized lgamma for x >= XSTIRLING: Stirling series, the truncation error is below 1e-15."""
    y = 1.0 / (x * x)
    series = (1 / 12 - y * (1 / 360 - y * (1 / 1260 - y * (1 / 1680 - y / 1188)))) / x
    return (x - 0.5) * np.log(x) - x + 0.5 * math.log(2 * math.pi) + series


def bisect_vectorized(g, a, b, iterations=64):
    """Zeros of the increasing function g on the brackets [a, b] (arrays), g(a) <= 0 <= g(b)."""
    a, b = a.copy(), b.copy()
    for _ in range(iterations):
        m = 0.5 * (a + b)
        positive = g(m) > 0
        b = np.where(positive, m, b)
        a = np.where(positive, a, m)
    return 0.5 * (a + b)


def reference_roots(x0, x1):
    """Solutions of lgamma(x) = k*pi on [x0, x1] with 0 < x0 < 1 and x1 > XSTIRLING: lgamma decreases on
    (0, XMIN] from +inf to lgamma(XMIN) > -pi and increases on [XMIN, +inf). The roots larger than XSTIRLING
    are all computed at once by a vectorized bisection."""
    roots = []
    for k in range(int(math.lgamma(x0) / math.pi), 0, -1):
        roots.append(bisect(lambda x: math.lgamma(x) - k * math.pi, x0, 1.0))
    roots += [1.0, 2.0]
    k_stirling = int(math.lgamma(XSTIRLING) / math.pi) + 1
    for k in range(1, k_stirling):
        roots.append(bisect(lambda x: math.lgamma(x) - k * math.pi, 2.0, XSTIRLING))
    K = np.arange(k_stirling, int(math.lgamma(x1) / math.pi) + 1) * math.pi
    a, b = np.full_like(K, XSTIRLING), np.full_like(K, x1)
    roots += bisect_vectorized(lambda x: lgamma_stirling(x) - K, a, b).tolist()
    return np.asarray(roots)


def benchmark(pmb, runs):
    pmb()  # preallocate buffers (and autotune)
    V, _ = pmb(to_numpy=True)
    times = [pmb(return_elapsed_time=True)[-1] for _ in range(runs)]
    return float(np.median(times)), np.sort(V[:, 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solutions of Gamma(x) = exp(k*pi) with a block predicate.")
    parser.add_argument("-N", type=int, default=2**32, help="grid size")
    parser.add_argument("--x0", type=float, default=2**-8, help="lower bound of the domain (0 < x0 < 1)")
    parser.add_argument("--x1", type=float, default=1e6, help=f"upper bound of the domain (x1 > {XSTIRLING:g})")
    parser.add_argument("--runs", type=int, default=10, help="number of timed runs")
    parser.add_argument("--no-plot", action="store_true", help="do not display the results with matplotlib")
    args = parser.parse_args()

    x0, x1, N = args.x0, args.x1, args.N
    assert 0 < x0 < 1 < XSTIRLING < x1, f"the domain has to contain [1, {XSTIRLING:g}] and be strictly positive"

    pmb = ParallelMarchingBlocks(
        1,
        fdef,
        x0=x0,
        L=x1 - x0,
        N=N,
        ftype=np.float64,
        itype=np.uint64 if N >= 2**32 else np.uint32,  # vertex indices of the grid go up to N
        default_allocation_policy=AllocationPolicy.ALLOCATE_EXACT_SIZE,
        enable_cuda_std_lib=True,
    )

    t0, X0 = benchmark(pmb, args.runs)
    pmb.pdef = pdef
    t1, X1 = benchmark(pmb, args.runs)

    logger.info(f"Kernel launcher finished in {time2str(t0)} without predicate ({X0.size} roots, N={N}).")
    logger.info(f"Kernel launcher finished in {time2str(t1)} with predicate ({X1.size} roots, N={N}).")
    logger.info(f"Speedup factor is {t0/t1:2.1f}.")

    R = reference_roots(x0, x1)
    if not (X0.size == X1.size == R.size):
        logger.error(f"Root count mismatch: {X0.size} without predicate, {X1.size} with predicate, {R.size} expected.")
        raise SystemExit(1)
    if not np.array_equal(X0, X1):
        logger.error("Roots found with and without predicate differ.")
        raise SystemExit(1)

    dx = pmb.dx[0]
    error = np.abs(X1 - R)
    logger.info(
        f"Maximal root error is {error.max():.3e} for a grid step dx={dx:.3e} (x=1: {error[R == 1.0][0]:.1e}, "
        f"x=2: {error[R == 2.0][0]:.1e})."
    )
    if error.max() > dx:
        logger.error("Roots are not within one grid step of the reference solutions.")
        raise SystemExit(1)

    if not args.no_plot:
        import matplotlib.pyplot as plt

        xmax = min(x1, 10.0)
        x = np.linspace(x0, xmax, 4096)
        lg = np.asarray([math.lgamma(xi) for xi in x])
        roots = X1[X1 <= xmax]

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        axes[0].semilogy(x, np.exp(lg), "k")
        for k in range(-1, int(lg.max() / math.pi) + 2):
            axes[0].axhline(math.exp(k * math.pi), color="0.8", lw=0.5)
        axes[0].semilogy(roots, [math.gamma(r) for r in roots], "r.")
        axes[0].set_ylabel(r"$\Gamma(x)$ and $e^{k\pi}$")
        axes[1].plot(x, np.sin(lg), "k")
        axes[1].plot(roots, np.zeros_like(roots), "r.")
        axes[1].axhline(0, color="0.8", lw=0.5)
        axes[1].set_xlabel("x")
        axes[1].set_xlim(0, xmax)
        axes[1].set_ylabel(r"$\sin(\ln\Gamma(x))$")
        fig.suptitle(rf"cupmb 1D: solutions of $\Gamma(x)=e^{{k\pi}}$ ({X1.size} roots on [{x0:g}, {x1:g}])")
        plt.show()
