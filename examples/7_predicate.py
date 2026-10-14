#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
# Block predicate example: a user defined predicate discards the blocks of the domain
# that certainly do not contain any part of the isosurface, before any scalar is evaluated.
#
# The predicate is evaluated with the affine arithmetic library cusvna (ext/cusvna): the
# shape function below is written once as a template and evaluated both on floating point
# numbers (scalar field) and on affine forms cusvna::AF1<T,3> (predicate). Functions such
# as sqr or sin are found by argument dependent lookup.
#
import os
import argparse

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

import numpy as np

from cupmb import (
    ParallelMarchingBlocks,
    Backend,
    AllocationPolicy,
    AutotuningRigor,
    logger,
)
from cupmb.utils.units import time2str

# Scalar field definition (fdef): a templated shape function S shape<T>(S x, S y, S z)
# where S is either T (float/double) or an affine form.
fdef = r"""
namespace cupmb {
    /* scalar version of sqr, the affine form version cusvna::sqr is found by ADL
     * (and preferred by partial ordering since it is more specialized) */
    template <typename S>
    __device__ __forceinline__ S sqr(const S& x) { return x*x; }

    /* Three interlaced tori, stretched along y and z and perturbed along x so that the
     * shape is not invariant under any axis permutation or reflection. */
    template <typename T, typename S>
    __device__ __forceinline__ S shape(const S& x, const S& y, const S& z) {
        const T c = T(0.85) * T(0.85);
        const T o = T(1);

        const S xx = sqr(x);
        const S yy = sqr(T(1.05) * y);
        const S zz = sqr(T(1.10) * z);

        return (sqr(xx + yy - c) + sqr(zz - o))
             * (sqr(yy + zz - c) + sqr(xx - o))
             * (sqr(zz + xx - c) + sqr(yy - o)) - T(0.01)
             + T(0.05) * sin(T(3) * x);
    }

    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i) {
        return shape<T>(trove::get<0>(p), trove::get<1>(p), trove::get<2>(p));
    }
}
"""

# Predicate definition (pdef): returns false only if the block domain certainly does not
# contain any zero of the shape function. domain.a[k] and domain.b[k] are the lower and upper
# bounds of the block along axis k (0=x, 1=y, 2=z).
pdef = r"""
#include <cusvna/af1.cuh>

namespace cupmb {
    template <typename T, typename I, I dim>
    __device__ __forceinline__ bool may_contain_zeros(const BoundingBox<T,dim>& domain) {
        using Interval = cusvna::Interval<T>;
        using AF = cusvna::AF1<T,3>;  /* 3 input noise symbols: x, y and z */

        const AF x = AF::template from_interval<0>(Interval(domain.a[0], domain.b[0]));
        const AF y = AF::template from_interval<1>(Interval(domain.a[1], domain.b[1]));
        const AF z = AF::template from_interval<2>(Interval(domain.a[2], domain.b[2]));

        return possibly_zero(shape<T>(x, y, z));
    }
}
"""


def benchmark(pmb, runs):
    pmb()  # preallocate buffers (and autotune)
    vertices, triangles, *_ = pmb()
    ntris = int(triangles.shape[0])
    times = [pmb(return_elapsed_time=True)[-1] for _ in range(runs)]
    return float(np.median(times)), ntris


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block predicate benchmark (cusvna affine arithmetic).")
    parser.add_argument("-N", type=int, default=2048, help="grid size, N=2048 requires ~12GB of VRAM")
    parser.add_argument(
        "--rigor",
        default="MEASURE",
        choices=[r.name for r in AutotuningRigor],
        help="autotuning rigor (ESTIMATE: ~8s, MEASURE: ~45s, EXHAUSTIVE: ~85s on a RTX 5090)",
    )
    parser.add_argument("--runs", type=int, default=10, help="number of timed runs")
    args = parser.parse_args()

    N = args.N
    logger.info(f"Benchmarking naive block mask extraction vs predicate based block extraction for N={N}.")

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        x0=-1.2,
        L=2.4,
        N=N,
        ftype=np.float32,
        backend=Backend.CUPY,
        default_allocation_policy=AllocationPolicy.ALLOCATE_EXACT_SIZE,
        autotuning_rigor=AutotuningRigor[args.rigor],
        enable_cuda_std_lib=True,
        # Disable FMA contraction so that the scalar field is evaluated identically by
        # all kernel variants: otherwise the sign of the few scalars that are very close
        # to zero may depend on the kernel, and so does the triangle count.
        extra_build_options=("--fmad=false",),
    )

    t0, n0 = benchmark(pmb, args.runs)

    pmb.pdef = pdef
    t1, n1 = benchmark(pmb, args.runs)

    logger.info(
        f"Kernel launcher finished in {time2str(t0)} without predicate ({n0} triangles, median of {args.runs} runs)."
    )
    logger.info(
        f"Kernel launcher finished in {time2str(t1)} with predicate ({n1} triangles, median of {args.runs} runs)."
    )
    logger.info(f"Speedup factor is {t0/t1:2.1f}.")

    if n0 != n1:
        logger.error(f"Triangle count mismatch: {n0} without predicate vs {n1} with predicate.")
        raise SystemExit(1)
