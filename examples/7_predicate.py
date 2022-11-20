#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

from cupmb import (
    ParallelMarchingBlocks,
    Backend,
    AllocationPolicy,
    AutotuningRigor,
    logger,
)
from cupmb.utils.units import time2str

if __name__ == "__main__":

    fdef = r"""
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i) {

        const auto& [x, y, z] = p;

        const T xx = x*x;
        const T yy = y*y;
        const T zz = z*z;

        const T c = T(0.85) * T(0.85);
        const T o = T(1);

        const T f = ((xx + yy - c)*(xx + yy - c) + (zz - o)*(zz - o))
                  * ((yy + zz - c)*(yy + zz - c) + (xx - o)*(xx - o))
                  * ((zz + xx - c)*(zz + xx - c) + (yy - o)*(yy - o)) - T(0.01);

        return f;
    }
}
    """

    pdef = r"""
#include <cusvna/aa.cuh>

namespace cupmb {
    template <typename T, typename I, I dim>
    __device__ __forceinline__ bool may_contain_zeros(const Box<T,dim>& domain) {
        using Interval = cusvna::Interval<T>;
        using AA = cusvna::AA<T>;

        const auto x = AA::from_interval<0>(Interval(domain.a[0], domain.b[0]));
        const auto y = AA::from_interval<1>(Interval(domain.a[1], domain.b[1]));
        const auto z = AA::from_interval<2>(Interval(domain.a[2], domain.b[2]));

        const auto xx = x*x;
        const auto yy = y*y;
        const auto zz = z*z;

        const T c = T(0.85) * T(0.85);
        const T o = T(1);

        const auto f = ((xx + yy - c)*(xx + yy - c) + (zz - o)*(zz - o))
                     * ((yy + zz - c)*(yy + zz - c) + (xx - o)*(xx - o))
                     * ((zz + xx - c)*(zz + xx - c) + (yy - o)*(yy - o)) - T(0.01);

        return (f.sign() == cusvna::Sign::UNKNOWN);
    }
}
    """

    N = 4096
    logger.info(
        f"Benchmarking naive block mask extraction vs predicate based block extraction for N={N}."
    )

    import numpy as np

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        x0=-1.2,
        L=2.4,
        N=N,
        ftype=np.float32,
        backend=Backend.CUPY,
        default_allocation_policy=AllocationPolicy.ALLOCATE_EXACT_SIZE,
        autotuning_rigor=AutotuningRigor.EXHAUSTIVE,
        enable_cuda_std_lib=True,
    )

    pmb()  # preallocate buffers
    t0 = pmb(return_elapsed_time=True)[-1]

    pmb.pdef = pdef

    pmb()  # preallocate buffers
    t1 = pmb(return_elapsed_time=True)[-1]

    logger.info(f"Kernel launcher finished in {time2str(t0)} without predicate.")
    logger.info(f"Kernel launcher finished in {time2str(t1)} with predicate.")
    logger.info(f"Speedup factor is {t0/t1:2.1f}.")

    logger.info("Rendering mesh.")
    import open3d

    (V, I) = pmb(to_numpy=True)

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(V)
    mesh.triangles = open3d.utility.Vector3iVector(I)

    bbox = pmb.bbox.as_open3d

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    render_options = vis.get_render_option()
    render_options.mesh_show_back_face = True
    render_options.mesh_show_wireframe = True
    render_options.show_coordinate_frame = True

    vis.add_geometry(mesh)
    vis.add_geometry(bbox)
    vis.update_renderer()

    running = True
    while running:
        running = vis.poll_events()
