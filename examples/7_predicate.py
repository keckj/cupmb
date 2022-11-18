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
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import (
    ParallelMarchingBlocks,
    Backend,
    AllocationPolicy,
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
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        const T r(0.5);
        return x*x + y*y + z*z - r*r;
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
        const T r(0.5);

        const auto f = x*x + y*y + z*z - r*r;

        return (f.sign() == cusvna::Sign::UNKNOWN);
    }
}
    """

    N = 4096
    logger.info(
        f"Benchmarking naive block mask extraction vs predicate based block extraction for N={N}."
    )

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        x0=-0.5,
        N=N,
        backend=Backend.CUPY,
        default_allocation_policy=AllocationPolicy.ALLOCATE_UPPER_POW2,
    )

    # import sys
    # import open3d
    # pmb.pdef = pdef
    # (V, I) = pmb(to_numpy=True)
    # mesh = open3d.geometry.TriangleMesh()
    # mesh.vertices = open3d.utility.Vector3dVector(V)
    # mesh.triangles = open3d.utility.Vector3iVector(I)

    # (V, I) = pmb()

    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    # render_options = vis.get_render_option()
    # render_options.mesh_show_back_face = True
    # render_options.mesh_show_wireframe = True
    # render_options.show_coordinate_frame = True

    # vis.add_geometry(mesh)
    # vis.add_geometry(pmb.bbox.as_open3d)
    # vis.update_renderer()

    # running = True
    # while running:
    # running = vis.poll_events()

    # sys.exit(1)

    pmb()
    t0 = pmb(return_elapsed_time=True)[-1]

    pmb.pdef = pdef

    pmb()
    t1 = pmb(return_elapsed_time=True)[-1]

    logger.info(f"Kernel launcher finished in {time2str(t0)} without predicate.")
    logger.info(f"Kernel launcher finished in {time2str(t1)} with predicate.")
    logger.info(f"Speedup factor is {t0/t1:2.1f}.")
