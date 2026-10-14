#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
import os, sys
import open3d
import numpy as np

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, Backend, logger

if __name__ == "__main__":

    fdef = """
namespace cupmb {
    template <typename T, typename I, I dim>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i,
            const T& r) {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        return (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5) - r*r;
    }
}
    """

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        backend=Backend.CUPY,
        extra_params=[("r", "T")],
        enable_cuda_std_lib=True,
    )

    V, I = pmb(r=0.5, to_numpy=True)

    if (V is None) or (I is None):
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit(1)

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64))
    mesh.triangles = open3d.utility.Vector3iVector(np.asarray(I, dtype=np.int32))

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    render_options = vis.get_render_option()
    render_options.mesh_show_back_face = True
    render_options.mesh_show_wireframe = True
    render_options.show_coordinate_frame = True

    vis.add_geometry(mesh)
    vis.add_geometry(pmb.bbox.as_open3d)
    vis.update_renderer()

    frame = 0
    params = {"r": (0.0, 0.75, 1000)}

    running = True
    geometry_removed = False
    while running:
        running = vis.poll_events()

        # kwds contains dynamic parameter values
        kwds = {}
        for pname, (minv, maxv, Nf) in params.items():
            dv = (maxv - minv) / Nf
            N = frame % (2 * Nf)
            if N <= Nf:
                pval = minv + N * dv
            else:
                pval = minv + (2 * Nf - N) * dv
            kwds[pname] = pval

        V, I = pmb(**kwds, to_numpy=True)

        if (V is None) or (I is None):
            if not geometry_removed:
                vis.remove_geometry(mesh, False)
                geometry_removed = True
        else:
            mesh.vertices = open3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64))
            mesh.triangles = open3d.utility.Vector3iVector(np.asarray(I, dtype=np.int32))
            if geometry_removed:
                vis.add_geometry(mesh, False)
                geometry_removed = False
            vis.update_geometry(mesh)

        vis.update_renderer()

        frame += 1

    vis.destroy_window()
