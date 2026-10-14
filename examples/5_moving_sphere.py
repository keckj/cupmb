#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, sys, collections
import open3d
import numpy as np

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, ExtraParameter, Backend, logger


class PingPong:
    def __init__(self, pmin, pmax, freq):
        assert freq > 0, freq
        self.pmin, self.pmax, self.freq, self.frame = (pmin, pmax, freq, -1)

    def __call__(self):
        self.frame += 1
        dp = (self.pmax - self.pmin) / self.freq
        N = self.frame % (2 * self.freq)
        if N <= self.freq:
            val = self.pmin + N * dp
        else:
            val = self.pmin + (2 * self.freq - N) * dp
        return val


if __name__ == "__main__":

    fdef = """
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i,
            const T& x0, const T& y0, const T& z0) {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        return (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0) - 0.25*0.25;
    }
}
    """

    extra_params = collections.OrderedDict()
    extra_params["x0"] = ExtraParameter("T", PingPong(0.25, 0.75, 113))
    extra_params["y0"] = ExtraParameter("T", PingPong(0.25, 0.75, 157))
    extra_params["z0"] = ExtraParameter("T", PingPong(0.25, 0.75, 193))

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        backend=Backend.CUPY,
        extra_params=extra_params,
        enable_cuda_std_lib=True,
    )
    bbox = pmb.bbox.as_open3d

    # Callables are not called when parameters are given
    # This is usefull for buffer initialization
    V, I = pmb(x0=0.5, y0=0.5, z0=0.5, to_numpy=True)

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
    vis.add_geometry(bbox)
    vis.update_renderer()

    running = True
    geometry_removed = False
    while running:
        running = vis.poll_events()

        # Call the callables for all missing extra parameters
        V, I = pmb(to_numpy=True)

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

    vis.destroy_window()
