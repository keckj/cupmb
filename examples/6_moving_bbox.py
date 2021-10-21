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

from cupmb import (
    ParallelMarchingBlocks,
    AllocationPolicy,
    ExtraParameter,
    Backend,
    logger,
)


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
        const auto& [x, y, z] = p;
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
        N=32,
        backend=Backend.CUPY,
        extra_params=extra_params,
        enable_cuda_std_lib=True,
        default_allocation_policy=AllocationPolicy.ALLOCATE_EXACT_SIZE,
    )

    scene_bbox = pmb.bbox.as_open3d
    moving_bbox = pmb.bbox.as_open3d
    moving_bbox.color = (1, 0, 0)

    def update_bbox(bbox, x0, y0, z0, moving_bbox=moving_bbox):
        center = np.asarray([x0, y0, z0], dtype=bbox.dtype)
        bbox.xmin = center - 0.25
        bbox.xmax = center + 0.25

        moving_bbox.min_bound = bbox.xmin
        moving_bbox.max_bound = bbox.xmax

    pmb.update_bbox = update_bbox

    # Callables are not called when parameters are explicitely given
    # This can be used for buffer initialization during setup (or first call)
    (V, I) = pmb(x0=0.5, y0=0.5, z0=0.5, to_numpy=True)

    # All buffers can be locked to check that they are never reallocated
    pmb.lock_all_buffers()

    if (V is None) or (I is None):
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit()

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(V)
    mesh.triangles = open3d.utility.Vector3iVector(I)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    render_options = vis.get_render_option()
    render_options.mesh_show_back_face = True
    render_options.mesh_show_wireframe = True
    render_options.show_coordinate_frame = True

    vis.add_geometry(mesh)
    vis.add_geometry(scene_bbox)
    vis.add_geometry(moving_bbox)
    vis.update_renderer()

    running = True
    geometry_removed = False
    while running:
        running = vis.poll_events()

        # Call the callables for all missing extra parameters
        (V, I) = pmb(to_numpy=True)

        if (V is None) or (I is None):
            if not geometry_removed:
                vis.remove_geometry(mesh, False)
                geometry_removed = True
        else:
            mesh.vertices = open3d.utility.Vector3dVector(V)
            mesh.triangles = open3d.utility.Vector3iVector(I)
            if geometry_removed:
                vis.add_geometry(mesh, False)
                geometry_removed = False
            vis.update_geometry(mesh)

        vis.update_geometry(moving_bbox)
        vis.update_renderer()

    vis.destroy_window()
