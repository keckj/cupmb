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
import open3d
import numpy as np

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

from cupmb import ParallelMarchingBlocks, Backend, logger


if __name__ == "__main__":

    fdef = r"""
    #include "csg.cuh"


    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz) {
        const Array<float,2> p(x,y);
        constexpr Box<float,2>      box1(7.,5.);
        constexpr Box<float,2>      box2(2.,10.);
        constexpr Parabola<float,2> par1(0.3);
        constexpr Parabola<float,2> par2(0.3);
        const auto& sdf = (translate(par1,0,-8)); //- (box2 - translate(par2,0,-2));
        return sdf.f(p);
    }
    """

    cuda_source_dir = "/home/keckj/Documents/poctix"
    extra_build_options = (f"-I{cuda_source_dir}",)

    pmb = ParallelMarchingBlocks(
        3,
        fdef,
        backend=Backend.CUPY,
        extra_build_options=extra_build_options,
        x0=-8,
        L=16,
        N=64,
    )
    T = pmb.ftype

    (V, I) = pmb(to_numpy=True)

    if V is None:
        msg = "Could not find any vertex, this is a bug."
        logger.error(msg)
        sys.exit()

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(V)
    mesh.triangles = open3d.utility.Vector3iVector(I)

    bbox = open3d.geometry.AxisAlignedBoundingBox.create_from_points(
        open3d.utility.Vector3dVector(np.vstack([pmb.xmin, pmb.xmax])),
    )

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    render_options = vis.get_render_option()
    render_options.mesh_show_back_face = True
    render_options.mesh_show_wireframe = True
    render_options.show_coordinate_frame = True

    vis.add_geometry(mesh)
    vis.add_geometry(bbox)
    vis.update_renderer()
    vis.run()

    frame = 0

    running = True
    while running:
        running = vis.poll_events()

        (V, I) = pmb(to_numpy=True)

        mesh.vertices = open3d.utility.Vector3dVector(V)
        mesh.triangles = open3d.utility.Vector3iVector(I)

        vis.update_geometry(mesh)
        vis.update_renderer()

        frame += 1

    vis.destroy_window()
