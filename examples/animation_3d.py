import os
import open3d

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

from cupmb import ParallelMarchingBlocks


if __name__ == "__main__":

    fdef = """
    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz,
            T r) {
        return (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5) - r*r;
    }
    """

    pmb = ParallelMarchingBlocks(3, fdef, extra_scalar_params=["T"])
    T = pmb.ftype

    rmin, rmax = 0.1, 0.5
    Nr = 1000
    dr = (rmax - rmin) / Nr

    (V, I) = pmb(T(rmax), to_numpy=True)

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
    vis.update_renderer()

    frame = 0

    running = True
    while running:
        running = vis.poll_events()

        if frame <= Nr:
            r = rmin + frame * dr
        else:
            r = rmin + (2 * Nr - frame) * dr

        (V, I) = pmb(T(r), to_numpy=True)

        mesh.vertices = open3d.utility.Vector3dVector(V)
        mesh.triangles = open3d.utility.Vector3iVector(I)

        vis.update_geometry(mesh)
        vis.update_renderer()

        frame += 1
        frame = frame % (2 * Nr)

    vis.destroy_window()
