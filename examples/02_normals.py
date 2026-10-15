#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
import os, sys, time, threading, collections
import open3d
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
ext_dir = os.path.join(script_dir, "..", "ext")
sys.path.append(ext_dir)
from vis_gui import AppWindow, gui, rendering

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

from cupmb import ParallelMarchingBlocks, Backend, VertexProperty, logger

Color = collections.namedtuple("Color", ["red", "green", "blue", "alpha"])


class ExampleAppWindow(AppWindow):

    fdef = r"""
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

    template <typename T, typename I, I dim>
    __device__ __forceinline__ trove::array<T,dim> compute_normal(
            const trove::array<T,dim>& p,
            const T& r)
    {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        T nx = x-0.5;
        T ny = y-0.5;
        T nz = z-0.5;
        T nn = rnorm3d(nx, ny, nz);
        return trove::make_array<T>(nx*nn, ny*nn, nz*nn);
    }
}
    """

    def __init__(self, width, height, running):
        super().__init__(width, height)
        self._running = running

        normal = VertexProperty("normal", fname="compute_normal", dtype="T", shape=("dim",))

        self._pmb = ParallelMarchingBlocks(
            3,
            self.fdef,
            N=32,
            backend=Backend.CUPY,
            extra_params=[("r", "T")],
            enable_cuda_std_lib=True,
            vertex_properties=(normal,),
        )

        self._animation_params = {"r": (0.0, 0.75, 1000)}

        # The grid size is requested by the GUI thread and applied by the mesh
        # generation thread (pmb must not be modified while it is being called).
        self._grid_lock = threading.Lock()
        self._grid_size = int(self._pmb.N[0])
        self._requested_grid_size = None

        # Reuse the point size slider of the base window as a grid size slider.
        self._point_size.set_limits(4, 128)
        self._point_size.int_value = self._grid_size
        self._point_size.set_on_value_changed(self._on_grid_change)
        self._point_size_label.text = "Grid size"

        self._load_initial_mesh()

    def _get_animation_parameters(self, frame):
        kwds = {}
        for pname, (minv, maxv, Nf) in self._animation_params.items():
            dv = (maxv - minv) / Nf
            N = (frame % (2 * Nf) + (2 * Nf)) % (2 * Nf)
            if N <= Nf:
                pval = minv + N * dv
            else:
                pval = minv + (2 * Nf - N) * dv
            kwds[pname] = pval
        return kwds

    def _on_grid_change(self, size):
        with self._grid_lock:
            self._grid_size = int(max(4, size))
            self._requested_grid_size = self._grid_size

    def _apply_settings(self):
        super()._apply_settings()
        # the base class resets the slider to the material point size
        # (also called by the base class constructor, before _grid_size exists)
        if hasattr(self, "_grid_size"):
            self._point_size.int_value = self._grid_size

    def _on_menu_open(self):
        logger.info("Cannot open external mesh sorry !")

    def _load_initial_mesh(self):
        V, I, N = self._pmb(**self._get_animation_parameters(-1), to_numpy=True)

        if (V is None) or (I is None):
            msg = "Could not find any vertex, this is a bug."
            logger.error(msg)
            raise RuntimeError(msg)

        bbox = open3d.geometry.AxisAlignedBoundingBox.create_from_points(
            open3d.utility.Vector3dVector(np.asarray([self._pmb.xmin, self._pmb.xmax], dtype=np.float64)),
        )

        mesh = open3d.geometry.TriangleMesh()

        mesh.vertices = open3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64))
        mesh.triangles = open3d.utility.Vector3iVector(np.asarray(I, dtype=np.int32))
        mesh.vertex_normals = open3d.utility.Vector3dVector(np.asarray(N, dtype=np.float64))

        bbox_material = rendering.MaterialRecord()

        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry(
            "mesh",
            mesh,
            self.settings.material,
        )
        self._scene.scene.add_geometry(
            "bbox",
            bbox,
            bbox_material,
        )

        self._scene.setup_camera(60, bbox, bbox.get_center())
        self._on_material_color(Color(0.0, 0.6, 0.2, 1.0))
        self._on_use_sun(False)

        self._thread = threading.Thread(target=self._generate_mesh)
        self._thread.start()

    def _generate_mesh(self):
        frame = 0
        self._can_update = True
        while self._running[0]:
            with self._grid_lock:
                grid_size, self._requested_grid_size = (
                    self._requested_grid_size,
                    None,
                )
            if grid_size is not None:
                self._pmb.N = grid_size

            V, I, N = self._pmb(**self._get_animation_parameters(frame), to_numpy=True)

            if (V is None) or (I is None):
                mesh = None
            else:
                mesh = open3d.geometry.TriangleMesh()
                mesh.vertices = open3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64))
                mesh.triangles = open3d.utility.Vector3iVector(np.asarray(I, dtype=np.int32))
                mesh.vertex_normals = open3d.utility.Vector3dVector(np.asarray(N, dtype=np.float64))
            self._mesh = mesh

            while self._running[0] and not self._can_update:
                time.sleep(0.001)
            self._can_update = False

            gui.Application.instance.post_to_main_thread(self.window, self.update_geometries)

            frame += 1

    def update_geometries(self):
        try:
            if self._scene.scene.has_geometry("mesh"):
                self._scene.scene.remove_geometry("mesh")
            if self._mesh is not None:
                self._scene.scene.add_geometry(
                    "mesh",
                    self._mesh,
                    self.settings.material,
                )
        except Exception as e:
            logger.exception(e)

        self._can_update = True


def main():
    gui.Application.instance.initialize()

    # dirty trick to know when to exit thread...
    running = [True]

    w = ExampleAppWindow(1024, 768, running)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error", "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()

    running[0] = False


if __name__ == "__main__":
    main()
