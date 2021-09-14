#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import numpy as np
import os, sys, time, threading
import open3d

script_dir = os.path.dirname(os.path.realpath(__file__))
ext_dir = os.path.join(script_dir, "ext")
sys.path.append(ext_dir)
from vis_gui import AppWindow, gui, rendering

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

from cupmb import ParallelMarchingBlocks, Backend, logger


class ExampleAppWindow(AppWindow):

    fdef = """
    template <typename T, typename I, I Nx, I Ny, I Nz>
    __device__ __forceinline__ T scalar(
            const T& x,  const T& y,  const T& z,
            const I& ix, const I& iy, const I& iz,
            T r) {
        return (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5) - r*r;
    }
    """

    def __init__(self, width, height, running):
        super().__init__(width, height)
        self._running = running

        self._pmb = ParallelMarchingBlocks(
            3, self.fdef, extra_params={"r": "T"}, backend=Backend.CUPY
        )

        self._point_size.set_limits(4, 128)
        self._point_size.int_value = self._pmb.N[0]
        self._point_size.set_on_value_changed(self._on_grid_change)

        point_size_label = (
            self._settings_panel.get_children()[-1]
            .get_children()[-1]
            .get_children()[-2]
        )
        point_size_label.text = "Grid size"

        self._load_initial_mesh()

    def _on_grid_change(self, size):
        self._pmb.N = int(size)

    def _on_menu_open(self):
        logger.info("Cannot open external mesh sorry !")

    def _load_initial_mesh(self):
        (V, I) = self._pmb(0.5, to_numpy=True)
        N = np.empty_like(V)
        N[:, 0] = (V[:, 0] - 0.5) / 0.5
        N[:, 1] = (V[:, 1] - 0.5) / 0.5
        N[:, 2] = (V[:, 2] - 0.5) / 0.5

        if V is None:
            msg = "Could not find any vertex, this is a bug."
            logger.error(msg)
            gui.Application.instance.quit()

        bbox = open3d.geometry.AxisAlignedBoundingBox.create_from_points(
            open3d.utility.Vector3dVector([self._pmb.xmin, self._pmb.xmax]),
        )

        mesh = open3d.geometry.TriangleMesh()

        mesh.vertices = open3d.utility.Vector3dVector(V)
        mesh.triangles = open3d.utility.Vector3iVector(I)
        mesh.vertex_normals = open3d.utility.Vector3dVector(N)

        bbox_material = rendering.Material()

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

        bounds = mesh.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

        self._thread = threading.Thread(target=self._generate_mesh)
        self._thread.start()

    def _generate_mesh(self):
        rmin, rmax = 0.0, 0.5
        Nr = 1000
        dr = (rmax - rmin) / Nr

        frame = 0

        self._can_update = True

        while self._running[0]:
            if frame <= Nr:
                r = rmin + frame * dr
            else:
                r = rmin + (2 * Nr - frame) * dr

            (V, I) = self._pmb(r, to_numpy=False)

            if (V is None) or (I is None):
                mesh = None
            else:
                N = 2 * (V - 0.5)
                N /= (((N * N).sum(axis=-1)) ** 0.5)[:, None]
                (V, I, N) = map(self._pmb.backend.asnumpy, (V, I, N))
                mesh = open3d.geometry.TriangleMesh()
                mesh.vertices = open3d.utility.Vector3dVector(V)
                mesh.triangles = open3d.utility.Vector3iVector(I)
                mesh.vertex_normals = open3d.utility.Vector3dVector(N)
            self._mesh = mesh

            while self._running[0] and not self._can_update:
                time.sleep(0.001)
            self._can_update = False

            gui.Application.instance.post_to_main_thread(
                self.window, self.update_geometries
            )

            frame += 1
            frame = frame % (2 * Nr)

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
