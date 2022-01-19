#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
import os, sys, ctypes
from cuda import cudart

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

from cupmb import ParallelMarchingBlocks, BufferType, Backend

import numpy as np

import pyrr
import glfw
from OpenGL.GL import *  # noqa F403
import OpenGL.GL.shaders

VERTEX_SHADER = """
#version 330

in vec3 position;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(position, 1.0f);
}
"""


FRAGMENT_SHADER = """
#version 330

out vec4 outColor;

void main() {
    outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
"""

CUPMB_SCALAR = """
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i,
            const T& r) {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        return x*x + y*y + z*z - r*r;
    }
}
"""


def setup_buffers():
    max_vertices = 10 * 1024 * 1024
    max_triangles = 10 * 1024 * 1024

    ftype = np.float32
    itype = np.uint32

    vertex_bytes = 3 * max_vertices * ftype().nbytes
    index_bytes = 3 * max_triangles * itype().nbytes

    pmb = ParallelMarchingBlocks(
        3,
        CUPMB_SCALAR,
        ftype=ftype,
        itype=itype,
        x0=-0.5,
        N=32,
        extra_params=[("r", "T")],
        backend=Backend.CUPY,
    )

    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
    shared_vertex_buffer = pmb.backend.CudaOpenGLMappedBuffer(VBO, flags)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_bytes, None, GL_DYNAMIC_DRAW)
    shared_index_buffer = pmb.backend.CudaOpenGLMappedBuffer(EBO, flags)

    pmb.vertex_buffer = shared_vertex_buffer.map()
    pmb.simplex_buffer = shared_index_buffer.map()
    pmb.lock_buffer(BufferType.VERTICES)
    pmb.lock_buffer(BufferType.SIMPLICES)

    assert pmb.allocated_vertex_count == max_vertices, (
        pmb.allocated_vertex_count,
        max_vertices,
    )
    assert pmb.allocated_simplices_count == max_triangles, (
        pmb.allocated_vertex_index_count,
        max_indices,
    )

    pmb.setup(0.5)

    shared_vertex_buffer.unmap()
    shared_index_buffer.unmap()

    def update_buffers(
        t,
        shared_index_buffer=shared_index_buffer,
        shared_vertex_buffer=shared_vertex_buffer,
        pmb=pmb,
    ):
        r = (t / 10.0) % 1.0
        r = (1 - r) if r > 0.5 else r

        pmb.replace_buffer(BufferType.VERTICES, shared_vertex_buffer.map())
        pmb.replace_buffer(BufferType.SIMPLICES, shared_index_buffer.map())

        (V, I) = pmb(r=r)

        shared_vertex_buffer.unmap()
        shared_index_buffer.unmap()

        Ntriangles = 0 if (I is None) else I.shape[0]
        assert Ntriangles <= max_triangles, (Ntriangles, max_triangles)

        return Ntriangles

    def unregister_buffers(
        shared_index_buffer=shared_index_buffer,
        shared_vertex_buffer=shared_vertex_buffer,
    ):
        shared_vertex_buffer.unregister()
        shared_index_buffer.unregister()

    return VBO, EBO, update_buffers, unregister_buffers


def main():
    if not glfw.init():
        return
    title = "Cupmb Cuda/OpenGL interop example"
    window = glfw.create_window(800, 800, title, None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )

    VBO, EBO, update_buffers, unregister_buffers = setup_buffers()

    position = glGetAttribLocation(shader, "position")

    glUseProgram(shader)
    glEnable(GL_DEPTH_TEST)

    fps = 0
    nframes = 0
    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        t = glfw.get_time()
        Ntriangles = update_buffers(t)

        dt = t - last_time
        if dt >= 1.0:
            fps = nframes / dt
            last_time = t
            nframes = 0

        width, height = glfw.get_window_size(window)
        glViewport(0, 0, width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        rot_x = pyrr.Matrix44.from_x_rotation(0)
        rot_y = pyrr.Matrix44.from_y_rotation(t)

        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glDrawElements(GL_TRIANGLES, int(3 * Ntriangles), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()
        glfw.set_window_title(window, f"{title} ({fps:.1f} fps)")
        nframes += 1

    unregister_buffers()

    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
