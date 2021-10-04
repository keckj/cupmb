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
    import pycuda
    import pycuda.gl.autoinit

    max_vertices = 10 * 1024 * 1024
    max_triangles = 10 * 1024 * 1024

    ftype = np.float32
    itype = np.uint32

    vertex_init = np.zeros(shape=(max_vertices, 3), dtype=ftype)
    vertex_bytes = vertex_init.nbytes

    index_init = np.zeros(shape=(max_triangles, 3), dtype=itype)
    index_bytes = index_init.nbytes

    flag = pycuda.gl.graphics_map_flags.WRITE_DISCARD

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
    shared_vertex_buffer = pycuda.gl.RegisteredBuffer(int(VBO), flag)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_bytes, None, GL_DYNAMIC_DRAW)
    shared_index_buffer = pycuda.gl.RegisteredBuffer(int(EBO), flag)

    pmb = ParallelMarchingBlocks(
        3,
        CUPMB_SCALAR,
        ftype=ftype,
        itype=itype,
        x0=-0.5,
        N=32,
        extra_params={"r": "T"},
        backend=Backend.PYCUDA,
        initialize_context=False,
    )

    def as_pycuda_buffer(mapped_buffer, backend=pmb.backend):
        return backend.Buffer(*mapped_buffer.device_ptr_and_size())

    vertex_mapping = shared_vertex_buffer.map()
    index_mapping = shared_index_buffer.map()
    mapped_vertex_buffer = as_pycuda_buffer(vertex_mapping)
    mapped_index_buffer = as_pycuda_buffer(index_mapping)
    assert mapped_vertex_buffer.size == vertex_bytes
    assert mapped_index_buffer.size == index_bytes

    pmb.vertex_buffer = mapped_vertex_buffer
    pmb.simplex_buffer = mapped_index_buffer
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

    vertex_mapping.unmap()
    index_mapping.unmap()

    def update_buffers(
        t,
        shared_index_buffer=shared_index_buffer,
        shared_vertex_buffer=shared_vertex_buffer,
        as_pycuda_buffer=as_pycuda_buffer,
        context=pycuda.gl.autoinit.context,
        pmb=pmb,
    ):
        vertex_mapping = shared_vertex_buffer.map()
        index_mapping = shared_index_buffer.map()
        mapped_vertex_buffer = as_pycuda_buffer(vertex_mapping)
        mapped_index_buffer = as_pycuda_buffer(index_mapping)
        pmb.replace_buffer(BufferType.VERTICES, mapped_vertex_buffer)
        pmb.replace_buffer(BufferType.SIMPLICES, mapped_index_buffer)

        r = (t / 10.0) % 1.0
        r = (1 - r) if r > 0.5 else r

        (V, I) = pmb(r=r)

        vertex_mapping.unmap()
        index_mapping.unmap()

        Ntriangles = 0 if (I is None) else I.shape[0]
        assert Ntriangles <= max_triangles, (Ntriangles, max_triangles)

        return Ntriangles

    return VBO, EBO, update_buffers


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

    VBO, EBO, update_buffers = setup_buffers()

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

    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
